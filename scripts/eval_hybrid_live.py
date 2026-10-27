"""Hybrid live replay: the event chain offline, on a simulated clock (MuJoCo only, CPU; GPU box for the GEM-X dumps).

    GEM-X dumps of a looping video (k1live/gemx_worker.py --dump [--realtime], one dump per loop: every run of the
    worker differs; a dump of every frame is replayed at the live ~30 fps timing) -> the live sender's mapping per
    processed frame at its completion time (k1live.sender_gemx: soma_det -> JumpGate -> HumanMapper(perception="gemx",
    mirror, dir IK, loco presenter), cut on each loop)
    -> k1live.sim_server.SimServer.tick() itself (ArmLimiter 12 / 150, WalkLimiter sonic preset x max_speed, heading
    hold, leash, staleness, attract loop, fall / respawn) -> hybrid (walk4 7000, PD arms; planner synchronous, emulating
    the live worker thread: one inference in flight, each result applied --latency..--latency-max ticks later)
    + the service's auto idle (20 s nobody) / auto calibrate (person 2 s) / auto start (k1live.service._idle_logic).

Every fall writes the ticks around it (phase, hand-over weights, commands, planner state and reference tilt, tilt,
foot contacts, robot self-contacts and their force, arm goal / reference / robot arms) to <out>/<run>_fall<k>.json, a
SETTLE whose tilt reached --near to <run>_near<k>.json; the run summary lists every WALK -> SETTLE episode (walk / settle
length, tilts, feet at the stop, max self-contact force), settle timeouts and max tilt per phase. docs/results.md.

--inject puts a walking intent on the person frames of a video window (one per loop): the live GEM-X stream sometimes
sent one as the person left the frame (event PC), offline dumps do not. --ablate / --hybrid / --heading-hold turn
suspects off one at a time.

    .venv/bin/python scripts/eval_hybrid_live.py --dumps dumps/k1live_rehearsal_full1.npz --loops 8 --runs 4 \
        --inject 86.5:0.6:0:-0.25 88.0:0.6:-0.25:0 --out logs/f/live --json logs/f/live.json \
        [--hybrid settle_arms=person] [--heading-hold 0] [--ablate arms] [--start-xy 3.4,-0.3]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DT = 0.02
WALK4 = ROOT / "models" / "k1" / "k1s_walk4" / "deploy" / "k1s_walk4_007000"
PRE_S, POST_S = 8.0, 1.0      # fall trace window


def parse_kv(items):
    out = {}
    for kv in items:
        k, v = kv.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        out[k] = v
    return out


# ------------------------------------------------------------------------------------------------ sender
class SenderReplay:
    """The live k1live.sender_gemx loop on GEM-X dumps: frame rows in completion order, one dump per video loop."""

    def __init__(self, dumps, loops, profile="presenter", mirror=True, arm_ik="dir", gemx_dist="pelvis",
                 send_ms=5.0, presenter_kw=None, retime=(34.0, 4.0), seed=0, inject=None):
        """retime (ms, jitter ms): a dump made without --realtime (every frame, slower than the camera on a shared GPU)
        is replayed as if each frame finished `ms` (+ uniform jitter) after its capture - the live worker at ~30 fps
        (event PC: 29.4 frames/s reached the server). --realtime dumps keep their own completion times."""
        from k1live.human import HumanMapper, JumpGate

        self.rows = []      # (t_arrive, t_capture, loop, dump index, row)
        self.tracks = [self._load(p) for p in dumps]
        rng = np.random.default_rng(seed)
        for tr in self.tracks:
            if not tr["realtime"] and retime:
                tr["t_done"] = tr["src"] / tr["fps"] + (retime[0] + rng.uniform(0, retime[1], len(tr["src"]))) / 1e3
        self.period = max(tr["n_video"] / tr["fps"] for tr in self.tracks)
        for L in range(loops):
            d = L % len(self.tracks)
            tr = self.tracks[d]
            t0 = L * self.period
            for i in range(len(tr["src"])):
                self.rows.append((t0 + float(tr["t_done"][i]) + send_ms / 1e3, t0 + tr["src"][i] / tr["fps"], L, d, i))
        self.rows.sort(key=lambda r: r[0])
        # inject: [(video t0, seconds, vx, vy)], one per loop (loop L: inject[L % n]) - a walking intent on the frames
        # with a person in that window, as the live sender sent when the person left the frame (event PC)
        self.inject = list(inject or [])
        self.k = 0
        self.loop = -1
        self.gate = JumpGate()
        # the sender starts with its own default profile; the service's first set switches it (as live)
        self.mapper = HumanMapper(mirror=mirror, loco=True, auto_calibrate=True, loco_profile="visitor", smpl=False,
                                  arm_ik=arm_ik, perception="gemx",
                                  loco_kw={"dist_src": gemx_dist, "profile_kw": {"presenter": dict(presenter_kw or {})}})
        self.mapper.set_loco_profile(profile)
        self.seq = 0
        self.last = None    # newest frame dict
        self.n_present = 0

    @staticmethod
    def _load(p):
        z = np.load(p, allow_pickle=True)
        meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
        return {"J": np.asarray(z["joints_incam"], np.float64), "kp": np.asarray(z["kp2d_vitpose"], np.float64),
                "valid": np.asarray(z["valid"], bool), "src": np.asarray(z["source_frame_index"]),
                "t_done": np.asarray(z["t_done"], np.float64), "K": np.asarray(z["K"], np.float64),
                "wh": tuple(int(v) for v in z["image_wh"]), "fps": float(z["fps"]),
                "n_video": int(meta.get("frames", int(np.asarray(z["source_frame_index"]).max()) + 1)), "path": str(p),
                "realtime": bool(meta.get("realtime", True))}

    def calibrate(self):
        self.mapper.request_calibration(explicit=True)

    def until(self, wall):
        """HumanFrame dicts whose sender publish time is <= wall (the newest is what the server keeps)."""
        from k1live.gemx import soma_det

        out = []
        while self.k < len(self.rows) and self.rows[self.k][0] <= wall:
            t_arr, t_cap, L, d, i = self.rows[self.k]
            self.k += 1
            tr = self.tracks[d]
            if L != self.loop:                     # a looping video restarted: a cut (sender grab())
                if self.loop >= 0:
                    self.gate.reset()
                    self.mapper.on_cut()
                self.loop = L
            w, h = tr["wh"]
            det = soma_det(tr["J"][i], tr["kp"][i], w, h, tr["K"]) if tr["valid"][i] else None
            det = self.gate(det)
            if self.gate.relocked:
                self.mapper.on_relock()
            r = self.mapper.process(det, t_cap, w, h)
            if self.inject and r["present"]:
                v0, dur, vx, vy = self.inject[L % len(self.inject)]
                tv = t_cap - L * self.period
                if v0 <= tv < v0 + dur:
                    r["loco"] = {"vx": vx, "vy": vy, "wz": 0.0}
            hf = {"seq": self.seq, "t_capture": t_cap, "present": r["present"], "conf": r["conf"], "arm_q": r["arm_q"],
                  "waist_yaw": r["waist_yaw"], "bow": r["bow"], "root": r["root"], "legs_visible": r["legs_visible"],
                  "loco": r["loco"], "calibrated": r["calibrated"], "t_sent": t_arr, "_t_recv": t_arr,
                  "_loop": L, "_frame": int(tr["src"][i])}
            self.seq += 1
            self.n_present += int(bool(r["present"]))
            self.last = hf
            out.append(hf)
        return out

    @property
    def done(self):
        return self.k >= len(self.rows)


# ------------------------------------------------------------------------------------------------ service
class ServiceLogic:
    """k1live.service auto flow for an unattended fake camera: start following at once; nobody for idle_after_s ->
    stop + mode idle; person present 2 s -> calibrate; sender calibrated (and seen uncalibrated, or 1.5 s) -> mode live
    + start. The server applies the session's set {max_speed, idle_after, walk}."""

    def __init__(self, srv, snd, idle_after_s=20.0, stale_s=3.0):
        self.srv, self.snd = srv, snd
        self.idle_after_s, self.stale_s = idle_after_s, stale_s
        self.running, self.auto_idle = True, False
        self.absent_since = self.present_since = None
        self.calib = None
        self.events = []

    def present(self, now):
        f = self.snd.last
        return bool(f is not None and now - f["t_sent"] < self.stale_s and f["present"])

    def step(self, now):
        srv = self.srv
        if self.present(now):
            self.absent_since = None
            self.present_since = self.present_since or now
        else:
            self.present_since = None
            self.absent_since = self.absent_since or now
        if (not self.auto_idle and self.running and self.absent_since and now - self.absent_since >= self.idle_after_s):
            self.auto_idle, self.running, self.calib = True, False, None
            srv.ctrl("stop")
            srv.ctrl("set", {"mode": "idle"})
            self.events.append((round(now, 2), "auto_idle"))
        if self.auto_idle and self.present_since and now - self.present_since >= 2.0:
            if self.calib is None:
                self.calib = {"t": now, "seen_false": False, "state": "running"}
                self.snd.calibrate()
                self.events.append((round(now, 2), "auto_calibrate"))
            c = self.calib
            if c["state"] == "running":
                cal = bool(self.snd.last and self.snd.last["calibrated"])
                if not cal:
                    c["seen_false"] = True
                if cal and (c["seen_false"] or now - c["t"] > 1.5):
                    c["state"] = "done"
                elif now - c["t"] > 6.0:
                    c["state"] = "failed"
            if c["state"] == "done":
                self.events.append((round(now, 2), "auto_start"))
                self.auto_idle, self.running = False, True
                srv.ctrl("set", {"mode": "live"})
                srv.ctrl("start")


# ------------------------------------------------------------------------------------------------ server
def make_server(cfg, log_dir):
    """k1live.sim_server.SimServer with injected inputs and a simulated clock (no render, sockets unused)."""
    import zmq

    from k1live import protocol as P
    from k1live import sim_server as S

    hyb = dict(cfg.get("hybrid") or {})
    lat = int(cfg.get("latency", 5))

    class Pub:
        def send_multipart(self, *_a, **_k):
            pass

    class ReplayServer(S.SimServer):
        def __init__(self, args):
            self.now = 0.0
            self.inbox, self.ctrl_q = [], []
            super().__init__(args, zmq.Context.instance(), Pub())
            self.t_start = self.t_person = 0.0

        def backend_kw(self, name):
            kw = super().backend_kw(name)
            if name == "hybrid":
                kw.update(planner_async=False, **hyb)
            return kw

        def event(self, ev, **kw):
            self.log.write(json.dumps({"t": round(self.now, 3), "event": ev, **kw}, default=str) + "\n")

        def ctrl(self, cmd, args=None):
            self.ctrl_q.append(P.CtrlMsg(cmd=cmd, args=dict(args or {})))

        def drain(self, wall):
            for d in self.inbox:
                self.hf = d
                self.n_human += 1
            self.inbox = []
            q, self.ctrl_q = self.ctrl_q, []
            for m in q:
                self.on_ctrl(m, wall)

    argv = ["--backend", "hybrid", "--no-render", "--no-autostart", "--idle-after", str(cfg.get("idle_after", 20.0)),
            "--arm-mode", "pd", "--arm-vmax", "12", "--arm-amax", "150", "--log-dir", str(log_dir),
            "--heading-hold", str(cfg.get("heading_hold", 2.0))]
    srv = ReplayServer(S.parse_args(argv))
    if getattr(srv.be, "kp", None) is not None:
        emulate_async(srv.be.kp.stream, lat, cfg.get("latency_max"), cfg.get("seed", 0))
    return srv


def emulate_async(st, lat, lat_max=None, seed=0):
    """The live planner worker thread on the synchronous stream: one inference in flight at a time (a replan due while
    one is pending waits for the next 10 Hz planner tick, k1live.planner.PlannerStream.planner_tick async_) and each
    result applied lat..lat_max ticks after its request (event PC: inference 27-35 ms + G1 -> K1 conversion)."""
    rng = np.random.default_rng(1000 + seed)
    lat_max = lat if lat_max is None else int(lat_max)
    tick0 = st.planner_tick

    def planner_tick():
        if st.pending:
            return
        st.latency_ticks = int(rng.integers(lat, lat_max + 1))
        tick0()

    st.planner_tick = planner_tick
    st.latency_ticks = lat


class Probe:
    """Per-tick record (ring buffer for fall traces + compact arrays for the run summary)."""

    def __init__(self, be):
        import mujoco

        self.mj, self.be = mujoco, be
        m = be.model
        self.feet = [m.body(f"{s}_ankle_roll_link").id for s in ("left", "right")]
        self.floor = m.geom("floor").id
        self.ring = collections.deque(maxlen=int((PRE_S + POST_S) / DT) + 5)
        self.cols = collections.defaultdict(list)

    def contacts(self):
        """Feet on the floor (left, right) + robot self-contacts (count, total normal force N, body names)."""
        d, m = self.be.data, self.be.model
        on = [False, False]
        self.self_n, self.self_f, self.self_pairs = 0, 0.0, set()
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            if c.geom1 != self.floor and c.geom2 != self.floor:
                self.mj.mj_contactForce(m, d, i, f6)
                self.self_n += 1
                self.self_f += float(abs(f6[0]))
                self.self_pairs.add(tuple(sorted((m.body(m.geom_bodyid[c.geom1]).name,
                                                  m.body(m.geom_bodyid[c.geom2]).name))))
                continue
            b = m.geom_bodyid[c.geom2 if c.geom1 == self.floor else c.geom1]
            for k, fb in enumerate(self.feet):
                if b == fb:
                    on[k] = True
        return on

    def log(self, srv, call):
        be, d = self.be, self.be.data
        info = srv.last_info or {}
        x, y, yaw = be.root_pose()
        kp = getattr(be, "kp", None)
        win = getattr(be, "last_window", None)
        ref_yaw = float(np.arctan2(2 * (win.quat[0, 0] * win.quat[0, 3] + win.quat[0, 1] * win.quat[0, 2]),
                                   1 - 2 * (win.quat[0, 2] ** 2 + win.quat[0, 3] ** 2))) if win is not None else np.nan
        st = kp.stream if kp is not None else None
        hf = srv.hf or {}
        rec = {"t": round(srv.now, 3), "phase": info.get("phase"), "wL": info.get("wL"), "wA": info.get("wA"),
               "wP": info.get("wP"), "walk_cmd": [round(float(v), 3) for v in call.get("walk_cmd", (0, 0, 0))],
               "cmd": [round(float(v), 3) for v in srv.cmd], "tracking": srv.tracking, "mode": srv.mode,
               "streaming": srv.streaming, "person": bool(hf.get("present")), "loop": hf.get("_loop"),
               "frame": hf.get("_frame"), "loco": hf.get("loco"),
               "tilt": round(be.tilt_deg(), 2), "z": round(be.pelvis_height(), 3), "xy": [round(x, 3), round(y, 3)],
               "yaw": round(yaw, 4), "v": [round(float(v), 3) for v in d.qvel[:3]],
               "w": [round(float(v), 3) for v in d.qvel[3:6]], "contact": self.contacts(),
               "self_n": self.self_n, "self_f": round(self.self_f, 1),
               "self_pairs": ["/".join(x) for x in sorted(self.self_pairs)][:4],
               "foot_z": [round(float(d.xpos[b][2]), 3) for b in self.feet],
               "goal": [round(float(v), 3) for v in srv.goal], "ref": [round(float(v), 3) for v in call["arm_q"]],
               "arm": [round(float(v), 3) for v in be.arm_q_now()],
               "t_intent": round(getattr(be, "t_intent", 0.0), 2), "t_phase": round(getattr(be, "t_phase", 0.0), 2),
               "t_still": round(getattr(be, "t_still", 0.0), 2)}
        if st is not None:
            rec.update({"pl_mode": int(st.last.mode), "pl_speed": round(float(st.last.speed), 3),
                        "pl_move": [round(float(v), 3) for v in st.last.move[:2]],
                        "pl_face": round(float(kp.face_yaw + kp.yaw0), 4), "pl_cur": int(st.cur), "pl_T": int(st.track.T),
                        "pl_replans": int(st.n_replans), "pl_pending": len(st.pending), "ref_yaw": round(ref_yaw, 4),
                        "ref_pos": [round(float(v), 3) for v in win.pos[0]] if win is not None else None,
                        "ref_tilt": round(float(np.degrees(np.arccos(np.clip(1 - 2 * (win.quat[0, 1] ** 2 + win.quat[0, 2] ** 2),
                                                                             -1, 1)))), 2) if win is not None else None})
        self.ring.append(rec)
        c = self.cols
        c["t"].append(srv.now)
        c["phase"].append({"stand": 0, "walk": 1, "settle": 2}.get(info.get("phase"), -1))
        c["tilt"].append(rec["tilt"])
        c["walk_cmd"].append(rec["walk_cmd"])
        c["cmd"].append(rec["cmd"])
        c["xy"].append(rec["xy"])
        c["yaw"].append(yaw)
        c["tracking"].append(srv.tracking)
        c["person"].append(rec["person"])
        c["goal_jump"].append(float(np.abs(np.asarray(rec["goal"]) - np.asarray(self._prev_goal)).max())
                              if getattr(self, "_prev_goal", None) is not None else 0.0)
        self._prev_goal = rec["goal"]
        c["arm_err"].append(float(np.degrees(np.abs(np.asarray(rec["arm"]) - np.asarray(rec["goal"]))).mean()))
        c["wL"].append(info.get("wL", 0.0))
        c["wA"].append(info.get("wA", 0.0))
        c["ref_tilt"].append(rec.get("ref_tilt") or 0.0)
        c["self_f"].append(rec["self_f"])
        c["contact"].append(rec["contact"])
        c["pl_walk_ticks"].append(int(rec.get("pl_mode", 0) != 0))
        return rec


# ------------------------------------------------------------------------------------------------ one run
def walk_episodes(ph, t, tilt, walk_cmd, ref_tilt=None, contact=None, pl_walk=None, self_f=None):
    """WALK -> SETTLE -> STAND episodes: walk length, settle length, max tilt in settle, timed out; the planner's
    reference pelvis tilt, the feet on the ground when SETTLE starts and how long the planner was asked to walk."""
    eps, i, n = [], 0, len(ph)
    ref_tilt = np.zeros(n) if ref_tilt is None else np.asarray(ref_tilt, float)
    while i < n:
        if ph[i] == 1 and (i == 0 or ph[i - 1] != 1):
            j = i
            while j < n and ph[j] == 1:
                j += 1
            walk_s = (j - i) * DT
            k = j
            while k < n and ph[k] == 2:
                k += 1
            settle = slice(j, k)
            eps.append({"t": round(float(t[i]), 2), "walk_s": round(walk_s, 2), "settle_s": round((k - j) * DT, 2),
                        "tilt_walk": round(float(tilt[i:j].max()), 1) if j > i else None,
                        "tilt_settle": round(float(tilt[settle].max()), 1) if k > j else None,
                        "rewalk": bool(k < n and ph[k] == 1), "cmd_max": round(float(np.abs(walk_cmd[i:j]).max()), 3),
                        "cmd_vx": round(float(walk_cmd[i:j, 0][np.argmax(np.abs(walk_cmd[i:j, 0]))]), 3) if j > i else 0.0,
                        "ref_tilt_settle": round(float(ref_tilt[settle].max()), 1) if k > j else None,
                        "contact_at_settle": "".join("1" if c else "0" for c in contact[j]) if (contact is not None
                                                                                               and j < n) else None,
                        "planner_walk_s": round(float(np.sum(pl_walk[i:j])) * DT, 2) if pl_walk is not None else None,
                        "self_f_max": round(float(np.max(self_f[i:max(k, i + 1)])), 1) if self_f is not None else None})
            i = j
        else:
            i += 1
    return eps


def run_one(cfg):
    out_dir = Path(cfg["out"])
    out_dir.mkdir(parents=True, exist_ok=True)
    name = cfg["name"]
    os.environ["K1LIVE_HYBRID_VARIANT"] = str(cfg.get("variant", WALK4))
    from k1live.backends import hybrid as H

    H.DEFAULT_VARIANT = Path(os.environ["K1LIVE_HYBRID_VARIANT"])
    snd = SenderReplay(cfg["dumps"], cfg["loops"], profile=cfg.get("profile", "presenter"),
                       presenter_kw=cfg.get("presenter_kw"), seed=cfg.get("seed", 0), inject=cfg.get("inject"))
    srv = make_server(cfg, out_dir / f"{name}_sim")
    svc = ServiceLogic(srv, snd, idle_after_s=cfg.get("idle_after", 20.0))
    # the session's first set (k1live.service on start: profile, max_speed, idle_after, mirror, walk) + start
    srv.ctrl("set", {"mode": "live", "backend": "hybrid", "walk": True, "max_speed": cfg.get("max_speed", 0.5),
                     "idle_after": cfg.get("idle_after", 20.0)})
    srv.ctrl("stop")
    srv.ctrl("set", {"mode": "live"})
    srv.ctrl("start")
    be = srv.be
    call = {}
    step0 = be.step
    abl = set(cfg.get("ablate") or [])
    if "leash" in abl:          # no 3 m leash (sim_server.leash): commands reach the backend as the person asked
        from k1live import sim_server as S

        S.leash = lambda cmd, *_a, **_k: cmd
    frozen = {"q": None}

    def step(arm_q, **kw):   # what the server handed the backend this tick (+ ablations)
        arm_q = np.asarray(arm_q, float)
        if "arms" in abl:       # arm goal frozen while not standing: no GEM-X arm jumps during WALK / SETTLE
            if be.phase == "stand" and be.wL <= 0.0:
                frozen["q"] = arm_q.copy()
            elif frozen["q"] is not None:
                arm_q, kw["arm_dq"] = frozen["q"], np.zeros(10)
        call["arm_q"] = arm_q
        call["walk_cmd"] = kw.get("walk_cmd", (0.0, 0.0, 0.0))
        return step0(arm_q, **kw)

    be.step = step
    if cfg.get("start_xy"):   # spawn elsewhere (the leash / arena edge)
        be.reset(*cfg["start_xy"])
    probe = Probe(be)
    T = snd.period * cfg["loops"]
    n = int(T / DT)
    falls, fall_files, near_files = [], [], []
    pending = None
    settle_max, prev_phase = 0.0, "stand"
    near = float(cfg.get("near", 12.0))
    t_wall = time.perf_counter()
    for i in range(n):
        wall = i * DT
        srv.now = wall
        srv.inbox.extend(snd.until(wall))
        if i % 5 == 0:
            svc.step(wall)
        f0 = srv.falls
        srv.tick(wall)
        srv.tick_ms.append(0.0)
        srv.ticks += 1
        rec = probe.log(srv, call)
        if i % 50 == 49:
            srv.summary(wall)
        if srv.falls > f0:
            pending = {"t": round(wall, 2), "i": i, "rec": rec, "k": len(falls)}
            falls.append({"t": round(wall, 2), "tilt": rec["tilt"], "z": rec["z"], "cmd": rec["cmd"],
                          "loop": rec["loop"], "frame": rec["frame"], "xy": rec["xy"]})
        ph = rec["phase"]
        if ph == "settle":
            settle_max = max(settle_max, rec["tilt"])
        elif prev_phase == "settle":   # a SETTLE just ended: keep the ticks of a near-fall (tilt >= --near)
            if near > 0 and settle_max >= near and not srv.fallen and len(near_files) < 30:
                p = out_dir / f"{name}_near{len(near_files)}.json"
                p.write_text(json.dumps({"run": name, "settle_tilt_max": settle_max, "ticks": list(probe.ring)}))
                near_files.append(str(p))
            settle_max = 0.0
        prev_phase = ph
        if pending is not None and i - pending["i"] >= int(POST_S / DT):
            p = out_dir / f"{name}_fall{pending['k']}.json"
            p.write_text(json.dumps({"run": name, "fall": falls[pending["k"]], "ticks": list(probe.ring)}))
            fall_files.append(str(p))
            pending = None
    srv.close()
    C = {k: np.asarray(v) for k, v in probe.cols.items()}
    eps = walk_episodes(C["phase"], C["t"], C["tilt"], C["walk_cmd"], C["ref_tilt"], C["contact"], C["pl_walk_ticks"],
                        C["self_f"])
    short = [e for e in eps if e["walk_s"] < 1.0]
    ph = C["phase"]
    res = {"name": name, "dumps": [Path(p).name for p in cfg["dumps"]], "loops": cfg["loops"], "seconds": round(T, 1),
           "hybrid": cfg.get("hybrid") or {}, "heading_hold": cfg.get("heading_hold", 2.0), "ablate": cfg.get("ablate") or [],
           "start_xy": cfg.get("start_xy"), "inject": cfg.get("inject") or [],
           "latency": [cfg.get("latency", 3), cfg.get("latency_max")],
           "falls": len(falls), "fall_list": falls, "fall_files": fall_files, "near_files": near_files,
           "stats": dict(be.stats),
           "walks": len(eps), "short_walks": len(short), "walk_s_total": round(float((ph == 1).sum() * DT), 1),
           "settle_timeouts": int(be.stats.get("settle_timeouts", 0)),
           "tilt_max": {p: (round(float(C["tilt"][ph == v].max()), 1) if (ph == v).any() else None)
                        for p, v in (("stand", 0), ("walk", 1), ("settle", 2))},
           "settle_tilt_after_short_walk_max": max([e["tilt_settle"] or 0 for e in short], default=None),
           "settle_s_max": max([e["settle_s"] for e in eps], default=None),
           "episodes": eps, "service": svc.events[:200], "present_frac": round(float(C["person"].mean()), 3),
           "tracking_frac": round(float(C["tracking"].mean()), 3), "wall_s": round(time.perf_counter() - t_wall, 1)}
    if cfg.get("trace"):
        np.savez_compressed(out_dir / f"{name}_trace.npz", **{k: v for k, v in C.items()})
    return res


def run_pool(cfgs, workers, json_path):
    res = []
    with ProcessPoolExecutor(workers, max_tasks_per_child=1) as ex:   # one run per process (module patches, ~3 GB)
        for r in ex.map(run_one, cfgs):
            print(json.dumps({k: r[k] for k in ("name", "falls", "walks", "short_walks", "settle_timeouts", "tilt_max",
                                                "settle_tilt_after_short_walk_max", "wall_s")}), flush=True)
            res.append(r)
            if json_path:
                Path(json_path).write_text(json.dumps(res, indent=1, default=float))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dumps", nargs="+", required=True, help="GEM-X --realtime dumps of the same looping video")
    ap.add_argument("--loops", type=int, default=6, help="video loops per run (dump k for loop k mod n)")
    ap.add_argument("--runs", type=int, default=4, help="runs; run r starts at dump r (rotated order)")
    ap.add_argument("--latency", type=int, default=3, help="planner result applied this many ticks after its request")
    ap.add_argument("--latency-max", type=int, default=6, help="... drawn uniformly from latency..latency-max per replan")
    ap.add_argument("--heading-hold", type=float, default=2.0)
    ap.add_argument("--max-speed", type=float, default=0.5, help="event.yaml max_speed (the service's set)")
    ap.add_argument("--hybrid", nargs="*", default=[], metavar="KEY=VALUE", help="HybridBackend constructor options")
    ap.add_argument("--presenter", nargs="*", default=[], metavar="KEY=VALUE", help="presenter walking overrides")
    ap.add_argument("--start-xy", default="", help="x,y spawn (e.g. 3.4,-0.3: next to the 3 m leash)")
    ap.add_argument("--inject", nargs="*", default=[], metavar="T0:SECONDS:VX:VY",
                    help="walking intent injected on the frames with a person, video time T0 for SECONDS, one per loop "
                         "(cycled); run r starts at entry r. e.g. 86:0.6:0:0.25 = 0.6 s sideways as the person leaves")
    ap.add_argument("--ablate", nargs="*", default=[], choices=["arms", "leash"],
                    help="turn a suspect off: arms = arm goal frozen while walking / settling, leash = no 3 m leash")
    ap.add_argument("--variant", default=str(WALK4))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--trace", action="store_true", help="also save the per-tick compact arrays")
    ap.add_argument("--near", type=float, default=12.0, help="save the ticks of a SETTLE whose tilt reached this (deg)")
    ap.add_argument("--tag", default="live")
    ap.add_argument("--out", default=str(ROOT / "logs" / "f" / "live"))
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    dumps = [str(Path(p).resolve()) for p in a.dumps]
    cfgs = []
    for r in range(a.runs):
        order = dumps[r % len(dumps):] + dumps[:r % len(dumps)]
        inj = [tuple(float(x) for x in v.split(":")) for v in a.inject]
        inj = inj[r % len(inj):] + inj[:r % len(inj)] if inj else []
        cfgs.append({"name": f"{a.tag}_r{r}", "dumps": order, "seed": r, "loops": a.loops, "latency": a.latency,
                     "inject": inj,
                     "latency_max": a.latency_max,
                     "heading_hold": a.heading_hold, "max_speed": a.max_speed, "hybrid": parse_kv(a.hybrid),
                     "presenter_kw": parse_kv(a.presenter), "variant": a.variant, "out": a.out, "trace": a.trace, "near": a.near,
                     "ablate": a.ablate,
                     "start_xy": [float(v) for v in a.start_xy.split(",")] if a.start_xy else None})
    res = run_pool(cfgs, a.workers, a.json)
    tot = {"runs": len(res), "seconds": round(sum(r["seconds"] for r in res), 1), "falls": sum(r["falls"] for r in res),
           "walks": sum(r["walks"] for r in res), "short_walks": sum(r["short_walks"] for r in res),
           "settle_timeouts": sum(r["settle_timeouts"] for r in res)}
    print(json.dumps(tot))


if __name__ == "__main__":
    main()
