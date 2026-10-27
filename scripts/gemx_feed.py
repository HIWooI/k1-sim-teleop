"""GEM-X real-time worker dumps as the live sender_gemx feeds the robot: offline, deterministic, MuJoCo only.

A dump (k1live/gemx_worker.py --dump [--realtime]) holds, per processed camera frame, the SOMA-77 skeleton in the
camera frame, the VitPose 2D keypoints, the source frame index and (real-time dumps) the time the result was ready.
This module replays it through exactly the live sender's per-frame mapping (k1live/sender_gemx.py main loop):

    soma_det(joints, kp2d, w, h, K) -> JumpGate -> HumanMapper(perception="gemx", arm IK, walking profile, SMPL) -> HumanFrame

only on the frames the worker processed. The input is an every-frame dump (causal, no
drops) replayed with live timing (Rows.retime: each result 34 + U(0, 4) ms after its capture, the event PC's ~30 fps
worker; --realtime dumps made on a shared GPU server run at 8-15 fps with 3 s CPU YOLOX stalls and are not used). The
HumanFrame of row i is captured at k_i / fps and reaches the sim_server at t_done_i + MAP_MS (mapping + publish); an
every-frame dump used without retiming arrives at capture + NONRT_LAT_S.

    rows = load_rows("dumps/full/pexels_10515017.npz").retime()                 # every-frame dump, live timing
    frames = sender_frames(rows, mirror=True, profile="presenter")               # one clip, live settings
    frames = playlist_frames([(rows_a, 2), (rows_b, 2)], minutes=10)             # k1live.rehearse-like playlist
    run = ServerRun("sonic_human", log_dir); run.play(frames); res = run.close()  # SimServer.tick on the sim clock

ServerRun drives k1live.sim_server.SimServer itself (event arm path --arm-mode pd --arm-vmax 12 --arm-amax 150, no
render, sockets never fed): the same arm limiter, walking limiter, heading hold, staleness, fall rule (tilt > 45 deg)
and 3 s reset as the live server, and the same summary.json numbers as k1live.rehearse (arm error vs the IK goal, hand
position error, falls / resets). The hybrid planner runs synchronously with its result applied PLANNER_TICKS later
(the live worker thread's ~100 ms), so a run does not depend on the machine's speed.
"""
from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DT = 0.02
MAP_MS = 6.0            # sender_gemx: soma_det + IK + SMPL + JSON + zmq after the worker's reply (event PC: 5-7 ms)
NONRT_LAT_S = 0.060     # every-frame dumps used as is: capture -> HumanFrame (event PC p50 58 ms)
LIVE_MS, LIVE_JITTER_MS = 34.0, 4.0   # every-frame dump -> live timing: result 34 + U(0, 4) ms after capture (event
                                      # PC: 29.4 frames/s reached the server)
PLANNER_TICKS = 5       # hybrid planner result applied 100 ms after the request (live worker thread)
EVENT_SIM_ARGS = ["--arm-mode", "pd", "--arm-vmax", "12", "--arm-amax", "150"]   # event.yaml arm_path improved
BASE_SIM_ARGS = ["--arm-mode", "student", "--arm-vmax", "6", "--arm-amax", "80"]  # the classic arm path (docs/architecture.md)


@dataclass
class Rows:
    """The processed frames of one dump (index = processing order)."""
    path: str
    k: np.ndarray            # (N,) source frame index
    t_done: Optional[np.ndarray]   # (N,) s since the dump started (frame 0 captured at 0); None = not real time
    J: np.ndarray            # (N, 77, 3) camera frame, m (NaN rows without 3D)
    KP: np.ndarray           # (N, 77, 3) VitPose px + confidence
    valid: np.ndarray        # (N,) 3D result
    K: np.ndarray
    wh: tuple                # image size the worker saw (after --max-h)
    fps: float
    n_video: int             # frames in the video
    meta: dict = field(default_factory=dict)

    @property
    def realtime(self):
        return bool(self.meta.get("realtime")) and self.t_done is not None

    @property
    def seconds(self):
        return self.n_video / self.fps

    def det(self, i):
        from k1live.gemx import soma_det

        if not self.valid[i]:
            return None
        return soma_det(self.J[i], self.KP[i], self.wh[0], self.wh[1], self.K)

    def arrive(self, i):
        """Seconds since the clip start at which row i's HumanFrame reaches the sim_server."""
        if self.realtime:
            return float(self.t_done[i]) + MAP_MS / 1e3
        return float(self.k[i]) / self.fps + NONRT_LAT_S

    def per_frame(self):
        """(n_video,) newest row index whose frame was captured at or before each video frame (-1 = none yet)."""
        return np.searchsorted(self.k, np.arange(self.n_video), side="right") - 1

    def retime(self, ms=LIVE_MS, jitter_ms=LIVE_JITTER_MS, seed=0):
        """An every-frame dump replayed with live timing (same as scripts/eval_hybrid_live.py): each
        frame's result reaches the sender `ms` + U(0, jitter_ms) after its capture - the live worker at ~30 fps
        (event PC: 29.4 frames/s reached the server). A --realtime dump keeps its own completion times."""
        if self.realtime or ms is None or ms <= 0:
            return self
        rng = np.random.default_rng(seed)
        done = self.k / self.fps + (ms + rng.uniform(0, jitter_ms, len(self.k))) / 1e3
        meta = {**self.meta, "realtime": True, "retimed_ms": ms, "retime_jitter_ms": jitter_ms}
        return Rows(self.path, self.k, done, self.J, self.KP, self.valid, self.K, self.wh, self.fps, self.n_video, meta)


def load_rows(path) -> Rows:
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    k = np.asarray(z["source_frame_index"], int) if "source_frame_index" in z.files else np.arange(len(z["joints_incam"]))
    J = np.asarray(z["joints_incam"], np.float64)
    valid = np.asarray(z["valid"], bool) if "valid" in z.files else np.isfinite(J[:, 0, 0])
    t_done = np.asarray(z["t_done"], np.float64) if "t_done" in z.files else None
    n_video = int(meta.get("frames") or (k.max() + 1 if len(k) else 0))
    return Rows(str(path), k, t_done, J, np.asarray(z["kp2d_vitpose"], np.float64), valid,
                np.asarray(z["K"], np.float64), tuple(int(v) for v in z["image_wh"]), float(z["fps"]), n_video, meta)


def find_dump(root, name):
    """<root>/<name>.npz or <root>/<video stem>.npz."""
    root = Path(root)
    for cand in (root / f"{name}.npz", root / f"{Path(str(name)).stem}.npz"):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"no GEM-X dump for {name!r} under {root}")


# ------------------------------------------------------------------------------------------------ sender
class SenderGemx:
    """k1live.sender_gemx's per-frame state (JumpGate + HumanMapper) without the camera, worker and sockets."""

    def __init__(self, mirror=True, profile="presenter", arm_ik="dir", gemx_dist="pelvis", smpl=True, loco=True,
                 presenter_kw=None):
        from k1live.human import HumanMapper, JumpGate

        self.gate = JumpGate()
        self.mapper = HumanMapper(mirror=mirror, loco=loco, loco_profile=profile, smpl=smpl, arm_ik=arm_ik,
                                  perception="gemx", loco_kw={"dist_src": gemx_dist,
                                                              "profile_kw": {"presenter": dict(presenter_kw or {})}})

    def cut(self):
        """The looping fake camera restarted (sender_gemx grab: gate reset, mapper.on_cut, worker FLAG_RESET)."""
        self.gate.reset()
        self.mapper.on_cut()

    def switch(self):
        """CTRL set video: a new source (sender_gemx switch: gate reset, mapper.new_scene, worker FLAG_RESET)."""
        self.gate.reset()
        self.mapper.new_scene()

    def frames(self, rows: Rows, t_off=0.0, t0=0.0, t1=None, seq0=0):
        """HumanFrame dicts of the rows captured in [t0, t1) of the clip, on a timeline where the clip's t0 is t_off.
        Extra keys: _k (source frame), _row, _t_arrive, _found (3D result), _gated, _det (soma_det for drawing)."""
        out = []
        w, h = rows.wh
        t_end = rows.seconds if t1 is None else t1
        for i in range(len(rows.k)):
            tc = rows.k[i] / rows.fps
            if tc < t0 - 1e-9 or tc >= t_end - 1e-9:
                continue
            raw = rows.det(i)
            det = self.gate(raw)
            if self.gate.relocked:
                self.mapper.on_relock()
            f = self.mapper.process(det, t_off + tc - t0, w, h)
            f.pop("obs", None)
            f.update(seq=seq0 + len(out), t_capture=t_off + tc - t0, _t_arrive=t_off + rows.arrive(i) - t0,
                     _k=int(rows.k[i]), _row=i, _found=raw is not None, _gated=raw is not None and det is None,
                     _det=raw)
            out.append(f)
        return out


def sender_frames(rows, t0=0.0, t1=None, **kw):
    """One clip segment through a fresh sender (the source was just switched to it)."""
    return SenderGemx(**kw).frames(rows, 0.0, t0, t1)


def playlist_frames(items, minutes, repeat=2, **kw):
    """k1live.rehearse's fake camera: each clip plays `repeat` times (loop cuts) before the next (CTRL set video),
    cycling until `minutes`. items: list of (name, Rows). -> (frames, timeline [{"t", "clip"}], seconds)."""
    snd = SenderGemx(**kw)
    frames, timeline = [], []
    t, total, ci = 0.0, minutes * 60.0, 0
    while t < total:
        name, rows = items[ci % len(items)]
        if ci:
            snd.switch()
        timeline.append({"t": t, "clip": name})
        for r in range(repeat):
            if r:
                snd.cut()
            frames += snd.frames(rows, t_off=t, seq0=len(frames))
            t += rows.seconds
            if t >= total:
                break
        ci += 1
    return [f for f in frames if f["t_capture"] < total], timeline, total


# ------------------------------------------------------------------------------------------------ server
def _free():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return str(s.getsockname()[1])


def make_server(backend, log_dir, sim_args=None, attract=False, variants=None):
    """k1live.sim_server.SimServer (no render, sockets on unused ports, never fed). Checkpoints: variants
    {"hybrid": dir, "sonic_human": dir} passed to the backend (else $K1LIVE_HYBRID_VARIANT / $K1LIVE_HUMAN_VARIANT, read
    when the backend module is first imported). hybrid: planner synchronous, applied PLANNER_TICKS later."""
    import zmq

    from k1live import sim_server as S

    args = S.parse_args(["--backend", backend, "--no-render", "--log-dir", str(log_dir), "--human-port", _free(),
                         "--ctrl-port", _free(), "--state-port", _free(), "--robot-port", _free()]
                        + ([] if attract else ["--no-attract"]) + list(EVENT_SIM_ARGS if sim_args is None else sim_args))

    var = {k: str(Path(v).resolve()) for k, v in (variants or {}).items() if v}

    class Offline(S.SimServer):
        def backend_kw(self, name):
            kw = super().backend_kw(name)
            if name in var:
                kw = dict(kw, variant_dir=var[name])
            return dict(kw, planner_async=False) if name == "hybrid" else kw

    ctx = zmq.Context()
    st = ctx.socket(zmq.PUB)
    st.setsockopt(zmq.LINGER, 0)
    srv = Offline(args, ctx, st)
    if backend == "hybrid":
        srv.be.kp.stream.latency_ticks = PLANNER_TICKS
    return srv, ctx


class ServerRun:
    """A SimServer ticked on a simulated clock from HumanFrames with arrival times; per-tick record for metrics."""

    T0 = 1000.0

    def __init__(self, backend, log_dir, sim_args=None, attract=False, label=None, variants=None):
        self.backend, self.label = backend, label or backend
        self.srv, self.ctx = make_server(backend, log_dir, sim_args, attract, variants)
        self.srv.t_start = self.srv.t_person = self.T0
        self.i = 0
        self.frames, self.k = [], 0
        self.rec = {k: [] for k in ("t", "track", "eg", "hand", "tilt", "fallen", "xy", "yaw", "pelvis_z", "goal", "q",
                                    "cmd", "person_cmd", "phase")}
        self.fall_t = []

    def load(self, frames):
        """Frames on this run's timeline (t_capture / _t_arrive in s from the run start)."""
        self.frames, self.k = frames, 0

    @property
    def t(self):
        return self.i * DT

    def step(self):
        srv = self.srv
        wall = self.T0 + self.i * DT
        fr = self.frames
        while self.k < len(fr) and fr[self.k]["_t_arrive"] <= self.i * DT + 1e-9:
            f = {k: v for k, v in fr[self.k].items() if not k.startswith("_")}
            f["t_capture"] = self.T0 + f["t_capture"]
            f["_t_recv"] = wall
            srv.hf = f
            srv.n_human += 1
            self.k += 1
        falls0 = srv.falls
        srv.tick(wall)
        be = srv.be
        r = self.rec
        x, y, yaw = be.root_pose()
        r["t"].append(self.i * DT)
        r["track"].append(srv.tracking)
        q = be.arm_q_now()
        r["eg"].append(float(np.degrees(np.mean(np.abs(q - srv.goal)))) if srv.tracking else np.nan)
        r["hand"].append(srv.hand_err(srv.goal) if (srv.tracking and srv.hand_err is not None and self.i % 5 == 0)
                         else np.nan)
        r["tilt"].append(be.tilt_deg())
        r["fallen"].append(srv.fallen)
        r["xy"].append((x, y))
        r["yaw"].append(yaw)
        r["pelvis_z"].append(be.pelvis_height())
        r["goal"].append(np.asarray(srv.goal, float).copy())
        r["q"].append(q)
        r["cmd"].append(np.asarray(srv.cmd, float).copy())
        hf = srv.hf
        lo = hf.get("loco") if (hf is not None and srv.tracking) else None
        r["person_cmd"].append(bool(lo) and max(abs(lo["vx"]), abs(lo["vy"]), abs(lo["wz"])) > 0.02)
        r["phase"].append(str(srv.last_info.get("phase", "")))
        if srv.falls > falls0:
            self.fall_t.append(round(self.i * DT, 2))
        self.i += 1
        if self.i % 50 == 0:      # the live server's 1 Hz summary (fills the histograms behind summary.json)
            srv.summary(wall)

    def play(self, frames, seconds=None, on_tick=None):
        self.load(frames)
        n = int(((frames[-1]["t_capture"] + 1.0) if seconds is None else seconds) / DT)
        for _ in range(n):
            self.step()
            if on_tick is not None:
                on_tick(self)

    def arrays(self):
        return {k: np.asarray(v) for k, v in self.rec.items()}

    def close(self):
        try:
            res = self.srv.close()
        finally:
            self.ctx.destroy(0)
        return res


def window_stats(A, sel):
    """Arm error vs the IK goal (deg) and hand error (cm) over the tracked ticks in sel (bool mask)."""
    m = sel & A["track"].astype(bool)
    eg = A["eg"][m]
    hd = A["hand"][m]
    hd = hd[np.isfinite(hd)]
    return {"track_s": round(float(m.sum() * DT), 1),
            "arm_deg": round(float(np.mean(eg)), 2) if len(eg) else None,
            "arm_p90_deg": round(float(np.percentile(eg, 90)), 2) if len(eg) else None,
            "hand_cm": round(float(np.mean(hd)), 2) if len(hd) else None,
            "hand_p90_cm": round(float(np.percentile(hd, 90)), 2) if len(hd) else None}


def stream_stats(frames):
    """What the sender produced: rows, 3D share, tracked share, rate, latency (arrival - capture)."""
    if not frames:
        return {}
    t = np.array([f["t_capture"] for f in frames])
    lat = np.array([f["_t_arrive"] - f["t_capture"] for f in frames])
    span = max(t[-1] - t[0], 1e-6)
    return {"frames": len(frames), "rate_hz": round(len(frames) / span, 2),
            "found_frac": round(float(np.mean([f["_found"] for f in frames])), 3),
            "present_frac": round(float(np.mean([f["present"] for f in frames])), 3),
            "gated": int(sum(f["_gated"] for f in frames)),
            "smpl_frac": round(float(np.mean([f.get("smpl") is not None for f in frames])), 3),
            "latency_ms_p50": round(float(np.median(lat)) * 1e3, 1),
            "latency_ms_p95": round(float(np.percentile(lat, 95)) * 1e3, 1)}


def env_variant(name, path):
    """The backends read their checkpoint folder from the environment at import: set it before importing them."""
    if path:
        os.environ[name] = str(Path(path).resolve())
