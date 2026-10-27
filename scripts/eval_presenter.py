"""Presenter (speaker) videos through the live chain, offline and deterministic: which backend / framing works.

At a talk-style event the camera will mostly see a presenter (podium, stage, slides, whiteboard), not a visitor standing
2.5-3.5 m in front of it. Per clip:

  1. perception = what k1live.sender does with each camera frame: frame scaled to <= 720 px high, MediaPipe Pose full
     (sonic/pose_mp.PoseTracker, VIDEO mode, one person, its own gate off) -> k1live.human.JumpGate ->
     k1live.human.HumanMapper (mirror, One-Euro, arm IK, waist, bow, LocoEstimator walking intent) -> HumanFrame.
     The sender keeps up with ~30 fps, so a 60 fps clip is processed every second frame. MediaPipe detections are
     cached (logs/presenter/mp_<clip>_<key>.npz); gate + mapper + estimator are re-run every time (cheap), so
     estimator settings can be compared without MediaPipe (--loco k=v, --loco-profile).
  2. control = k1live.sim_server.SimServer itself on a simulated clock (no sockets used, no render): HumanFrame k
     arrives at t_capture + 35 ms (live p50 latency), 50 Hz ticks -> arm limiter, walking (VelocityLimiter, heading
     hold, leash), hold / relax when the person is lost, fall reset. One run per backend:
       split       student arms + ROBOTIS loco legs (event default)
       sonic       K1 SONIC gesture checkpoint (models/k1/k1s_la4/deploy_002000), standing lower body
       sonic_walk  K1 SONIC walking checkpoint (--walk-variant, folder name contains "walk" -> planner + walking)

Metrics (JSON): tracking (person present / MediaPipe found someone / JumpGate drops / person switches), calibration
(arms: HumanMapper; walking: LocoEstimator home) and hip / leg visibility, walking intent onsets per axis (on a
"stay" clip every onset is a false walk), the robot's arm error vs the IK goal (+ best lag), walk time, net
displacement / turn, falls, max tilt. Clip kinds: stay = the presenter stays at one spot (gestures, turning to the
slides, writing on the board), move = walks on the stage / room.

  .venv/bin/python scripts/eval_presenter.py --json logs/presenter/eval.json            # default clip set (manifest)
  .venv/bin/python scripts/eval_presenter.py --clips p01=<path> --backends split --perception-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

DT = 0.02
LATENCY_S = 0.035          # camera capture -> HumanFrame at the sim_server (live rehearsal p50 ~ 31 ms)
SENDER_FPS = 30.0
MAX_H = 720
EVENT = Path(os.environ.get("K1_EVENT_VIDEOS", str(ROOT / "videos" / "event")))
WALK_VARIANT = ROOT / "models" / "k1" / "k1s_walk" / "deploy" / "k1s_walk_005500"

# Presenter clip set (<event videos>/presenter/, Pexels). type: 1 stage wide, 2 podium, 3 slides / pointing,
# 4 microphone or clicker in hand, 5 whiteboard writing (back to camera), 6 webinar-style upper body, 7 audience heads
# / several people in view. kind: stay | move (see module doc).
MANIFEST = [
    ("p01_stage_mic", "pexels_34917491", "1 4 7", "stay", "stage, full body, handheld mic, audience heads in front"),
    ("p02_seminar_screen", "pexels_8716575", "1 6 7", "stay", "in front of the screen, full body, open gestures"),
    ("p03_slide_point", "pexels_3252973", "1 3 7", "stay", "points at the projected slide, turns to it, audience"),
    ("p04_walk_to_stage", "pexels_8716432", "1 7", "move", "walks from the back of the room to the front, back view"),
    ("p05_mic_walk", "pexels_8244322", "1 4 7", "move", "handheld mic, walks around in front of a panel"),
    ("p06_mic_panel", "pexels_8244304", "4 7", "stay", "handheld mic, standing, seated panel beside"),
    ("p07_podium_close", "pexels_6952134", "2 7", "stay", "podium, waist up, seated panel beside"),
    ("p08_podium_wide", "pexels_6952136", "2 7", "stay", "podium seen wide, seated panel"),
    ("p09_podium_mic", "pexels_8873232", "2 4", "stay", "podium, handheld mic and tablet, turns in at the start"),
    ("p10_podium_gesture", "pexels_6173219", "2 6", "stay", "podium, waist up, large hand gestures"),
    ("p11_whiteboard_talk", "pexels_7648344", "3 7", "stay", "presents at a whiteboard, turns to it, audience"),
    ("p12_pointer_screen", "pexels_9034887", "3 7", "stay", "pointer stick at the screen, audience heads in front"),
    ("p13_flipchart_room", "pexels_7552608", "3", "move", "flip chart in a meeting room, small, turns / steps"),
    ("p14_flipchart_side", "pexels_9034405", "3", "stay", "flip chart, side view, waist up"),
    ("p15_board_writing", "pexels_31657397", "5", "stay", "writes on a whiteboard, side / back to the camera"),
    ("p16_board_back", "pexels_5897643", "5", "stay", "writes on a whiteboard, back to the camera (vertical)"),
    ("p17_webinar_front", "pexels_8048256", "6", "stay", "talks to the camera, chest up"),
    ("p18_explain_vertical", "pexels_8042706", "6", "stay", "explains with hand gestures, thighs up (vertical)"),
    ("p19_seminar_far", "pexels_8716579", "1 7", "stay", "seminar room, small presenter, large heads in front"),
    ("p20_applause", "pexels_3252986", "3 7", "stay", "presenter at the screen, audience applauding in front"),
]


# person region (the sender's --crop) for the clips where someone else is in view: the stage / podium area without
# the seated panel or the audience heads in front (chosen from the first frames, 0..1 of the frame)
CROPS = {"p05_mic_walk": (0.38, 0.05, 0.72, 1.0), "p07_podium_close": (0.0, 0.0, 0.55, 1.0),
         "p19_seminar_far": (0.3, 0.05, 0.8, 0.7), "p20_applause": (0.4, 0.0, 0.85, 0.8)}


def manifest_clips(root=EVENT / "presenter"):
    return [{"name": n, "video": str(root / f"{f}.mp4"), "types": t.split(), "kind": k, "desc": d}
            for n, f, t, k, d in MANIFEST]


# ------------------------------------------------------------------------------------------------ perception
def _key(video, *extra):
    st = Path(video).stat()
    return hashlib.md5("|".join([str(Path(video).resolve()), str(st.st_size), str(int(st.st_mtime))]
                                + [str(e) for e in extra]).encode()).hexdigest()[:10]


def detections(video, crop=None, cache_dir=ROOT / "logs" / "presenter", model="full", max_fps=SENDER_FPS):
    """MediaPipe detections as the live sender would get them (<= 720 px high, <= max_fps, optional person region
    `crop` = the sender's --crop). Cached npz.
    -> dict(t (n,), world (n,33,3), vis (n,33), image (n,33,3) [NaN = nobody], w, h, fps, src_fps)."""
    import cv2

    key = _key(video, model, max_fps, MAX_H, "v1", *([] if crop is None else ["crop", *crop]))
    cache = Path(cache_dir) / f"mp_{Path(video).stem}_{key}.npz"
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    from sonic.pose_mp import PoseTracker

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tr = PoseTracker(model, crop=crop)
    tr.max_jump = float("inf")          # continuity is JumpGate's job (k1live/sender.py)
    T, Wl, V, I = [], [], [], []
    k, last_slot, wh = 0, -1, None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        t = k / fps
        k += 1
        slot = int(math.floor(t * max_fps + 1e-6))
        if slot == last_slot:           # the sender would not have finished the previous frame yet
            continue
        last_slot = slot
        if f.shape[0] > MAX_H:
            s = MAX_H / f.shape[0]
            f = cv2.resize(f, (int(round(f.shape[1] * s)), MAX_H), interpolation=cv2.INTER_AREA)
        wh = (f.shape[1], f.shape[0])
        d = tr.process(f, t)
        T.append(t)
        if d is None:
            Wl.append(np.full((33, 3), np.nan))
            V.append(np.zeros(33))
            I.append(np.full((33, 3), np.nan))
        else:
            Wl.append(d["world"])
            V.append(d["vis"])
            I.append(d["image"])
    tr.close()
    cap.release()
    out = {"t": np.array(T), "world": np.array(Wl), "vis": np.array(V), "image": np.array(I),
           "w": np.array(wh[0]), "h": np.array(wh[1]), "fps": np.array(min(fps, max_fps)), "src_fps": np.array(fps)}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    return out


def camera_motion(video, D, cache_dir=ROOT / "logs" / "presenter"):
    """Stock clips are often hand-held / panning / zooming; the event camera is fixed. Background motion per
    detection frame as a similarity transform A_k (2x3, frame-0 px -> frame-k px, at the detection resolution):
    LK optical flow on corners, the tracked person masked out, RANSAC (other people moving are outliers). Cached."""
    import cv2

    cache = Path(cache_dir) / f"cam_{Path(video).stem}_{_key(video, 'cam', len(D['t']), 'v2')}.npz"
    if cache.exists():
        return np.load(cache)["A"]
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(D["w"]), int(D["h"])
    want = {int(round(t * fps)): i for i, t in enumerate(D["t"])}
    A = np.zeros((len(D["t"]), 2, 3))
    A[:, 0, 0] = A[:, 1, 1] = 1.0
    acc = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    prev = None
    k = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        i = want.get(k)
        k += 1
        if i is None:
            continue
        g = cv2.cvtColor(cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        if prev is not None:
            mask = np.full(g.shape, 255, np.uint8)
            mask[int(0.62 * h):] = 0   # the bottom band is where the (blurred, moving) audience heads are
            for j in (i - 1, i):     # person box (detections of both frames), with a margin
                im = D["image"][j]
                if not np.isnan(im[0, 0]):
                    x0, y0 = np.nanmin(im[:, :2], 0) * (w, h)
                    x1, y1 = np.nanmax(im[:, :2], 0) * (w, h)
                    mx, my = 0.15 * (x1 - x0) + 10, 0.1 * (y1 - y0) + 10
                    mask[max(0, int(y0 - my)):int(y1 + my), max(0, int(x0 - mx)):int(x1 + mx)] = 0
            pts = cv2.goodFeaturesToTrack(prev, 400, 0.01, 8, mask=mask)
            M = None
            if pts is not None and len(pts) >= 12:
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, pts, None, winSize=(21, 21), maxLevel=3)
                good = st.ravel() == 1
                if good.sum() >= 12:
                    M, inl = cv2.estimateAffinePartial2D(pts[good], nxt[good], method=cv2.RANSAC,
                                                         ransacReprojThreshold=1.5)
            if M is None:
                M = np.array([[1.0, 0, 0], [0, 1.0, 0]])
            acc = np.vstack([M, [0, 0, 1]]) @ acc
            A[i] = acc[:2]
        prev = g
    cap.release()
    np.savez(cache, A=A)
    return A


def motion_summary(A, w, h):
    """Largest background shift (fraction of the frame width) and zoom (%) over the clip vs its first frame."""
    c = np.array([w / 2, h / 2, 1.0])
    shift = np.array([(Ak @ c)[:2] - c[:2] for Ak in A])
    zoom = np.sqrt(np.abs(np.linalg.det(A[:, :, :2])))
    return {"max_shift_frac": round(float(np.abs(shift[:, 0]).max() / w), 3),
            "max_shift_y_frac": round(float(np.abs(shift[:, 1]).max() / h), 3),
            "max_zoom_pct": round(float(100 * np.abs(zoom - 1).max()), 1)}


def stabilize(D, A):
    """Image landmarks mapped back into frame 0 (as if the camera had not moved). World landmarks unchanged."""
    D = dict(D)
    w, h = float(D["w"]), float(D["h"])
    I = D["image"].copy()
    for i in range(len(I)):
        if np.isnan(I[i, 0, 0]):
            continue
        Ai = np.linalg.inv(np.vstack([A[i], [0, 0, 1]]))[:2]
        px = I[i, :, :2] * (w, h)
        I[i, :, :2] = (px @ Ai[:, :2].T + Ai[:, 2]) / (w, h)
    D["image"] = I
    return D


def person_stream(D, mirror=True, loco_params=None, profile="visitor", t0=0.0, t1=None, arm_ik="shape18"):
    """Cached detections -> JumpGate -> HumanMapper (+ LocoEstimator) -> list of HumanFrame dicts with diagnostics.
    arm_ik = the sender's --arm-ik (event arm path: dir)."""
    from k1live.human import L_HIP, R_HIP, HumanMapper, JumpGate

    gate = JumpGate()
    hm = HumanMapper(mirror=mirror, loco=True, loco_profile=profile, arm_ik=arm_ik)
    if loco_params:
        from k1live.locomotion import LocoEstimator

        hm.loco.est = LocoEstimator(mirror=mirror, profile=profile, **loco_params)
    est = hm.loco.est
    w, h = int(D["w"]), int(D["h"])
    out, prev_c = [], None
    t_end = float(D["t"][-1]) if t1 is None else t1
    for k, t in enumerate(D["t"]):
        if t < t0 - 1e-9 or t > t_end + 1e-9:
            continue
        raw = None if np.isnan(D["world"][k, 0, 0]) else {"world": D["world"][k], "vis": D["vis"][k],
                                                          "image": D["image"][k]}
        det = gate(raw)
        if gate.relocked:   # k1live/sender.py: tracking jumped (someone else) -> walking homes again
            hm.on_relock()
        f = hm.process(det, float(t - t0), w, h)
        f.pop("obs", None)
        c = None
        if det is not None:
            c = 0.5 * (np.asarray(det["image"])[L_HIP, :2] + np.asarray(det["image"])[R_HIP, :2])
        jump = bool(c is not None and prev_c is not None and np.linalg.norm(c - prev_c) > gate.max_jump)
        if c is not None:
            prev_c = c
        vis = D["vis"][k]
        f.update(seq=len(out), t_capture=float(t - t0), _found=raw is not None, _gated=raw is not None and det is None,
                 _switch=jump, _relock=gate.relocked, _hips=bool(raw is not None and min(vis[23], vis[24]) >= 0.5),
                 _legs=bool(raw is not None and min(vis[25], vis[26], vis[27], vis[28]) >= 0.5),
                 _home=bool(est is not None and est.home is not None),
                 _offset=None if (est is None or est.offset is None or est.home is None) else dict(est.offset),
                 _hip_c=None if c is None else [float(c[0]), float(c[1])])
        out.append(f)
    return out


def onsets(x):
    nz = np.abs(np.asarray(x)) > 0.02
    return int(np.sum(nz[1:] & ~nz[:-1]) + (1 if len(nz) and nz[0] else 0))


def perception_metrics(frames, kind):
    n = len(frames)
    if not n:
        return {"frames": 0}
    t = np.array([f["t_capture"] for f in frames])
    pres = np.array([f["present"] for f in frames])
    found = np.array([f["_found"] for f in frames])
    loco = np.array([[0.0, 0.0, 0.0] if not f.get("loco") else [f["loco"]["vx"], f["loco"]["vy"], f["loco"]["wz"]]
                     for f in frames])
    home = np.array([f["_home"] for f in frames])
    cal = np.array([f["calibrated"] for f in frames])
    offs = [f["_offset"] for f in frames if f["_offset"] is not None]
    walk = np.any(np.abs(loco) > 0.02, axis=1)
    ev = []
    for i, ax in enumerate(("vx", "vy", "wz")):
        nz = np.abs(loco[:, i]) > 0.02
        for k in np.flatnonzero(nz & ~np.r_[False, nz[:-1]]):
            ke = k
            while ke < n and nz[ke]:
                ke += 1
            ev.append({"axis": ax, "t": round(float(t[k]), 2), "dur_s": round(float(t[min(ke, n - 1)] - t[k]), 2),
                       "sign": int(np.sign(loco[k, i]))})
    ev.sort(key=lambda e: e["t"])
    return {
        "frames": n, "seconds": round(float(t[-1] - t[0]), 1),
        "found_frac": round(float(found.mean()), 3), "tracked_frac": round(float(pres.mean()), 3),
        "gate_drops": int(sum(f["_gated"] for f in frames)), "switches": int(sum(f["_switch"] for f in frames)),
        "relocks": int(sum(f["_relock"] for f in frames)),
        "hips_frac": round(float(np.mean([f["_hips"] for f in frames])), 3),
        "legs_frac": round(float(np.mean([f["_legs"] for f in frames])), 3),
        "arms_calibrated_at": None if not cal.any() else round(float(t[np.argmax(cal)]), 2),
        "walk_home_at": None if not home.any() else round(float(t[np.argmax(home)]), 2),
        "walk_home_frac": round(float(home.mean()), 3),
        "intent_walk_frac": round(float(walk.mean()), 4),
        "intent_onsets": {ax: onsets(loco[:, i]) for i, ax in enumerate(("vx", "vy", "wz"))},
        "false_onsets": ({ax: onsets(loco[:, i]) for i, ax in enumerate(("vx", "vy", "wz"))} if kind == "stay"
                         else None),
        "intent_events": ev[:40],
        "max_abs_offset": None if not offs else {
            "fwd_m": round(float(max(abs(o["fwd"]) for o in offs)), 3),
            "side_m": round(float(max(abs(o["side"]) for o in offs)), 3),
            "yaw_deg": round(float(math.degrees(max(abs(o["yaw"]) for o in offs))), 1)},
    }


# ------------------------------------------------------------------------------------------------ control
def make_server(backend, log_dir, variant=None):
    """k1live.sim_server.SimServer without render / attract; sockets connect to unused ports and are never fed."""
    import socket

    import zmq

    from k1live import sim_server as S

    def free():
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            return str(s.getsockname()[1])

    be = "sonic" if backend.startswith("sonic") else "split"
    args = S.parse_args(["--backend", be, "--no-render", "--no-attract", "--log-dir", str(log_dir),
                         "--human-port", free(), "--ctrl-port", free(), "--state-port", free(), "--robot-port", free()])
    extra = {}
    if backend == "sonic_walk":
        extra["variant_dir"] = str(variant or WALK_VARIANT)
    elif backend == "sonic" and variant:
        extra["variant_dir"] = str(variant)

    class Offline(S.SimServer):
        def backend_kw(self, name):
            return dict(super().backend_kw(name), **extra)

    ctx = zmq.Context()
    st = ctx.socket(zmq.PUB)
    st.setsockopt(zmq.LINGER, 0)
    srv = Offline(args, ctx, st)
    return srv, ctx


class ServerRun:
    """Drive a SimServer tick by tick on a simulated clock from a HumanFrame list."""

    T0 = 1000.0

    def __init__(self, backend, log_dir, variant=None):
        self.srv, self.ctx = make_server(backend, log_dir, variant)
        self.srv.t_start = self.srv.t_person = self.T0
        self.backend = backend
        self.i = 0
        self.k = 0
        self.rec = {k: [] for k in ("t", "goal", "q", "cmd", "track", "xy", "yaw", "tilt", "fallen", "person_cmd")}

    def step(self, frames):
        srv = self.srv
        wall = self.T0 + self.i * DT
        while self.k < len(frames) and frames[self.k]["t_capture"] + LATENCY_S <= self.i * DT + 1e-9:
            f = dict(frames[self.k])
            f["t_capture"] = self.T0 + f["t_capture"]
            f["_t_recv"] = wall
            srv.hf = f
            self.k += 1
        srv.tick(wall)
        x, y, yaw = srv.be.root_pose()
        r = self.rec
        r["t"].append(self.i * DT)
        r["goal"].append(np.asarray(srv.goal, float).copy())
        r["q"].append(srv.be.arm_q_now())
        r["cmd"].append(np.asarray(srv.cmd, float).copy())
        r["track"].append(srv.tracking)
        r["xy"].append((x, y))
        r["yaw"].append(yaw)
        r["tilt"].append(srv.be.tilt_deg())
        r["fallen"].append(srv.fallen)
        hf = srv.hf
        lo = hf.get("loco") if (hf is not None and srv.tracking) else None
        r["person_cmd"].append(bool(lo) and max(abs(lo["vx"]), abs(lo["vy"]), abs(lo["wz"])) > 0.02)
        self.i += 1

    def close(self):
        try:
            self.srv.close()
        finally:
            self.ctx.destroy(0)

    def metrics(self):
        R = {k: np.asarray(v) for k, v in self.rec.items()}
        tr = R["track"].astype(bool)
        err = np.degrees(np.abs(R["q"] - R["goal"])).mean(1)
        lag_err = []
        for L in range(0, 21):
            m = tr[:len(tr) - L] if L else tr
            if L and m.sum() > 10:
                lag_err.append(float(np.degrees(np.abs(R["q"][L:] - R["goal"][:len(R["goal"]) - L]))[m].mean()))
            elif not L and tr.sum() > 10:
                lag_err.append(float(err[tr].mean()))
            else:
                lag_err.append(np.inf)
        best = int(np.argmin(lag_err))
        cmd = R["cmd"]
        walking = np.any(np.abs(cmd) > 0.02, axis=1) & R["person_cmd"].astype(bool)
        yaw = np.unwrap(R["yaw"])
        return {"backend": self.backend, "seconds": round(float(R["t"][-1]), 1),
                "track_frac": round(float(tr.mean()), 3),
                "arm_err_deg": round(float(err[tr].mean()), 2) if tr.any() else None,
                "arm_err_p90_deg": round(float(np.percentile(err[tr], 90)), 2) if tr.any() else None,
                "arm_err_bestlag_deg": None if not np.isfinite(lag_err[best]) else round(lag_err[best], 2),
                "best_lag_ms": best * 20,
                "falls": int(self.srv.falls), "max_tilt_deg": round(float(R["tilt"].max()), 1),
                "person_walk_s": round(float(walking.sum() * DT), 2),
                "cmd_onsets": {ax: onsets(np.where(R["person_cmd"].astype(bool), cmd[:, i], 0.0))
                               for i, ax in enumerate(("vx", "vy", "wz"))},
                "moved_m": round(float(np.linalg.norm(R["xy"][-1] - R["xy"][0])), 2),
                "path_m": round(float(np.linalg.norm(np.diff(R["xy"], axis=0), axis=1).sum()), 2),
                "turned_deg": round(float(np.degrees(yaw[-1] - yaw[0])), 1),
                "max_turn_deg": round(float(np.degrees(np.max(np.abs(yaw - yaw[0])))), 1)}


def run_backend(frames, backend, log_dir, variant=None, tail_s=1.0):
    run = ServerRun(backend, log_dir, variant)
    try:
        n = int((frames[-1]["t_capture"] + tail_s) / DT) if frames else 0
        for _ in range(n):
            run.step(frames)
        return run.metrics()
    finally:
        run.close()


# ------------------------------------------------------------------------------------------------ jobs
def job(cfg):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    t0 = time.perf_counter()
    D = detections(cfg["video"], crop=cfg.get("crop"))
    A = camera_motion(cfg["video"], D)
    if cfg.get("stabilize"):
        D = stabilize(D, A)
    frames = person_stream(D, mirror=cfg.get("mirror", True), loco_params=cfg.get("loco"),
                           profile=cfg.get("profile", "visitor"), t0=cfg.get("t0", 0.0), t1=cfg.get("t1"))
    out = {"clip": cfg["name"], "video": Path(cfg["video"]).name, "types": cfg.get("types", []),
           "kind": cfg.get("kind", "stay"), "desc": cfg.get("desc", ""), "crop": cfg.get("crop"),
           "frame": [int(D["w"]), int(D["h"])], "fps": round(float(D["fps"]), 2),
           "camera_motion": motion_summary(A, int(D["w"]), int(D["h"])),
           "perception": perception_metrics(frames, cfg.get("kind", "stay"))}
    if cfg.get("backend"):
        log_dir = Path(cfg["log_root"]) / f"{cfg['name']}_{cfg['backend']}"
        out["control"] = run_backend(frames, cfg["backend"], log_dir, cfg.get("variant"))
    out["wall_s"] = round(time.perf_counter() - t0, 1)
    return out


def framing(p):
    """Framing group from what MediaPipe saw: full (legs visible), waist (hips, no legs), upper (no hips)."""
    if p.get("legs_frac", 0) >= 0.5:
        return "full"
    if p.get("hips_frac", 0) >= 0.5:
        return "waist"
    return "upper"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", nargs="*", default=None, help="name=path[@t0-t1][:stay|move] (default: MANIFEST)")
    ap.add_argument("--only", nargs="*", default=None, help="manifest clip names to run")
    ap.add_argument("--backends", nargs="*", default=["split", "sonic", "sonic_walk"])
    ap.add_argument("--walk-variant", default=str(WALK_VARIANT))
    ap.add_argument("--perception-only", action="store_true")
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--loco", action="append", default=[], help="LocoEstimator param override k=v")
    ap.add_argument("--loco-profile", default="visitor", help="LocoEstimator profile (k1live.locomotion PROFILES)")
    ap.add_argument("--stabilize", action="store_true", help="undo the stock camera's pan / zoom (fixed camera)")
    ap.add_argument("--crops", action="store_true", help="person region (CROPS) on the multi-person clips")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    if a.clips:
        clips = []
        for spec in a.clips:
            name, _, rest = spec.partition("=")
            kind = "stay"
            if rest.endswith((":stay", ":move")):
                rest, kind = rest[:-5], rest[-4:]
            path, _, rng = rest.partition("@")
            c = {"name": name, "video": path, "kind": kind}
            if rng:
                x0, _, x1 = rng.partition("-")
                c.update(t0=float(x0 or 0), t1=float(x1) if x1 else None)
            clips.append(c)
    else:
        clips = manifest_clips()
        if a.only:
            clips = [c for c in clips if c["name"] in a.only]
    loco = {}
    for kv in a.loco:
        k, v = kv.split("=")
        loco[k] = float(v) if v.replace(".", "", 1).replace("-", "", 1).isdigit() else v
    log_root = ROOT / "logs" / "presenter" / "server"
    # MediaPipe first (one process per clip, cached), then the control runs
    for c in clips:
        if a.crops and c["name"] in CROPS:
            c["crop"] = CROPS[c["name"]]
    with ProcessPoolExecutor(a.workers) as ex:
        list(ex.map(detections, [c["video"] for c in clips], [c.get("crop") for c in clips]))
    cfgs = []
    for c in clips:
        base = dict(c, mirror=not a.no_mirror, loco=loco or None, log_root=str(log_root), variant=None,
                    profile=a.loco_profile, stabilize=a.stabilize)
        if a.perception_only:
            cfgs.append(base)
        for be in ([] if a.perception_only else a.backends):
            cfgs.append(dict(base, backend=be, variant=a.walk_variant if be == "sonic_walk" else None))
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(job, cfgs))
    clips_out = {}
    for r in res:
        c = clips_out.setdefault(r["clip"], {k: r[k] for k in ("clip", "video", "types", "kind", "desc", "crop", "frame",
                                                               "fps", "camera_motion", "perception")})
        c["framing"] = framing(r["perception"])
        if "control" in r:
            c.setdefault("backends", {})[r["control"]["backend"]] = r["control"]
    for c in clips_out.values():
        p = c["perception"]
        fo = p.get("false_onsets")
        line = (f"{c['clip']:22s} {c['framing']:5s} {c['kind']:4s} trk {p['tracked_frac']:.2f} hips {p['hips_frac']:.2f} "
                f"legs {p['legs_frac']:.2f} home {p['walk_home_at']} sw {p['switches']} relock {p['relocks']} "
                f"cam {c['camera_motion']['max_shift_frac']}/{c['camera_motion']['max_zoom_pct']}% "
                f"onsets {p['intent_onsets']}{' FALSE' if fo and any(fo.values()) else ''}")
        for be, m in c.get("backends", {}).items():
            line += (f"\n    {be:10s} arm {m['arm_err_deg']} (lag {m['arm_err_bestlag_deg']}@{m['best_lag_ms']}ms) "
                     f"walk {m['person_walk_s']}s moved {m['moved_m']} m turned {m['turned_deg']} deg falls {m['falls']} "
                     f"tilt {m['max_tilt_deg']}")
        print(line, flush=True)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps({"loco_profile": a.loco_profile, "loco_params": loco,
                                            "stabilize": a.stabilize, "crops": a.crops,
                                            "mirror": not a.no_mirror, "latency_s": LATENCY_S,
                                            "backends": [] if a.perception_only else a.backends,
                                            "walk_variant": a.walk_variant, "clips": list(clips_out.values())},
                                           indent=1, default=float))


if __name__ == "__main__":
    main()
