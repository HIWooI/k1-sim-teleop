"""Perception process with GEM-X instead of MediaPipe: camera or video file -> GEM-X (GPU worker) -> K1 targets.

Same contract as k1live/sender.py (protocol.py): HumanFrame on HUMAN_PORT ~30 Hz, SenderState on SENDER_STATE_PORT
(first one < 1 s after start, note "loading" until GEM-X is ready), CTRL calibrate / set {mirror, camera, video, loop,
loco_profile, crop} / quit with the same no-op rule, /cam.mjpg /cam.jpg /state on CAM_HTTP_PORT, --source, ports
shifted by K1LIVE_PORT_OFFSET.

    .venv/bin/python -m k1live.sender_gemx --source 0                       # camera, GEM-X on this machine's GPU
    .venv/bin/python -m k1live.sender_gemx --source clip.mp4 --loop         # fake camera
    .venv/bin/python -m k1live.sender_gemx --source 0 --gemx-cmd "ssh gpu-host <GEM-X venv python> \
        <repo>/k1live/gemx_worker.py --serve" --gemx-fmt jpeg               # perception on another machine

Per frame: the newest camera frame -> k1live/gemx_worker.py (a child process in the GEM-X virtualenv: VitPose
SOMA-77 -> GEM sliding window -> SOMA FK, demo_webcam.py's real-time path) -> k1live.gemx.soma_det (MediaPipe-33 layout,
VitPose 2D + GEM-X 3D) -> JumpGate -> k1live.human.HumanMapper(perception="gemx"): arms by the bone-direction IK
(frontal-plane angle from the 2D keypoints, depth from GEM-X), or --arm-src mlp = the distilled retarget MLP on the SOMA
articulation; waist / bow from the 3D skeleton; walking intent k1live.gemx.GemxLocoEstimator (camera-frame pelvis)
-> mirror -> HumanFrame. Until the GEM-X window has --min-window frames (~0.5 s after the person appears) the
frame counts as nobody (the camera view shows the 2D keypoints).

Exit codes: as k1live.sender (0, 2 bad arguments, 3 source failed with --exit-on-fail, 4 port in use) plus
5 = GEM-X unavailable: no CUDA, no GEM-X environment / checkpoint, or the worker died at start-up or keeps dying.
The service then starts k1live.sender (MediaPipe) instead. MuJoCo-free; torch only for --arm-src mlp.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import shlex
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import zmq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from k1live import gemx_worker as GW  # noqa: E402  (wire format only: no torch at import)
from k1live import protocol as P  # noqa: E402
from k1live.sender import CamView, FrameSource, make_http, parse_crop, source_spec  # noqa: E402

EXIT_GEMX = 5
DEFAULT_PERCEPTION = GW.DEFAULT_PERCEPTION


def default_python(perception):
    env = os.environ.get("K1LIVE_GEMX_PYTHON")
    if env:
        return env
    return str(Path(perception).expanduser() / "GEM-X" / ".venv" / "bin" / "python")


# ------------------------------------------------------------------------------------------------ worker client
class GemxClient:
    """The GEM-X worker child: start (wait for ready), one request at a time with a timeout, stderr -> our log."""

    def __init__(self, cmd, fmt=GW.FMT_RAW, start_timeout=240.0, frame_timeout=5.0, log=None):
        self.cmd, self.fmt = list(cmd), fmt
        self.start_timeout, self.frame_timeout = start_timeout, frame_timeout
        self.log = log or (lambda m: print(m, file=sys.stderr, flush=True))
        self.proc, self.info, self.error = None, {}, ""
        self.q = queue.Queue()
        self.seq = 0

    def start(self, stop=None):
        """-> True when the worker reported ready; False with self.error otherwise (worker gone, or timeout)."""
        self.close()
        self.q = queue.Queue()
        try:
            self.proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, bufsize=0)
        except OSError as e:
            self.error = f"cannot start GEM-X worker: {e}"
            return False
        threading.Thread(target=self._stderr, args=(self.proc,), daemon=True, name="gemx-stderr").start()
        threading.Thread(target=self._reader, args=(self.proc, self.q), daemon=True, name="gemx-reader").start()
        t0 = time.time()
        while time.time() - t0 < self.start_timeout:
            if stop is not None and stop.is_set():
                self.error = "stopped"
                return False
            try:
                kind, obj = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            if kind == "ready":
                self.info = obj
                return True
            self.error = obj.get("error", "worker error") if isinstance(obj, dict) else str(obj)
            return False
        self.error = f"GEM-X worker not ready after {self.start_timeout:.0f} s"
        return False

    def _stderr(self, proc):
        for line in iter(proc.stderr.readline, b""):
            self.log("[gemx-worker] " + line.decode(errors="replace").rstrip())

    def _reader(self, proc, q):
        f = proc.stdout
        try:
            magic, obj = GW.read_msg(f)
            if magic == GW.MAGIC_READY:
                q.put(("ready", obj))
            else:
                q.put(("error", obj))
                return
            while True:
                q.put(("reply", GW.read_reply(f)))
        except (EOFError, ValueError, OSError) as e:
            code = proc.poll()
            q.put(("error", {"error": f"worker exited ({code}): {e}", "code": code}))

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def submit(self, frame, t, flags=0, crop=None):
        """Send one frame (does not wait) -> its sequence number. Raises RuntimeError if the worker is gone."""
        if not self.alive():
            raise RuntimeError(self.error or "GEM-X worker not running")
        self.seq += 1
        try:
            self.proc.stdin.write(GW.pack_request(self.seq, t, frame, self.fmt, flags, crop))
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise RuntimeError(f"GEM-X worker pipe: {e}") from e
        return self.seq & 0xFFFFFFFF

    def wait(self, seq):
        """Reply of request seq (earlier replies are dropped); RuntimeError if the worker is gone / too slow."""
        deadline = time.time() + self.frame_timeout
        while True:
            try:
                kind, obj = self.q.get(timeout=max(0.01, deadline - time.time()))
            except queue.Empty:
                raise RuntimeError(f"GEM-X worker: no reply in {self.frame_timeout:.0f} s") from None
            if kind == "error":
                self.error = obj.get("error", "worker error")
                raise RuntimeError(self.error)
            if kind == "reply" and obj["seq"] == seq:
                return obj

    def process(self, frame, t, flags=0, crop=None):
        """submit + wait: -> reply dict (k1live.gemx_worker.read_reply)."""
        return self.wait(self.submit(frame, t, flags, crop))

    def close(self):
        p, self.proc = self.proc, None
        if p is None:
            return
        try:
            p.stdin.close()
        except OSError:
            pass
        try:
            p.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            p.terminate()
            try:
                p.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                p.kill()


def view_draw(frame, det):
    """Camera view overlay: GEM-X SOMA skeleton (orange), or the VitPose keypoints while the window warms up."""
    import cv2

    from k1live.gemx import draw_soma

    if det.get("kp_only") is not None:
        out = frame.copy()
        kp = det["kp_only"]
        r = max(2, frame.shape[0] // 240)
        for x, y, c in kp:
            if c > 0.35:
                cv2.circle(out, (int(x), int(y)), r, (40, 150, 255), -1, cv2.LINE_AA)
        return out
    return draw_soma(frame, det["soma_uv"])


def loco_overrides(items):
    """["fwd_start=0.25", ...] -> {"fwd_start": 0.25, ...} for LocoEstimator. Unknown keys or values are reported and
    skipped (an unknown key would make the estimator refuse to start, i.e. no walking at all)."""
    from k1live.locomotion import LocoEstimator

    out = {}
    for item in items or []:
        k, _, v = str(item).partition("=")
        k, v = k.strip(), v.strip()
        default = LocoEstimator.DEFAULTS.get(k, KeyError)
        if default is KeyError or not v:
            print(f"[sender_gemx] ignoring walking parameter {item!r} (unknown key or no value)", flush=True)
            continue
        try:
            out[k] = (v.lower() in ("1", "true", "yes", "on")) if isinstance(default, bool) else float(v)
        except ValueError:
            print(f"[sender_gemx] ignoring walking parameter {item!r} (not a number)", flush=True)
    if out:
        print(f"[sender_gemx] presenter walking overrides: {out}", flush=True)
    return out


# ------------------------------------------------------------------------------------------------ main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--source", default="", help="camera index (digits) or video file (played in real time)")
    src.add_argument("--camera", type=int, default=None, help="alias of --source N")
    src.add_argument("--video", default="", help="alias of --source PATH")
    ap.add_argument("--loop", action="store_true", help="loop the video file")
    ap.add_argument("--no-mirror", action="store_true", help="robot copies the person's own left/right")
    ap.add_argument("--arm-ik", default=os.environ.get("K1LIVE_ARM_IK", "dir"), choices=["shape18", "dir"],
                    help="arm IK on the GEM-X skeleton (default dir: frontal angle from 2D, depth from GEM-X)")
    ap.add_argument("--arm-src", default="ik", choices=["ik", "mlp"],
                    help="ik = --arm-ik on the joints; mlp = the distilled retarget MLP on the SOMA articulation")
    ap.add_argument("--mlp", default="", help="--arm-src mlp checkpoint (default k1_retarget.pt, k1live/gemx_mlp.py)")
    ap.add_argument("--no-loco", action="store_true", help="no walking intent (loco always None)")
    ap.add_argument("--loco-profile", default=os.environ.get("K1LIVE_LOCO_PROFILE", "visitor"),
                    choices=["visitor", "presenter"])
    ap.add_argument("--gemx-dist", default=os.environ.get("K1LIVE_GEMX_DIST", "pelvis"), choices=["pelvis", "ground"],
                    help="walking distance: GEM-X pelvis depth, or the ankles' ray on the floor plane fixed at home")
    ap.add_argument("--presenter-param", action="append", default=[], metavar="KEY=VALUE",
                    help="presenter walking profile override, repeatable (venue tuning, docs/runbook.md), "
                         "e.g. fwd_start=0.25; only applies while the walking mode is presenter")
    ap.add_argument("--crop", default=None, help="x0,y0,x1,y1 (0..1): look for the person only in this region")
    ap.add_argument("--no-smpl", action="store_true", help="do not compute the whole-body stream (HumanFrame.smpl)")
    ap.add_argument("--no-auto-calibrate", action="store_true", help="calibrate only on CTRL calibrate")
    ap.add_argument("--max-h", type=int, default=int(os.environ.get("K1LIVE_GEMX_MAX_H", "1080")),
                    help="frames taller than this are downscaled before GEM-X (default 1080: far persons need pixels)")
    # GEM-X worker
    ap.add_argument("--gemx-perception", default=DEFAULT_PERCEPTION,
                    help="GEM-X perception folder (demo_webcam.py, GEM-X/ inside); $K1LIVE_GEMX_PERCEPTION")
    ap.add_argument("--gemx-python", default="", help="GEM-X venv python (default $K1LIVE_GEMX_PYTHON or "
                    "<perception>/GEM-X/.venv/bin/python)")
    ap.add_argument("--gemx-cmd", default=os.environ.get("K1LIVE_GEMX_CMD", ""),
                    help="full worker command instead (e.g. through ssh to a GPU machine); --gemx-fmt jpeg then")
    ap.add_argument("--gemx-fmt", default="", choices=["", "raw", "jpeg"], help="frame transport (default raw "
                    "locally, jpeg with --gemx-cmd)")
    ap.add_argument("--gemx-ckpt", default="", help="GEM-X checkpoint (default: GEM-X's own)")
    ap.add_argument("--window", type=int, default=64, help="GEM-X sliding window (frames)")
    ap.add_argument("--min-window", type=int, default=16, help="frames before the first 3D result")
    ap.add_argument("--hfov", type=float, default=float(os.environ.get("K1LIVE_GEMX_HFOV", "0")),
                    help="camera horizontal FOV, deg (0 = GEM-X's estimate_K guess); scales the metric depth")
    ap.add_argument("--detector", default="auto", choices=["auto", "yolox", "vitpose"])
    ap.add_argument("--start-timeout", type=float, default=240.0, help="GEM-X model loading limit, s")
    ap.add_argument("--worker-restarts", type=int, default=3, help="worker deaths tolerated before exiting 5")
    # same as k1live.sender
    ap.add_argument("--http-port", type=int, default=P.CAM_HTTP_PORT)
    ap.add_argument("--no-http", action="store_true")
    ap.add_argument("--human-port", type=int, default=P.HUMAN_PORT)
    ap.add_argument("--state-port", type=int, default=P.SENDER_STATE_PORT)
    ap.add_argument("--ctrl-port", type=int, default=P.CTRL_PORT)
    ap.add_argument("--max-seconds", type=float, default=0.0)
    ap.add_argument("--exit-on-fail", action="store_true")
    ap.add_argument("--log", default="", help="one JSON line per frame (timing) to this file")
    a = ap.parse_args(argv)
    if a.camera is not None:
        a.source = str(a.camera)
    elif a.video:
        a.source = a.video
    a.spec = source_spec(a.source)
    if a.crop is not None:
        a.crop_box = parse_crop(a.crop)
        if a.crop and a.crop_box is None:
            ap.error(f"--crop {a.crop}: want x0,y0,x1,y1 with 0 <= x0 < x1 <= 1, 0 <= y0 < y1 <= 1")
    else:
        a.crop_box = parse_crop(os.environ.get("K1LIVE_CROP", ""))
    if a.spec and not a.spec.startswith("camera:") and not Path(a.spec).exists():
        ap.error(f"--source {a.source}: no such file")
    a.fmt = (GW.FMT_JPEG if (a.gemx_fmt == "jpeg" or (not a.gemx_fmt and a.gemx_cmd)) else GW.FMT_RAW)
    return a


def worker_cmd(a):
    if a.gemx_cmd:
        return shlex.split(a.gemx_cmd)
    cmd = [a.gemx_python or default_python(a.gemx_perception), str(ROOT / "k1live" / "gemx_worker.py"), "--serve",
           "--perception", str(Path(a.gemx_perception).expanduser()), "--window", str(a.window),
           "--min-window", str(a.min_window), "--hfov", f"{a.hfov:g}", "--detector", a.detector]
    if a.gemx_ckpt:
        cmd += ["--ckpt", a.gemx_ckpt]
    return cmd


def preflight(a):
    """Cheap checks before spawning anything: '' if worth trying, else the reason (-> exit 5)."""
    if a.gemx_cmd:
        return ""
    py = Path(a.gemx_python or default_python(a.gemx_perception)).expanduser()
    if not py.exists():
        return f"GEM-X python not found: {py}"
    if not (Path(a.gemx_perception).expanduser() / "demo_webcam.py").exists():
        return f"GEM-X perception folder (demo_webcam.py) not found: {a.gemx_perception}"
    return ""


def main(argv=None):
    args = parse_args(argv)
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    ctx = zmq.Context()
    try:
        pub = ctx.socket(zmq.PUB)
        pub.setsockopt(zmq.SNDHWM, 10)
        pub.setsockopt(zmq.LINGER, 0)
        pub.bind(f"tcp://{P.HOST}:{args.human_port}")
        st_pub = ctx.socket(zmq.PUB)
        st_pub.setsockopt(zmq.LINGER, 0)
        st_pub.bind(f"tcp://{P.HOST}:{args.state_port}")
    except zmq.ZMQError as e:
        print(f"[sender_gemx] bind failed: {e}", file=sys.stderr, flush=True)
        ctx.destroy(0)
        return 4
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.LINGER, 0)
    sub.connect(f"tcp://{P.HOST}:{args.ctrl_port}")
    sub.setsockopt(zmq.SUBSCRIBE, P.TOPIC_CTRL)

    st = {"spec": args.spec, "mirror": not args.no_mirror, "loop": args.loop, "fps": 0.0, "present": False,
          "calibrated": False, "pose_ms": 0.0, "note": "loading", "source": None, "mapper": None, "frames": 0,
          "sent": 0, "loco_profile": args.loco_profile, "crop": args.crop_box, "gemx": "loading", "gemx_info": {},
          "gemx_ms": {}, "status3d": False}
    st_lock = threading.Lock()

    def sender_state():
        src, mp = st["source"], st["mapper"]
        note = st["note"] or ("" if src is not None else "no source")
        if st["gemx"] == "warming" and not note:
            note = "GEM-X: 3D 준비 중"
        if mp is not None and mp.loco is not None and not mp.loco.available:
            note = (note + "; " if note else "") + "loco: none"
        if src is not None and src.error:
            note = (note + "; " if note else "") + src.error
        return P.SenderState(t=P.now(), cam_ok=bool(src is not None and src.ok), source=st["spec"],
                             fps=round(st["fps"], 1), present=st["present"], calibrated=st["calibrated"],
                             pose_ms=round(st["pose_ms"], 1), note=note)

    def state_loop():
        while not stop.is_set():
            with st_lock:
                msg = sender_state()
            try:
                st_pub.send_multipart(P.encode(P.TOPIC_SENDER_STATE, msg))
            except zmq.ZMQError:
                return
            stop.wait(0.25)

    threading.Thread(target=state_loop, daemon=True, name="state").start()
    print(f"[sender_gemx] state on {args.state_port}, HUMAN on {args.human_port}, source {args.spec or '(none)'}, "
          f"walking profile {args.loco_profile}, arms {args.arm_src if args.arm_src == 'mlp' else args.arm_ik}",
          flush=True)

    def fail(msg):
        print(f"[sender_gemx] GEM-X unavailable: {msg}", file=sys.stderr, flush=True)
        with st_lock:
            st["note"] = f"GEM-X 사용 불가: {msg}"
        time.sleep(0.6)          # let one state message with the reason out
        stop.set()
        time.sleep(0.3)
        ctx.destroy(0)
        return EXIT_GEMX

    why = preflight(args)
    if why:
        return fail(why)
    # the camera starts while GEM-X loads (the view shows it), the robot gets nothing until GEM-X is ready
    view = CamView(draw=view_draw)
    http = None
    if not args.no_http:
        try:
            http = make_http(view, lambda: {**asdict(sender_state()), "clients": view.clients, "perception": "gemx",
                                            "gemx": st["gemx"], "gemx_info": st["gemx_info"],
                                            "gemx_ms": st["gemx_ms"], "loco_profile": st["loco_profile"],
                                            "crop": None if st["crop"] is None else list(st["crop"])},
                             args.http_port, stop)
        except OSError as e:
            print(f"[sender_gemx] http port {args.http_port}: {e}", file=sys.stderr, flush=True)
            stop.set()
            ctx.destroy(0)
            return 4
    source = None
    if args.spec:
        source = FrameSource(args.spec, loop=args.loop, max_h=args.max_h)
        source.start()
        st["source"] = source

    client = GemxClient(worker_cmd(args), fmt=args.fmt, start_timeout=args.start_timeout,
                        log=lambda m: print(m, file=sys.stderr, flush=True))
    t_load = time.time()
    ok = client.start(stop)
    if not ok:
        if source is not None:
            source.stop()
        if http is not None:
            http.shutdown()
            http.server_close()
        return fail(client.error) if not stop.is_set() else 0
    print(f"[sender_gemx] GEM-X ready in {time.time() - t_load:.1f} s: {json.dumps(client.info)}", flush=True)

    from k1live.gemx import soma_det
    from k1live.human import HumanMapper, JumpGate

    mlp = None
    if args.arm_src == "mlp":
        try:
            from k1live.gemx_mlp import MlpArms

            mlp = MlpArms(args.mlp or None)
        except Exception as e:  # no torch / no weights: the IK arms
            print(f"[sender_gemx] --arm-src mlp unavailable ({type(e).__name__}: {e}); using the IK", flush=True)
    gate = JumpGate()
    mapper = HumanMapper(mirror=not args.no_mirror, loco=not args.no_loco, auto_calibrate=not args.no_auto_calibrate,
                         loco_profile=args.loco_profile, smpl=not args.no_smpl, arm_ik=args.arm_ik,
                         perception="gemx", loco_kw={"dist_src": args.gemx_dist,
                                                     "profile_kw": {"presenter": loco_overrides(args.presenter_param)}})
    with st_lock:
        st.update(mapper=mapper, note="", gemx="ready", gemx_info=client.info)
    log = open(args.log, "a") if args.log else None
    pending = {"flags": GW.FLAG_RESET}

    def switch(new_spec, loop=True):
        nonlocal source, last_n, loops_seen
        if new_spec == st["spec"] and source is not None and loop == source.loop and not source.ended:
            return
        if source is not None:
            source.stop()
        gate.reset()
        source = FrameSource(new_spec, loop=loop, max_h=args.max_h)
        source.start()
        last_n, loops_seen = 0, 0
        mapper.new_scene()
        if mlp is not None:
            mlp.reset()
        pending["flags"] |= GW.FLAG_RESET
        with st_lock:
            st.update(spec=new_spec, source=source, loop=loop, note="")
        print(f"[sender_gemx] source -> {new_spec} (loop={loop})", flush=True)

    def set_crop(value):
        box = parse_crop(value)
        if box is None and value not in (None, "", []):
            st["note"] = f"영역 무시: {value!r}"
            return
        if box == st["crop"]:
            return
        st["crop"] = box
        gate.reset()
        mapper.on_relock()
        pending["flags"] |= GW.FLAG_RESET      # re-acquire the person inside the new region
        print(f"[sender_gemx] person region {box or 'whole frame'}", flush=True)

    def handle_ctrl():
        while True:
            try:
                frames = sub.recv_multipart(zmq.NOBLOCK)
            except zmq.Again:
                return
            try:
                m = P.decode(frames, P.CtrlMsg)
            except Exception:
                continue
            if m.cmd == P.CMD_CALIBRATE:
                mapper.request_calibration(explicit=True)
                pending["flags"] |= GW.FLAG_IDENTITY   # re-measure the body shape of whoever is there now
                print("[sender_gemx] calibrating (stand still facing the camera)", flush=True)
            elif m.cmd == P.CMD_SET:
                a = m.args or {}
                if "mirror" in a and bool(a["mirror"]) != mapper.mirror:
                    mapper.set_mirror(bool(a["mirror"]))
                    st["mirror"] = bool(a["mirror"])
                if a.get("camera") is not None and str(a["camera"]).strip() != "":
                    switch(source_spec(a["camera"]), loop=False)
                elif a.get("video"):
                    v = source_spec(a["video"])
                    if Path(v).exists():
                        switch(v, loop=bool(a.get("loop", True)))
                    else:
                        st["note"] = f"no such video: {a['video']}"
                elif "loop" in a and source is not None and not source.is_camera:
                    source.loop = bool(a["loop"])
                if a.get("loco_profile") in P.LOCO_PROFILES and a["loco_profile"] != st["loco_profile"]:
                    mapper.set_loco_profile(a["loco_profile"])
                    st["loco_profile"] = a["loco_profile"]
                if "crop" in a:
                    set_crop(a["crop"])
            elif m.cmd == P.CMD_QUIT:
                stop.set()

    t_start = time.time()
    last_n, seq, fps_ema, t_prev, loops_seen = 0, 0, 0.0, None, 0
    restarts, code = 0, 0
    inflight = None    # the frame the GPU is working on: dict(seq, frame, t_cap, n, skipped, t0)

    def grab(timeout):
        """Newest camera frame not sent yet -> item dict, or None. Handles a looping video's cut."""
        nonlocal last_n, loops_seen
        frame, t_cap, n = source.wait_frame(last_n, timeout=timeout)
        if frame is None or n == last_n:
            return None
        if source.loops != loops_seen:   # a looping video restarted: a cut, not the person stepping back
            loops_seen = source.loops
            gate.reset()
            mapper.on_cut()
            pending["flags"] |= GW.FLAG_RESET
        item = {"frame": frame, "t_cap": t_cap, "n": n, "skipped": n - last_n - 1, "t0": time.perf_counter()}
        last_n = n
        return item

    def send(item):
        flags, pending["flags"] = pending["flags"], 0
        item["seq"] = client.submit(item["frame"], item["t_cap"], flags, st["crop"])
        return item

    try:
        while not stop.is_set():
            handle_ctrl()
            if args.max_seconds and time.time() - t_start > args.max_seconds:
                break
            if source is None:
                stop.wait(0.05)
                continue
            if args.exit_on_fail and not source.ok and time.time() - t_start > 5.0 and source.n == 0:
                print(f"[sender_gemx] source failed: {source.error}", file=sys.stderr, flush=True)
                code = 3
                break
            if source.ended and not source.is_alive() and inflight is None:
                print("[sender_gemx] video ended", flush=True)
                break
            item = inflight
            try:
                if item is None:
                    item = grab(0.2)
                    if item is None:
                        continue
                    send(item)
                rep = client.wait(item["seq"])
                # pipelining: the next frame goes to the GPU before this result is mapped (~5-7 ms of IK, SMPL, JSON
                # off the critical path); it is the newest frame at this moment, like the non-pipelined loop
                nxt = grab(0.0)
                inflight = send(nxt) if nxt is not None else None
            except RuntimeError as e:   # worker died / hung: restart it (the robot sees nobody meanwhile)
                inflight = None
                restarts += 1
                print(f"[sender_gemx] {e} -> restarting the worker ({restarts}/{args.worker_restarts})", flush=True)
                pub.send_multipart(P.encode(P.TOPIC_HUMAN, P.HumanFrame(seq=seq, t_capture=item["t_cap"] if item
                                                                       else time.time(), present=False)))
                seq += 1
                if restarts > args.worker_restarts:
                    code = EXIT_GEMX
                    break
                with st_lock:
                    st.update(note="loading", gemx="loading", present=False)
                if not client.start(stop):
                    if stop.is_set():
                        break
                    code = EXIT_GEMX
                    break
                with st_lock:
                    st.update(note="", gemx="ready")
                pending["flags"] |= GW.FLAG_RESET
                continue
            frame, t_cap, n, skipped, t0 = item["frame"], item["t_cap"], item["n"], item["skipped"], item["t0"]
            t1 = time.perf_counter()
            h, w = frame.shape[:2]
            det, view_det = None, None
            if rep["status"] == GW.ST_3D:
                det = soma_det(rep["joints"], rep["kp2d"], w, h, rep["K"])
                view_det = det
            elif rep["status"] == GW.ST_2D:
                view_det = {"kp_only": rep["kp2d"]}
            if rep["flags"] & GW.FLAG_RESET:
                gate.reset()
            det = gate(det)
            if gate.relocked:
                mapper.on_relock()
            r = mapper.process(det, t_cap, w, h)
            if mlp is not None and det is not None and r["arm_q"] is not None:
                from k1live.human import mirror_arms

                q = mlp(rep["body_pose"], rep["kp2d"], t_cap)
                r["arm_q"] = [float(v) for v in (mirror_arms(q) if mapper.mirror else q)]
            pose_ms = (time.perf_counter() - t0) * 1e3
            hf = P.HumanFrame(seq=seq, t_capture=t_cap, present=r["present"], conf=r["conf"], arm_q=r["arm_q"],
                              waist_yaw=r["waist_yaw"], bow=r["bow"], root=r["root"],
                              legs_visible=r["legs_visible"], loco=r["loco"], calibrated=r["calibrated"],
                              smpl=r["smpl"])
            pub.send_multipart(P.encode(P.TOPIC_HUMAN, hf))
            seq += 1
            view.put(frame, view_det, mapper.mirror)
            now = time.time()
            if t_prev is not None:
                fps_ema = 0.9 * fps_ema + 0.1 / max(now - t_prev, 1e-3) if fps_ema else 1.0 / max(now - t_prev, 1e-3)
            t_prev = now
            with st_lock:
                st.update(fps=fps_ema, present=r["present"], calibrated=r["calibrated"],
                          pose_ms=0.8 * st["pose_ms"] + 0.2 * pose_ms if st["pose_ms"] else pose_ms, frames=n,
                          sent=seq, gemx="ready" if rep["status"] == GW.ST_3D else
                          ("warming" if rep["status"] == GW.ST_2D else "ready"),
                          gemx_ms={k: round(v, 1) for k, v in rep["ms"].items()})
            if log is not None:
                log.write(json.dumps({"seq": hf.seq, "t_capture": t_cap, "t_sent": hf.t_sent,
                                      "gemx_ms": round((t1 - t0) * 1e3, 2), "worker_ms": rep["ms"],
                                      "status": rep["status"], "ik_ms": round(mapper.ik_ms, 2),
                                      "pose_ms": round(pose_ms, 2), "skipped": skipped,
                                      "present": r["present"]}) + "\n")
    finally:
        stop.set()
        if source is not None:
            source.stop()
        client.close()
        if http is not None:
            http.shutdown()
            http.server_close()
        if log is not None:
            log.close()
        ctx.destroy(0)
    print(f"[sender_gemx] exit {code} after {seq} frames", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
