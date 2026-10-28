"""Perception process: camera or video file -> MediaPipe -> K1 targets -> HUMAN topic (~30 Hz). MuJoCo-free.

    .venv/bin/python -m k1live.sender --source 0                  # live camera index 0 (the demo laptop)
    .venv/bin/python -m k1live.sender --source clip.mp4 --loop    # "fake camera": the file played in real time

Per frame (k1live/human.py): MediaPipe Pose full (sonic/pose_mp.py, single-person tracking) -> One-Euro -> K1 arm IK
(the offline reel's retargeting, every frame) + waist yaw / lean / pelvis offset from the calibrated home + walking
intent (k1live.locomotion.LocoEstimator if present) -> mirror -> protocol.HumanFrame on tcp://127.0.0.1:5701.

The source runs in its own thread and always hands the newest frame to the pose stage (older frames are dropped), so
latency does not grow when a frame takes longer than the camera period. A video file is decoded on its own clock
(frame i is "captured" at start + i / fps, frames the reader is late for are skipped), i.e. it behaves like a camera.

Also: /cam.mjpg (camera with skeleton overlay, <= 720p, mirrored like the robot when mirror is on), /cam.jpg,
/state on http://127.0.0.1:8701; SenderState on 5704 at 4 Hz (first one before the heavy imports, < 1 s after
start, then 4 Hz). CTRL (connects to the service on 5702): calibrate, set {mirror, camera, video, loop}, quit; a set
that repeats the current value is a no-op (the service re-sends all settings on reconnect). Without --source no
source is opened until a set camera / video arrives (never a default camera).

Exit codes: 0 = quit / SIGTERM / SIGINT / --max-seconds / end of a non-looping video, 2 = bad arguments,
3 = source failed and --exit-on-fail, 4 = a port is already in use. A camera that fails or disappears does not
end the process: SenderState.cam_ok turns false and the source is retried every 2 s (or switch with set video).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import zmq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from k1live import protocol as P  # noqa: E402
from k1live.timing import sleep_until  # noqa: E402

MAX_H = 720


# ------------------------------------------------------------------------------------------------ frame sources
class FrameSource(threading.Thread):
    """Camera index or video file, newest-frame slot. spec: "camera:N" or a path."""

    def __init__(self, spec: str, loop: bool = True, max_h: int = MAX_H, cam_size=(1280, 720), cam_fps=30):
        super().__init__(daemon=True, name="source")
        self.spec, self.loop, self.max_h, self.cam_size, self.cam_fps = spec, loop, max_h, cam_size, cam_fps
        self.is_camera = spec.startswith("camera:")
        self.cond = threading.Condition()
        self.frame, self.t_capture, self.n = None, 0.0, 0
        self.ok = False
        self.ended = False
        self.error = ""
        self.fps_nominal = 0.0
        self.loops = 0
        self.skipped = 0
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def wait_frame(self, last_n: int, timeout: float = 0.5):
        with self.cond:
            if self.n == last_n:
                self.cond.wait(timeout)
            return self.frame, self.t_capture, self.n

    def _publish(self, frame, t):
        import cv2

        if frame.shape[0] > self.max_h:
            s = self.max_h / frame.shape[0]
            frame = cv2.resize(frame, (int(round(frame.shape[1] * s)), self.max_h), interpolation=cv2.INTER_AREA)
        with self.cond:
            self.frame, self.t_capture, self.n = frame, t, self.n + 1
            self.cond.notify_all()

    def _open(self):
        import cv2

        if self.is_camera:
            cap = cv2.VideoCapture(int(self.spec.split(":", 1)[1]))
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_size[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_size[1])
                cap.set(cv2.CAP_PROP_FPS, self.cam_fps)
        else:
            cap = cv2.VideoCapture(self.spec)
        if not cap.isOpened():
            self.error = f"cannot open {self.spec}"
            return None
        self.fps_nominal = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if not (1.0 <= self.fps_nominal <= 240.0):
            self.fps_nominal = 30.0
        return cap

    def run(self):
        while not self._stop.is_set():
            cap = self._open()
            if cap is None:
                self.ok = False
                if self._stop.wait(2.0):
                    break
                continue
            self.ok, self.error = True, ""
            try:
                more = self._run_camera(cap) if self.is_camera else self._run_video(cap)
            finally:
                cap.release()
            if not more:
                break
        self.ok = False

    def _run_camera(self, cap):
        fails = 0
        while not self._stop.is_set():
            ok, frame = cap.read()
            t = time.time()
            if not ok or frame is None:
                fails += 1
                if fails > 15:  # camera gone: reopen
                    self.ok, self.error = False, "camera read failed"
                    return not self._stop.wait(2.0)
                time.sleep(0.02)
                continue
            fails = 0
            self.ok = True
            self._publish(frame, t)
        return False

    def _run_video(self, cap):
        import cv2

        fps = self.fps_nominal
        t0, i = time.time(), 0  # frame i is due at t0 + i / fps
        while not self._stop.is_set():
            due_idx = int((time.time() - t0) * fps)
            while i < due_idx:  # late: skip without decoding into Python
                if not cap.grab():
                    break
                i += 1
                self.skipped += 1
            ok, frame = cap.read()
            if not ok or frame is None:
                if not self.loop:
                    self.ended = True
                    return False
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.loops += 1
                t0, i = time.time(), 0
                continue
            t_due = t0 + i / fps
            if not sleep_until(t_due, clock=time.time, stop=self._stop):
                return False
            self._publish(frame, t_due)  # captured "at" its due time (a late read counts as latency)
            i += 1
        return False


# ------------------------------------------------------------------------------------------------ MJPEG server
class CamView:
    """Latest (frame, detection) -> lazily drawn + JPEG-encoded once per frame for every /cam.mjpg client."""

    def __init__(self, max_fps=30.0, quality=80, draw=None):
        self.draw = draw   # draw(frame, det) -> frame with the skeleton; None = MediaPipe overlay (k1live.sender_gemx)
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.frame, self.det, self.mirror, self.seq = None, None, True, 0
        self._jpeg, self._jpeg_seq = None, -1
        self.max_fps, self.quality = max_fps, quality
        self.clients = 0

    def put(self, frame, det, mirror):
        with self.cond:
            self.frame, self.det, self.mirror = frame, det, mirror
            self.seq += 1
            self.cond.notify_all()

    def jpeg(self, last_seq=-1, timeout=1.0):
        import cv2

        from sonic.pose_mp import draw_overlay

        with self.cond:
            if self.seq == last_seq:
                self.cond.wait(timeout)
            if self.frame is None:
                return None, self.seq
            if self._jpeg_seq == self.seq:
                return self._jpeg, self.seq
            frame, det, mirror, seq = self.frame, self.det, self.mirror, self.seq
        if det is None:
            img = frame
        elif self.draw is not None:
            img = self.draw(frame, det)
        else:
            img = draw_overlay(frame, det, "", color=(80, 220, 120))
        if mirror:
            img = cv2.flip(img, 1)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        with self.cond:
            if ok and seq >= self._jpeg_seq:
                self._jpeg, self._jpeg_seq = buf.tobytes(), seq
            return self._jpeg, seq


def make_http(view: CamView, state_fn, port: int, stop: threading.Event):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path == "/cam.mjpg":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                view.clients += 1
                seq, t_last = -1, 0.0
                try:
                    while not stop.is_set():
                        jpg, seq2 = view.jpeg(seq)
                        if jpg is None or seq2 == seq:
                            continue
                        seq = seq2
                        dt = 1.0 / view.max_fps - (time.monotonic() - t_last)
                        if dt > 0:
                            time.sleep(dt)
                        t_last = time.monotonic()
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    view.clients -= 1
            elif path == "/cam.jpg":
                jpg, _ = view.jpeg(-2, timeout=0)
                if jpg is None:
                    self.send_error(503, "no frame yet")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpg)))
                self.end_headers()
                self.wfile.write(jpg)
            elif path in ("/state", "/healthz"):
                body = json.dumps(state_fn()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_error(404)

    srv = ThreadingHTTPServer((P.HOST, port), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.2}, daemon=True, name="http").start()
    return srv


# ------------------------------------------------------------------------------------------------ main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--source", default="", help="camera index (digits) or video file (played in real time)")
    src.add_argument("--camera", type=int, default=None, help="alias of --source N")
    src.add_argument("--video", default="", help="alias of --source PATH")
    ap.add_argument("--loop", action="store_true", help="loop the video file")
    ap.add_argument("--no-mirror", action="store_true", help="robot copies the person's own left/right")
    ap.add_argument("--model", default="full", choices=["lite", "full", "heavy"], help="MediaPipe Pose model")
    ap.add_argument("--ik-smoothing", type=float, default=0.6, help="IK EMA weight of the new solution (1 = none)")
    ap.add_argument("--arm-filter-cutoff", type=float, default=1.5,
                    help="dir IK One-Euro minimum cutoff in Hz (higher = less smoothing / less lag)")
    ap.add_argument("--arm-filter-beta", type=float, default=1.0,
                    help="dir IK One-Euro response to motion speed")
    ap.add_argument("--arm-ik", default=os.environ.get("K1LIVE_ARM_IK", "shape18"), choices=["shape18", "dir"],
                    help="arm IK: shape18 = position IK, or dir = bone directions with frontal-plane angles from the image "
                    "(k1live/arm_ik.py, docs/architecture.md). Default $K1LIVE_ARM_IK or shape18")
    ap.add_argument("--no-loco", action="store_true", help="do not load k1live.locomotion (loco always None)")
    ap.add_argument("--loco-profile", default=os.environ.get("K1LIVE_LOCO_PROFILE", "visitor"),
                    choices=["visitor", "presenter"], help="walking intent: visitor = floor joystick (step / turn "
                    "-> walk / turn), presenter = no turning, walk along while the person strolls (stage, podium; "
                    "docs/architecture.md). Default $K1LIVE_LOCO_PROFILE or visitor")
    ap.add_argument("--crop", default=None,
                    help="x0,y0,x1,y1 (0..1 of the frame): look for the person only in this region, e.g. the stage "
                    "without the audience heads in front (default $K1LIVE_CROP or the whole frame)")
    ap.add_argument("--no-smpl", action="store_true", help="do not compute the whole-body stream (HumanFrame.smpl)")
    ap.add_argument("--no-auto-calibrate", action="store_true", help="calibrate only on CTRL calibrate")
    ap.add_argument("--http-port", type=int, default=P.CAM_HTTP_PORT)
    ap.add_argument("--no-http", action="store_true")
    ap.add_argument("--human-port", type=int, default=P.HUMAN_PORT)
    ap.add_argument("--state-port", type=int, default=P.SENDER_STATE_PORT)
    ap.add_argument("--ctrl-port", type=int, default=P.CTRL_PORT)
    ap.add_argument("--max-seconds", type=float, default=0.0, help="exit after this long (0 = run until stopped)")
    ap.add_argument("--exit-on-fail", action="store_true", help="exit 3 if the source cannot be opened in 5 s")
    ap.add_argument("--log", default="", help="write one JSON line per frame (timing) to this file")
    a = ap.parse_args(argv)
    if not (0 < a.arm_filter_cutoff <= 120 and 0 <= a.arm_filter_beta <= 20):
        ap.error("arm filter requires 0 < cutoff <= 120 and 0 <= beta <= 20")
    if a.camera is not None:
        a.source = str(a.camera)
    elif a.video:
        a.source = a.video
    a.spec = source_spec(a.source)
    if a.crop is not None:
        a.crop_box = parse_crop(a.crop)
        if a.crop and a.crop_box is None:
            ap.error(f"--crop {a.crop}: want x0,y0,x1,y1 with 0 <= x0 < x1 <= 1, 0 <= y0 < y1 <= 1")
    else:  # from the environment (the service's child): a bad value must not stop the sender - whole frame
        env = os.environ.get("K1LIVE_CROP", "")
        a.crop_box = parse_crop(env)
        if env and a.crop_box is None:
            print(f"[sender] ignoring K1LIVE_CROP={env!r} (want x0,y0,x1,y1 in 0..1)", file=sys.stderr, flush=True)
    if a.loco_profile not in ("visitor", "presenter"):  # a bad $K1LIVE_LOCO_PROFILE: the default, not an exit
        print(f"[sender] ignoring walking profile {a.loco_profile!r}", file=sys.stderr, flush=True)
        a.loco_profile = "visitor"
    if a.spec and not a.spec.startswith("camera:") and not Path(a.spec).exists():
        ap.error(f"--source {a.source}: no such file")
    return a


def parse_crop(s):
    """ "x0,y0,x1,y1" or [x0, y0, x1, y1] -> tuple of 4 floats in [0, 1] (a box of at least 10 % per side);
    "" / None / [] (whole frame) or anything invalid -> None."""
    if s is None or s == "" or (isinstance(s, (list, tuple)) and not s):
        return None
    try:
        b = tuple(float(v) for v in (s if isinstance(s, (list, tuple)) else str(s).split(",")))
    except (TypeError, ValueError):
        return None
    if len(b) != 4 or not (0.0 <= b[0] < b[2] <= 1.0 and 0.0 <= b[1] < b[3] <= 1.0) or \
            b[2] - b[0] < 0.1 or b[3] - b[1] < 0.1:
        return None
    return b


def source_spec(s) -> str:
    """CLI / CTRL source -> "camera:N", a video path, or "" (none)."""
    s = "" if s is None else str(s).strip()
    if not s:
        return ""
    return f"camera:{int(s)}" if s.isdigit() else str(Path(s).expanduser().resolve())


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
        print(f"[sender] bind failed: {e}", file=sys.stderr, flush=True)
        ctx.destroy(0)
        return 4
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.LINGER, 0)
    sub.connect(f"tcp://{P.HOST}:{args.ctrl_port}")
    sub.setsockopt(zmq.SUBSCRIBE, P.TOPIC_CTRL)

    spec = args.spec
    st = {"spec": spec, "mirror": not args.no_mirror, "loop": args.loop, "fps": 0.0, "present": False,
          "calibrated": False, "pose_ms": 0.0, "note": "loading", "source": None, "mapper": None, "frames": 0,
          "sent": 0, "loco_profile": args.loco_profile, "crop": args.crop_box}
    st_lock = threading.Lock()

    def sender_state():
        src, mp = st["source"], st["mapper"]
        note = st["note"] or ("" if src is not None else "no source")
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
            st_pub.send_multipart(P.encode(P.TOPIC_SENDER_STATE, msg))
            stop.wait(0.25)

    threading.Thread(target=state_loop, daemon=True, name="state").start()
    print(f"[sender] state on {args.state_port}, HUMAN on {args.human_port}, source {spec or '(none)'}, "
          f"walking profile {args.loco_profile}, arm IK {args.arm_ik}"
          + (f", person region {args.crop_box}" if args.crop_box else ""), flush=True)

    # heavy imports after the first SenderState is out

    from k1live.human import HumanMapper, JumpGate
    from sonic.pose_mp import PoseTracker

    def new_tracker():
        tr = PoseTracker(args.model, crop=st["crop"])  # landmarks come back in full-frame coordinates
        tr.max_jump = float("inf")  # continuity is JumpGate's job (self-healing)
        return tr

    view = CamView()
    http = None
    if not args.no_http:
        try:
            http = make_http(view, lambda: {**asdict(sender_state()), "clients": view.clients,
                                            "loco_profile": st["loco_profile"],
                                            "crop": None if st["crop"] is None else list(st["crop"])},
                             args.http_port, stop)
        except OSError as e:
            print(f"[sender] http port {args.http_port}: {e}", file=sys.stderr, flush=True)
            stop.set()
            ctx.destroy(0)
            return 4

    tracker = new_tracker()
    gate = JumpGate()
    mapper = HumanMapper(mirror=not args.no_mirror, ik_smoothing=args.ik_smoothing, loco=not args.no_loco,
                         auto_calibrate=not args.no_auto_calibrate, loco_profile=args.loco_profile,
                         smpl=not args.no_smpl, arm_ik=args.arm_ik,
                         arm_filter=(args.arm_filter_cutoff, args.arm_filter_beta))
    source = None
    if spec:
        source = FrameSource(spec, loop=args.loop)
        source.start()
    with st_lock:
        st.update(source=source, mapper=mapper, note="")
    if mapper.loco is not None and not mapper.loco.available:
        print(f"[sender] no walking intent: {mapper.loco.error}", flush=True)
    log = open(args.log, "a") if args.log else None

    def switch(new_spec, loop=True):
        nonlocal source, last_n, tracker, loops_seen
        if new_spec == st["spec"] and source is not None and loop == source.loop and not source.ended:
            return  # idempotent: the service re-sends its settings on reconnect
        if source is not None:
            source.stop()
        # a fresh MediaPipe graph: in VIDEO mode it keeps tracking the previous clip's ROI, whose bogus first
        # result used to lock the single-person gate out for the whole clip (10-min rehearsal, vertical clip)
        tracker.close()
        tracker = new_tracker()
        gate.reset()
        source = FrameSource(new_spec, loop=loop)
        source.start()
        last_n, loops_seen = 0, 0
        mapper.new_scene()  # new person / framing: both homes again (automatic, stillness-gated)
        with st_lock:
            st.update(spec=new_spec, source=source, loop=loop, note="")
        print(f"[sender] source -> {new_spec} (loop={loop})", flush=True)

    def set_crop(value):
        """Person search region from CTRL set (None / [] = whole frame). Same box = no-op; a bad box is ignored."""
        nonlocal tracker
        box = parse_crop(value)
        if box is None and value not in (None, "", []):
            st["note"] = f"영역 무시: {value!r}"
            return
        if box == st["crop"]:
            return
        st["crop"] = box
        tracker.close()             # MediaPipe keeps tracking its last ROI: a new graph for the new region
        tracker = new_tracker()
        gate.reset()
        mapper.on_relock()          # the person may now be someone else / elsewhere: walking homes again
        print(f"[sender] person region {box or 'whole frame'}", flush=True)

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
                print("[sender] calibrating (stand still facing the camera)", flush=True)
            elif m.cmd == P.CMD_SET:
                a = m.args or {}
                if "mirror" in a and bool(a["mirror"]) != mapper.mirror:
                    mapper.set_mirror(bool(a["mirror"]))
                    st["mirror"] = bool(a["mirror"])
                    print(f"[sender] mirror {mapper.mirror}", flush=True)
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
                    print(f"[sender] walking profile {a['loco_profile']}", flush=True)
                if "crop" in a:
                    set_crop(a["crop"])
            elif m.cmd == P.CMD_QUIT:
                stop.set()

    t_start = time.time()
    t_t0 = time.monotonic()
    last_n, seq, fps_ema, t_prev, loops_seen = 0, 0, 0.0, None, 0
    code = 0
    try:
        while not stop.is_set():
            handle_ctrl()
            if args.max_seconds and time.time() - t_start > args.max_seconds:
                break
            if source is None:
                stop.wait(0.05)
                continue
            if args.exit_on_fail and not source.ok and time.time() - t_start > 5.0 and source.n == 0:
                print(f"[sender] source failed: {source.error}", file=sys.stderr, flush=True)
                code = 3
                break
            if source.ended and not source.is_alive():
                print("[sender] video ended", flush=True)
                break
            frame, t_cap, n = source.wait_frame(last_n, timeout=0.2)
            if frame is None or n == last_n:
                continue
            if source.loops != loops_seen:  # a looping video restarted: a cut, not the person stepping back
                loops_seen = source.loops
                gate.reset()
                mapper.on_cut()
            skipped = n - last_n - 1
            last_n = n
            t0 = time.perf_counter()
            det = gate(tracker.process(frame, time.monotonic() - t_t0))
            if gate.relocked:  # tracking jumped (someone else): not a step, walking homes again
                mapper.on_relock()
            t1 = time.perf_counter()
            h, w = frame.shape[:2]
            r = mapper.process(det, t_cap, w, h)
            pose_ms = (time.perf_counter() - t0) * 1e3
            hf = P.HumanFrame(seq=seq, t_capture=t_cap, present=r["present"], conf=r["conf"], arm_q=r["arm_q"],
                              waist_yaw=r["waist_yaw"], bow=r["bow"], root=r["root"],
                              legs_visible=r["legs_visible"], loco=r["loco"], calibrated=r["calibrated"],
                              smpl=r["smpl"])
            pub.send_multipart(P.encode(P.TOPIC_HUMAN, hf))
            seq += 1
            view.put(frame, det, mapper.mirror)
            now = time.time()
            if t_prev is not None:
                fps_ema = 0.9 * fps_ema + 0.1 / max(now - t_prev, 1e-3) if fps_ema else 1.0 / max(now - t_prev, 1e-3)
            t_prev = now
            with st_lock:
                st.update(fps=fps_ema, present=r["present"], calibrated=r["calibrated"],
                          pose_ms=0.8 * st["pose_ms"] + 0.2 * pose_ms if st["pose_ms"] else pose_ms, frames=n,
                          sent=seq)
            if log is not None:
                log.write(json.dumps({"seq": hf.seq, "t_capture": t_cap, "t_sent": hf.t_sent, "mp_ms": round(
                    (t1 - t0) * 1e3, 2), "ik_ms": round(mapper.ik_ms, 2), "smpl_ms": round(mapper.smpl_ms, 2),
                    "pose_ms": round(pose_ms, 2),
                    "skipped": skipped, "present": r["present"]}) + "\n")
    finally:
        stop.set()
        if source is not None:
            source.stop()
        if http is not None:
            http.shutdown()
            http.server_close()
        tracker.close()
        if log is not None:
            log.close()
        ctx.destroy(0)
    print(f"[sender] exit {code} after {seq} frames", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
