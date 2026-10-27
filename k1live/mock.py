"""Protocol-faithful stand-ins for k1live.sender / k1live.sim_server (service development and fault tests).

    python -m k1live.mock sender --source <video> [--loop] [--present 40:20] [--crash-after S] [--hang-after S]
    python -m k1live.mock sim [--load-s 3] [--fall-every S] [--crash-after S] [--hang-after S]

Same ports, topics and messages as k1live/protocol.py. The mock sender never opens a camera: a camera source just
reports cam_ok=False ("mock: camera disabled"). The mock sim has no physics; it draws a stick figure from the
arm targets and follows the CTRL state machine (start/stop/reset/estop/set/gesture/drive), with optional injected
falls so the service's fall/reset handling can be exercised.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import zmq

from . import protocol as P


class MjpegServer:
    """Latest-frame multipart MJPEG server (/<path>.mjpg and a single-shot /<path>.jpg)."""

    def __init__(self, port, path):
        self.cond = threading.Condition()
        self.jpg = None
        self.seq = 0
        self.enabled = True
        srv = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path.split("?")[0] == f"/{path}.jpg":
                    jpg = srv.jpg
                    if jpg is None:
                        self.send_error(503)
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpg)))
                    self.end_headers()
                    self.wfile.write(jpg)
                    return
                if self.path.split("?")[0] != f"/{path}.mjpg":
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                last = -1
                try:
                    while True:
                        with srv.cond:
                            srv.cond.wait_for(lambda: srv.seq != last, timeout=1.0)
                            jpg, last = srv.jpg, srv.seq
                        if jpg is None or not srv.enabled:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return

        self.httpd = ThreadingHTTPServer((P.HOST, port), H)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def push(self, img_bgr, quality=75):
        ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            with self.cond:
                self.jpg = buf.tobytes()
                self.seq += 1
                self.cond.notify_all()


def _ctrl_sub(ctx):
    s = ctx.socket(zmq.SUB)
    s.connect(f"tcp://{P.HOST}:{P.CTRL_PORT}")
    s.setsockopt(zmq.SUBSCRIBE, P.TOPIC_CTRL)
    return s


def _drain(sock, cls=None):
    out = []
    while True:
        try:
            out.append(P.decode(sock.recv_multipart(zmq.NOBLOCK), cls))
        except zmq.Again:
            return out


class Faults:
    def __init__(self, args):
        self.t0 = time.time()
        self.crash_after = args.crash_after
        self.hang_after = args.hang_after

    def check(self):
        el = time.time() - self.t0
        if self.crash_after and el > self.crash_after:
            print(f"[mock] injected crash after {el:.1f} s", flush=True)
            os._exit(3)
        if self.hang_after and el > self.hang_after:
            print(f"[mock] injected hang after {el:.1f} s", flush=True)
            while True:
                time.sleep(3600)


# ---------------------------------------------------------------------------------------------------------
def run_sender(args):
    ctx = zmq.Context()
    pub_h = ctx.socket(zmq.PUB)
    pub_h.bind(f"tcp://{P.HOST}:{P.HUMAN_PORT}")
    pub_s = ctx.socket(zmq.PUB)
    pub_s.bind(f"tcp://{P.HOST}:{P.SENDER_STATE_PORT}")
    ctrl = _ctrl_sub(ctx)
    cam = MjpegServer(P.CAM_HTTP_PORT, "cam")
    faults = Faults(args)
    on_s, off_s = (float(x) for x in args.present.split(":")) if args.present else (1.0, 0.0)

    source = args.source
    mirror = not args.no_mirror
    cap, vfps, t_vid0, n_read = None, 30.0, 0.0, 0
    calibrated, calib_at = False, None
    seq, fps_ema, t_prev = 0, 0.0, time.time()
    t_state = 0.0
    t_start = time.time()

    def open_source(src):
        nonlocal cap, vfps, t_vid0, n_read
        if cap is not None:
            cap.release()
        cap = None
        if str(src).isdigit() or str(src).startswith("camera:"):
            return  # the mock never touches a real camera
        c = cv2.VideoCapture(str(src))
        if c.isOpened():
            cap, vfps, t_vid0, n_read = c, c.get(cv2.CAP_PROP_FPS) or 30.0, time.time(), 0

    loading_until = time.time() + args.load_s
    while time.time() < loading_until:  # "loading" while the (mock) model loads
        pub_s.send_multipart(P.encode(P.TOPIC_SENDER_STATE, P.SenderState(
            t=P.now(), cam_ok=False, source=str(source), fps=0.0, present=False, calibrated=False, pose_ms=0.0,
            note="loading")))
        time.sleep(0.25)
    open_source(source)
    print(f"[mock sender] source={source} opened={cap is not None}", flush=True)
    while True:
        faults.check()
        for m in _drain(ctrl):
            cmd, a = m.get("cmd"), m.get("args", {})
            if cmd == P.CMD_CALIBRATE:
                calibrated, calib_at = False, time.time() + 1.0
            elif cmd == P.CMD_SET:
                new = None
                if "camera" in a:
                    new = f"camera:{int(a['camera'])}"
                elif "video" in a:
                    new = str(a["video"])
                if new is not None and new != source:  # idempotent set: same source is a no-op
                    source = new
                    open_source(source)
                    print(f"[mock sender] source -> {source} opened={cap is not None}", flush=True)
                if "mirror" in a:
                    mirror = bool(a["mirror"])
            elif cmd == P.CMD_QUIT:
                return
        if calib_at and time.time() >= calib_at:
            calibrated, calib_at = True, None
        cam_ok = cap is not None
        frame = None
        if cap is not None:
            ok, frame = cap.read()
            if not ok:
                if args.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    t_vid0, n_read = time.time(), 0
                    continue
                cap.release()
                cap, cam_ok = None, False
            else:
                n_read += 1
                dt = t_vid0 + n_read / vfps - time.time()
                if dt > 0:
                    time.sleep(dt)
        else:
            time.sleep(1 / 30)
        t_cap = time.time()
        if args.cam_fail_after and t_cap - t_start > args.cam_fail_after:
            cam_ok, frame = False, None
        cyc = (t_cap - t_start) % (on_s + off_s) if off_s > 0 else 0.0
        present = cam_ok and cyc < on_s
        ph = 2 * math.pi * 0.25 * t_cap
        arm = [-1.2 * (0.5 + 0.5 * math.sin(ph)), 0.3, 0.0, 0.6, 0.0,
               -1.2 * (0.5 + 0.5 * math.cos(ph)), -0.3, 0.0, 0.6, 0.0] if present else None
        step = present and (cyc % 20.0) > 14.0
        hf = P.HumanFrame(seq=seq, t_capture=t_cap, present=present, conf=0.9 if present else 0.0, arm_q=arm,
                          waist_yaw=0.0 if present else None, bow=0.0 if present else None,
                          root={"x": 0.45 if step else 0.0, "y": 0.0, "yaw": 0.0} if present else None,
                          legs_visible=present, loco={"vx": 0.4 if step else 0.0, "vy": 0.0, "wz": 0.0}
                          if present else None, calibrated=calibrated)
        pub_h.send_multipart(P.encode(P.TOPIC_HUMAN, hf))
        seq += 1
        fps_ema = 0.9 * fps_ema + 0.1 / max(t_cap - t_prev, 1e-3)
        t_prev = t_cap
        if frame is not None:
            h, w = frame.shape[:2]
            sc = 640 / max(w, h)
            img = cv2.resize(frame, (int(w * sc), int(h * sc)))
            if mirror:
                img = cv2.flip(img, 1)
            if present:
                c = (img.shape[1] // 2, img.shape[0] // 2)
                cv2.circle(img, c, 12, (60, 220, 60), -1)
                cv2.line(img, (c[0] - 80, c[1]), (c[0] + 80, c[1]), (60, 220, 60), 4)
            cv2.putText(img, f"MOCK {seq}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cam.push(img)
        if t_cap - t_state >= 0.25:
            t_state = t_cap
            pub_s.send_multipart(P.encode(P.TOPIC_SENDER_STATE, P.SenderState(
                t=t_cap, cam_ok=cam_ok, source=source, fps=round(fps_ema, 1), present=present,
                calibrated=calibrated, pose_ms=18.0,
                note="" if cam_ok else ("mock: camera disabled" if "camera" in source else "no frames"))))


# ---------------------------------------------------------------------------------------------------------
def run_sim(args):
    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://{P.HOST}:{P.SIM_STATE_PORT}")
    ctrl = _ctrl_sub(ctx)
    hum = ctx.socket(zmq.SUB)
    hum.connect(f"tcp://{P.HOST}:{P.HUMAN_PORT}")
    hum.setsockopt(zmq.SUBSCRIBE, P.TOPIC_HUMAN)
    robot = MjpegServer(P.ROBOT_HTTP_PORT, "robot")
    faults = Faults(args)
    backends = ["split", "sonic"] if args.sonic else ["split"]
    st = dict(mode="idle", backend=args.backend, streaming=False, walk_enabled=True, estop=False, fallen=False,
              resets=0, gesture=None, max_speed=0.5)
    last_h = None
    drive, drive_t = (0.0, 0.0, 0.0), 0.0
    fall_t = None
    gesture_end = 0.0
    t0 = time.time()
    loading_until = t0 + args.load_s
    tick, t_state, t_render = 0, 0.0, 0.0
    hz_n, hz_t, hz = 0, time.time(), 0.0
    next_fall = t0 + args.fall_every if args.fall_every else None
    x = 0.0
    while time.time() < loading_until:
        pub.send_multipart(P.encode(P.TOPIC_SIM_STATE, P.SimState(
            t=P.now(), mode=st["mode"], backend=st["backend"], streaming=False, walk_enabled=True, estop=False,
            fallen=False, resets=0, pelvis_z=0.0, tilt_deg=0.0, loco_cmd=[0, 0, 0], human_present=False,
            human_age_ms=None, ctrl_hz=0.0, rt_factor=0.0, note="loading", backends=[])))
        time.sleep(0.25)
    print("[mock sim] ready", flush=True)
    t_next = time.time()
    while True:
        faults.check()
        now = time.time()
        for m in _drain(ctrl):
            cmd, a = m.get("cmd"), m.get("args", {})
            if cmd == P.CMD_START:
                st.update(streaming=True, estop=False)
            elif cmd == P.CMD_STOP:
                st["streaming"] = False
            elif cmd == P.CMD_RESET:
                st.update(fallen=False, resets=st["resets"] + 1)
                fall_t = None
                x = 0.0
            elif cmd == P.CMD_ESTOP:
                st.update(streaming=False, estop=True)
                drive = (0.0, 0.0, 0.0)
            elif cmd == P.CMD_SET:
                if "mode" in a and a["mode"] in P.MODES:
                    st["mode"] = a["mode"]
                if "backend" in a and a["backend"] in backends:
                    st["backend"] = a["backend"]
                if "walk" in a:
                    st["walk_enabled"] = bool(a["walk"])
                if "max_speed" in a:
                    st["max_speed"] = float(a["max_speed"])
            elif cmd == P.CMD_GESTURE:
                st.update(gesture=a.get("name"), mode="gesture")
                gesture_end = now + args.gesture_s
            elif cmd == P.CMD_DRIVE:
                drive, drive_t = (float(a.get("vx", 0)), float(a.get("vy", 0)), float(a.get("wz", 0))), now
            elif cmd == P.CMD_QUIT:
                return
        for h in _drain(hum):
            last_h = h
        if st["gesture"] and now > gesture_end:
            st["gesture"] = None
        if next_fall and now >= next_fall and not st["fallen"]:
            st["fallen"], fall_t = True, now
            next_fall = now + args.fall_every
        if st["fallen"] and fall_t and now - fall_t > args.auto_reset_s:
            st.update(fallen=False, resets=st["resets"] + 1)
            fall_t = None
            x = 0.0
        present = bool(last_h and last_h["present"] and now - last_h["t_capture"] < 0.5)
        if now - drive_t < 0.3:
            v = drive
        elif st["streaming"] and st["walk_enabled"] and st["mode"] == "live" and present and last_h.get("loco"):
            lo = last_h["loco"]
            v = (lo["vx"], lo["vy"], lo["wz"])
        else:
            v = (0.0, 0.0, 0.0)
        v = tuple(float(np.clip(c, -st["max_speed"], st["max_speed"])) for c in v)
        if st["fallen"] or st["estop"]:
            v = (0.0, 0.0, 0.0)
        x += v[0] * 0.02
        tick += 1
        hz_n += 1
        if now - hz_t >= 1.0:
            hz, hz_n, hz_t = hz_n / (now - hz_t), 0, now
        if now - t_render >= 1 / 25:
            t_render = now
            img = np.full((540, 960, 3), 235, np.uint8)
            cx, cy = 480 + int(60 * math.sin(x)), 300
            arm = last_h["arm_q"] if (present and st["streaming"] and st["mode"] == "live") else None
            col = (60, 60, 200) if st["fallen"] else (90, 90, 90)
            if st["fallen"]:
                cv2.line(img, (cx - 120, 470), (cx + 120, 470), col, 18)
            else:
                cv2.line(img, (cx, cy - 120), (cx, cy + 60), col, 18)
                cv2.line(img, (cx, cy + 60), (cx - 40, cy + 170), col, 14)
                cv2.line(img, (cx, cy + 60), (cx + 40, cy + 170), col, 14)
                cv2.circle(img, (cx, cy - 160), 32, col, -1)
                for side, sgn in ((0, 1), (5, -1)):
                    a = (arm[side] if arm else 0.0) + (0.6 * math.sin(now * 3) if st["gesture"] else 0.0)
                    ang = math.pi / 2 + sgn * (-a)
                    ex, ey = cx + sgn * 30 + int(110 * math.cos(ang) * sgn), cy - 100 + int(110 * math.sin(ang))
                    cv2.line(img, (cx + sgn * 30, cy - 100), (ex, ey), col, 12)
            cv2.putText(img, f"MOCK SIM {st['backend']} {st['mode']} {'STREAM' if st['streaming'] else ''}",
                        (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)
            robot.push(img)
        if now - t_state >= 0.1:
            t_state = now
            pub.send_multipart(P.encode(P.TOPIC_SIM_STATE, P.SimState(
                t=now, mode=st["mode"], backend=st["backend"], streaming=st["streaming"],
                walk_enabled=st["walk_enabled"], estop=st["estop"], fallen=st["fallen"], resets=st["resets"],
                pelvis_z=0.3 if st["fallen"] else 0.78, tilt_deg=80.0 if st["fallen"] else 2.0,
                loco_cmd=list(v), human_present=present,
                human_age_ms=round((now - last_h["t_capture"]) * 1e3, 1) if last_h else None,
                ctrl_hz=round(hz, 1), rt_factor=1.0, gesture=st["gesture"], arm_err_deg=3.0 if present else None,
                note="mock", backends=backends)))
        t_next += 0.02
        sl = t_next - time.time()
        if sl > 0:
            time.sleep(sl)
        elif sl < -0.2:
            t_next = time.time()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("role", choices=["sender", "sim"])
    ap.add_argument("--source", default="camera:0")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--present", default="", help="ON:OFF seconds cycle of the person being in view")
    ap.add_argument("--backend", default="split")
    ap.add_argument("--sonic", action="store_true", help="mock sim reports the sonic backend as available")
    ap.add_argument("--load-s", type=float, default=1.0)
    ap.add_argument("--fall-every", type=float, default=0.0)
    ap.add_argument("--auto-reset-s", type=float, default=2.0)
    ap.add_argument("--gesture-s", type=float, default=6.0)
    ap.add_argument("--crash-after", type=float, default=0.0)
    ap.add_argument("--hang-after", type=float, default=0.0)
    ap.add_argument("--cam-fail-after", type=float, default=0.0)
    ap.add_argument("--no-autostart", action="store_true", help="(sim_server compatibility; the mock waits anyway)")
    ap.add_argument("--idle-after", type=float, default=8.0, help="(sim_server compatibility, unused)")
    ap.add_argument("--log-dir", default="", help="(sim_server compatibility, unused)")
    args = ap.parse_args(argv)
    if args.source.isdigit():
        args.source = f"camera:{args.source}"
    try:
        (run_sender if args.role == "sender" else run_sim)(args)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    sys.exit(main())
