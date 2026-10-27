"""Robot view for the audience: offscreen MuJoCo render of the live K1 -> /robot.mjpg (MuJoCo simulation only).

Runs in its own process (multiprocessing "spawn") so rendering never holds the control loop's GIL: on a Mac (CGL) a
1280x720 frame takes ~15 ms + ~4 ms JPEG, which would be most of a 20 ms control tick. The control loop only copies
qpos into shared memory each tick (RenderLink.push, a few microseconds); the render process rebuilds the same MjModel
(the backend's `render_model_fn`), runs forward kinematics on the snapshot and draws it with the bright event scene
of k1/sim.py, then serves:

    http://127.0.0.1:8702/robot.mjpg   multipart MJPEG, 1280x720, up to 30 fps (frames only while a client watches)
    http://127.0.0.1:8702/robot.jpg    one frame
    http://127.0.0.1:8702/state        {"fps", "render_ms", "clients", "frames"}

Camera: front, slightly to the side (azimuth 160 deg = 20 deg off the robot's front at spawn), follows the pelvis
smoothly when the robot walks (critically damped, ~0.5 s), and slowly turns with the robot when it has turned away
by more than 50 deg for a while, so the audience keeps seeing its front.
"""
from __future__ import annotations

import importlib
import json
import multiprocessing as mp
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from k1live.protocol import HOST, ROBOT_HTTP_PORT
from k1live.timing import sleep_until

CAM = dict(azimuth=160.0, distance=2.7, elevation=-8.0, lookat_z=0.68)


def load_fn(spec: str):
    mod, fn = spec.split(":")
    return getattr(importlib.import_module(mod), fn)


class RenderLink:
    """Control-process side: shared qpos buffer + the render process handle."""

    def __init__(self, model_fn: str, nq: int, port: int = ROBOT_HTTP_PORT, size=(1280, 720), fps=30.0,
                 quality=80):
        ctx = mp.get_context("spawn")
        self.nq = nq
        self.buf = ctx.Array("d", nq, lock=False)
        self.seq = ctx.Value("L", 0, lock=False)
        self.lock = ctx.Lock()
        self.stop_ev = ctx.Event()
        self.ready = ctx.Event()
        self.proc = ctx.Process(target=_render_main, name="k1live-render", daemon=True,
                                args=(model_fn, nq, self.buf, self.seq, self.lock, self.stop_ev, self.ready, port,
                                      size, fps, quality, os.getpid()))
        self.proc.start()
        self.model_fn = model_fn

    def push(self, qpos):
        if self.lock.acquire(timeout=0.002):  # never block the control loop
            try:
                self.buf[:] = qpos[: self.nq]
                self.seq.value += 1
            finally:
                self.lock.release()

    def alive(self):
        return self.proc.is_alive()

    def close(self, timeout=2.0):
        self.stop_ev.set()
        self.proc.join(timeout)
        if self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(1.0)


class FollowCam:
    """Pelvis-following free camera (no shake): critically damped position, slow heading follow."""

    def __init__(self, azimuth=CAM["azimuth"], distance=CAM["distance"], elevation=CAM["elevation"],
                 lookat_z=CAM["lookat_z"], tau=0.5, yaw_follow_deg=50.0):
        import mujoco

        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.az0, self.cam.distance, self.cam.elevation = azimuth, distance, elevation
        self.lookat_z, self.tau, self.yaw_follow = lookat_z, tau, np.radians(yaw_follow_deg)
        self.p = self.v = self.vel = self.xy_prev = None
        self.yaw_off = 0.0
        self.t = None

    def update(self, qpos, t):
        xy = np.asarray(qpos[:2], np.float64)
        w, x, y, z = qpos[3:7]
        yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        if self.p is None or t - self.t > 0.5:  # first frame, or resuming after nobody watched: snap, no swing
            if self.p is None:
                self.yaw_off = 0.0
            err = (yaw - self.yaw_off + np.pi) % (2 * np.pi) - np.pi
            if abs(err) > self.yaw_follow:
                self.yaw_off += np.sign(err) * (abs(err) - self.yaw_follow)
            self.p, self.v, self.vel, self.xy_prev, self.t = xy.copy(), np.zeros(2), np.zeros(2), xy.copy(), t
        dt = float(np.clip(t - self.t, 0.0, 0.2))
        self.t = t
        if dt > 0:  # pelvis velocity, smoothed over ~0.5 s (steps sway the pelvis sideways)
            a_v = dt / (0.5 + dt)
            self.vel += a_v * ((xy - self.xy_prev) / dt - self.vel)
        self.xy_prev = xy.copy()
        k = 1.0 / self.tau  # critically damped spring toward the pelvis + 2 tau v: no lag at constant speed
        a = k * k * (xy + 2 * self.tau * self.vel - self.p) - 2 * k * self.v
        self.v += a * dt
        self.p += self.v * dt
        err = (yaw - self.yaw_off + np.pi) % (2 * np.pi) - np.pi
        if abs(err) > self.yaw_follow:  # robot turned away: swing round slowly (0.5 rad/s)
            self.yaw_off += np.sign(err) * min(abs(err) - self.yaw_follow + 0.05, 0.5 * dt)
        self.cam.lookat[:] = [self.p[0], self.p[1], self.lookat_z]
        self.cam.azimuth = self.az0 + np.degrees(self.yaw_off)
        return self.cam


def _render_main(model_fn, nq, buf, seq, lock, stop_ev, ready, port, size, fps, quality, parent_pid):
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the parent decides when we stop
    signal.signal(signal.SIGTERM, lambda *_: stop_ev.set())
    if sys.platform.startswith("linux") and "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"  # headless Ubuntu: no X display needed
    import cv2
    import mujoco

    model = load_fn(model_fn)()
    data = mujoco.MjData(model)
    W, H = size
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, W)
    model.vis.global_.offheight = max(model.vis.global_.offheight, H)
    renderer = mujoco.Renderer(model, height=H, width=W)
    opt = mujoco.MjvOption()
    cam = FollowCam()
    st = {"fps": 0.0, "render_ms": 0.0, "clients": 0, "frames": 0, "jpeg": None, "seq": 0}
    cond = threading.Condition()
    want = threading.Event()  # set while someone needs frames

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path == "/robot.mjpg":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                with cond:
                    st["clients"] += 1
                want.set()
                last = -1
                try:
                    while not stop_ev.is_set():
                        with cond:
                            if st["seq"] == last:
                                cond.wait(1.0)
                            jpg, s = st["jpeg"], st["seq"]
                        if jpg is None or s == last:
                            continue
                        last = s
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    with cond:
                        st["clients"] -= 1
            elif path == "/robot.jpg":
                want.set()
                with cond:
                    if st["jpeg"] is None or st["clients"] == 0:
                        cond.wait(1.0)
                    jpg = st["jpeg"]
                if jpg is None:
                    self.send_error(503, "no frame yet")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpg)))
                self.end_headers()
                self.wfile.write(jpg)
            elif path in ("/state", "/healthz"):
                body = json.dumps({k: v for k, v in st.items() if k != "jpeg"}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_error(404)

    try:
        srv = ThreadingHTTPServer((HOST, port), Handler)
    except OSError as e:
        print(f"[render] port {port}: {e}", flush=True)
        return
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.2}, daemon=True).start()
    ready.set()
    q = np.zeros(nq)
    period = 1.0 / fps
    t_next = time.monotonic()
    t_snap_frames, n_frames, t_want = time.monotonic(), 0, 0.0
    last_seq = -1
    try:
        while not stop_ev.is_set():
            if os.getppid() != parent_pid:  # parent died without telling us
                break
            if want.is_set():
                t_want = time.monotonic()
                want.clear()
            with cond:
                active = st["clients"] > 0
            if not active and time.monotonic() - t_want > 2.0:  # nobody watching: idle cheaply
                time.sleep(0.05)
                t_next = time.monotonic()
                continue
            t_next += period
            with lock:
                q[:] = buf[:]
                s = seq.value
            t0 = time.perf_counter()
            if s != last_seq or s == 0:
                data.qpos[:] = q
                mujoco.mj_forward(model, data)
                last_seq = s
            renderer.update_scene(data, camera=cam.update(q, time.monotonic()), scene_option=opt)
            rgb = renderer.render()
            ok, jpg = cv2.imencode(".jpg", rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
            dt_ms = (time.perf_counter() - t0) * 1e3
            if ok:
                with cond:
                    st["jpeg"], st["seq"] = jpg.tobytes(), st["seq"] + 1
                    st["frames"] += 1
                    st["render_ms"] = round(0.9 * st["render_ms"] + 0.1 * dt_ms if st["render_ms"] else dt_ms, 2)
                    cond.notify_all()
            n_frames += 1
            now = time.monotonic()
            if now - t_snap_frames >= 1.0:
                st["fps"] = round(n_frames / (now - t_snap_frames), 1)
                t_snap_frames, n_frames = now, 0
            sl = t_next - time.monotonic()
            if sl > 0:
                sleep_until(t_next, clock=time.monotonic, spin=0.0)
            elif sl < -period:
                t_next = time.monotonic()  # behind: drop frames instead of bursting
    finally:
        srv.shutdown()
        srv.server_close()
        try:
            renderer.close()
        except Exception:
            pass
