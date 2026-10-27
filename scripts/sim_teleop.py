"""M3 deploy+sim process: ZMQ protocol v3 (SMPL) -> SONIC encoder/decoder -> MuJoCo G1, real time.

Equivalent of `run_sim_loop.py` + `deploy.sh --input-type zmq sim` in one Python process.
Run under mjpython on macOS for the interactive viewer:
    .venv/bin/mjpython scripts/sim_teleop.py
Headless (records an mp4, for testing):
    .venv/bin/python scripts/sim_teleop.py --headless --record videos/live.mp4 --max-seconds 30
Another embodiment (e.g. a ROBOTIS K1 SONIC checkpoint in <dir>/release/model_{encoder,decoder}.onnx):
    .venv/bin/mjpython scripts/sim_teleop.py --robot k1 --models-dir <dir>

Viewer keys: ENTER = toggle streaming (reference standing pose <-> camera)
             9 = elastic band on/off,  BACKSPACE = reset robot,  I = re-init heading
             , / . = heading +/- 0.1 rad
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np
import zmq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sonic.deploy import SonicController, parse_gain_specs  # noqa: E402
from sonic.motion import standing_motion  # noqa: E402
from sonic.params import CONTROL_DT, MODELS  # noqa: E402
from sonic.robots import get_robot  # noqa: E402
from sonic.runner import SUBSTEPS  # noqa: E402
from sonic.sim import G1Sim, Recorder  # noqa: E402
from sonic.stream import StreamBuffer, attach, unpack_pose_message  # noqa: E402

WINDOW = {"release": 10, "sonic_v1_1": 10, "low_latency": 4}
KEY_ENTER, KEY_BACKSPACE = 257, 259


class Keys:
    def __init__(self):
        self.q = []
        self.lock = threading.Lock()

    def __call__(self, key):
        with self.lock:
            self.q.append(key)

    def pop(self):
        with self.lock:
            q, self.q = self.q, []
        return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="release", choices=list(WINDOW))
    ap.add_argument("--robot", default="g1", choices=["g1", "k1"], help="embodiment (sonic/robots.py)")
    ap.add_argument("--models-dir", default=str(MODELS), help="dir holding <variant>/model_{encoder,decoder}.onnx")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--delay", type=float, default=0.03, help="resampling delay behind newest sample (s)")
    ap.add_argument("--auto-stream", type=float, default=1.5,
                    help="enable streaming this many s after the band drop once data flows (<0: wait for ENTER)")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--record", default="")
    ap.add_argument("--max-seconds", type=float, default=0.0)
    ap.add_argument("--motor-kp-scale", action="append", default=[], help='e.g. "0-11=1.2" (hardware motor idx)')
    ap.add_argument("--motor-kd-scale", action="append", default=[])
    args = ap.parse_args()

    robot = get_robot(args.robot)
    sim = G1Sim(elastic_band=True, robot=robot)
    ctrl = SonicController(args.variant, models_dir=args.models_dir, robot=robot,
                           motor_kp_scale=parse_gain_specs(args.motor_kp_scale),
                           motor_kd_scale=parse_gain_specs(args.motor_kd_scale))
    stand = standing_motion(robot=robot)
    ctrl.set_motion(stand, play=False)
    buf = StreamBuffer(window=WINDOW[args.variant], delay=args.delay, robot=robot)

    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://{args.host}:{args.port}")
    sub.setsockopt(zmq.SUBSCRIBE, b"pose")
    print(f"[sim] subscribed tcp://{args.host}:{args.port} 'pose' (protocol v3 SMPL)")

    keys = Keys()
    viewer = None
    if not args.headless:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=keys,
                                              show_left_ui=False, show_right_ui=False)
        import mujoco

        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = sim.pelvis
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 180.0
        viewer.cam.elevation = -10.0
    rec = Recorder(sim, azimuth=180.0, elevation=-10.0, distance=2.8) if args.record else None

    t_ctrl0 = t_drop = None
    n_msgs = 0
    last_print = 0.0
    wall0 = time.monotonic()
    t_wall_next = wall0
    tick = 0
    lat_ms = []
    try:
        while viewer is None or viewer.is_running():
            now_w = time.monotonic()
            # ---- input: drain ZMQ ---------------------------------------------------
            while True:
                try:
                    msg = sub.recv(zmq.NOBLOCK)
                except zmq.Again:
                    break
                ver, f = unpack_pose_message(msg)
                if ver != 3:
                    continue
                ts = float(f["timestamp_monotonic"][0]) if "timestamp_monotonic" in f else now_w
                wr = f["joint_pos"][0, 23:29] if robot.name == "g1" else None  # v3 wire format carries G1 wrists
                buf.add_sample(ts, f["body_quat"][0], f["smpl_joints"][0], wr)
                lat_ms.append((now_w - ts) * 1e3)
                n_msgs += 1
            # ---- keys -----------------------------------------------------------------
            for k in keys.pop():
                if k == KEY_ENTER:
                    if ctrl.current_motion is buf.motion:
                        buf.active = False
                        ctrl.set_motion(stand, play=False)
                        print("[sim] streaming OFF")
                    elif attach(ctrl, buf, now_w):
                        print("[sim] streaming ON")
                elif k == ord("9"):
                    sim.set_band(not sim.band.enable)
                    t_drop = None if sim.band.enable else sim.time
                    print("[sim] band", "ON" if sim.band.enable else "OFF")
                elif k == KEY_BACKSPACE:
                    import mujoco

                    mujoco.mj_resetData(sim.model, sim.data)
                    ctrl.reset()
                    ctrl.set_motion(stand, play=False)
                    buf.active = False
                    sim.set_band(True)
                    t_ctrl0 = t_drop = None
                    print("[sim] reset")
                elif k == ord("I"):
                    ctrl.reinitialize_heading = True
                elif k == ord(","):
                    ctrl.heading.delta_heading += 0.1
                elif k == ord("."):
                    ctrl.heading.delta_heading -= 0.1
            # ---- state machine like the official procedure -----------------------------
            if ctrl.state == "WAIT_FOR_CONTROL":
                ctrl.start_control()
                t_ctrl0 = sim.time
                print("[sim] control started (']')")
            if t_ctrl0 is not None and t_drop is None and sim.band.enable and sim.time - t_ctrl0 >= 2.0:
                sim.set_band(False)
                t_drop = sim.time
                print("[sim] band released ('9')")
            if (args.auto_stream >= 0 and t_drop is not None and not buf.active and buf.samples
                    and sim.time - t_drop >= args.auto_stream and now_w - buf.last_sample_time < 0.5):
                if attach(ctrl, buf, now_w):
                    print("[sim] streaming ON (auto)")
            # ---- one control tick -------------------------------------------------------
            if buf.active and ctrl.current_motion is buf.motion:
                ctrl.current_frame = buf.tick(now_w)
                ctrl.play = True
            ls = sim.low_state()
            q_t, kp, kd = ctrl.step(ls)
            sim.set_command(q_t, kp, kd)
            sim.step(SUBSTEPS)
            if viewer is not None:
                viewer.sync()
            if rec is not None and tick % 2 == 0:
                rec.grab()
            tick += 1
            if now_w - last_print > 2.0:
                last_print = now_w
                lat = np.mean(lat_ms[-50:]) if lat_ms else float("nan")
                print(f"[sim] t={sim.time:6.1f} state={ctrl.state} stream={'ON' if buf.active else 'off'} "
                      f"msgs={n_msgs} input_age={lat:5.1f}ms pelvis_z={sim.pelvis_height():.2f} "
                      f"tilt={sim.tilt_deg():4.1f}")
            if args.max_seconds and now_w - wall0 > args.max_seconds:
                break
            # ---- real-time pacing -----------------------------------------------------
            t_wall_next += CONTROL_DT
            sl = t_wall_next - time.monotonic()
            if sl > 0:
                time.sleep(sl)
            elif sl < -0.2:
                t_wall_next = time.monotonic()
    finally:
        if viewer is not None:
            viewer.close()
        if rec is not None:
            import imageio.v2 as imageio

            imageio.mimwrite(args.record, rec.frames, fps=25, quality=7, macro_block_size=1)
            print("[sim] video ->", args.record)
        sub.close(0)
        ctx.term()


if __name__ == "__main__":
    main()
