"""M3 perception process: webcam (or --video) -> MediaPipe Pose -> SMPL-24 -> ZMQ protocol v3.

Plays the role of gear_sonic/examples/live_camera_teleop/webcam_stream.py (GEM-X) on a Mac CPU:
same topic ("pose"), same port (5556), same v3 fields, so the official C++ deploy
(`deploy.sh --input-type zmq`) could consume it too. Adds `timestamp_monotonic` (an optional
field of the official protocol) so the consumer can resample to 50 Hz.

Keys (camera window): c = calibrate (stand upright facing the camera, arms relaxed or T-pose)
                      l = legs auto/lock/track,  m = mirror preview,  q / ESC = quit
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import zmq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sonic.mp2smpl import MediaPipeToSmpl, OneEuro  # noqa: E402
from sonic.pose_mp import PoseTracker, draw_overlay  # noqa: E402
from sonic.stream import v3_message  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="camera index or video file")
    ap.add_argument("--video", default="", help="alias of --source FILE (played in real time)")
    ap.add_argument("--loop", action="store_true", help="loop the video file")
    ap.add_argument("--model", default="full", choices=["lite", "full", "heavy"])
    ap.add_argument("--crop", default="")
    ap.add_argument("--num-poses", type=int, default=1)
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--pelvis", default="j0", choices=["zero", "j0"])
    ap.add_argument("--root-pitch-share", type=float, default=0.5)
    ap.add_argument("--legs", default="auto", choices=["auto", "track", "lock"])
    ap.add_argument("--arm-depth-scale", type=float, default=0.7,
                    help="<1 shrinks MediaPipe's arm depth (arms reaching forward make SONIC step forward)")
    ap.add_argument("--max-root-tilt", type=float, default=40.0, help="safety cap on commanded root tilt (deg)")
    ap.add_argument("--min-cutoff", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.4)
    ap.add_argument("--auto-calib", type=int, default=20, help="auto-calibrate after N good frames (0=off)")
    ap.add_argument("--resolution", default="1280x720")
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--save", default="", help="write the overlay preview to this mp4")
    ap.add_argument("--max-seconds", type=float, default=0.0)
    args = ap.parse_args()

    src = args.video or args.source
    is_file = not src.isdigit()
    cap = cv2.VideoCapture(src if is_file else int(src))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {src} (macOS: allow camera access for your terminal app)")
    if not is_file and args.resolution:
        w, h = (int(x) for x in args.resolution.split("x"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[sender] source={src} {int(cap.get(3))}x{int(cap.get(4))} @ {vfps:.0f} fps")

    crop = tuple(float(x) for x in args.crop.split(",")) if args.crop else None
    tracker = PoseTracker(args.model, num_poses=args.num_poses, crop=crop)
    filt = OneEuro(args.min_cutoff, args.beta)
    conv = MediaPipeToSmpl(root_pitch_share=args.root_pitch_share, pelvis_mode=args.pelvis, legs=args.legs,
                           arm_depth_scale=args.arm_depth_scale, max_root_tilt_deg=args.max_root_tilt)
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.bind(f"tcp://*:{args.port}")
    print(f"[sender] publishing SMPL protocol v3 on tcp://*:{args.port} topic 'pose'")

    writer = None
    calib_buf, calibrated = [], False
    mirror = not is_file
    frame_index = 0
    t_start = time.monotonic()
    fps_ema, t_prev = 0.0, time.monotonic()
    video_t0 = time.monotonic()
    n_read = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            if is_file and args.loop:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                video_t0 = time.monotonic()
                n_read = 0
                continue
            break
        n_read += 1
        if is_file:  # pace the file in real time
            due = video_t0 + n_read / vfps
            dt = due - time.monotonic()
            if dt > 0:
                time.sleep(dt)
        t_cap = time.monotonic()
        det = tracker.process(frame, t_cap - t_start)
        status = []
        if det is not None:
            world = filt(det["world"], t_cap)
            good = det["vis"][[11, 12, 23, 24]].min() > 0.6
            if not calibrated and args.auto_calib and good:
                calib_buf.append(world)
                if len(calib_buf) >= args.auto_calib:
                    conv.calibrate(calib_buf)
                    calibrated = True
                    print("[sender] auto-calibrated")
            out = conv.convert(world, det["vis"])
            sock.send(v3_message(out["smpl_joints"], out["body_quat"], frame_index, t_mono=t_cap))
            frame_index += 1
            status.append("legs LOCKED" if conv.legs_locked else "legs tracked")
        else:
            status.append("NO PERSON")
        now = time.monotonic()
        fps_ema = 0.9 * fps_ema + 0.1 / max(now - t_prev, 1e-3)
        t_prev = now
        text = f"{fps_ema:4.1f} fps  sent {frame_index}  {'CALIBRATED' if calibrated else 'press c to calibrate'}\n" + " | ".join(status)
        if not args.no_window or args.save:
            vis_img = draw_overlay(frame, det, "")
            if mirror:
                vis_img = cv2.flip(vis_img, 1)
            vis_img = draw_overlay(vis_img, None, text)
            if args.save:
                if writer is None:
                    h, w = vis_img.shape[:2]
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
                writer.write(vis_img)
            if not args.no_window:
                if vis_img.shape[0] > 720:
                    sc = 720 / vis_img.shape[0]
                    vis_img = cv2.resize(vis_img, (int(vis_img.shape[1] * sc), 720))
                cv2.imshow("SONIC webcam teleop (MediaPipe -> SMPL)", vis_img)
                k = cv2.waitKey(1) & 0xFF
                if k in (27, ord("q")):
                    break
                if k == ord("c") and det is not None:
                    calib_buf, calibrated = [world], False
                    args.auto_calib = max(args.auto_calib, 15)
                    print("[sender] calibrating... stand still")
                if k == ord("l"):
                    conv.legs = {"auto": "lock", "lock": "track", "track": "auto"}[conv.legs]
                    print("[sender] legs:", conv.legs)
                if k == ord("m"):
                    mirror = not mirror
        if args.max_seconds and now - t_start > args.max_seconds:
            break
    if writer is not None:
        writer.release()
    sock.close(0)
    ctx.term()
    tracker.close()


if __name__ == "__main__":
    main()
