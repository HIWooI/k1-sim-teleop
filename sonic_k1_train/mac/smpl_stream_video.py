"""The live sender's whole-body stream (HumanFrame.smpl) computed offline from a video file, frame by frame (not real
time), plus statistics to compare with the training data - is what the webcam produces inside the distribution the
K1 SMPL encoder is trained on?

    MUJOCO_GL=egl python sonic_k1_train/mac/smpl_stream_video.py --videos a.mp4 b.mp4 --out logs/smpl_streams \
        [--train-set <converted set> ] [--overlay]

Per video: <out>/<stem>.npz (t, joints (T,24,3) root-local, quat (T,4), wrist (T,2), arm_q (T,10); NaN rows when
nobody is tracked) and, with --overlay, <stem>.mp4 (camera frame | SMPL skeleton front view).
Stats (also for --train-set's SMPL files): root tilt, wrist / ankle heights, per-frame joint speed, bone-direction
jitter while standing still - printed as a table and written to <out>/stats.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p in (str(REPO), str(REPO / "sonic_k1_train" / "data")):
    if p not in sys.path:
        sys.path.insert(0, p)

from k1live import smpl_stream  # noqa: E402


def stream_video(path: Path, mirror=False, max_s=60.0, overlay=None):
    import cv2

    from k1live.human import HumanMapper, JumpGate
    from sonic.pose_mp import PoseTracker

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tr = PoseTracker("full")
    tr.max_jump = float("inf")
    gate = JumpGate()
    m = HumanMapper(mirror=mirror, loco=False)
    out = {k: [] for k in ("t", "joints", "quat", "wrist", "arm_q")}
    wr = None
    i = 0
    while i < max_s * fps:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[0] > 720:
            s = 720 / frame.shape[0]
            frame = cv2.resize(frame, (int(frame.shape[1] * s), 720))
        t = i / fps
        det = gate(tr.process(frame, t))
        r = m.process(det, t, frame.shape[1], frame.shape[0])
        out["t"].append(t)
        if r["smpl"] is not None:
            J, q, w = smpl_stream.from_msg(r["smpl"])
            out["joints"].append(J), out["quat"].append(q), out["wrist"].append(w), out["arm_q"].append(r["arm_q"])
        else:
            out["joints"].append(np.full((24, 3), np.nan)), out["quat"].append(np.full(4, np.nan))
            out["wrist"].append(np.full(2, np.nan)), out["arm_q"].append(np.full(10, np.nan))
        if overlay is not None:
            from bones_preview import BONES, LEFT

            h = frame.shape[0]
            panel = np.full((h, h * 3 // 4, 3), 245, np.uint8)
            if r["smpl"] is not None:
                Jw = smpl_stream.world_joints(J, q)
                z0 = -Jw[:, 2].min()
                px = lambda p: (int(panel.shape[1] / 2 + p[1] * h / 2.2), int(h * 0.93 - (p[2] + z0) * h / 2.2))  # noqa: E731
                for a, b in BONES:
                    cv2.line(panel, px(Jw[a]), px(Jw[b]), (40, 90, 220) if b in LEFT else (220, 80, 40), 3, cv2.LINE_AA)
            img = np.concatenate([frame, panel], 1)
            if wr is None:
                import imageio.v2 as imageio

                wr = imageio.get_writer(str(overlay), fps=fps, codec="libx264", quality=6, macro_block_size=8)
            wr.append_data(img[:, :, ::-1])
        i += 1
    cap.release()
    tr.close()
    if wr is not None:
        wr.close()
    res = {k: np.asarray(v, np.float64) for k, v in out.items()}
    res["fps"] = fps
    res["calibrated"] = m.calibrated
    res["smpl_ms"] = m.smpl_ms
    return res


def stats(J, Q, fps):
    """Distribution summary of a stream / clip set: J (T,24,3) local, Q (T,4)."""
    ok = np.all(np.isfinite(J.reshape(len(J), -1)), axis=1)
    J, Q = J[ok], Q[ok]
    if len(J) < 3:
        return {"frames": int(len(J))}
    tilt = np.degrees(np.arccos(np.clip(1 - 2 * (Q[:, 1] ** 2 + Q[:, 2] ** 2), -1, 1)))
    sp = np.linalg.norm(np.diff(J, axis=0), axis=-1).mean(1) * fps
    # jitter: frame-to-frame change of the forearm directions, in deg/frame (high-pass proxy)
    fa = J[:, [20, 21]] - J[:, [18, 19]]
    fa /= np.linalg.norm(fa, axis=-1, keepdims=True)
    dang = np.degrees(np.arccos(np.clip(np.sum(fa[1:] * fa[:-1], -1), -1, 1)))
    acc = np.linalg.norm(np.diff(J, 2, axis=0), axis=-1).mean(1) * fps * fps
    return {"frames": int(len(J)), "root_tilt_p50": round(float(np.median(tilt)), 1),
            "root_tilt_p95": round(float(np.percentile(tilt, 95)), 1),
            "wrist_z_p05": round(float(np.percentile(J[:, [20, 21], 2], 5)), 3),
            "wrist_z_p95": round(float(np.percentile(J[:, [20, 21], 2], 95)), 3),
            "wrist_x_p95": round(float(np.percentile(J[:, [20, 21], 0], 95)), 3),
            "ankle_z_p50": round(float(np.median(J[:, [7, 8], 2])), 3),
            "knee_bend_p50_deg": round(float(np.median(_knee(J))), 1),
            "joint_speed_p50": round(float(np.median(sp)), 3), "joint_speed_p95": round(float(np.percentile(sp, 95)), 3),
            "forearm_step_deg_p50": round(float(np.median(dang)), 2),
            "joint_acc_p50": round(float(np.median(acc)), 2)}


def _knee(J):
    a = J[:, [1, 2]] - J[:, [4, 5]]
    b = J[:, [7, 8]] - J[:, [4, 5]]
    c = np.sum(a * b, -1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    return 180.0 - np.degrees(np.arccos(np.clip(c, -1, 1))).ravel()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", nargs="*", default=[])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train-set", type=Path, default=None)
    ap.add_argument("--overlay", action="store_true")
    ap.add_argument("--max-s", type=float, default=60.0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    res = {}
    for v in map(Path, a.videos):
        r = stream_video(v, max_s=a.max_s, overlay=(a.out / f"{v.stem}.mp4") if a.overlay else None)
        np.savez_compressed(a.out / f"{v.stem}.npz", **{k: r[k] for k in ("t", "joints", "quat", "wrist", "arm_q")},
                            fps=r["fps"])
        res[f"video:{v.stem}"] = dict(stats(r["joints"], r["quat"], r["fps"]), tracked=round(float(
            np.isfinite(r["joints"][:, 0, 0]).mean()), 3), smpl_ms=round(float(r["smpl_ms"]), 2))
        print(v.stem, json.dumps(res[f"video:{v.stem}"]), flush=True)
    if a.train_set is not None:
        import joblib

        Js, Qs = [], []
        for p in sorted((a.train_set / "smpl").glob("*.pkl")):
            s = joblib.load(p)
            q = smpl_stream.root_quat_from_aa_yup(s["pose_aa"][:, :3])
            Js.append(smpl_stream.world_joints(s["smpl_joints"], q * np.array([1.0, -1.0, -1.0, -1.0])))
            Qs.append(q)
            res[f"train:{p.stem}"] = stats(Js[-1], q, s["fps"])
        res["train:ALL"] = stats(np.concatenate(Js), np.concatenate(Qs), 50.0)
        print("train:ALL", json.dumps(res["train:ALL"]))
    (a.out / "stats.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
