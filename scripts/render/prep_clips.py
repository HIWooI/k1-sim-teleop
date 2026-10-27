"""Normalise picked source clips for demo v2: one copy per clip that every step uses (MediaPipe, GEM-X, reel).

Long side <= 1920 px (lanczos), frame rate <= 30 fps (50/60 fps sources halved), H.264 CRF 18 yuv420p, no audio,
optional trim. Output <out>/<name>.mp4 (+ <name>.json with the source, trim and resulting size / fps).

    python scripts/demo_v2/prep_clips.py --out videos/demo_v2/clips name=<src>[@t0-t1] ...
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2


def probe(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, n, w, h


def main():
    import imageio_ffmpeg

    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+", help="name=path[@t0-t1]")
    ap.add_argument("--out", default="videos/demo_v2/clips")
    ap.add_argument("--max-side", type=int, default=1920)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for spec in a.clips:
        name, rest = spec.split("=", 1)
        src, _, rng = rest.partition("@")
        dst = Path(a.out) / f"{name}.mp4"
        if dst.exists() and not a.force:
            print(f"{name}: exists")
            continue
        fps, n, w, h = probe(src)
        s = min(1.0, a.max_side / max(w, h))
        W, H = int(round(w * s / 2)) * 2, int(round(h * s / 2)) * 2
        ofps = fps / 2 if fps > 31 else fps
        cmd = [ff, "-y", "-loglevel", "error"]
        t0 = t1 = None
        if rng:
            x, _, y = rng.partition("-")
            t0 = float(x or 0)
            cmd += ["-ss", f"{t0:.3f}"]
            if y:
                t1 = float(y)
                cmd += ["-t", f"{t1 - t0:.3f}"]
        cmd += ["-i", src, "-an", "-vf", f"scale={W}:{H}:flags=lanczos,fps={ofps:.6f}", "-c:v", "libx264", "-crf", "18",
                "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)]
        subprocess.run(cmd, check=True)
        f2, n2, w2, h2 = probe(dst)
        info = {"name": name, "source": str(src), "source_wh": [w, h], "source_fps": round(fps, 3), "t0": t0, "t1": t1,
                "wh": [w2, h2], "fps": round(f2, 3), "frames": n2, "seconds": round(n2 / f2, 2)}
        (Path(a.out) / f"{name}.json").write_text(json.dumps(info, indent=1))
        print(json.dumps(info))


if __name__ == "__main__":
    main()
