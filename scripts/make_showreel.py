"""1080p event showreels (calm gestures only).

  robot   videos/showreel_robot.mp4   the 8 hand-authored gestures (sonic/gestures.py) in event order, one fixed
                                      camera, Korean captions, title/end cards                      (60-75 s)
  teleop  videos/showreel_teleop.mp4  human video | robot (video -> MediaPipe -> SMPL -> SONIC -> G1), curated
                                      formal-looking stock clips, small captions, title/end cards   (45-60 s)

Each part is simulated once into videos/showreel_parts/ with the display options of render_event.py /
run_video.py and reused; --rerender simulates again. Text is drawn here (Korean needs a CJK font: Apple SD Gothic
Neo on macOS, Noto Sans CJK elsewhere). No logos or trademark images, only plain text.

  .venv/bin/python scripts/make_showreel.py robot
  .venv/bin/python scripts/make_showreel.py teleop
  .venv/bin/python scripts/make_showreel.py all --check-frames 8     # + contact sheet of frames for review
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
PARTS = ROOT / "videos" / "showreel_parts"
EVENT_VIDEOS = Path(os.environ.get("K1_EVENT_VIDEOS", str(ROOT / "videos" / "event")))

W, H, FPS = 1920, 1080, 25
XFADE = 10  # frames (0.4 s) of cross-dissolve between all segments

# one camera for every gesture: 40 deg off the front so the bow has depth, far enough for arms overhead
ROBOT_CAMERA = "140,2.45,0.82,-6"
ROBOT_PRE, ROBOT_POST = 0.5, 0.6  # s recorded around each gesture

# event order: greeting -> bow -> welcome, applause -> guiding -> heart -> hooray -> standing
# (clip, caption, English line, gesture-time in/out in s; out<0 means "from the end")
ROBOT_REEL = [
    ("wave_hello", "손 흔들어 인사", "Waving hello", 0.3, -0.2),
    ("bow", "정중하게 목례", "A polite bow", 0.3, -0.2),
    ("welcome_open_arms", "두 팔 벌려 환영", "Welcome with open arms", 0.3, -0.2),
    ("clap", "축하의 박수", "Applause", 0.3, -0.2),
    ("point_guide", "손바닥으로 안내", "This way, please", 0.3, -0.2),
    ("heart", "머리 위로 하트", "A heart overhead", 0.3, -0.2),
    ("raise_both_arms", "두 팔 들어 만세", "Hooray, both arms up", 0.3, -0.2),
    ("weight_shift", "편안하게 서 있기", "Standing at ease", 0.3, 8.0),
]

# curated human clips (Pexels; calm, frontal, full body).
# (part id, source under EVENT_VIDEOS, source credit, caption, English line, human-video start s, duration s)
TELEOP_REEL = [
    ("pexels_10374292", "raw/pexels_10374292.mp4", "Pexels 10374292", "손 흔들어 인사", "Waving hello", 0.4, 7.4),
    ("pexels_10149176", "raw/pexels_10149176.mp4", "Pexels 10149176", "두 손으로 환영", "Welcome", 1.0, 8.0),
    ("pexels_10149187", "raw/pexels_10149187.mp4", "Pexels 10149187", "축하의 박수", "Applause", 1.0, 7.0),
    ("pexels_10515017", "raw/pexels_10515017.mp4", "Pexels 10515017", "손바닥으로 안내", "This way, please", 3.0, 7.5),
    ("event_raise_arms", "event_raise_arms.mp4", "Pexels 8939359 (역재생)", "두 팔 천천히 들기", "Raising both arms",
     4.0, 9.0),
    ("event_stand", "event_stand.mp4", "Pexels 9558954", "편안하게 서 있기", "Standing at ease", 0.5, 6.5),
]

TITLE_TOP = "built with NVIDIA GEAR-SONIC"
TITLE_MAIN = "사람 동작을 따라 하는 휴머노이드"

INK = (30, 33, 40)
MUTED = (112, 117, 128)
ACCENT = (58, 96, 160)


# ------------------------------------------------------------------------------------------------- text
FONT_FILES = [
    ("/System/Library/Fonts/AppleSDGothicNeo.ttc", {"regular": 0, "medium": 2, "semibold": 4, "bold": 6}),
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", {"regular": 1, "medium": 1, "semibold": 1, "bold": 1}),
]
if os.environ.get("K1_FONT_CJK"):   # a Noto Sans CJK .ttc elsewhere (machines without system CJK fonts, e.g. the GPU box)
    FONT_FILES.append((os.environ["K1_FONT_CJK"], {"regular": 1, "medium": 1, "semibold": 1, "bold": 1}))
_fonts = {}


def font(weight, size):
    key = (weight, size)
    if key not in _fonts:
        for path, idx in FONT_FILES:
            if Path(path).exists():
                _fonts[key] = ImageFont.truetype(path, size, index=idx[weight])
                break
        else:
            raise SystemExit("no Korean font found (Apple SD Gothic Neo / Noto Sans CJK)")
    return _fonts[key]


class Overlay:
    """Pre-rendered RGBA sprite, alpha-blended into frames (only its bounding box is touched)."""

    def __init__(self, rgba: Image.Image, x: int, y: int):
        a = np.asarray(rgba).astype(np.float32)
        self.rgb, self.a = a[..., :3], a[..., 3:] / 255.0
        self.x, self.y = x, y

    def blend(self, frame, alpha=1.0):
        if alpha <= 0:
            return frame
        h, w = self.a.shape[:2]
        roi = frame[self.y:self.y + h, self.x:self.x + w].astype(np.float32)
        a = self.a * alpha
        frame[self.y:self.y + h, self.x:self.x + w] = (roi * (1 - a) + self.rgb * a + 0.5).astype(np.uint8)
        return frame


def text_size(txt, f):
    l, t, r, b = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), txt, font=f)
    return r - l, b - t, l, t


def pill(lines, pad=(30, 20), gap=8, radius=18, bg=(255, 255, 255, 224), bar=ACCENT, align="left"):
    """Rounded caption box. lines: [(text, font, rgb)]."""
    sizes = [text_size(t, f) for t, f, _ in lines]
    tw = max(s[0] for s in sizes)
    th = sum(s[1] for s in sizes) + gap * (len(lines) - 1)
    bw = 8 if bar else 0
    w, h = tw + 2 * pad[0] + bw, th + 2 * pad[1]
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, w - 1, h - 1), radius=radius, fill=bg)
    if bar:
        d.rounded_rectangle((0, 0, bw + radius, h - 1), radius=radius, fill=bar + (255,))
        d.rectangle((bw, 0, bw + radius, h - 1), fill=bg)
    y = pad[1]
    for (t, f, c), (sw, sh, l, top) in zip(lines, sizes):
        x = bw + pad[0] + ((tw - sw) // 2 if align == "center" else 0)
        d.text((x - l, y - top), t, font=f, fill=c + (255,))
        y += sh + gap
    return im


def plain_text(txt, f, rgb, alpha=255):
    sw, sh, l, t = text_size(txt, f)
    im = Image.new("RGBA", (sw + 4, sh + 4), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((2 - l, 2 - t), txt, font=f, fill=rgb + (alpha,))
    return im


# ------------------------------------------------------------------------------------------------- cards
def card_background():
    y = np.linspace(0, 1, H)[:, None, None]
    top, bot = np.array([248, 249, 251], np.float32), np.array([226, 230, 236], np.float32)
    return np.broadcast_to(top * (1 - y) + bot * y, (H, W, 3)).astype(np.uint8).copy()


def card(lines, seconds, small=None):
    """Title/end card: centred lines [(text, font, rgb, gap_after_px)], fading/sliding in over the first 0.8 s."""
    bg = card_background()
    sprites = [plain_text(t, f, c) for t, f, c, _ in lines]
    total = sum(s.height for s in sprites) + sum(g for *_, g in lines[:-1])
    y = (H - total) // 2 - 20
    ovs = []
    for s, (*_, g) in zip(sprites, lines):
        ovs.append(Overlay(s, (W - s.width) // 2, y))
        y += s.height + g
    rule = Image.new("RGBA", (120, 4), ACCENT + (255,))
    rule_ov = Overlay(rule, (W - 120) // 2, ovs[0].y + sprites[0].height + lines[0][3] // 2 - 2)
    small_ov = None
    if small:
        sp = plain_text(small, font("regular", 26), MUTED)
        small_ov = Overlay(sp, (W - sp.width) // 2, H - 110)
    n = int(round(seconds * FPS))
    for i in range(n):
        f = bg.copy()
        for k, ov in enumerate(ovs):
            u = np.clip((i - 4 - 5 * k) / (0.8 * FPS), 0, 1)
            e = u * u * (3 - 2 * u)
            y0 = ov.y
            ov.y = int(round(y0 + (1 - e) * 18))
            ov.blend(f, e)
            ov.y = y0
        rule_ov.blend(f, float(np.clip((i - 2) / (0.6 * FPS), 0, 1)))
        if small_ov is not None:
            small_ov.blend(f, float(np.clip((i - 12) / (0.8 * FPS), 0, 1)))
        yield f


def title_card(subtitle, seconds=4.5):
    return card([(TITLE_TOP, font("semibold", 40), MUTED, 46),
                 (TITLE_MAIN, font("bold", 92), INK, 34),
                 (subtitle, font("regular", 38), MUTED, 0)], seconds)


def end_card(lines2, small, seconds=5.0):
    return card([(TITLE_TOP, font("bold", 64), INK, 40),
                 (TITLE_MAIN, font("medium", 46), INK, 26)] + [(t, font("regular", 32), MUTED, 14) for t in lines2],
                seconds, small=small)


# ------------------------------------------------------------------------------------------------- video io
def read_frames(path, t0, t1):
    """RGB frames of an mp4 in [t0, t1) seconds (sequential decode; parts are short)."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    a, b = int(round(t0 * fps)), int(round(t1 * fps))
    i = 0
    while i < b:
        ok, f = cap.read()
        if not ok:
            break
        if i >= a:
            if f.shape[1] != W or f.shape[0] != H:
                f = cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
            yield cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()


class Reel:
    """Streams frames into ffmpeg (H.264 High, yuv420p, BT.709 tagged) with cross-dissolves between segments."""

    def __init__(self, path, xfade=XFADE):
        ff = shutil.which("ffmpeg")
        if ff is None:
            import imageio_ffmpeg

            ff = imageio_ffmpeg.get_ffmpeg_exe()
        self.path = Path(path)
        self.proc = subprocess.Popen(
            [ff, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
             "-i", "-", "-vf", "scale=out_color_matrix=bt709:out_range=tv", "-c:v", "libx264", "-preset", "slow",
             "-crf", "17", "-profile:v", "high", "-pix_fmt", "yuv420p", "-color_primaries", "bt709",
             "-color_trc", "bt709", "-colorspace", "bt709", "-movflags", "+faststart", str(self.path)],
            stdin=subprocess.PIPE)
        self.xfade, self.tail, self.n = xfade, [], 0
        self.marks = []  # (segment name, start s, end s) in the final timeline

    def _write(self, f):
        self.proc.stdin.write(np.ascontiguousarray(f, dtype=np.uint8).tobytes())
        self.n += 1

    def add(self, name, frames):
        it = iter(frames)
        start = self.n
        k = len(self.tail)
        for i, a in enumerate(self.tail):
            b = next(it, None)
            if b is None:
                self._write(a)
                continue
            w = (i + 1) / (k + 1)
            self._write((a.astype(np.float32) * (1 - w) + b.astype(np.float32) * w + 0.5).astype(np.uint8))
        buf = deque()
        for f in it:
            buf.append(f)
            if len(buf) > self.xfade:
                self._write(buf.popleft())
        self.tail = list(buf)
        self.marks.append((name, start / FPS, (self.n + len(self.tail)) / FPS))

    def close(self):
        for f in self.tail:
            self._write(f)
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise SystemExit(f"ffmpeg failed writing {self.path}")
        return self.n / FPS


def fade_env(i, n, fin=0.5, fout=0.5, delay=0.25):
    t = i / FPS
    u = min(np.clip((t - delay) / fin, 0, 1), np.clip((n / FPS - t - 0.15) / fout, 0, 1))
    return float(u * u * (3 - 2 * u))


def with_overlays(frames, n, ovs_fade, ovs_static=()):
    for i, f in enumerate(frames):
        f = f.copy()
        for ov in ovs_static:
            ov.blend(f)
        e = fade_env(i, n)
        for ov in ovs_fade:
            ov.blend(f, e)
        yield f


def corner_tag():
    sp = plain_text(f"{TITLE_TOP}  ·  Unitree G1 물리 시뮬레이션", font("semibold", 26), MUTED, alpha=230)
    return Overlay(sp, 56, 44)


# ------------------------------------------------------------------------------------------------- robot
def robot_parts(rerender):
    names = [c[0] for c in ROBOT_REEL]
    todo = [n for n in names if rerender or not (PARTS / f"event_{n}.json").exists()]
    if todo:
        cmd = [sys.executable, str(ROOT / "scripts" / "render_event.py"), "--clips", *todo, "--size", f"{W}x{H}",
               "--camera", ROBOT_CAMERA, "--no-caption", "--pre", str(ROBOT_PRE), "--post", str(ROBOT_POST),
               "--quality", "9", "--fps", str(FPS), "--out-dir", str(PARTS)]
        print("$", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=ROOT)
    return {n: json.loads((PARTS / f"event_{n}.json").read_text()) for n in names}


def make_robot(out, rerender):
    meta = robot_parts(rerender)
    reel = Reel(out)
    reel.add("title", title_card("행사용 제스처 8종  ·  Unitree G1 물리 시뮬레이션"))
    tag = corner_tag()
    segs = []
    for i, (name, ko, en, g_in, g_out) in enumerate(ROBOT_REEL):
        m = meta[name]
        dur = m["duration_s"]
        t0 = m["video"]["pre_s"] + g_in
        t1 = m["video"]["pre_s"] + (g_out if g_out > 0 else dur + g_out)
        n = int(round((t1 - t0) * FPS))
        cap = pill([(f"{i + 1:02d} / {len(ROBOT_REEL):02d}", font("semibold", 24), ACCENT),
                    (ko, font("bold", 58), INK), (en, font("regular", 28), MUTED)], pad=(34, 24), gap=12)
        ov = Overlay(cap, 96, H - 96 - cap.height)
        reel.add(name, with_overlays(read_frames(PARTS / f"event_{name}.mp4", t0, t1), n, [ov], [tag]))
        segs.append({"clip": name, "caption": ko, "english": en, "part": f"videos/showreel_parts/event_{name}.mp4",
                     "part_in_s": round(t0, 2), "part_out_s": round(t1, 2), "fell": m["fell"],
                     "joint_err_deg": m["tracking_joint_err_deg"].get("all_deg"),
                     "max_pelvis_tilt_deg": round(m["max_pelvis_tilt_deg"], 1)})
    reel.add("end", end_card(["로봇 동작 8종은 키프레임으로 만들고, NVIDIA GEAR-SONIC 정책이 물리 시뮬레이션에서 실시간으로 제어했습니다."],
                             "Unitree G1 · MuJoCo 물리 시뮬레이션 · 실제 로봇 영상이 아닙니다"))
    total = reel.close()
    return total, segs, reel.marks


# ------------------------------------------------------------------------------------------------- teleop
def teleop_part(pid, src, rerender):
    out, js = PARTS / f"teleop_{pid}.mp4", PARTS / f"teleop_{pid}.json"
    if rerender or not js.exists():
        cmd = [sys.executable, "-u", str(ROOT / "scripts" / "run_video.py"), "--video", str(EVENT_VIDEOS / src),
               "--num-poses", "1", "--clean", "--clean-size", f"{W}x{H}", "--no-caption", "--quality", "9",
               "--max-video-seconds", "20", "--fps", str(FPS), "--out", str(out), "--json", str(js)]
        print("$", " ".join(cmd))
        log = ROOT / "logs" / f"showreel_teleop_{pid}.log"
        log.parent.mkdir(exist_ok=True)
        with open(log, "w") as fh:
            subprocess.run(cmd, check=True, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    return json.loads(js.read_text())


def make_teleop(out, rerender):
    reel = Reel(out)
    reel.add("title", title_card("사람 영상  →  휴머노이드가 실시간으로 따라 하기"))
    hw = W // 2
    lab_h = pill([("사람 영상", font("semibold", 26), INK)], pad=(20, 12), radius=14, bar=None,
                 bg=(255, 255, 255, 210))
    lab_r = pill([("휴머노이드 로봇 (실시간 따라 하기)", font("semibold", 26), INK)], pad=(20, 12), radius=14, bar=None,
                 bg=(255, 255, 255, 210))
    statics = [Overlay(lab_h, 40, 36), Overlay(lab_r, hw + 40, 36),
               Overlay(Image.new("RGBA", (4, H), (255, 255, 255, 255)), hw - 2, 0)]
    segs = []
    for pid, src, credit, ko, en, s0, dur in TELEOP_REEL:
        m = teleop_part(pid, src, rerender)
        if m["fell"]:
            raise SystemExit(f"{pid}: robot fell in the teleop part - do not use it")
        v = m["video_out"]
        t0 = v["human_t0_in_video_s"] + s0
        t1 = min(t0 + dur, v["frames"] / v["fps"])
        n = int(round((t1 - t0) * FPS))
        cap = pill([(ko, font("bold", 42), INK), (en, font("regular", 24), MUTED)], pad=(30, 16), gap=8,
                   align="center", bar=None)
        ov = Overlay(cap, (W - cap.width) // 2, H - 56 - cap.height)
        reel.add(pid, with_overlays(read_frames(PARTS / f"teleop_{pid}.mp4", t0, t1), n, [ov], statics))
        lm = m["limb_dir_err_deg"]
        segs.append({"part_id": pid, "source": src, "credit": credit, "caption": ko, "english": en,
                     "human_in_s": s0, "human_out_s": round(s0 + (t1 - t0), 2), "fell": m["fell"],
                     "legs_locked_frames": m["legs_locked_frames"],
                     "arms_dir_err_deg": round(lm.get("arms_mean", 0), 1), "legs_dir_err_deg": round(lm.get("legs_mean", 0), 1),
                     "max_tilt_deg": round(m["max_tilt_deg"], 1)})
    reel.add("end", end_card(["사람 영상 → MediaPipe 자세 추정 → SMPL 인체 모델 → NVIDIA GEAR-SONIC → Unitree G1"],
                             "사람 영상: Pexels 무료 스톡 영상(Pexels License) · 로봇: MuJoCo 물리 시뮬레이션"))
    total = reel.close()
    return total, segs, reel.marks


# ------------------------------------------------------------------------------------------------- checks
def check_sheet(path, n):
    """n evenly spaced frames (skipping the first/last 2 s) -> <name>_check/ + one contact sheet."""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    want = set(np.linspace(2 * FPS, total - 2 * FPS, n).astype(int).tolist())
    d = path.with_name(path.stem + "_check")
    d.mkdir(exist_ok=True)
    thumbs, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i in want:
            cv2.imwrite(str(d / f"t{i / FPS:05.1f}s.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 90])
            thumbs.append(cv2.resize(f, (640, 360), interpolation=cv2.INTER_AREA))
        i += 1
    while len(thumbs) % 2:
        thumbs.append(np.full_like(thumbs[0], 255))
    rows = [np.concatenate(thumbs[k:k + 2], axis=1) for k in range(0, len(thumbs), 2)]
    sheet = path.with_name(path.stem + "_check.jpg")
    cv2.imwrite(str(sheet), np.concatenate(rows, axis=0), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["robot", "teleop", "all"])
    ap.add_argument("--rerender", action="store_true", help="simulate the parts again instead of reusing them")
    ap.add_argument("--check-frames", type=int, default=0, help="also write N review frames + a contact sheet")
    args = ap.parse_args()
    PARTS.mkdir(parents=True, exist_ok=True)
    jobs = {"robot": make_robot, "teleop": make_teleop}
    for which in (["robot", "teleop"] if args.which == "all" else [args.which]):
        out = ROOT / "videos" / f"showreel_{which}.mp4"
        total, segs, marks = jobs[which](out, args.rerender)
        manifest = {"video": f"videos/showreel_{which}.mp4", "size": f"{W}x{H}", "fps": FPS, "duration_s": round(total, 2),
                    "command": f".venv/bin/python scripts/make_showreel.py {which}",
                    "timeline": [{"segment": n, "start_s": round(a, 2), "end_s": round(b, 2)} for n, a, b in marks],
                    "segments": segs}
        out.with_suffix(".json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        print(f"{which}: {out} {total:.1f} s, {len(segs)} clips, falls {sum(s['fell'] for s in segs)}")
        if args.check_frames:
            print("  check frames ->", check_sheet(out, args.check_frames))


if __name__ == "__main__":
    main()
