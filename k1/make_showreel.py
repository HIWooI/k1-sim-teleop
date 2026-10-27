"""K1 event showreels (1920x1080, H.264), composed from the clips k1/render_event.py and k1/run_video.py wrote.

  robot   videos/k1_showreel_robot.mp4   8 IK-authored gestures in event order, one camera, Korean captions,
                                         title/end cards (60-75 s)
  teleop  videos/k1_showreel_teleop.mp4  human video | K1, the curated formal stock clips, small captions (45-60 s)
          --variant soma_ik|soma_csv|soma_mlp: the same reel from the GEM-X (SOMA) runs of k1/run_video.py
          (k1_sbs_<clip>_<variant>.mp4) -> videos/k1_showreel_teleop_soma.mp4, end card names that pipeline

Cards, captions, cross-dissolves and the ffmpeg writer are shared with scripts/make_showreel.py (the G1 reel).
No logos or trademark images, plain text only; the title line is --title-top.

  .venv/bin/python -m k1.render_event && .venv/bin/python -m k1.run_video --preset   # clips (once)
  .venv/bin/python -m k1.make_showreel all --check-frames 8
  .venv/bin/python -m k1.run_video --preset --source soma_ik && .venv/bin/python -m k1.make_showreel teleop --variant soma_ik
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import make_showreel as ms  # noqa: E402

VIDEOS = ROOT / "videos"
TITLE_TOP = "AI Sapiens K1 텔레옵"

# event order: greeting -> bow -> welcome, applause -> guiding -> heart -> hooray -> standing
# (clip, caption, English line, gesture-time in/out s; out<0 = from the end)
ROBOT_REEL = [
    ("wave_hello", "손 흔들어 인사", "Waving hello", 0.4, -0.3),
    ("bow", "정중하게 목례", "A polite bow", 0.4, -0.3),
    ("welcome_open_arms", "두 팔 벌려 환영", "Welcome with open arms", 0.4, -0.3),
    ("clap", "축하의 박수", "Applause", 0.4, -0.3),
    ("point_guide", "손바닥으로 안내", "This way, please", 0.4, -0.3),
    ("heart", "머리 위로 하트", "A heart overhead", 0.4, -0.3),
    ("raise_both_arms", "두 팔 들어 만세", "Hooray, both arms up", 0.4, -0.3),
    ("stand_at_ease", "공손하게 서 있기", "Standing at ease", 0.4, 7.6),
]

# (clip, credit, caption, English line, human-video start s, duration s)
TELEOP_REEL = [
    ("pexels_10374292", "Pexels 10374292", "손 흔들어 인사", "Waving hello", 0.4, 7.4),
    ("pexels_10149176", "Pexels 10149176", "두 손으로 환영", "Welcome", 1.0, 8.0),
    ("pexels_10149187", "Pexels 10149187", "축하의 박수", "Applause", 1.0, 7.0),
    ("pexels_10515017", "Pexels 10515017", "손바닥으로 안내", "This way, please", 3.0, 7.5),
    ("event_raise_arms", "Pexels 8939359 (역재생)", "두 팔 천천히 들기", "Raising both arms", 4.0, 9.0),
    ("event_stand", "Pexels 9558954", "편안하게 서 있기", "Standing at ease", 0.5, 6.5),
]


# end-card line per perception/retarget path (what actually ran, see k1/run_video.py --source)
PIPELINE_LINE = {
    "": "사람 영상 → MediaPipe 자세 추정 → K1 팔 역기구학 → 모방·균형 정책 → AI Sapiens K1",
    "soma_ik": "사람 영상 → GEM-X(SOMA) 3D 자세 추정 → K1 팔 역기구학 → 모방·균형 정책 → AI Sapiens K1",
    "soma_csv": "사람 영상 → GEM-X(SOMA) → soma-retargeter → 모방·균형 정책 → AI Sapiens K1",
    "soma_mlp": "사람 영상 → GEM-X(SOMA) → 증류 리타깃 MLP → 모방·균형 정책 → AI Sapiens K1",
    "soma_dir": "사람 영상 → GEM-X(SOMA) 3D 자세 추정 → K1 팔 방향 역기구학 → 모방·균형 정책 → AI Sapiens K1",
}
# per-clip tag under the caption when one reel mixes perception paths (--clip-variant)
PATH_TAG = {"": "MediaPipe 자세 추정", "soma_ik": "GEM-X(SOMA) · K1 역기구학", "soma_csv": "GEM-X(SOMA) · soma-retargeter",
            "soma_mlp": "GEM-X(SOMA) · 증류 MLP", "soma_dir": "GEM-X(SOMA) · K1 팔 방향 역기구학"}


def corner_tag(title_top):
    sp = ms.plain_text(f"{title_top}  ·  물리 시뮬레이션", ms.font("semibold", 26), ms.MUTED, alpha=230)
    return ms.Overlay(sp, 56, 44)


def title_card(title_top, subtitle, seconds=4.5):
    return ms.card([(title_top, ms.font("semibold", 40), ms.MUTED, 46),
                    (ms.TITLE_MAIN, ms.font("bold", 92), ms.INK, 34),
                    (subtitle, ms.font("regular", 38), ms.MUTED, 0)], seconds)


def end_card(title_top, lines2, small, seconds=5.0):
    return ms.card([(title_top, ms.font("bold", 64), ms.INK, 40),
                    (ms.TITLE_MAIN, ms.font("medium", 46), ms.INK, 26)]
                   + [(t, ms.font("regular", 32), ms.MUTED, 14) for t in lines2], seconds, small=small)


def make_robot(out, title_top):
    reel = ms.Reel(out)
    reel.add("title", title_card(title_top, "행사용 제스처 8종  ·  AI Sapiens K1 물리 시뮬레이션"))
    tag = corner_tag(title_top)
    segs = []
    for i, (name, ko, en, g_in, g_out) in enumerate(ROBOT_REEL):
        m = json.loads((VIDEOS / f"k1_event_{name}.json").read_text())
        if m["fell"]:
            raise SystemExit(f"{name}: the robot fell - fix the gesture before composing")
        pre = m["video"]["pre_s"]
        t0 = pre + g_in
        t1 = pre + (g_out if g_out > 0 else m["duration_s"] + g_out)
        n = int(round((t1 - t0) * ms.FPS))
        cap = ms.pill([(f"{i + 1:02d} / {len(ROBOT_REEL):02d}", ms.font("semibold", 24), ms.ACCENT),
                       (ko, ms.font("bold", 58), ms.INK), (en, ms.font("regular", 28), ms.MUTED)], pad=(34, 24), gap=12)
        ov = ms.Overlay(cap, 96, ms.H - 96 - cap.height)
        reel.add(name, ms.with_overlays(ms.read_frames(VIDEOS / f"k1_event_{name}.mp4", t0, t1), n, [ov], [tag]))
        segs.append({"clip": name, "caption": ko, "english": en, "part": f"videos/k1_event_{name}.mp4",
                     "part_in_s": round(t0, 2), "part_out_s": round(t1, 2), "mode": m["mode"], "fell": m["fell"],
                     "arms_err_deg": m["tracking_err_deg"]["arms"], "max_pelvis_tilt_deg": m["max_pelvis_tilt_deg"]})
    reel.add("end", end_card(title_top, [
        "동작 8종은 관절 키프레임(IK)으로 만들고, 팔은 모방 정책 · 다리는 균형 정책이 실시간으로 제어했습니다."],
        "AI Sapiens K1 · MuJoCo 물리 시뮬레이션 · 실제 로봇 영상이 아닙니다"))
    return reel.close(), segs, reel.marks


def make_teleop(out, title_top, variant="", per_clip=None):
    """variant: k1.run_video --source of every clip ('' = MediaPipe); per_clip {clip: variant} overrides it. A mixed
    reel says which path each clip took (small line in its caption) and lists every path on the end card."""
    per_clip = per_clip or {}
    used = {pid: per_clip.get(pid, variant) for pid, *_ in TELEOP_REEL}
    mixed = len(set(used.values())) > 1
    reel = ms.Reel(out)
    reel.add("title", title_card(title_top, "사람 영상  →  휴머노이드가 그대로 따라 하기"))
    segs = []
    for pid, credit, ko, en, s0, dur in TELEOP_REEL:
        v_ = used[pid]
        sfx = f"_{v_}" if v_ else ""
        m = json.loads((VIDEOS / f"k1_sbs_{pid}{sfx}.json").read_text())
        if m["fell"]:
            raise SystemExit(f"{pid}{sfx}: the robot fell - do not use it")
        v = m["video_out"]
        t0 = v["human_t0_in_video_s"] + s0
        t1 = min(t0 + dur, v["frames"] / v["fps"])
        n = int(round((t1 - t0) * ms.FPS))
        lines = [(ko, ms.font("bold", 42), ms.INK), (en, ms.font("regular", 24), ms.MUTED)]
        if mixed:
            lines.append((PATH_TAG[v_], ms.font("regular", 20), ms.ACCENT))
        cap = ms.pill(lines, pad=(30, 16), gap=8, align="center", bar=None)
        ov = ms.Overlay(cap, (ms.W - cap.width) // 2, ms.H - 56 - cap.height)
        reel.add(pid, ms.with_overlays(ms.read_frames(VIDEOS / f"k1_sbs_{pid}{sfx}.mp4", t0, t1), n, [ov]))
        segs.append({"clip": pid, "credit": credit, "caption": ko, "english": en, "human_in_s": s0,
                     "human_out_s": round(s0 + (t1 - t0), 2), "part": f"videos/k1_sbs_{pid}{sfx}.mp4",
                     "variant": v_ or "mediapipe", "pipeline": m.get("pipeline"), "fell": m["fell"],
                     "arms_err_deg": m["tracking_err_deg"]["arms"], "max_pelvis_tilt_deg": m["max_pelvis_tilt_deg"],
                     "arm2d_vs_human_deg": {k: v["mean"] for k, v in m.get("arm2d_vs_human_deg", {}).items()}})
    if mixed:
        lines2 = []
        for v_ in dict.fromkeys(used.values()):  # first-use order
            names = " · ".join(ko for pid, _, ko, *_ in TELEOP_REEL if used[pid] == v_)
            lines2 += [f"[{names}]", PIPELINE_LINE[v_]]
    else:
        lines2 = [PIPELINE_LINE[variant]]
    reel.add("end", end_card(title_top, lines2,
                             "사람 영상: Pexels 무료 스톡 영상(Pexels License) · 로봇: MuJoCo 물리 시뮬레이션"))
    return reel.close(), segs, reel.marks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["robot", "teleop", "all"])
    ap.add_argument("--title-top", default=TITLE_TOP)
    ap.add_argument("--check-frames", type=int, default=0)
    ap.add_argument("--variant", default="", choices=sorted(PIPELINE_LINE), help="teleop: which k1.run_video "
                    "--source runs to compose ('' = MediaPipe); a soma_* variant writes k1_showreel_teleop_soma.mp4")
    ap.add_argument("--clip-variant", nargs="*", default=[], metavar="CLIP=VARIANT",
                    help="teleop: per-clip override, e.g. pexels_10149187=mediapipe (mixed reel, paths labelled)")
    ap.add_argument("--out-stem", default="", help="teleop: output name stem (default k1_showreel_teleop[_soma])")
    args = ap.parse_args()
    per_clip = {}
    for kv in args.clip_variant:
        k, v = kv.split("=", 1)
        per_clip[k] = "" if v in ("", "mediapipe") else v
        if per_clip[k] not in PIPELINE_LINE:
            raise SystemExit(f"unknown variant {v}")
    for which in (["robot", "teleop"] if args.which == "all" else [args.which]):
        soma = any(v for v in [args.variant, *per_clip.values()])
        stem = f"k1_showreel_{which}" + ("_soma" if which == "teleop" and soma else "")
        if which == "teleop" and args.out_stem:
            stem = args.out_stem
        out = VIDEOS / f"{stem}.mp4"
        if which == "teleop":
            total, segs, marks = make_teleop(out, args.title_top, args.variant, per_clip)
        else:
            total, segs, marks = make_robot(out, args.title_top)
        cmd = f".venv/bin/python -m k1.make_showreel {which}" + (
            (f" --variant {args.variant}" if args.variant else "")
            + (" --clip-variant " + " ".join(args.clip_variant) if args.clip_variant else "")
            + (f" --out-stem {args.out_stem}" if args.out_stem else "") if which == "teleop" else "")
        manifest = {"video": f"videos/{stem}.mp4", "size": f"{ms.W}x{ms.H}", "fps": ms.FPS,
                    "duration_s": round(total, 2), "title_top": args.title_top, "command": cmd,
                    **({"variant": args.variant or "mediapipe", "clip_variant": per_clip} if which == "teleop"
                       else {}),
                    "timeline": [{"segment": n, "start_s": round(a, 2), "end_s": round(b, 2)} for n, a, b in marks],
                    "segments": segs}
        out.with_suffix(".json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        print(f"{which}: {out} {total:.1f} s, {len(segs)} clips, falls {sum(s['fell'] for s in segs)}")
        if args.check_frames:
            print("  check frames ->", ms.check_sheet(out, args.check_frames))


if __name__ == "__main__":
    main()
