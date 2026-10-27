"""Rehearsal / latency measurement of the live loop (MuJoCo only, no camera): sender on a playlist of event clips
played in real time (fake camera) + sim_server, driven over CTRL like the service would, on free ports.

    .venv/bin/python -m k1live.rehearse --minutes 10                    # the 6 event clips, walking mixed in
    .venv/bin/python -m k1live.rehearse --minutes 1 --clips a.mp4 b.mp4 --no-walk

Walking: unless --no-walk, an operator `drive` pattern runs every 60 s while the arms keep following the clip
(forward 0.4 m/s, turn, back, sideways, turn back), so arms + walking are exercised together. If
k1live.locomotion is installed, walking intent from the clips themselves also reaches the robot (gesture clips
should give none). Output: <out>/rehearsal.json (+ the sim_server log dir and sender.jsonl) and a markdown table.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import zmq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from k1live import protocol as P  # noqa: E402

EVENT = Path(os.environ.get("K1_EVENT_VIDEOS", str(ROOT / "videos" / "event")))
CLIPS = ["raw/pexels_10374292.mp4", "raw/pexels_10149176.mp4", "raw/pexels_10149187.mp4", "raw/pexels_10515017.mp4",
         "event_raise_arms.mp4", "event_open_arms.mp4", "event_stand.mp4"]
# (start s within each minute, duration s, vx, vy, wz)
WALK = [(15, 6, 0.4, 0.0, 0.0), (21, 4, 0.0, 0.0, 0.5), (25, 5, -0.25, 0.0, 0.0), (30, 4, 0.0, 0.2, 0.0),
        (34, 4, 0.0, 0.0, -0.5), (38, 5, 0.6, 0.0, 0.0), (43, 3, 0.3, 0.0, 0.4)]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def clip_seconds(path):
    import cv2

    c = cv2.VideoCapture(str(path))
    n, fps = c.get(cv2.CAP_PROP_FRAME_COUNT), c.get(cv2.CAP_PROP_FPS) or 30.0
    c.release()
    return n / fps


def pct(a, q):
    return round(float(np.percentile(a, q)), 2) if len(a) else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--clips", nargs="*", default=[], help="video files (default: the event clips)")
    ap.add_argument("--repeat", type=int, default=2, help="play each clip this many times before switching")
    ap.add_argument("--no-walk", action="store_true")
    ap.add_argument("--backend", default="split")
    ap.add_argument("--out", default="")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--render-client", action="store_true", help="keep a /robot.mjpg client connected (as the "
                    "audience screen would), so rendering load is part of the measurement")
    ap.add_argument("--sim-args", default="", help="extra sim_server arguments")
    ap.add_argument("--sender-args", default="", help="extra sender arguments")
    a = ap.parse_args(argv)
    clips = [Path(c) for c in a.clips] or [EVENT / c for c in CLIPS]
    clips = [c for c in clips if c.exists()]
    if not clips:
        raise SystemExit("no clips found")
    dur = {c: clip_seconds(c) for c in clips}
    out = Path(a.out) if a.out else ROOT / "logs" / "k1live" / f"rehearsal_{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    ports = {k: free_port() for k in ("human", "ctrl", "sim", "snd", "cam", "robot")}
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    ctx = zmq.Context()
    ctrl = ctx.socket(zmq.PUB)
    ctrl.setsockopt(zmq.LINGER, 0)
    ctrl.bind(f"tcp://127.0.0.1:{ports['ctrl']}")
    load0 = os.getloadavg()
    sim_cmd = [sys.executable, "-m", "k1live.sim_server", "--backend", a.backend, "--human-port", str(ports["human"]),
               "--ctrl-port", str(ports["ctrl"]), "--state-port", str(ports["sim"]), "--robot-port",
               str(ports["robot"]), "--log-dir", str(out / "sim")] + (["--no-render"] if a.no_render else []) + \
        a.sim_args.split()
    snd_cmd = [sys.executable, "-m", "k1live.sender", "--source", str(clips[0]), "--loop", "--human-port",
               str(ports["human"]), "--ctrl-port", str(ports["ctrl"]), "--state-port", str(ports["snd"]),
               "--http-port", str(ports["cam"]), "--log", str(out / "sender.jsonl")] + a.sender_args.split()
    sim = subprocess.Popen(sim_cmd, cwd=ROOT, env=env, stdout=open(out / "sim.out", "w"), stderr=subprocess.STDOUT)
    snd = subprocess.Popen(snd_cmd, cwd=ROOT, env=env, stdout=open(out / "sender.out", "w"), stderr=subprocess.STDOUT)
    subs = []
    for port, topic in ((ports["sim"], P.TOPIC_SIM_STATE), (ports["snd"], P.TOPIC_SENDER_STATE)):
        s = ctx.socket(zmq.SUB)
        s.setsockopt(zmq.LINGER, 0)
        s.connect(f"tcp://127.0.0.1:{port}")
        s.setsockopt(zmq.SUBSCRIBE, topic)
        subs.append(s)
    poller = zmq.Poller()
    for s in subs:
        poller.register(s, zmq.POLLIN)
    reader = None
    if a.render_client and not a.no_render:
        reader = subprocess.Popen([sys.executable, "-c", (
            "import sys,time,urllib.request\n"
            "time.sleep(4)\n"
            "while True:\n"
            "  try:\n"
            f"    r=urllib.request.urlopen('http://127.0.0.1:{ports['robot']}/robot.mjpg',timeout=5)\n"
            "    while r.read(65536): pass\n"
            "  except Exception: time.sleep(0.5)\n")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    first = {}
    sims, snds, timeline = [], [], []
    ci, clip_t0, t_run0 = 0, None, None
    timeline.append({"t": t0, "clip": clips[0].name})
    total = a.minutes * 60.0
    try:
        while True:
            for s, _ in poller.poll(50):
                fr = s.recv_multipart()
                if fr[0] == P.TOPIC_SIM_STATE:
                    m = P.decode(fr, P.SimState)
                    first.setdefault("sim_state_s", round(time.time() - t0, 2))
                    if m.note != "loading":
                        sims.append(m)
                        if t_run0 is None:
                            t_run0 = time.time()
                else:
                    first.setdefault("sender_state_s", round(time.time() - t0, 2))
                    snds.append(P.decode(fr, P.SenderState))
            now = time.time()
            if sim.poll() is not None or snd.poll() is not None:
                print("[rehearse] a process exited early", flush=True)
                break
            if t_run0 is None:
                if now - t0 > 60:
                    print("[rehearse] sim_server never came up", flush=True)
                    break
                continue
            el = now - t_run0
            if el >= total:
                break
            if clip_t0 is None:
                clip_t0 = now
            if now - clip_t0 > dur[clips[ci % len(clips)]] * a.repeat and len(clips) > 1:
                ci += 1
                c = clips[ci % len(clips)]
                ctrl.send_multipart(P.encode(P.TOPIC_CTRL, P.CtrlMsg("set", {"video": str(c), "loop": True})))
                timeline.append({"t": now, "clip": c.name})
                clip_t0 = now
            if not a.no_walk:
                sec = el % 60.0
                for st, d, vx, vy, wz in WALK:
                    if st <= sec < st + d:
                        ctrl.send_multipart(P.encode(P.TOPIC_CTRL, P.CtrlMsg("drive", {"vx": vx, "vy": vy, "wz": wz})))
                        break
                time.sleep(0.05)
    finally:
        load1 = os.getloadavg()
        for p in (snd, sim):
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)
        codes = {}
        for name, p in (("sender", snd), ("sim_server", sim)):
            try:
                codes[name] = p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill()
                codes[name] = "killed"
        if reader is not None:
            reader.kill()
        ctx.destroy(0)
    res = analyse(out, sims, snds, timeline, first, codes, (load0, load1), a, clips)
    (out / "rehearsal.json").write_text(json.dumps(res, indent=2))
    print(table(res))
    return 0


def analyse(out, sims, snds, timeline, first, codes, loads, a, clips):
    summ = json.loads((out / "sim" / "summary.json").read_text()) if (out / "sim" / "summary.json").exists() else {}
    secs = [json.loads(l) for l in open(out / "sim" / "sim.jsonl") if '"event": "sec"' in l] \
        if (out / "sim" / "sim.jsonl").exists() else []
    events = [json.loads(l) for l in open(out / "sim" / "sim.jsonl") if '"event": "sec"' not in l] \
        if (out / "sim" / "sim.jsonl").exists() else []
    secs = secs[3:]  # warm-up
    hz = [s["hz"] for s in secs]
    rt = [s["rt"] for s in secs]
    snd = [json.loads(l) for l in open(out / "sender.jsonl")] if (out / "sender.jsonl").exists() else []
    pres = [r for r in snd if r["present"]]
    # per clip arm error (1 Hz windows assigned to the clip playing then)
    per_clip = {}
    for i, seg in enumerate(timeline):
        t1 = timeline[i + 1]["t"] if i + 1 < len(timeline) else 1e18
        w = [s for s in secs if seg["t"] + 2 <= s["t"] < t1 and s["arm_err_goal"] is not None and s["play"] is None]
        pc = per_clip.setdefault(seg["clip"], {"deg": [], "cm": [], "track": []})
        pc["deg"] += [s["arm_err_goal"] for s in w]
        pc["cm"] += [s["hand_err_cm"] for s in w if s.get("hand_err_cm") is not None]
        pc["track"] += [s["tracking_frac"] for s in secs if seg["t"] + 2 <= s["t"] < t1]
    walking = [s for s in secs if abs(s["cmd"][0]) > 0.05 or abs(s["cmd"][1]) > 0.05 or abs(s["cmd"][2]) > 0.05]
    span = (snd[-1]["t_capture"] - snd[0]["t_capture"]) if len(snd) > 1 else 0
    return {
        "minutes": a.minutes, "backend": a.backend, "clips": [c.name for c in clips], "walk_pattern": not a.no_walk,
        "render": not a.no_render, "render_client": a.render_client, "host": os.uname().nodename,
        "loadavg_start_end": [round(loads[0][0], 2), round(loads[1][0], 2)], "exit_codes": codes, **first,
        "latency_ms": summ.get("lat_ms"),
        "ctrl_hz": {"mean": round(float(np.mean(hz)), 3) if hz else None, "min": min(hz) if hz else None,
                    "p5": pct(hz, 5)},
        "rt_factor": {"mean": round(float(np.mean(rt)), 4) if rt else None, "min": min(rt) if rt else None,
                      "p5": pct(rt, 5)},
        "skipped_ticks": summ.get("skipped"), "ticks": summ.get("ticks"),
        "policy_ms": summ.get("policy_ms"), "tick_ms": summ.get("tick_ms"),
        "arm_err_deg": summ.get("arm_err_deg"),
        "hand_err_cm": summ.get("hand_err_cm"),
        "per_clip": {k: {"arm_err_deg": round(float(np.mean(v["deg"])), 2) if v["deg"] else None,
                         "hand_err_cm": round(float(np.mean(v["cm"])), 2) if v["cm"] else None,
                         "tracking_frac": round(float(np.mean(v["track"])), 3) if v["track"] else None}
                     for k, v in per_clip.items()},
        "falls": summ.get("falls"), "resets": summ.get("resets"),
        "walking_seconds": len(walking), "person_walk_s": summ.get("person_walk_s"),
        "loco_estimator": "k1live.locomotion" if (ROOT / "k1live" / "locomotion.py").exists() else None, "max_cmd": [round(max(abs(s["cmd"][i]) for s in secs), 3) for i in range(3)]
        if secs else None,
        "heading_err_deg": {"mean_abs": round(float(np.mean([abs(s["yaw_err_deg"]) for s in secs
                                                             if s.get("yaw_err_deg") is not None])), 2)
                            if any(s.get("yaw_err_deg") is not None for s in secs) else None,
                            "max_abs": max((abs(s["yaw_err_deg"]) for s in secs if s.get("yaw_err_deg") is not None),
                                           default=None)},
        "max_tilt_deg": max((s["tilt"] for s in secs), default=None),
        "min_pelvis_z": min((s["pelvis_z"] for s in secs), default=None),
        "sender": {"frames": len(snd), "fps": round(len(snd) / span, 2) if span else None,
                   "present_frac": round(len(pres) / max(len(snd), 1), 3),
                   "pose_ms_p50": pct([r["pose_ms"] for r in snd], 50), "pose_ms_p95": pct([r["pose_ms"] for r in snd], 95),
                   "mp_ms_p50": pct([r["mp_ms"] for r in snd], 50), "ik_ms_p50": pct([r["ik_ms"] for r in pres], 50),
                   "ik_ms_p95": pct([r["ik_ms"] for r in pres], 95),
                   "send_ms_p50": pct([(r["t_sent"] - r["t_capture"]) * 1e3 for r in snd], 50),
                   "send_ms_p95": pct([(r["t_sent"] - r["t_capture"]) * 1e3 for r in snd], 95),
                   "source_frames_skipped": int(sum(r["skipped"] for r in snd))},
        "events": [e for e in events if e.get("event") in ("fall", "reset", "skip", "backend_failed")][:50],
        "out": str(out),
    }


def table(r):
    lat = r["latency_ms"] or {}
    ae = r["arm_err_deg"] or {}
    rows = [("지연 p50 / p95 (캡처 → 제어 반영)", f"{lat.get('p50')} / {lat.get('p95')} ms", "p95 ≤ 150 ms"),
            ("제어 주기 평균 / 최소(1 s 창)", f"{r['ctrl_hz']['mean']} / {r['ctrl_hz']['min']} Hz", "50 ± 2"),
            ("실시간 비율 평균 / 최소", f"{r['rt_factor']['mean']} / {r['rt_factor']['min']}", "≥ 0.98"),
            ("건너뛴 틱", f"{r['skipped_ticks']} / {r['ticks']}", ""),
            ("정책 1틱 p50 / p95", f"{r['policy_ms']['p50']} / {r['policy_ms']['p95']} ms", "≤ 5 ms"),
            ("팔 오차 (IK 목표 대비) 평균 / p90", f"{ae.get('vs_ik_goal_mean')} / {ae.get('vs_ik_goal_p90')}°", "≤ 8°"),
            ("팔 오차 (서버 레퍼런스 대비)", f"{ae.get('vs_reference_mean')}°", ""),
            ("손 위치 오차 (IK 목표 자세 대비, 몸통 좌표) 평균 / p90",
             f"{(r.get('hand_err_cm') or {}).get('mean')} / {(r.get('hand_err_cm') or {}).get('p90')} cm", ""),
            ("낙상 / 리셋", f"{r['falls']} / {r['resets']}", "0"),
            ("사람 영상에서 나온 보행 (제스처 영상 = 오보행)", f"{r.get('person_walk_s')} s", "0"),
            ("운영자 drive 보행 시간", f"{r['walking_seconds']} s", ""),
            ("방향 유지 오차 (yaw_ref − yaw) 평균 / 최대 (1 s 샘플)",
             f"{r['heading_err_deg']['mean_abs']} / {r['heading_err_deg']['max_abs']}°", ""),
            ("sender fps, 사람 검출 비율", f"{r['sender']['fps']}, {r['sender']['present_frac']}", "~30"),
            ("sender MediaPipe+IK p50 / p95", f"{r['sender']['pose_ms_p50']} / {r['sender']['pose_ms_p95']} ms", "")]
    s = "| 항목 | 결과 | 기준 |\n|---|---|---|\n" + "\n".join(f"| {a} | {b} | {c} |" for a, b, c in rows)
    return s


if __name__ == "__main__":
    sys.exit(main())
