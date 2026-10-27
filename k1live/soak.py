"""Long-run test of the K1 live service: rehearsal video as the camera, viewer clients, injected faults.

    python -m k1live.soak --minutes 20 --source videos/k1live_rehearsal.mp4 --out logs/k1live/soak_20min
    python -m k1live.soak --minutes 120 --source ... --chaos default --out logs/k1live/soak_2h
    python -m k1live.soak --analyze logs/k1live/soak_2h          # summary of a finished run again
    python -m k1live.soak --make-rehearsal videos/k1live_rehearsal.mp4 --event-videos <event videos dir>

The service runs as a child (`python -m k1live.service --start --session-dir <out>/session`). Two viewer threads
read /cam.mjpg + /robot.mjpg and /api/events like the audience display does. Faults (`--chaos`, a comma list of
`<minute>:<kill|hang>:<sender|sim>`, or "default") are injected by pid:
kill = SIGKILL (crash), hang = SIGSTOP (a frozen process; the supervisor must restart it after 3 s of silence).
"default" puts all four kinds in the first 20 minutes and again in the last 20 (fault-free middle for the memory trend).
For each fault the summary gives the time to the fallback video, to the restart, and back to the live view.

Output (<out>/): session/ (service logs, metrics.csv, events.jsonl), soak.json (summary + PASS/FAIL per criterion,
CRITERIA below), soak.md (tables for docs/results.md), viewer.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from . import protocol as P

REPO = Path(__file__).resolve().parents[1]


def get_json(url, timeout=3.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


class Viewer(threading.Thread):
    """A display-like client: counts frames on an MJPEG stream (or messages on the SSE stream), reconnects."""

    def __init__(self, url, kind):
        super().__init__(daemon=True, name=f"viewer-{kind}")
        self.url, self.kind = url, kind
        self.n = 0
        self.reconnects = 0
        self.stop = threading.Event()

    def run(self):
        while not self.stop.is_set():
            try:
                with urllib.request.urlopen(self.url, timeout=5) as r:
                    while not self.stop.is_set():
                        line = r.readline()
                        if not line:
                            break
                        if self.kind == "sse":
                            self.n += line.startswith(b"data: ")
                        elif line.lower().startswith(b"content-length:"):
                            r.read(int(line.split(b":")[1]) + 2)  # blank line + JPEG
                            self.n += 1
            except Exception:
                pass
            if not self.stop.is_set():
                self.reconnects += 1
                self.stop.wait(1.0)


def parse_chaos(spec, minutes):
    if not spec:
        return []
    if spec == "default":
        # each fault kind at the start and again at the end; the middle stays fault-free so the memory trend gets a
        # long restart-free stretch (>= 60 min of a 2 h run)
        acts = [("kill", "sender"), ("kill", "sim"), ("hang", "sender"), ("hang", "sim")]
        if minutes >= 60:
            times = [5.0, 10.0, 15.0, 20.0, minutes - 20, minutes - 15, minutes - 10, minutes - 5]
        else:
            times = [minutes * f for f in (0.2, 0.4, 0.6, 0.8)]
        return [(t, *acts[i % len(acts)]) for i, t in enumerate(times)]
    out = []
    for item in spec.split(","):
        m, act, proc = item.split(":")
        assert act in ("kill", "hang") and proc in ("sender", "sim"), item
        out.append((float(m), act, proc))
    return sorted(out)


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sess = out / "session"
    port = P.UI_HTTP_PORT
    base = f"http://{P.HOST}:{port}"
    cmd = [sys.executable, "-m", "k1live.service", "--start", "--session-dir", str(sess), "--source", args.source,
           "--max-seconds", str(args.minutes * 60 + 30)] + (["--mock"] if args.mock else []) + args.service_args
    print("[soak]", " ".join(cmd), flush=True)
    svc = subprocess.Popen(cmd, cwd=str(REPO), stdout=open(out / "service.out", "ab"), stderr=subprocess.STDOUT)
    for _ in range(120):
        try:
            get_json(base + "/api/health")
            break
        except Exception:
            time.sleep(0.5)
    viewers = {k: Viewer(base + p, k if k == "sse" else "mjpeg") for k, p in
               (("cam", "/cam.mjpg?fps=30"), ("robot", "/robot.mjpg?fps=30"), ("sse", "/api/events"))}
    for v in viewers.values():
        v.start()
    chaos = parse_chaos(args.chaos, args.minutes)
    faults = []
    t0 = time.time()
    t_end = t0 + args.minutes * 60
    vf = open(out / "viewer.csv", "w", newline="")
    vw = csv.writer(vf)
    vw.writerow(["t", "elapsed_s", "cam_frames", "robot_frames", "sse_msgs", "reconnects"])
    last_print = 0.0
    try:
        while time.time() < t_end and svc.poll() is None:
            now = time.time()
            el = now - t0
            while chaos and el >= chaos[0][0] * 60:
                _, act, proc = chaos.pop(0)
                try:
                    pid = get_json(base + "/api/state")["procs"][proc]["pid"]
                except Exception:
                    pid = None
                if pid:
                    os.kill(pid, signal.SIGKILL if act == "kill" else signal.SIGSTOP)
                    faults.append({"t": now, "act": act, "proc": proc, "pid": pid})
                    print(f"[soak] {el / 60:6.1f} min: {act} {proc} (pid {pid})", flush=True)
            vw.writerow([round(now, 1), round(el, 1), viewers["cam"].n, viewers["robot"].n, viewers["sse"].n,
                         sum(v.reconnects for v in viewers.values())])
            vf.flush()
            if now - last_print > 60:
                last_print = now
                try:
                    s = get_json(base + "/api/state")
                    sim = s.get("sim") or {}
                    print(f"[soak] {el / 60:6.1f} min view={s['view']} hz={sim.get('ctrl_hz')} "
                          f"p95={s['lat_ms']['p95']} falls={s['counters']['falls']} resets={sim.get('resets')} "
                          f"restarts={s['procs']['sender']['restarts']}/{s['procs']['sim']['restarts']} "
                          f"rss={s['procs']['sender']['rss_mb']}/{s['procs']['sim']['rss_mb']}", flush=True)
                except Exception as e:
                    print(f"[soak] {el / 60:6.1f} min state unavailable: {e}", flush=True)
            time.sleep(2.0)
    finally:
        for f in faults:  # never leave a stopped process behind
            if f["act"] == "hang":
                try:
                    os.kill(f["pid"], signal.SIGCONT)
                    os.kill(f["pid"], signal.SIGKILL)
                except ProcessLookupError:
                    pass
        for v in viewers.values():
            v.stop.set()
        if svc.poll() is None:
            svc.send_signal(signal.SIGTERM)
            try:
                svc.wait(20)
            except subprocess.TimeoutExpired:
                svc.kill()
        vf.close()
    (out / "faults.json").write_text(json.dumps(faults, indent=1))
    summary = analyze(out, args.minutes)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


# ---------------------------------------------------------------------------------------------------------
def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _slope_mb_per_h(ts, ys):
    pts = [(t, y) for t, y in zip(ts, ys) if y is not None]
    if len(pts) < 10:
        return None
    n = len(pts)
    mt = sum(t for t, _ in pts) / n
    my = sum(y for _, y in pts) / n
    den = sum((t - mt) ** 2 for t, _ in pts)
    return round(sum((t - mt) * (y - my) for t, y in pts) / den * 3600, 2) if den else None


def analyze(out, minutes=None):
    out = Path(out)
    sess = out / "session" if (out / "session").exists() else out
    rows = list(csv.DictReader(open(sess / "metrics.csv")))
    events = [json.loads(x) for x in open(sess / "events.jsonl") if x.strip()]
    faults = json.loads((out / "faults.json").read_text()) if (out / "faults.json").exists() else []
    t = [_f(r["t"]) for r in rows]
    dur = (t[-1] - t[0]) / 60 if len(t) > 1 else 0.0

    def col(name):
        return [_f(r[name]) for r in rows]

    # trend per process lifetime: a restart starts a new process (new baseline), so fit the longest
    # stretch without a restart (warm-up skipped) instead of the whole run
    mem = {}
    for name in ("sender_rss_mb", "sim_rss_mb", "service_rss_mb"):
        proc = name.replace("_rss_mb", "")
        ys = col(name)
        rc = col(f"{proc}_restarts") if proc != "service" else [0.0] * len(rows)
        segs, cur, last_rc, seg_t0 = [], [], None, None
        for a, b, r in zip(t, ys, rc):
            if r != last_rc:
                if cur:
                    segs.append(cur)
                cur, last_rc, seg_t0 = [], r, a
            if a is not None and b is not None and seg_t0 is not None and a >= seg_t0 + WARMUP_S:
                cur.append((a, b))
        if cur:
            segs.append(cur)
        segs = [sg for sg in segs if len(sg) >= 2]
        if segs:
            best = max(segs, key=lambda sg: sg[-1][0] - sg[0][0])
            vals = [b for sg in segs for _, b in sg]
            mem[proc] = {"start_mb": round(best[0][1], 1), "end_mb": round(best[-1][1], 1), "max_mb": round(max(vals), 1),
                         "slope_mb_per_h": _slope_mb_per_h([a for a, _ in best], [b for _, b in best]),
                         "trend_window_min": round((best[-1][0] - best[0][0]) / 60, 1)}
    p95 = [v for v in col("lat_p95") if v is not None]
    hz = [v for v in col("ctrl_hz") if v]
    rt = [v for v in col("rt_factor") if v]
    live_rows = [r for r in rows if r["view"] == "live"]
    proc_fail = [e for e in events if e["kind"] == "proc_fail"]
    injected = len(faults)
    fault_rows = []
    for i, f in enumerate(faults):
        t_next = faults[i + 1]["t"] if i + 1 < len(faults) else float("inf")
        after = [e for e in events if f["t"] <= e["t"] < min(t_next, f["t"] + 120)]

        def first(pred):
            for e in after:
                if pred(e):
                    return round(e["t"] - f["t"], 2)
            return None
        t_fail = first(lambda e: e["kind"] == "proc_fail" and e.get("proc") == f["proc"])
        t_fb = first(lambda e: e["kind"] == "view" and e.get("view") == "fallback")
        t_healthy = first(lambda e: e["kind"] == "proc_healthy" and e.get("proc") == f["proc"])
        t_live = None
        if t_fb is not None:
            t_live = first(lambda e: e["kind"] == "view" and e.get("view") == "live" and e["t"] - f["t"] > t_fb)
        fault_rows.append({"min": round((f["t"] - t[0]) / 60, 1) if t else None, "act": f["act"], "proc": f["proc"],
                           "detected_s": t_fail, "fallback_s": t_fb, "healthy_s": t_healthy, "live_s": t_live})
    last = rows[-1] if rows else {}
    summary = {
        "duration_min": round(dur, 1),
        "planned_min": minutes,
        "restarts": {"sender": int(_f(last.get("sender_restarts")) or 0), "sim": int(_f(last.get("sim_restarts")) or 0)},
        "injected_faults": injected,
        "unexpected_failures": max(0, len(proc_fail) - injected),
        "proc_fail_reasons": [f"{e['proc']}: {e.get('reason')}" for e in proc_fail][:20],
        "memory": mem,
        "lat_p95_ms": {"median": _pct(p95, 0.5), "max": max(p95) if p95 else None, "share_le_150": (
            round(sum(v <= 150 for v in p95) / len(p95), 3) if p95 else None)},
        "ctrl_hz": {"median": _pct(hz, 0.5), "p01": _pct(hz, 0.01), "min": min(hz) if hz else None},
        "rt_factor": {"median": _pct(rt, 0.5), "p01": _pct(rt, 0.01), "min": min(rt) if rt else None},
        "falls": int(_f(last.get("falls")) or 0),
        "resets": int(_f(last.get("resets")) or 0) if last.get("resets") not in (None, "") else None,
        "watchdog_resets": int(_f(last.get("svc_resets")) or 0),
        "fallback_switches": int(_f(last.get("fallback_switches")) or 0),
        "auto_idle": int(_f(last.get("auto_idle")) or 0),
        "live_share": round(len(live_rows) / len(rows), 3) if rows else None,
        "faults": fault_rows,
    }
    vpath = out / "viewer.csv"
    if vpath.exists():
        v = list(csv.DictReader(open(vpath)))
        if len(v) > 1:
            el = _f(v[-1]["elapsed_s"]) - _f(v[0]["elapsed_s"])
            summary["viewer"] = {k: round((_f(v[-1][c]) - _f(v[0][c])) / el, 1) for k, c in
                                 (("cam_fps", "cam_frames"), ("robot_fps", "robot_frames"), ("sse_hz", "sse_msgs"))}
            summary["viewer"]["reconnects"] = int(_f(v[-1]["reconnects"]))
    summary["verdict"] = verdict(summary)
    (out / "soak.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    (out / "soak.md").write_text(to_markdown(summary))
    return summary


CRITERIA = [  # (key, Korean label, test) — docs/runbook.md
    ("no_unexpected_failures", "예기치 않은 프로세스 실패 0", lambda s: s["unexpected_failures"] == 0),
    ("faults_recovered", "장애마다 10 s 안에 정상 복귀, 15 s 안에 실시간 화면",
     lambda s: all((f["healthy_s"] is not None and f["healthy_s"] <= 10)
                   and (f["fallback_s"] is None or (f["live_s"] is not None and f["live_s"] <= 15))
                   for f in s["faults"])),
    ("latency", "지연 p95 ≤ 150 ms (99 % 이상의 시간)",
     lambda s: s["lat_p95_ms"]["share_le_150"] is not None and s["lat_p95_ms"]["share_le_150"] >= 0.99),
    ("ctrl_hz", "제어 50 ± 2 Hz (하위 1 % ≥ 48)",
     lambda s: s["ctrl_hz"]["median"] is not None and abs(s["ctrl_hz"]["median"] - 50) <= 2
     and (s["ctrl_hz"]["p01"] or 0) >= 48),
    ("rt_factor", "실시간 비율 하위 1 % ≥ 0.98", lambda s: (s["rt_factor"]["p01"] or 0) >= 0.98),
    ("falls", "낙상 0 · 감시 리셋 0", lambda s: s["falls"] == 0 and s["watchdog_resets"] == 0),
    ("memory", "메모리 추세 sender·sim ≤ 30 MB/h, service ≤ 5 MB/h (재시작 없는 30분 이상 구간)",
     lambda s: None if any((m.get("trend_window_min") or 0) < MEM_WINDOW_MIN for m in s["memory"].values())
     else all(m["slope_mb_per_h"] is not None and m["slope_mb_per_h"] <= (5 if k == "service" else 30)
              for k, m in s["memory"].items())),
    ("viewer", "관객 화면 클라이언트 카메라·로봇 ≥ 20 fps", lambda s: "viewer" not in s or (
        s["viewer"]["cam_fps"] >= 20 and s["viewer"]["robot_fps"] >= 20)),
]


def verdict(s):
    out = {}
    for key, _, test in CRITERIA:
        try:
            r = test(s)
            out[key] = None if r is None else bool(r)
        except (KeyError, TypeError):
            out[key] = False
    out["pass"] = all(v is not False for v in out.values())
    out["complete"] = all(v is not None for k, v in out.items() if k != "pass")
    return out


def _pct(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def to_markdown(s):
    s = json.loads(json.dumps(s), object_hook=lambda d: {k: ("–" if v is None else v) for k, v in d.items()})
    L = [f"| 항목 | 값 |", "|---|---|",
         f"| 시험 시간 | {s['duration_min']} 분 |",
         f"| 프로세스 재시작 (sender / sim) | {s['restarts']['sender']} / {s['restarts']['sim']} "
         f"(주입한 장애 {s['injected_faults']}, 예기치 않은 실패 {s['unexpected_failures']}) |",
         f"| 지연 p95 (30초 창) 중앙값 / 최대 | {s['lat_p95_ms']['median']} / {s['lat_p95_ms']['max']} ms "
         f"(≤150 ms 비율 {s['lat_p95_ms']['share_le_150']}) |",
         f"| 제어 Hz 중앙값 / 하위 1% / 최소 | {s['ctrl_hz']['median']} / {s['ctrl_hz']['p01']} / "
         f"{s['ctrl_hz']['min']} |",
         f"| 실시간 비율 중앙값 / 하위 1% / 최소 | {s['rt_factor']['median']} / {s['rt_factor']['p01']} / "
         f"{s['rt_factor']['min']} |",
         f"| 낙상 / 리셋 / 감시 리셋 | {s['falls']} / {s['resets']} / {s['watchdog_resets']} |",
         f"| 대체 영상 전환 / 자동 대기 | {s['fallback_switches']} / {s['auto_idle']} |",
         f"| 실시간 화면 비율 | {s['live_share']} |"]
    if "viewer" in s:
        v = s["viewer"]
        L.append(f"| 관객 화면 클라이언트 (카메라 / 로봇 fps, SSE Hz, 재연결) | {v['cam_fps']} / {v['robot_fps']}, "
                 f"{v['sse_hz']}, {v['reconnects']} |")
    L += ["", "| 프로세스 | 시작 MB | 끝 MB | 최대 MB | 추세 MB/h (재시작 없는 가장 긴 구간, 분) |", "|---|---|---|---|---|"]
    for k, m in s["memory"].items():
        L.append(f"| {k} | {m['start_mb']} | {m['end_mb']} | {m['max_mb']} | {m['slope_mb_per_h']} "
                 f"({m['trend_window_min']}) |")
    if "verdict" in s:
        L += ["", "| 기준 | 결과 |", "|---|---|"]
        for key, label, _ in CRITERIA:
            v = s["verdict"].get(key)
            L.append(f"| {label} | {'보류 (구간 부족)' if v in (None, '–') else ('통과' if v else '**실패**')} |")
        L.append(f"| **종합** | {('**통과**' if s['verdict'].get('pass') else '**실패**')}"
                 f"{'' if s['verdict'].get('complete') else ' (보류 항목 제외)'} |")
    if s["faults"]:
        L += ["", "| 분 | 장애 | 감지 s | 대체 영상 s | 정상 복귀 s | 실시간 화면 s |", "|---|---|---|---|---|---|"]
        for f in s["faults"]:
            L.append(f"| {f['min']} | {f['proc']} {'강제 종료' if f['act'] == 'kill' else '멈춤(SIGSTOP)'} | "
                     f"{f['detected_s']} | {f['fallback_s']} | {f['healthy_s']} | {f['live_s']} |")
    return "\n".join(L) + "\n"


WARMUP_S = 300.0      # memory trend: skip each process's first 5 min (first video loop, allocator settling)
MEM_WINDOW_MIN = 30.0  # a shorter restart-free stretch cannot tell a slow leak from settling -> "deferred"

REHEARSAL = [  # (file under the event videos dir, seconds) -> 115.6 s loop; all Pexels clips
    ("event_open_arms.mp4", None), ("raw/pexels_6764051.mp4", None), ("event_raise_arms.mp4", None),
    ("raw/pexels_10149187.mp4", None), ("raw/pexels_9558217.mp4", None)]


def make_rehearsal(event_videos, out, empty_s=25.0):
    """Arms, walking towards the camera, arms, clapping, turning, then an empty scene (idle mode), 1280x720 @ 30."""
    import shutil
    ff = shutil.which("ffmpeg")
    if ff is None:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
    ev = Path(event_videos)
    V = ("scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=0x303338,"
         "fps=30,format=yuv420p,setsar=1")
    cmd = [ff, "-hide_banner", "-loglevel", "error", "-y"]
    for name, _ in REHEARSAL:
        cmd += ["-i", str(ev / name)]
    cmd += ["-f", "lavfi", "-t", str(empty_s), "-i", "color=c=0x303338:s=1280x720:r=30"]
    n = len(REHEARSAL)
    chains = [f"[{i}:v]{V}[v{i}]" for i in range(n)] + [f"[{n}:v]format=yuv420p,setsar=1[v{n}]"]
    graph = ";".join(chains) + ";" + "".join(f"[v{i}]" for i in range(n + 1)) + f"concat=n={n + 1}:v=1:a=0[out]"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd + ["-filter_complex", graph, "-map", "[out]", "-c:v", "libx264", "-preset", "veryfast",
                          "-crf", "23", "-an", str(out)], check=True)
    print(f"[soak] rehearsal video -> {out}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="K1 live service soak test")
    ap.add_argument("--make-rehearsal", default="", help="write the rehearsal loop video to this path and exit")
    ap.add_argument("--event-videos", default="", help="event videos dir (with raw/) for --make-rehearsal "
                    "(default $K1_EVENT_VIDEOS or videos/event)")
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--source", default="")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--chaos", default="", help='"default" or "<min>:<kill|hang>:<sender|sim>,..."')
    ap.add_argument("--out", default="")
    ap.add_argument("--analyze", default="", help="summarise an existing run directory")
    ap.add_argument("service_args", nargs="*", help="extra arguments for k1live.service (after --)")
    args = ap.parse_args(argv)
    if args.make_rehearsal:
        make_rehearsal(args.event_videos or os.environ.get("K1_EVENT_VIDEOS", str(REPO / "videos" / "event")),
                       args.make_rehearsal)
        return 0
    if args.analyze:
        print(json.dumps(analyze(args.analyze), ensure_ascii=False, indent=1))
        return 0
    if not args.source or not args.out:
        ap.error("--source and --out are required for a run")
    args.source = str(Path(args.source).resolve()) if not args.source.isdigit() else args.source
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
