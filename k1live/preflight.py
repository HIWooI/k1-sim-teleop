"""Pre-event checks for the K1 live demo, printed as a Korean table (macOS and Ubuntu).

    .venv/bin/python -m k1live.preflight               # event day: includes the camera check
    .venv/bin/python -m k1live.preflight --no-camera   # everything except opening the camera
    .venv/bin/python -m k1live.preflight --json        # machine-readable

Checks: Python/venv, packages, model files, GEM-X perception (when event.yaml / the panel chose it: CUDA, GEM-X
environment and checkpoint; --gemx-full also loads the models), ONNX policies load and run, MuJoCo offscreen render of the K1 scene,
the service ports are free, disk space, CPU load, memory (budget per backend: hybrid ~2.5 GB more), power, fallback/rehearsal videos, browser, sender/sim_server
modules, camera (open + one frame). Exit code 1 when any check fails (warnings do not fail).
"""
from __future__ import annotations

import argparse
import glob
import importlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

if sys.platform.startswith("linux") and "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "egl"

from k1live import protocol as P  # noqa: E402

PASS, WARN, FAIL, SKIP = "통과", "경고", "실패", "건너뜀"
PACKAGES = ["numpy", "scipy", "yaml", "mujoco", "onnxruntime", "cv2", "mediapipe", "zmq"]
POLICIES = ["k1/assets/student_asset/exported/policy.onnx", "k1/assets/loco_armdriven_r2/exported/policy.onnx"]
PORTS = [(P.UI_HTTP_PORT, "운영 패널"), (P.CAM_HTTP_PORT, "카메라 영상"), (P.ROBOT_HTTP_PORT, "로봇 영상"),
         (P.HUMAN_PORT, "HUMAN"), (P.CTRL_PORT, "CTRL"), (P.SIM_STATE_PORT, "SIM_STATE"),
         (P.SENDER_STATE_PORT, "SENDER_STATE")]


class Report:
    def __init__(self):
        self.rows = []

    def add(self, item, status, detail=""):
        self.rows.append({"항목": item, "결과": status, "내용": detail})
        return status

    def run(self, item, fn):
        t0 = time.time()
        try:
            res = fn()
        except Exception as e:  # a check that crashes is a failed check, not a crashed preflight
            res = (FAIL, f"{type(e).__name__}: {e}")
        status, detail = res
        if time.time() - t0 > 3:
            detail += f" ({time.time() - t0:.0f}초)"
        return self.add(item, status, detail)

    @property
    def failed(self):
        return any(r["결과"] == FAIL for r in self.rows)


def _w(s):
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def _pad(s, n):
    return s + " " * max(0, n - _w(s))


def table(rows, max_detail=78):
    cols = ["항목", "결과", "내용"]
    mark = {PASS: "✓ ", WARN: "! ", FAIL: "✗ ", SKIP: "- "}
    rows = [{k: str(r[k]) for k in cols} for r in rows]
    for r in rows:
        r["결과"] = mark.get(r["결과"], "") + r["결과"]
        if _w(r["내용"]) > max_detail:
            d = r["내용"]
            while _w(d) > max_detail - 1:
                d = d[:-1]
            r["내용"] = d + "…"
    w = {k: max(_w(k), *(_w(r[k]) for r in rows)) for k in cols}
    line = "+" + "+".join("-" * (w[k] + 2) for k in cols) + "+"
    out = [line, "| " + " | ".join(_pad(k, w[k]) for k in cols) + " |", line]
    for r in rows:
        out.append("| " + " | ".join(_pad(r[k], w[k]) for k in cols) + " |")
    out.append(line)
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------------------
def check_python():
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    v = platform.python_version()
    if sys.version_info < (3, 10):
        return FAIL, f"Python {v} (3.10 이상 필요)"
    return (PASS if in_venv else WARN), f"Python {v} · {'venv' if in_venv else 'venv 아님'} · {sys.executable}"


def check_packages():
    vers, missing = [], []
    for m in PACKAGES:
        try:
            mod = importlib.import_module(m)
            vers.append(f"{m} {getattr(mod, '__version__', '?')}")
        except Exception as e:
            missing.append(f"{m} ({type(e).__name__})")
    if missing:
        return FAIL, "없음: " + ", ".join(missing)
    return PASS, ", ".join(vers)


def check_models(mock=False):
    miss, have = [], []
    for p in POLICIES:
        (have if (REPO / p).is_file() else miss).append(Path(p).parts[2])
    tasks = sorted(glob.glob(str(REPO / "models" / "pose_landmarker_*.task")))
    (have if tasks else miss).append("pose_landmarker" + (f"({len(tasks)})" if tasks else ""))
    try:
        from k1 import sim as k1sim
        xml_ok = k1sim.K1_XML.is_file() and k1sim.MESHDIR.is_dir()
        (have if xml_ok else miss).append("K1 MJCF·메시" + ("" if xml_ok else f" ({k1sim.K1_XML})"))
    except Exception as e:
        miss.append(f"k1.sim ({e})")
    hint = "K1 정책은 scripts/fetch_weights.sh · " if any(not (REPO / p).is_file() for p in POLICIES) else ""
    if miss and not mock:
        return FAIL, hint + "없음: " + ", ".join(miss)
    return (WARN if miss else PASS), hint + "있음: " + ", ".join(have) + (f" · 없음: {', '.join(miss)}" if miss else "")


def check_sonic():
    from k1live.service import DEFAULTS, resolve
    ck = sorted(glob.glob(str(resolve(DEFAULTS["sonic_checkpoint_glob"]))))
    backend = (REPO / "k1live" / "backends" / "sonic.py").exists()
    if ck and backend:
        return PASS, f"sonic 백엔드 사용 가능 · 체크포인트 {len(ck)}개 (최신 {Path(ck[-1]).parent.relative_to(REPO)})"
    why = ([] if ck else ["체크포인트 없음"]) + ([] if backend else ["k1live/backends/sonic.py 없음"])
    return WARN, "sonic 비활성 (split만 사용): " + ", ".join(why)


def check_hybrid():
    from k1live.service import hybrid_available_locally, load_config
    load_config()   # event.yaml hybrid_variant -> $K1LIVE_HYBRID_VARIANT, as the service does
    if hybrid_available_locally():
        v = os.environ.get("K1LIVE_HYBRID_VARIANT", "models/k1/k1s_walk3/deploy/k1s_walk3_005000")
        return PASS, f"hybrid 백엔드 사용 가능 (행사 기본) · SONIC 보행 {Path(v).name} · 플래너 · 스테이징 메시"
    return WARN, ("hybrid 비활성 → split으로 시작(scripts/fetch_weights.sh): SONIC 보행 체크포인트(models/k1/k1s_walk3/deploy/"
                  "k1s_walk3_005000 또는 K1LIVE_HYBRID_VARIANT), models/planner/planner_sonic.onnx, 스테이징 K1 메시 중 없음")


def check_onnx():
    import numpy as np
    import onnxruntime as ort
    out = []
    for p in POLICIES:
        f = REPO / p
        if not f.is_file():
            return FAIL, f"scripts/fetch_weights.sh 필요 · {p} 없음"
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        sess = ort.InferenceSession(str(f), so, providers=["CPUExecutionProvider"])
        feed = {}
        for i in sess.get_inputs():
            shape = [d if isinstance(d, int) and d > 0 else 1 for d in i.shape]
            feed[i.name] = np.zeros(shape, np.float32)
        sess.run(None, feed)
        t0 = time.perf_counter()
        for _ in range(20):
            sess.run(None, feed)
        ms = (time.perf_counter() - t0) / 20 * 1e3
        out.append(f"{Path(p).parts[2]} {ms:.2f} ms")
        if ms > 5.0:
            return WARN, "느림 (1틱 5 ms 초과): " + ", ".join(out)
    return PASS, "추론 1회: " + ", ".join(out)


def check_render():
    import mujoco
    from k1 import sim as k1sim
    t0 = time.perf_counter()
    model = k1sim.build_model(P.POLICY_JOINTS)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    t_build = time.perf_counter() - t0
    r = mujoco.Renderer(model, height=360, width=640)
    try:
        r.update_scene(data)
        r.render()
        t1 = time.perf_counter()
        for _ in range(10):
            r.update_scene(data)
            img = r.render()
        ms = (time.perf_counter() - t1) / 10 * 1e3
    finally:
        r.close()
    if img.max() == 0:
        return FAIL, "렌더 결과가 검은 화면"
    gl = os.environ.get("MUJOCO_GL", "기본")
    return (PASS if ms < 30 else WARN), f"K1 장면 {t_build:.1f} s 로드, 640×360 {ms:.1f} ms/프레임 (GL {gl})"


def port_busy(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind((P.HOST, port))
        return False
    except OSError:
        return True
    finally:
        s.close()


def check_ports():
    busy = [f"{p}({n})" for p, n in PORTS if port_busy(p)]
    if not busy:
        return PASS, "모두 비어 있음: " + ", ".join(str(p) for p, _ in PORTS)
    if port_busy(P.UI_HTTP_PORT):
        return WARN, f"사용 중: {', '.join(busy)} — 서비스가 이미 실행 중일 수 있음 (http://127.0.0.1:{P.UI_HTTP_PORT}/)"
    return FAIL, f"사용 중: {', '.join(busy)} — 다른 프로그램이 포트를 쓰고 있음"


def check_disk():
    path = REPO / "logs"
    path = path if path.exists() else REPO
    free = shutil.disk_usage(path).free / 1e9
    if free < 2:
        return FAIL, f"남은 공간 {free:.1f} GB (2 GB 이상 필요)"
    return (PASS if free >= 10 else WARN), f"남은 공간 {free:.0f} GB ({path})"


def check_load():
    n = os.cpu_count() or 1
    l1, l5, _ = os.getloadavg()
    msg = f"부하 {l1:.1f} / {l5:.1f} (1분/5분), 코어 {n}개"
    if l1 > 0.7 * n:
        return WARN, msg + " — 다른 무거운 프로그램을 끄세요"
    return PASS, msg


# resident memory of the processes of one show (GB, measured on the event PC): sim_server split 0.76 /
# hybrid 3.12 (planner ONNX + its retarget models), sender (MediaPipe) ~1, render process ~0.5, service + browser ~1.5
SIM_GB = {"split": 0.8, "hybrid": 3.2, "sonic": 3.2, "sonic_human": 1.5}
OTHER_GB = 3.0


def memory_gb():
    """(total, available) GB, or None."""
    try:
        if sys.platform == "darwin":
            total = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5).stdout) / 1e9
            vm = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
            page = int(vm.split("page size of")[1].split()[0]) if "page size of" in vm else 4096
            pages = {l.split(":")[0].strip(): int(l.split(":")[1].strip().rstrip(".")) for l in vm.splitlines()[1:] if ":" in l}
            avail = sum(pages.get(k, 0) for k in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"))
            return total, avail * page / 1e9
        info = {l.split(":")[0]: int(l.split()[1]) for l in Path("/proc/meminfo").read_text().splitlines() if l.split()}
        return info["MemTotal"] / 1e6, info.get("MemAvailable", info.get("MemFree", 0)) / 1e6
    except (OSError, ValueError, IndexError, KeyError, subprocess.TimeoutExpired):
        return None


def check_memory():
    from k1live.service import load_config
    m = memory_gb()
    if m is None:
        return SKIP, "메모리 정보 없음"
    total, avail = m
    backend = load_config().get("backend", "split")
    need = SIM_GB.get(backend, 3.2) + OTHER_GB
    msg = f"전체 {total:.0f} GB · 지금 쓸 수 있는 {avail:.1f} GB · 필요 약 {need:.1f} GB (백엔드 {backend})"
    if avail < 1.5:
        return FAIL, msg + " — 다른 앱을 닫으세요"
    if avail < need:
        tip = " — hybrid는 split보다 약 2.5 GB 더 씀: 다른 앱·브라우저 탭을 닫거나 패널에서 split" if backend == "hybrid" else \
            " — 다른 앱·브라우저 탭을 닫으세요"
        return WARN, msg + tip
    return PASS, msg


def check_power():
    if sys.platform == "darwin":
        try:
            out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=5).stdout
        except (OSError, subprocess.TimeoutExpired):
            return SKIP, "pmset 없음"
        first = out.splitlines()[0] if out else ""
        pct = next((w for w in out.split() if w.endswith("%;")), "").rstrip(";")
        if "AC Power" in first:
            return PASS, f"전원 연결됨 {pct}".strip()
        return WARN, f"배터리 사용 중 {pct} — 전원 어댑터를 연결하세요".strip()
    acs = glob.glob("/sys/class/power_supply/A*/online")
    if not acs:
        return SKIP, "전원 정보 없음 (데스크톱)"
    on = any(Path(a).read_text().strip() == "1" for a in acs)
    return (PASS, "전원 연결됨") if on else (WARN, "배터리 사용 중 — 전원 어댑터를 연결하세요")


def check_videos():
    from k1live.service import list_videos, load_config
    cfg = load_config()
    fb = [Path(p) for p in cfg["fallback_videos"]]
    ok = [p.name for p in fb if p.is_file()]
    reh = list_videos(cfg["rehearsal_videos"])
    detail = f"대체 영상 {len(ok)}/{len(fb)} · 리허설 영상 {len(reh)}개"
    if not ok:
        return WARN, detail + " — 대체 영상이 없으면 장애 시 '잠시 후 다시 시작합니다' 화면만 나옴"
    return (PASS if len(ok) == len(fb) else WARN), detail


def find_browser():
    if sys.platform == "darwin":
        for app in ("Google Chrome", "Chromium", "Microsoft Edge"):
            for base in ("/Applications", os.path.expanduser("~/Applications")):
                if Path(f"{base}/{app}.app").exists():
                    return app
        return "Safari" if Path("/Applications/Safari.app").exists() else None
    for b in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "microsoft-edge"):
        if shutil.which(b):
            return b
    return "firefox" if shutil.which("firefox") else None


def check_browser():
    b = find_browser()
    if b is None:
        return WARN, "브라우저를 찾지 못함 — 주소를 직접 여세요"
    chrome = any(k in b.lower() for k in ("chrome", "chromium", "edge"))
    return (PASS if chrome else WARN), b + ("" if chrome else " (Chrome 권장: 관객 화면 앱 창·전체 화면)")


def check_gemx(full=False):
    """perception gemx (event.yaml / panel): the GEM-X worker's own check, run the way the service would start it.
    Light (default): CUDA, GEM-X imports, checkpoint on disk (a few seconds); full: load + warm up the models."""
    from k1live.sender_gemx import parse_args, worker_cmd
    from k1live.service import gemx_args, load_config

    cfg = load_config()
    if cfg.get("perception") != "gemx":
        return SKIP, "인식 = MediaPipe (GEM-X를 쓰려면 패널 설정 › 인식 또는 event.yaml perception: gemx)"
    cmd = worker_cmd(parse_args(gemx_args(cfg)))
    cmd = [c for c in cmd if c != "--serve"] + ["--check"] + ([] if full else ["--light"])
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300 if full else 90, cwd=str(REPO))
    except FileNotFoundError as e:
        return WARN, f"GEM-X 환경 없음 ({e.filename}) → 행사 중에는 MediaPipe로 자동 전환"
    except subprocess.TimeoutExpired:
        return WARN, "GEM-X 점검 시간 초과 → MediaPipe로 자동 전환될 수 있음"
    try:
        info = json.loads((r.stdout.strip().splitlines() or ["{}"])[-1])
    except ValueError:
        info = {"error": (r.stderr.strip().splitlines() or ["?"])[-1]}
    if r.returncode != 0:
        why = {5: "CUDA GPU 없음", 6: "GEM-X 환경·체크포인트 문제"}.get(r.returncode, f"exit {r.returncode}")
        return WARN, f"GEM-X 사용 불가 ({why}: {info.get('error', '')}) → 행사 중에는 MediaPipe로 자동 전환"
    extra = f" · 모델 준비 {info['ready_s']} s" if "ready_s" in info else ""
    return PASS, f"GEM-X 사용 가능: {info.get('device')} {info.get('vram_gb', '')} GB · 체크포인트 {Path(str(info.get('ckpt', '?'))).name}{extra}"


def check_modules(mock=False):
    have = {m: (REPO / "k1live" / f"{m}.py").exists() for m in ("sender", "sim_server")}
    if all(have.values()):
        return PASS, "k1live.sender, k1live.sim_server"
    miss = [m for m, ok in have.items() if not ok]
    return (WARN if mock else FAIL), "없음: " + ", ".join(f"k1live/{m}.py" for m in miss) + (
        " (mock 모드로만 실행 가능)" if mock else "")


def check_camera(index):
    import cv2
    t0 = time.time()
    cap = cv2.VideoCapture(index)
    try:
        if not cap.isOpened():
            hint = (" — 시스템 설정 › 개인정보 보호 및 보안 › 카메라에서 터미널을 허용하세요"
                    if sys.platform == "darwin" else " — 카메라 연결·/dev/video* 권한을 확인하세요")
            return FAIL, f"카메라 {index} 열 수 없음" + hint
        frame = None
        while time.time() - t0 < 5:
            ok, frame = cap.read()
            if ok and frame is not None and frame.size:
                break
            frame = None
        if frame is None:
            return FAIL, f"카메라 {index}: 5초 안에 프레임 없음"
        h, w = frame.shape[:2]
        fps = cap.get(cv2.CAP_PROP_FPS)
        return PASS, f"카메라 {index}: {w}×{h} {fps:.0f} fps, 첫 프레임 {time.time() - t0:.1f} s"
    finally:
        cap.release()


def main(argv=None):
    ap = argparse.ArgumentParser(description="K1 live demo pre-event checks")
    ap.add_argument("--no-camera", action="store_true", help="skip opening the camera")
    ap.add_argument("--camera", type=int, default=None, help="camera index (default: the one the service uses)")
    ap.add_argument("--quick", action="store_true", help="skip the ONNX and render checks")
    ap.add_argument("--mock", action="store_true", help="sender/sim_server may be missing (mock rehearsal)")
    ap.add_argument("--gemx-full", action="store_true", help="GEM-X: load and warm up the models (20-60 s)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    rep = Report()
    rep.run("파이썬", check_python)
    rep.run("패키지", check_packages)
    rep.run("모델 파일", lambda: check_models(args.mock))
    rep.run("sonic 백엔드", check_sonic)
    rep.run("hybrid 백엔드", check_hybrid)
    rep.run("하위 프로그램", lambda: check_modules(args.mock))
    if args.quick and not args.gemx_full:
        rep.add("GEM-X 인식", SKIP, "--quick")
    else:
        rep.run("GEM-X 인식", lambda: check_gemx(args.gemx_full))
    if args.quick:
        rep.add("ONNX 정책", SKIP, "--quick")
        rep.add("MuJoCo 렌더", SKIP, "--quick")
    else:
        rep.run("ONNX 정책", check_onnx)
        rep.run("MuJoCo 렌더", check_render)
    rep.run("포트", check_ports)
    rep.run("디스크", check_disk)
    rep.run("CPU 부하", check_load)
    rep.run("메모리", check_memory)
    rep.run("전원", check_power)
    rep.run("영상 파일", check_videos)
    rep.run("브라우저", check_browser)
    if args.no_camera:
        rep.add("카메라", SKIP, "--no-camera")
    else:
        idx = args.camera
        if idx is None:
            from k1live.service import load_config
            idx = int(load_config()["camera"])
        rep.run("카메라", lambda: check_camera(idx))

    if args.json:
        print(json.dumps({"ok": not rep.failed, "rows": rep.rows}, ensure_ascii=False, indent=1))
    else:
        print("\nK1 실시간 데모 사전 점검  " + time.strftime("%Y-%m-%d %H:%M") + f"  ({platform.system()})")
        print(table(rep.rows))
        n = {s: sum(r["결과"] == s for r in rep.rows) for s in (PASS, WARN, FAIL, SKIP)}
        print(f"통과 {n[PASS]} · 경고 {n[WARN]} · 실패 {n[FAIL]} · 건너뜀 {n[SKIP]}")
        print("→ 실패 항목을 해결한 뒤 다시 실행하세요." if rep.failed else "→ 데모를 시작할 수 있습니다.")
    return 1 if rep.failed else 0


if __name__ == "__main__":
    sys.exit(main())
