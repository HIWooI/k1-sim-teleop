"""K1 live demo service: supervises sender + sim_server and serves the operator panel and the audience display.

    .venv/bin/python -m k1live.service                      # event: camera (last used index), real processes
    .venv/bin/python -m k1live.service --source videos/x.mp4 # rehearsal: a video file played as a camera
    .venv/bin/python -m k1live.service --mock               # protocol mocks instead of sender/sim_server

HTTP on 127.0.0.1:8700 (k1live/protocol.py UI_HTTP_PORT):
    /            operator panel (Korean)           /display        audience display (16:9)
    /api/state   status JSON                       /api/events     status stream (SSE, 5 Hz)
    /api/cmd     POST {"cmd": ..., ...}            /api/config     static page configuration
    /cam.mjpg    camera relay (one upstream connection to the sender, fanned out)
    /robot.mjpg  robot render relay                /cam.jpg, /robot.jpg  latest frame
    /fallback/N  fallback video N (Range requests)

Session logs: logs/k1live/<YYYYmmdd-HHMMSS>/ (service.log, sender.log, sim_server.log, events.jsonl, metrics.csv).
Event settings: k1live/web/event.yaml (texts, fallback videos, timings); operator choices persist in
logs/k1live/settings.json. MuJoCo simulation only: nothing here talks to a robot.
"""
from __future__ import annotations

import argparse
import copy
import glob
import http.client
import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import yaml
import zmq

from . import protocol as P
from .supervisor import ProcSpec, Supervisor, rss_mb

REPO = Path(__file__).resolve().parents[1]
WEB = Path(__file__).resolve().parent / "web"
LOG_ROOT = REPO / "logs" / "k1live"
SETTINGS_PATH = LOG_ROOT / "settings.json"
EVENT_DIR = os.environ.get("K1_EVENT_VIDEOS", str(REPO / "videos" / "event"))  # "{event_dir}" in paths (event media)

GESTURES = [  # k1/gestures.py GESTURES, with operator labels
    ("wave_hello", "손 흔들기"), ("bow", "인사"), ("welcome_open_arms", "환영"), ("clap", "박수"),
    ("point_guide", "안내"), ("heart", "하트"), ("raise_both_arms", "만세"), ("stand_at_ease", "편히 서기"),
]

DEFAULTS = {
    "title": "K1 텔레옵",
    "subtitle_sonic": "GEAR-SONIC 방식 전신 제어",
    "guide": "카메라 앞에 서서 팔을 움직여 보세요 · 앞으로 한 걸음 나오면 로봇이 걸어요",
    "guide_no_walk": "카메라 앞에 서서 팔을 움직여 보세요",
    "guide_presenter": "발표자의 팔 동작을 로봇이 실시간으로 따라 합니다",   # walking mode "presenter"
    "msg_welcome": "안녕하세요! 정면을 보고 편하게 서 주세요",
    "msg_calibrating": "그대로 잠시만 서 계세요",
    "msg_come_closer": "카메라 앞에 서 주세요",
    "label_cam": "카메라",
    "label_robot": "AI Sapiens K1 · 물리 시뮬레이션",
    "label_fallback": "녹화 영상",
    "event_dir": EVENT_DIR,       # env K1LIVE_EVENT_DIR overrides (e.g. on another machine)
    "fallback_videos": ["{event_dir}/fallback/demo_1.mp4", "{event_dir}/fallback/demo_2.mp4"],   # recorded demo reels
    "rehearsal_videos": ["{event_dir}/event_*.mp4", "videos/k1live_*.mp4"],
    "cameras": [0, 1, 2],
    "camera": 0,                  # camera index used at start (the last one chosen in the panel persists)
    "loop_video": True,
    "sender_cmd": ["{python}", "-m", "k1live.sender", "--source", "{source}", "{loop}", "{no_mirror}", "{arm_ik}"],
    # perception (panel 설정 › 인식): mediapipe = MediaPipe Pose on the CPU (any laptop; default); gemx = GEM-X on an NVIDIA
    # GPU (k1live/sender_gemx.py, docs/architecture.md): whole-body 3D with depth, steadier at a distance. If GEM-X cannot
    # start (no CUDA / GEM-X environment / checkpoint: the sender exits 5) the service switches to MediaPipe by itself.
    "perception": "mediapipe",
    "gemx": {"perception_dir": None,  # GEM-X perception folder (demo_webcam.py, GEM-X/ inside); None =
                                      # $K1LIVE_GEMX_PERCEPTION or third_party/gemx_perception (docs/setup.md)
             "python": None,          # GEM-X venv python; None = <perception_dir>/GEM-X/.venv/bin/python
             "cmd": None,             # whole worker command instead, e.g. through ssh to a GPU machine on the LAN
             "fmt": None,             # frame transport raw | jpeg (default raw locally, jpeg with cmd)
             "hfov": 0,               # camera horizontal FOV, deg (0 = GEM-X's guess); scales the depth in metres
             "arm_src": "ik",         # ik = bone-direction IK on the GEM-X skeleton | mlp = distilled retarget MLP
             "max_h": 1080,           # frames taller than this are downscaled first (a far person needs pixels)
             "start_timeout_s": 240,  # model loading limit (typically 20-40 s; the display shows the fallback video)
             "args": []},             # extra k1live.sender_gemx arguments
    "sender_gemx_cmd": ["{python}", "-m", "k1live.sender_gemx", "--source", "{source}", "{loop}", "{no_mirror}",
                        "{arm_ik}", "{gemx}"],
    "sim_cmd": ["{python}", "-m", "k1live.sim_server", "--backend", "{backend}", "--no-autostart",
                "--idle-after", "{idle_after}", "--log-dir", "{session}/sim", "{arm_mode}", "{arm_vmax}", "{arm_amax}"],
    "mock_sender_cmd": ["{python}", "-m", "k1live.mock", "sender", "--source", "{source}", "{loop}",
                        "{no_mirror}"],
    "mock_sim_cmd": ["{python}", "-m", "k1live.mock", "sim", "--backend", "{backend}", "--no-autostart",
                     "--idle-after", "{idle_after}", "--log-dir", "{session}/sim"],
    "grace_s": 45.0,              # first status after (re)start may take this long (models loading)
    "stale_s": 3.0,               # then: no status for this long -> restart the process
    "idle_after_s": 20.0,         # nobody in view this long while following -> idle (greeting loop)
    "auto_calibrate": False,      # idle + person stays 2 s -> calibrate
    "auto_start": False,          # ... then start following
    "fallback": "auto",           # auto | on | off
    "fallback_after_s": 2.0,      # camera or sim bad this long -> fallback video
    "live_after_s": 3.0,          # both good again this long -> live
    "fall_reset_s": 6.0,          # sim still fallen after this -> the service sends reset (sim auto-resets in 3 s)
    "cam_restart_s": 20.0,        # camera source reports no frames this long -> restart the sender
    "mem_limit_mb": {"sender": 1500, "sim": 4000, "sender_gemx": 8000},  # a child above this is restarted while
                                                                         # nobody is followed (GEM-X: torch + CUDA)
    "mode": "live",
    "walk": True,
    "loco_profile": "visitor",    # walking mode: visitor (step off home = walk, turn = turn) | presenter (no turning,
                                  # walk along while the presenter strolls; docs/architecture.md)
    "crop": None,                 # person search region [x0, y0, x1, y1] (0..1 of the raw camera image) or None
    "max_speed": 0.5,             # m/s, operator slider
    "speed_limit": 0.8,           # slider maximum
    "turn_speed": 0.6,            # rad/s for the operator's Q/E keys
    "mirror": True,
    "backend": "split",
    "sonic_checkpoint_glob": "models/k1/*/deploy_*/model_decoder.onnx",
    "hybrid_variant": None,     # hybrid backend: SONIC walking checkpoint dir (None = $K1LIVE_HYBRID_VARIANT or walk3 5000)
    # arm path (panel "팔 경로"): improved (docs/architecture.md): bone-direction IK in the sender, PD arms with
    # gravity compensation in sim_server, faster arm reference; classic = position IK + mimic student arms, 6 rad/s
    "arm_path": "improved",
    "arm_ik": "dir", "arm_mode": "pd", "arm_vmax": 12.0, "arm_amax": 150.0,   # what "improved" means (event.yaml)
    "display_cam_flip": False,    # flip the camera image on the display (if the sender does not mirror it)
    "cam_fit": "cover",           # cover | contain
    "robot_fit": "cover",
    "gestures": [g for g, _ in GESTURES],
}
PERSIST = ("camera", "walk", "loco_profile", "crop", "max_speed", "mirror", "backend", "idle_after_s",
           "auto_calibrate", "auto_start", "fallback", "arm_path", "perception")
ARM_PATHS = ("improved", "classic")
PERCEPTIONS = ("mediapipe", "gemx")
EXIT_GEMX = 5   # k1live.sender_gemx: GEM-X unavailable -> run k1live.sender (MediaPipe)
CLASSIC_ARMS = {"arm_ik": "shape18", "arm_mode": "student", "arm_vmax": 6.0, "arm_amax": 80.0}
SET_KEYS = PERSIST + ("source",)


def load_config(path=None, cli=None, *, read_saved=True):
    cfg = copy.deepcopy(DEFAULTS)
    p = Path(path) if path else WEB / "event.yaml"
    if p.exists():
        cfg.update(yaml.safe_load(p.read_text()) or {})
    try:
        saved = json.loads(SETTINGS_PATH.read_text()) if read_saved else {}
        cfg.update({k: v for k, v in saved.items() if k in PERSIST})
    except (OSError, ValueError):
        pass
    cli = {k: v for k, v in (cli or {}).items() if v is not None}
    for env, key in (("K1LIVE_LOCO_PROFILE", "loco_profile"), ("K1LIVE_CROP", "crop")):  # like a CLI value: used,
        if os.environ.get(env) and key not in cli:                                       # never saved
            cli[key] = os.environ[env]
    if cli.get("loco_profile") not in (None,) + P.LOCO_PROFILES:
        print(f"[service] ignoring walking profile {cli.pop('loco_profile')!r}", flush=True)
    if "crop" in cli:
        cli["crop"] = norm_crop(cli["crop"])
    cfg.update(cli)
    if cfg.get("loco_profile") not in P.LOCO_PROFILES:
        cfg["loco_profile"] = "visitor"
    if cfg.get("arm_path") not in ARM_PATHS:
        cfg["arm_path"] = "improved"
    if cfg.get("perception") not in PERCEPTIONS:
        print(f"[service] ignoring perception {cfg.get('perception')!r} -> mediapipe", flush=True)
        cfg["perception"] = "mediapipe"
    cfg["gemx"] = {**DEFAULTS["gemx"], **(cfg.get("gemx") or {})}
    cfg["crop"] = norm_crop(cfg.get("crop"))
    if cfg.get("hybrid_variant"):   # event.yaml checkpoint for the hybrid backend: this process (checks) and children
        v = resolve(str(cfg["hybrid_variant"]))
        if (v / "model_decoder.onnx").exists():
            os.environ["K1LIVE_HYBRID_VARIANT"] = str(v)
        else:                       # a missing checkpoint must not take the whole hybrid down: the built-in walk3 5000
            print(f"[service] hybrid_variant {v} missing -> default SONIC walking checkpoint", flush=True)
            cfg["hybrid_variant"] = None
    if cfg.get("backend") == "hybrid" and not hybrid_available_locally():   # event default; never crash-loop on it
        print("[service] hybrid backend files missing (SONIC walking checkpoint / planner / staged meshes) -> split",
              flush=True)
        cfg["backend"] = "split"
    cfg["_cli_keys"] = sorted(cli)
    if os.environ.get("K1LIVE_EVENT_DIR"):
        cfg["event_dir"] = os.environ["K1LIVE_EVENT_DIR"]
    for k in ("fallback_videos", "rehearsal_videos"):
        cfg[k] = [str(resolve(str(v).replace("{event_dir}", str(cfg["event_dir"])))) for v in cfg[k]]
    if not cfg.get("source"):
        cfg["source"] = f"camera:{int(cfg['camera'])}"
    return cfg


def norm_crop(v):
    """Person region as stored / sent: [x0, y0, x1, y1] rounded to 0.001, or None (whole frame / invalid)."""
    from .sender import parse_crop  # same rule as the sender (inside 0..1, >= 10 % per side)

    b = parse_crop(v)
    return None if b is None else [round(x, 3) for x in b]


def save_settings(cfg):
    """Operator choices only: values that came from the command line (rehearsal, soak) are not written back."""
    try:
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        try:
            old = json.loads(SETTINGS_PATH.read_text())
        except (OSError, ValueError):
            old = {}
        cli = set(cfg.get("_cli_keys", ()))
        out = {k: (old[k] if k in cli and k in old else cfg[k]) for k in PERSIST if k not in cli or k in old}
        SETTINGS_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    except OSError as e:
        print(f"[service] settings not saved: {e}", flush=True)


def resolve(path):
    p = Path(os.path.expanduser(str(path)))
    return p if p.is_absolute() else REPO / p


def list_videos(patterns):
    out = []
    for pat in patterns:
        for f in sorted(glob.glob(str(resolve(pat)))):
            if f not in out:
                out.append(f)
    return out


def source_cli(source):
    """'camera:1' -> '1' (sender CLI), a video path stays a path."""
    return source.split(":", 1)[1] if source.startswith("camera:") else source


def arm_settings(cfg):
    """Effective arm path settings: the event.yaml arm_* values for "improved", the old chain for "classic"."""
    if cfg.get("arm_path", "improved") == "classic":
        return dict(CLASSIC_ARMS)
    return {k: cfg.get(k, DEFAULTS[k]) for k in CLASSIC_ARMS}


def gemx_args(cfg):
    """event.yaml `gemx` settings -> k1live.sender_gemx arguments."""
    g = {**DEFAULTS["gemx"], **(cfg.get("gemx") or {})}
    out = []
    if g.get("cmd"):
        out += ["--gemx-cmd", str(g["cmd"])]
    else:
        if g.get("perception_dir"):
            out += ["--gemx-perception", str(resolve(g["perception_dir"]))]
        if g.get("python"):
            out += ["--gemx-python", str(resolve(g["python"]))]
    if g.get("fmt") in ("raw", "jpeg"):
        out += ["--gemx-fmt", g["fmt"]]
    if float(g.get("hfov") or 0) > 0:
        out += ["--hfov", f"{float(g['hfov']):g}"]
    if g.get("arm_src") == "mlp":
        out += ["--arm-src", "mlp"]
    if g.get("max_h"):
        out += ["--max-h", str(int(g["max_h"]))]
    out += ["--start-timeout", f"{float(g.get('start_timeout_s') or 240):g}"]
    return out + [str(x) for x in (g.get("args") or [])]


def gemx_env_missing(cfg):
    """'' when the GEM-X sender is worth starting, else why not: its own cheap check (GEM-X python and perception
    folder on disk). Without them (a fresh clone) it would only start, exit 5 and hand over to MediaPipe."""
    if cfg["sender_gemx_cmd"] != DEFAULTS["sender_gemx_cmd"]:
        return ""
    from k1live.sender_gemx import parse_args, preflight

    return preflight(parse_args(gemx_args(cfg)))


def build_argv(template, cfg, session=""):
    src = cfg["source"]
    arm = arm_settings(cfg)
    subs = {"{python}": sys.executable, "{source}": source_cli(src), "{backend}": cfg["backend"],
            "{arm_ik}": f"--arm-ik={arm['arm_ik']}", "{arm_mode}": f"--arm-mode={arm['arm_mode']}",
            "{arm_vmax}": f"--arm-vmax={float(arm['arm_vmax']):g}", "{arm_amax}": f"--arm-amax={float(arm['arm_amax']):g}",
            "{idle_after}": f"{float(cfg['idle_after_s']):g}",
            "{loop}": "--loop" if (cfg.get("loop_video") and not src.startswith("camera:")) else "",
            "{no_mirror}": "" if cfg.get("mirror", True) else "--no-mirror"}
    argv = []
    for a in template:
        if a == "{gemx}":
            argv += gemx_args(cfg)
            continue
        a = subs.get(a, a).replace("{session}", str(session))
        if a != "":
            argv.append(a)
    return argv


def child_env(cfg=None):
    """Extra environment for sender / sim_server: headless EGL rendering on Linux unless the user chose a GL; the
    hybrid backend's SONIC checkpoint from event.yaml `hybrid_variant` (else $K1LIVE_HYBRID_VARIANT passes through)."""
    env = {}
    if cfg and cfg.get("hybrid_variant") and (resolve(str(cfg["hybrid_variant"])) / "model_decoder.onnx").exists():
        env["K1LIVE_HYBRID_VARIANT"] = str(resolve(str(cfg["hybrid_variant"])))
    if sys.platform.startswith("linux") and "MUJOCO_GL" not in os.environ:
        env["MUJOCO_GL"] = "egl"
    return env


def sonic_available_locally(cfg):
    return (REPO / "k1live" / "backends" / "sonic.py").exists() and bool(
        glob.glob(str(resolve(cfg["sonic_checkpoint_glob"]))))


def hybrid_available_locally():
    """k1live/backends/hybrid.py, its SONIC walking checkpoint ($K1LIVE_HYBRID_VARIANT or walk3 5000), the planner ONNX
    and the staged K1 meshes the planner retargets with - file checks only (before the sim_server reports its list)."""
    variant = Path(os.environ.get("K1LIVE_HYBRID_VARIANT",
                                  REPO / "models" / "k1" / "k1s_walk3" / "deploy" / "k1s_walk3_005000"))
    meshes = REPO / "sonic_k1" / "overlay" / "gear_sonic" / "data" / "assets" / "robot_description" / "urdf" / "k1" / "meshes"
    return ((REPO / "k1live" / "backends" / "hybrid.py").exists() and (variant / "model_decoder.onnx").exists()
            and (REPO / "models" / "planner" / "planner_sonic.onnx").exists() and any(meshes.glob("*.stl")))


def pct(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


# ---------------------------------------------------------------------------------------------------------
class Controller:
    """All service decisions (no I/O): desired state, re-sync of restarted processes, idle mode, fall watchdog,
    gesture playlist, fallback-video hysteresis. `send(cmd, args)` publishes CTRL; `event(kind, **d)` logs."""

    FRESH_S = 2.0

    def __init__(self, cfg, send, event, restart_proc=None, set_argv=None, frame_age=None, proc_up=None):
        self.cfg = cfg
        self._send = send
        self.event = event
        self.restart_proc = restart_proc or (lambda name, reason: None)
        self.set_argv = set_argv or (lambda name: None)
        self.frame_age = frame_age or (lambda name: None)
        self.proc_up = proc_up or (lambda name: True)
        self.lock = threading.RLock()
        self.d = dict(running=False, estop=False, mode=cfg["mode"], backend=cfg["backend"], walk=cfg["walk"],
                      loco_profile=cfg["loco_profile"], crop=cfg["crop"], max_speed=float(cfg["max_speed"]),
                      mirror=cfg["mirror"], source=cfg["source"], perception=cfg.get("perception", "mediapipe"))
        self.perc_active = self.d["perception"]   # what the sender runs now (GEM-X may have fallen back to MediaPipe)
        self.perc_note = ""
        self.gemx_failed = None       # set by the supervisor thread (sender exit 5), handled in tick()
        self.sim = None
        self.sim_t = 0.0
        self.snd = None
        self.snd_t = 0.0
        self.replay = {"sim": True, "sender": True}
        self.auto_idle = False
        self.absent_since = None
        self.present_since = None
        self.view = "fallback"
        self.view_reason = "시작 중"
        self.bad_since = None
        self.good_since = None
        self.lat = deque()
        self.playlist = None          # index into cfg["gestures"] while the operator's playlist runs
        self.pl_gesture_seen = False
        self.pl_sent_t = 0.0
        self.calib = None             # {"t", "seen_false", "state": running|done|failed}
        self.fallen_since = None
        self.last_svc_reset = 0.0
        self.cam_bad_since = None
        self.last_cam_restart = 0.0
        self.counters = dict(falls=0, svc_resets=0, fallback_switches=0, auto_idle=0, cam_restarts=0,
                             gemx_fallbacks=0)
        self.ensure = {}              # reconcile bookkeeping: key -> [tries, last_sent_t, first_mismatch_t]
        self.t_start = time.time()

    # ---------------------------------------------------------------- inputs
    def send(self, cmd, args=None):
        self._send(cmd, args or {})

    def on_proc_event(self, name, ev, d):
        if ev == "start":
            with self.lock:
                self.replay["sim" if name == "sim" else "sender"] = True

    def sender_template(self):
        """Which sender_cmd the next sender (re)start uses: GEM-X while it is the active perception."""
        return "sender_gemx_cmd" if self.perc_active == "gemx" else "sender_cmd"

    def on_gemx_unavailable(self, reason):
        """The GEM-X sender exited 5 (no CUDA / environment / checkpoint, or its worker keeps dying): run MediaPipe.
        The operator's choice stays gemx (selecting it again retries)."""
        with self.lock:
            if self.perc_active != "gemx":
                return False
            self.perc_active = "mediapipe"
            self.perc_note = reason or "GEM-X 시작 실패"
            self.counters["gemx_fallbacks"] += 1
            self.event("gemx_fallback", reason=self.perc_note)
            self.set_argv("sender")
            return True

    def on_sim(self, s, now):
        with self.lock:
            if self.sim is not None and now - self.sim_t > 1.5:
                self.replay["sim"] = True
            if self.sim is not None and s.get("fallen") and not self.sim.get("fallen"):
                self.counters["falls"] += 1
                self.event("fall", resets=s.get("resets"))
            self.sim, self.sim_t = s, now
            if s.get("human_present") and s.get("human_age_ms") is not None:
                self.lat.append((now, float(s["human_age_ms"])))
            if self.replay["sim"] and s.get("note") != "loading":
                self.replay["sim"] = False
                self._replay("sim")

    def on_snd(self, s, now):
        with self.lock:
            if self.snd is not None and now - self.snd_t > 1.5:
                self.replay["sender"] = True
            self.snd, self.snd_t = s, now
            if self.replay["sender"] and s.get("note") != "loading":
                self.replay["sender"] = False
                self._replay("sender")

    def _source_args(self):
        src = self.d["source"]
        return {"camera": int(src.split(":", 1)[1])} if src.startswith("camera:") else {"video": src}

    def effective_mode(self):
        """The sim's base mode: live (follow) or idle (stand, attract loop). Gestures play on top of idle."""
        return "live" if (self.d["mode"] == "live" and not self.auto_idle) else "idle"

    def full_settings(self):
        a = dict(mode=self.effective_mode(), backend=self.d["backend"], walk=self.d["walk"],
                 loco_profile=self.d["loco_profile"], crop=self.d["crop"], max_speed=self.d["max_speed"],
                 mirror=self.d["mirror"], idle_after=float(self.cfg["idle_after_s"]))
        a.update(self._source_args())
        return a

    def _replay(self, who):
        self.send(P.CMD_SET, self.full_settings())
        if who == "sim":
            if self.d["estop"]:
                self.send(P.CMD_ESTOP)
            elif self.d["running"]:
                self.send(P.CMD_START)
            else:
                self.send(P.CMD_STOP)
        self.event("resync", proc=who)

    # ---------------------------------------------------------------- operator commands
    def command(self, cmd, a):
        """Returns (ok, message)."""
        with self.lock:
            now = time.time()
            if cmd == "toggle":
                cmd = "stop" if self.d["running"] else "start"
            if cmd == "start":
                self.d.update(running=True, estop=False, mode="live")
                self.auto_idle = False
                self.playlist = None
                self.absent_since = None
                self.send(P.CMD_SET, {"mode": "live"})
                self.send(P.CMD_START)
            elif cmd == "stop":
                self.d["running"] = False
                self.send(P.CMD_STOP)
            elif cmd == "estop":
                self.d.update(running=False, estop=True)
                self.playlist = None
                self.send(P.CMD_ESTOP)
            elif cmd == "reset":
                self.send(P.CMD_RESET)
            elif cmd == "calibrate":
                self.calib = {"t": now, "seen_false": False, "state": "running"}
                self.send(P.CMD_CALIBRATE)
            elif cmd == "mode":
                m = a.get("mode")
                if m not in P.MODES:
                    return False, f"모드 {m!r} 없음"
                self.d["mode"] = m
                self.auto_idle = False
                if m != "gesture":
                    self.playlist = None
                if m != "live" and self.d["running"]:
                    self.d["running"] = False
                    self.send(P.CMD_STOP)
                self.send(P.CMD_SET, {"mode": self.effective_mode()})
            elif cmd == "gesture":
                name = a.get("name")
                if name not in self.cfg["gestures"]:
                    return False, f"제스처 {name!r} 없음"
                self._to_gesture_mode()
                self.playlist = None
                self.send(P.CMD_GESTURE, {"name": name})
            elif cmd == "playlist":
                if a.get("on", True):
                    self._to_gesture_mode()
                    self.playlist = 0
                    self._play_next(now)
                else:
                    self.playlist = None
            elif cmd == "drive":
                lim = float(self.d["max_speed"])
                tl = float(self.cfg["turn_speed"])
                clip = lambda v, m: max(-m, min(m, float(v)))  # noqa: E731
                self.send(P.CMD_DRIVE, {"vx": clip(a.get("vx", 0), lim), "vy": clip(a.get("vy", 0), lim),
                                        "wz": clip(a.get("wz", 0), tl)})
            elif cmd == "set":
                return self.set_values({k: v for k, v in a.items() if k != "cmd"})
            else:
                return False, f"알 수 없는 명령 {cmd!r}"
            if cmd != "drive":
                self.event("cmd", cmd=cmd, **{k: v for k, v in a.items() if k != "cmd"})
            return True, "ok"

    def _to_gesture_mode(self):
        if self.d["running"]:
            self.d["running"] = False
            self.send(P.CMD_STOP)
        self.auto_idle = False
        if self.d["mode"] != "gesture":
            self.d["mode"] = "gesture"
            self.send(P.CMD_SET, {"mode": self.effective_mode()})

    def _play_next(self, now):
        g = self.cfg["gestures"][self.playlist % len(self.cfg["gestures"])]
        self.send(P.CMD_GESTURE, {"name": g})
        self.pl_gesture_seen = False
        self.pl_sent_t = now

    def set_values(self, a):
        args = {}
        for k, v in a.items():
            if k not in SET_KEYS:
                return False, f"설정 {k!r} 없음"
            if k == "backend":
                if v not in P.BACKENDS:
                    return False, f"백엔드 {v!r} 없음"
                if v not in self.backends_available():
                    return False, "sonic 백엔드를 쓸 수 없음 (체크포인트 또는 sonic 백엔드 없음)"
                self.d["backend"] = self.cfg["backend"] = v
                args["backend"] = v
                self.ensure.pop("backend", None)
                self.set_argv("sim")
            elif k == "walk":
                self.d["walk"] = self.cfg["walk"] = bool(v)
                args["walk"] = bool(v)
                self.ensure.pop("walk", None)
            elif k == "loco_profile":
                if v not in P.LOCO_PROFILES:
                    return False, f"보행 모드 {v!r} 없음 ({' / '.join(P.LOCO_PROFILES)})"
                self.d["loco_profile"] = self.cfg["loco_profile"] = v
                args["loco_profile"] = v
            elif k == "crop":
                box = norm_crop(v)
                if box is None and v not in (None, "", []):
                    return False, "영역은 [x0, y0, x1, y1] (0~1, 가로·세로 10 % 이상)"
                self.d["crop"] = self.cfg["crop"] = box
                args["crop"] = box
            elif k == "max_speed":
                v = max(0.05, min(float(self.cfg["speed_limit"]), float(v)))
                self.d["max_speed"] = self.cfg["max_speed"] = round(v, 2)
                args["max_speed"] = self.d["max_speed"]
            elif k == "mirror":
                self.d["mirror"] = self.cfg["mirror"] = bool(v)
                args["mirror"] = bool(v)
                self.set_argv("sender")
            elif k in ("source", "camera"):
                src = f"camera:{int(v)}" if k == "camera" or str(v).isdigit() else str(v)
                if src.startswith("camera:"):
                    self.cfg["camera"] = int(src.split(":")[1])
                elif not Path(src).exists():
                    return False, f"영상 파일 없음: {src}"
                self.d["source"] = self.cfg["source"] = src
                args.update(self._source_args())
                self.set_argv("sender")
            elif k == "fallback":
                if v not in ("auto", "on", "off"):
                    return False, "fallback = auto | on | off"
                self.cfg["fallback"] = v
            elif k == "idle_after_s":
                self.cfg["idle_after_s"] = max(5.0, float(v))
                args["idle_after"] = self.cfg["idle_after_s"]
                self.set_argv("sim")
            elif k in ("auto_calibrate", "auto_start"):
                self.cfg[k] = bool(v)
            elif k == "perception":
                if v not in PERCEPTIONS:
                    return False, f"인식 {v!r} 없음 ({' / '.join(PERCEPTIONS)})"
                self.d["perception"] = self.cfg["perception"] = v
                if v != self.perc_active or self.perc_note:
                    self.perc_active, self.perc_note = v, ""
                    self.set_argv("sender")
                    self.restart_proc("sender", f"인식 {v}")
            elif k == "arm_path":
                if v not in ARM_PATHS:
                    return False, f"팔 경로 {v!r} 없음 ({' / '.join(ARM_PATHS)})"
                if v != self.cfg.get("arm_path"):
                    self.cfg["arm_path"] = v
                    for name in ("sender", "sim"):   # IK (sender) and arm control (sim_server) are start-up options
                        self.set_argv(name)
                        self.restart_proc(name, f"팔 경로 {v}")
        if args:
            self.send(P.CMD_SET, args)
        save_settings(self.cfg)
        self.event("set", **a)
        return True, "ok"

    # ---------------------------------------------------------------- periodic logic
    def sim_fresh(self, now):
        return self.sim is not None and now - self.sim_t < self.FRESH_S and self.sim.get("note") != "loading"

    def snd_fresh(self, now):
        return self.snd is not None and now - self.snd_t < self.FRESH_S and self.snd.get("note") != "loading"

    def present(self, now):
        if self.snd_fresh(now):
            return bool(self.snd.get("present"))
        return bool(self.sim_fresh(now) and self.sim.get("human_present"))

    def backends_available(self):
        if self.sim is not None and self.sim.get("backends"):
            return list(self.sim["backends"])
        return ["split"] + (["sonic"] if sonic_available_locally(self.cfg) else []) + \
            (["hybrid"] if hybrid_available_locally() else [])

    def _ensure(self, key, mismatch, fix, now, retry_s=2.0, tries=3, give_up_s=20.0, on_give_up=None):
        e = self.ensure.get(key)
        if not mismatch:
            self.ensure.pop(key, None)
            return
        if e is None:
            e = self.ensure[key] = [0, now, now]  # first mismatch: give the process one retry interval
            return
        if e[0] < tries and now - e[1] >= retry_s:
            e[0] += 1
            e[1] = now
            fix()
            self.event("resend", key=key, attempt=e[0])
        elif e[0] >= tries and now - e[2] > give_up_s and on_give_up is not None:
            on_give_up()
            self.ensure.pop(key, None)

    def tick(self, now=None):
        now = time.time() if now is None else now
        if self.gemx_failed is not None:   # lock order controller -> supervisor, like every other restart
            why, self.gemx_failed = self.gemx_failed, None
            self.on_gemx_unavailable(why)
        with self.lock:
            while self.lat and now - self.lat[0][0] > 30.0:
                self.lat.popleft()
            sim_ok = self.sim_fresh(now)
            s = self.sim
            if sim_ok:
                eff = self.effective_mode()

                def give_up_backend():
                    self.event("backend_failed", wanted=self.d["backend"], got=s["backend"])
                    self.d["backend"] = self.cfg["backend"] = s["backend"]
                    self.set_argv("sim")

                self._ensure("backend", s["backend"] != self.d["backend"],
                             lambda: self.send(P.CMD_SET, {"backend": self.d["backend"]}), now,
                             retry_s=5.0, give_up_s=30.0, on_give_up=give_up_backend)
                self._ensure("walk", bool(s["walk_enabled"]) != bool(self.d["walk"]),
                             lambda: self.send(P.CMD_SET, {"walk": self.d["walk"]}), now)
                self._ensure("mode", s["mode"] not in (eff, "gesture"),
                             lambda: self.send(P.CMD_SET, {"mode": self.effective_mode()}), now)
                self._ensure("estop", self.d["estop"] and not s["estop"], lambda: self.send(P.CMD_ESTOP), now)
                self._ensure("streaming", (not self.d["estop"]) and bool(s["streaming"]) != bool(self.d["running"]),
                             lambda: self.send(P.CMD_START if self.d["running"] else P.CMD_STOP), now)
                # fall watchdog (the sim resets itself within 3 s; this is the backstop)
                if s.get("fallen"):
                    self.fallen_since = self.fallen_since or now
                    if (now - self.fallen_since > self.cfg["fall_reset_s"]
                            and now - self.last_svc_reset > self.cfg["fall_reset_s"]):
                        self.last_svc_reset = now
                        self.counters["svc_resets"] += 1
                        self.send(P.CMD_RESET)
                        self.event("watchdog_reset", fallen_s=round(now - self.fallen_since, 1))
                else:
                    self.fallen_since = None
                # gesture playlist
                if self.playlist is not None and self.d["mode"] == "gesture":
                    if s.get("gesture"):
                        self.pl_gesture_seen = True
                    elif (self.pl_gesture_seen and now - self.pl_sent_t > 0.5) or now - self.pl_sent_t > 30.0:
                        self.playlist += 1
                        self._play_next(now)
            self._idle_logic(now)
            self._calib_logic(now)
            self._camera_watchdog(now)
            self._view_logic(now, sim_ok)

    def _idle_logic(self, now):
        pres = self.present(now)
        if pres:
            self.absent_since = None
            self.present_since = self.present_since or now
        else:
            self.present_since = None
            self.absent_since = self.absent_since or now
        if (not self.auto_idle and self.d["running"] and self.d["mode"] == "live" and self.absent_since
                and now - self.absent_since >= float(self.cfg["idle_after_s"])):
            self.auto_idle = True
            self.d["running"] = False
            self.calib = None                  # the next visitor needs their own calibration
            self.counters["auto_idle"] += 1
            self.send(P.CMD_STOP)
            self.send(P.CMD_SET, {"mode": "idle"})
            self.event("auto_idle", absent_s=round(now - self.absent_since, 1))
        if self.auto_idle and self.present_since and now - self.present_since >= 2.0:
            if self.cfg["auto_calibrate"] and self.calib is None:
                self.calib = {"t": now, "seen_false": False, "state": "running"}
                self.send(P.CMD_CALIBRATE)
                self.event("auto_calibrate")
            calibrated_now = (self.calib is not None and self.calib["state"] == "done") or (
                not self.cfg["auto_calibrate"] and self.snd is not None and self.snd.get("calibrated"))
            if self.cfg["auto_start"] and calibrated_now:
                self.event("auto_start")
                self.command("start", {})

    def _calib_logic(self, now):
        c = self.calib
        if c is None or c["state"] != "running" or self.snd is None:
            if c is not None and c["state"] != "running" and now - c["t"] > 8.0 and not self.auto_idle:
                self.calib = None
            return
        if not self.snd.get("calibrated") and self.snd_t > c["t"]:
            c["seen_false"] = True
        if self.snd.get("calibrated") and self.snd_t > c["t"] and (c["seen_false"] or now - c["t"] > 1.5):
            c["state"] = "done"
            self.event("calibrated", after_s=round(now - c["t"], 2))
        elif now - c["t"] > 6.0:
            c["state"] = "failed"
            self.event("calibrate_failed")

    def _camera_watchdog(self, now):
        if not self.d["source"].startswith("camera:") or not self.snd_fresh(now):
            self.cam_bad_since = None
            return
        if self.snd.get("cam_ok"):
            self.cam_bad_since = None
            return
        self.cam_bad_since = self.cam_bad_since or now
        if (now - self.cam_bad_since > float(self.cfg["cam_restart_s"])
                and now - self.last_cam_restart > 3 * float(self.cfg["cam_restart_s"])):
            self.last_cam_restart = now
            self.counters["cam_restarts"] += 1
            self.restart_proc("sender", "카메라 프레임 없음")
            self.event("camera_restart", bad_s=round(now - self.cam_bad_since, 1))

    def health(self, now):
        cam_age, robot_age = self.frame_age("cam"), self.frame_age("robot")
        cam_ok = (self.proc_up("sender") and self.snd_fresh(now) and bool(self.snd.get("cam_ok"))
                  and cam_age is not None and cam_age < self.FRESH_S)
        sim_ok = (self.proc_up("sim") and self.sim_fresh(now)
                  and robot_age is not None and robot_age < self.FRESH_S)
        reasons = []
        loading = self.snd is not None and now - self.snd_t < self.FRESH_S and self.snd.get("note") == "loading"
        if not cam_ok:
            reasons.append("카메라 신호 없음" if self.snd_fresh(now) else
                           ("GEM-X 인식 모델 불러오는 중" if loading and self.perc_active == "gemx" else
                            "인식 준비 중" if loading else "카메라 프로세스 응답 없음"))
        if not sim_ok:
            reasons.append("시뮬레이션 화면 없음" if self.sim_fresh(now) else "시뮬레이션 응답 없음")
        return cam_ok, sim_ok, " · ".join(reasons)

    def _view_logic(self, now, _sim_ok):
        cam_ok, sim_ok, why = self.health(now)
        good = cam_ok and sim_ok
        if good:
            self.bad_since = None
            self.good_since = self.good_since or now
        else:
            self.good_since = None
            self.bad_since = self.bad_since or now
        mode = self.cfg["fallback"]
        if mode == "on":
            new, reason = "fallback", "운영자 설정"
        elif mode == "off":
            new, reason = "live", "운영자 설정 (대체 영상 끔)"
        elif self.view == "live" and not good and now - self.bad_since >= float(self.cfg["fallback_after_s"]):
            new, reason = "fallback", why
        elif self.view == "fallback" and good and now - self.good_since >= float(self.cfg["live_after_s"]):
            new, reason = "live", ""
        else:
            new, reason = self.view, (why if self.view == "fallback" and not good else self.view_reason)
        if new != self.view:
            self.counters["fallback_switches"] += new == "fallback"
            self.event("view", view=new, reason=reason)
        self.view, self.view_reason = new, reason

    # ---------------------------------------------------------------- status
    def audience_message(self, now):
        """Short line on the camera side of the display (or "")."""
        if self.calib is not None and self.calib["state"] == "running":
            return self.cfg["msg_calibrating"]
        if self.presenter_mode():  # the camera watches a presenter: no "please stand in front" bubbles
            return ""
        if self.auto_idle and self.present_since and now - self.present_since >= 1.0:
            return self.cfg["msg_welcome"]
        if self.d["running"] and self.d["mode"] == "live" and self.absent_since and now - self.absent_since > 2.0:
            return self.cfg["msg_come_closer"]
        return ""

    def presenter_mode(self):
        return self.d["walk"] and self.d["loco_profile"] == "presenter"

    def guide(self):
        if not self.d["walk"]:
            return self.cfg["guide_no_walk"]
        return self.cfg["guide_presenter"] if self.d["loco_profile"] == "presenter" else self.cfg["guide"]

    def snapshot(self, now=None):
        now = time.time() if now is None else now
        with self.lock:
            lat = [v for _, v in self.lat]
            sim = dict(self.sim) if self.sim else None
            snd = dict(self.snd) if self.snd else None
            cam_ok, sim_ok, why = self.health(now)
            walking = bool(sim and any(abs(float(v)) > 0.05 for v in (sim.get("loco_cmd") or [0, 0, 0])[:2]))
            return {
                "t": now, "uptime_s": round(now - self.t_start, 1),
                "view": self.view, "view_reason": self.view_reason, "health": why,
                "cam_ok": cam_ok, "sim_ok": sim_ok,
                "desired": dict(self.d), "effective_mode": self.effective_mode(), "auto_idle": self.auto_idle,
                "present": self.present(now),
                "absent_s": round(now - self.absent_since, 1) if self.absent_since else None,
                "attention": bool(self.auto_idle and self.present_since and now - self.present_since >= 1.0),
                "calib": self.calib["state"] if self.calib else None,
                "playlist": self.playlist is not None,
                "sim": sim, "sim_age_s": round(now - self.sim_t, 2) if self.sim else None,
                "snd": snd, "snd_age_s": round(now - self.snd_t, 2) if self.snd else None,
                "lat_ms": {"p50": pct(lat, 0.5), "p95": pct(lat, 0.95), "n": len(lat)},
                "walking": walking,
                "backends": self.backends_available(),
                "counters": dict(self.counters),
                "audience_msg": self.audience_message(now),
                "settings": {k: self.cfg[k] for k in PERSIST + ("turn_speed", "speed_limit")},
                "arm": arm_settings(self.cfg),
                "perception": {"want": self.d["perception"], "active": self.perc_active, "note": self.perc_note},
                "title": self.cfg["title"],
                "subtitle": self.cfg["subtitle_sonic"] if (sim or {}).get("backend") in ("sonic", "sonic_human") else "",
                "guide": self.guide(),
            }


# ---------------------------------------------------------------------------------------------------------
class MjpegRelay:
    """One upstream MJPEG connection (sender /cam.mjpg or sim /robot.mjpg); keeps the latest JPEG for clients."""

    def __init__(self, name, port, path):
        self.name, self.port, self.path = name, port, path
        self.cond = threading.Condition()
        self.jpg = None
        self.seq = 0
        self.t_frame = None
        self.fps = 0.0
        self.connected = False
        self._stop = threading.Event()
        threading.Thread(target=self._run, name=f"relay-{name}", daemon=True).start()

    def age(self):
        return None if self.t_frame is None else time.time() - self.t_frame

    def _put(self, jpg):
        now = time.time()
        with self.cond:
            if self.t_frame is not None:
                self.fps = 0.9 * self.fps + 0.1 / max(now - self.t_frame, 1e-3)
            self.jpg, self.t_frame = jpg, now
            self.seq += 1
            self.cond.notify_all()

    def _run(self):
        while not self._stop.is_set():
            conn = None
            try:
                conn = http.client.HTTPConnection(P.HOST, self.port, timeout=3.0)
                conn.request("GET", self.path)
                resp = conn.getresponse()
                if resp.status != 200:
                    raise OSError(f"HTTP {resp.status}")
                self.connected = True
                self._read_parts(resp)
            except (OSError, http.client.HTTPException, ValueError):
                pass
            finally:
                self.connected = False
                if conn is not None:
                    conn.close()
            self._stop.wait(0.5)

    def _read_parts(self, resp):
        while not self._stop.is_set():
            line = resp.readline(4096)
            if not line:
                return
            if not line.startswith(b"--"):
                continue
            n = 0
            while True:
                h = resp.readline(4096)
                if not h:
                    return
                h = h.strip()
                if not h:
                    break
                k, _, v = h.partition(b":")
                if k.strip().lower() == b"content-length":
                    n = int(v.strip())
            if n > 0:
                jpg = resp.read(n)
                if len(jpg) == n:
                    self._put(jpg)
            else:  # no Content-Length: read up to the JPEG end marker
                buf = b""
                while b"\xff\xd9" not in buf and len(buf) < 8_000_000:
                    chunk = resp.read1(65536)
                    if not chunk:
                        return
                    buf += chunk
                s, e = buf.find(b"\xff\xd8"), buf.find(b"\xff\xd9")
                if 0 <= s < e:
                    self._put(buf[s:e + 2])

    def wait(self, last_seq, timeout):
        with self.cond:
            self.cond.wait_for(lambda: self.seq != last_seq, timeout=timeout)
            return self.jpg, self.seq

    def stop(self):
        self._stop.set()


# ---------------------------------------------------------------------------------------------------------
class Bus(threading.Thread):
    """Owns every ZMQ socket: CTRL publisher, SIM_STATE / SENDER_STATE subscribers; runs the controller tick."""

    def __init__(self, ctl: Controller, beat, on_msg=None):
        super().__init__(name="bus", daemon=True)
        self.ctl = ctl
        self.beat = beat
        self.on_msg = on_msg or (lambda m: None)
        self.q = queue.Queue()
        self.ctx = zmq.Context.instance()
        self.pub = self.ctx.socket(zmq.PUB)
        self.pub.setsockopt(zmq.LINGER, 0)
        self.pub.bind(f"tcp://{P.HOST}:{P.CTRL_PORT}")
        self.subs = {}
        for name, port, topic in (("sim", P.SIM_STATE_PORT, P.TOPIC_SIM_STATE),
                                  ("sender", P.SENDER_STATE_PORT, P.TOPIC_SENDER_STATE)):
            s = self.ctx.socket(zmq.SUB)
            s.setsockopt(zmq.LINGER, 0)
            s.connect(f"tcp://{P.HOST}:{port}")
            s.setsockopt(zmq.SUBSCRIBE, topic)
            self.subs[name] = s
        self._stop = threading.Event()
        self.n_ctrl = 0

    def send(self, cmd, args):
        self.q.put(P.CtrlMsg(cmd=cmd, args=dict(args)))

    def run(self):
        poller = zmq.Poller()
        for s in self.subs.values():
            poller.register(s, zmq.POLLIN)
        t_tick = 0.0
        while not self._stop.is_set():
            try:
                ready = dict(poller.poll(50))
            except zmq.ZMQError:
                break
            now = time.time()
            for name, s in self.subs.items():
                if s in ready:
                    while True:
                        try:
                            frames = s.recv_multipart(zmq.NOBLOCK)
                        except zmq.Again:
                            break
                        try:
                            msg = P.decode(frames)
                        except (ValueError, IndexError):
                            continue
                        self.beat(name, now)
                        (self.ctl.on_sim if name == "sim" else self.ctl.on_snd)(msg, now)
            if now - t_tick >= 0.1:
                t_tick = now
                try:
                    self.ctl.tick(now)
                except Exception as e:  # keep the bus alive; the error goes to the service log
                    print(f"[service] tick error: {e!r}", flush=True)
            while True:
                try:
                    m = self.q.get_nowait()
                except queue.Empty:
                    break
                self.pub.send_multipart(P.encode(P.TOPIC_CTRL, m))
                self.n_ctrl += 1
                self.on_msg(m)
        for s in list(self.subs.values()) + [self.pub]:
            s.close(0)

    def stop(self):
        self._stop.set()


# ---------------------------------------------------------------------------------------------------------
class Service:
    def __init__(self, cfg, session_dir: Path, mock=False, mock_sender_args="", mock_sim_args="", host=P.HOST,
                 port=P.UI_HTTP_PORT, start_procs=True):
        self.cfg = cfg
        self.dir = Path(session_dir).resolve()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.mock = mock
        self.host, self.port = host, port
        self.events = deque(maxlen=300)
        self._ev_lock = threading.Lock()
        self._ev_file = open(self.dir / "events.jsonl", "a", buffering=1, encoding="utf-8")
        self.t0 = time.time()
        self.stopping = threading.Event()
        snd_t = cfg["mock_sender_cmd"] if mock else cfg["sender_cmd"]
        sim_t = cfg["mock_sim_cmd"] if mock else cfg["sim_cmd"]
        self.templates = {"sender": list(snd_t) + mock_sender_args.split(),
                          "sender_gemx": list(cfg["mock_sender_cmd"] if mock else cfg["sender_gemx_cmd"])
                          + mock_sender_args.split(),
                          "sim": list(sim_t) + mock_sim_args.split()}
        self.relays = {"cam": MjpegRelay("cam", P.CAM_HTTP_PORT, "/cam.mjpg"),
                       "robot": MjpegRelay("robot", P.ROBOT_HTTP_PORT, "/robot.mjpg")}
        gemx_missing = gemx_env_missing(cfg) if cfg.get("perception") == "gemx" and not mock else ""
        first = {"sender": "sender_gemx" if cfg.get("perception") == "gemx" and not gemx_missing else "sender",
                 "sim": "sim"}
        specs = [ProcSpec(name, build_argv(self.templates[first[name]], cfg, self.dir), self.dir / f"{log}.log",
                          cwd=str(REPO),
                          env=child_env(cfg), grace_s=float(cfg["grace_s"]), stale_s=float(cfg["stale_s"]))
                 for name, log in (("sender", "sender"), ("sim", "sim_server"))]
        self.ctl = Controller(cfg, send=self._send, event=self.event,
                              restart_proc=lambda n, r: self.sup.restart(n, r),
                              set_argv=lambda n: self.sup.set_argv(n, self.argv(n)),
                              frame_age=lambda n: self.relays[n].age(),
                              proc_up=lambda n: self.sup.procs[n].state == "running")
        self.sup = Supervisor(specs, on_event=self._proc_event)
        if gemx_missing:
            self.ctl.on_gemx_unavailable(gemx_missing)   # starts on MediaPipe; choosing GEM-X in the panel retries it
        self.bus = Bus(self.ctl, beat=self.sup.beat)
        self.start_procs = start_procs
        self._mem_restart_t = {}
        self.metrics_path = self.dir / "metrics.csv"
        (self.dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1))

    # ------------------------------------------------------------------------------------------------
    def event(self, kind, **d):
        rec = {"t": round(time.time(), 3), "kind": kind, **d}
        with self._ev_lock:
            self.events.append(rec)
            try:
                self._ev_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except ValueError:
                pass
        if kind not in ("resend",):
            print(f"[service] {kind} {json.dumps(d, ensure_ascii=False)}", flush=True)

    def argv(self, name):
        """Command line of the next (re)start of a child (the sender's depends on the active perception)."""
        key = name
        if name == "sender" and self.ctl.sender_template() == "sender_gemx_cmd":
            key = "sender_gemx"
        return build_argv(self.templates[key], self.cfg, self.dir)

    def _proc_event(self, name, ev, d):
        self.ctl.on_proc_event(name, ev, d)
        self.event(f"proc_{ev}", proc=name, **d)
        if name == "sender" and ev == "fail" and self.ctl.perc_active == "gemx":
            p = self.sup.procs.get("sender")
            if p is not None and p.last_exit == EXIT_GEMX:
                why = next((ln.split("unavailable:", 1)[1].strip() for ln in reversed(self.log_tail("sender", 60))
                            if "GEM-X unavailable:" in ln), "")
                self.ctl.gemx_failed = why or "GEM-X 시작 실패"   # handled on the bus thread (Controller.tick)

    def _send(self, cmd, args):
        self.bus.send(cmd, args)

    def start(self):
        self.bus.start()
        if self.start_procs:
            self.sup.start()
        threading.Thread(target=self._metrics_loop, name="metrics", daemon=True).start()
        self.httpd = ThreadingHTTPServer((self.host, self.port), make_handler(self))
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, name="http", daemon=True).start()
        self.event("service_start", port=self.port, mock=self.mock, dir=str(self.dir), sender=self.argv("sender"),
                   sim=build_argv(self.templates["sim"], self.cfg, self.dir))

    def shutdown(self):
        if self.stopping.is_set():
            return
        self.stopping.set()
        self.event("service_stop")
        self.sup.stop_watching()
        try:
            self.bus.send(P.CMD_QUIT, {})
            time.sleep(0.5)
        except Exception:
            pass
        self.sup.shutdown()
        for r in self.relays.values():
            r.stop()
        self.bus.stop()
        try:
            self.httpd.shutdown()
        except Exception:
            pass
        self._ev_file.close()

    def _metrics_loop(self):
        cols = ["t", "uptime_s", "view", "present", "streaming", "mode", "ctrl_hz", "rt_factor", "lat_p50",
                "lat_p95", "human_age_ms", "arm_err_deg", "fallen", "resets", "falls", "svc_resets",
                "fallback_switches", "auto_idle", "sender_restarts", "sim_restarts", "sender_rss_mb", "sim_rss_mb",
                "service_rss_mb", "cam_fps", "robot_fps", "loco_vx"]
        new = not self.metrics_path.exists()
        with open(self.metrics_path, "a", buffering=1) as f:
            if new:
                f.write(",".join(cols) + "\n")
            while not self.stopping.wait(5.0):
                s = self.ctl.snapshot()
                sim = s["sim"] or {}
                ps = self.sup.status()
                self._memory_guard(ps)
                me = rss_mb([os.getpid()]).get(os.getpid())
                row = [round(s["t"], 1), s["uptime_s"], s["view"], int(s["present"]), int(bool(sim.get("streaming"))),
                       sim.get("mode", ""), sim.get("ctrl_hz", ""), sim.get("rt_factor", ""), s["lat_ms"]["p50"],
                       s["lat_ms"]["p95"], sim.get("human_age_ms", ""), sim.get("arm_err_deg", ""),
                       int(bool(sim.get("fallen"))), sim.get("resets", ""), s["counters"]["falls"],
                       s["counters"]["svc_resets"], s["counters"]["fallback_switches"], s["counters"]["auto_idle"],
                       ps["sender"]["restarts"], ps["sim"]["restarts"], ps["sender"]["rss_mb"], ps["sim"]["rss_mb"],
                       round(me, 1) if me else "", round(self.relays["cam"].fps, 1),
                       round(self.relays["robot"].fps, 1), (sim.get("loco_cmd") or [""])[0]]
                f.write(",".join("" if v is None else str(v) for v in row) + "\n")

    def _memory_guard(self, ps):
        """Safety net for a long event: a child that grew past its limit is restarted, but only when the robot is
        not following anyone (stopped / idle), so no visitor sees it. At most once per 10 minutes per child."""
        limits = self.cfg.get("mem_limit_mb") or {}
        now = time.time()
        for name, st in ps.items():
            key = "sender_gemx" if (name == "sender" and self.ctl.perc_active == "gemx") else name
            lim, rss = limits.get(key, limits.get(name)), st.get("rss_mb")
            if not lim or not rss or rss <= lim or st["state"] != "running":
                continue
            if self.ctl.d["running"] and not self.ctl.auto_idle:
                continue
            if now - self._mem_restart_t.get(name, 0.0) < 600:
                continue
            self._mem_restart_t[name] = now
            self.event("memory_restart", proc=name, rss_mb=rss, limit_mb=lim)
            self.sup.restart(name, f"메모리 {rss:.0f} MB > {lim} MB")

    # ------------------------------------------------------------------------------------------------
    def state(self, with_events=True):
        s = self.ctl.snapshot()
        s["procs"] = self.sup.status()
        s["relay"] = {n: {"age_s": None if r.age() is None else round(r.age(), 2), "fps": round(r.fps, 1),
                          "connected": r.connected} for n, r in self.relays.items()}
        s["session"] = str(self.dir)
        s["mock"] = self.mock
        if with_events:
            with self._ev_lock:
                s["events"] = list(self.events)[-40:]
        return s

    def page_config(self):
        c = self.cfg
        return {
            "title": c["title"], "subtitle_sonic": c["subtitle_sonic"], "guide": c["guide"],
            "guide_no_walk": c["guide_no_walk"], "guide_presenter": c["guide_presenter"],
            "label_cam": c["label_cam"], "label_robot": c["label_robot"],
            "label_fallback": c["label_fallback"],
            "fallback": [{"i": i, "name": Path(p).name, "exists": Path(p).exists()}
                         for i, p in enumerate(c["fallback_videos"])],
            "gestures": [{"name": g, "label": dict(GESTURES).get(g, g)} for g in c["gestures"]],
            "cameras": c["cameras"],
            "videos": [{"path": p, "name": Path(p).name} for p in list_videos(c["rehearsal_videos"])],
            "display_cam_flip": c["display_cam_flip"], "cam_fit": c["cam_fit"], "robot_fit": c["robot_fit"],
            "speed_limit": c["speed_limit"], "turn_speed": c["turn_speed"],
        }

    def handle_cmd(self, body):
        cmd = body.get("cmd")
        if cmd == "restart":
            name = body.get("proc")
            if name not in self.sup.procs:
                return False, f"프로세스 {name!r} 없음"
            self.sup.restart(name, "운영자")
            return True, "ok"
        if cmd == "shutdown":
            threading.Thread(target=lambda: (time.sleep(0.3), os.kill(os.getpid(), signal.SIGTERM)),
                             daemon=True).start()
            return True, "종료합니다"
        return self.ctl.command(cmd, body)

    def log_tail(self, name, n=40):
        p = self.dir / {"sender": "sender.log", "sim": "sim_server.log", "service": "service.log"}.get(name, "")
        try:
            with open(p, "rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 16384))
                return f.read().decode("utf-8", "replace").splitlines()[-n:]
        except OSError:
            return []


# ---------------------------------------------------------------------------------------------------------
STATIC_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml", ".png": "image/png",
                ".json": "application/json", ".ico": "image/x-icon"}


def make_handler(svc: Service):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "K1Live/1.0"

        def log_message(self, fmt, *a):
            pass

        def _bytes(self, code, data, ctype, extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(data)

        def _json(self, obj, code=200):
            self._bytes(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

        def _static(self, name):
            p = (WEB / name).resolve()
            if WEB.resolve() not in p.parents or not p.is_file():
                self._bytes(404, b"not found", "text/plain")
                return
            self._bytes(200, p.read_bytes(), STATIC_TYPES.get(p.suffix, "application/octet-stream"))

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            u = urlparse(self.path)
            path, qs = u.path, parse_qs(u.query)
            try:
                if path in ("/", "/panel"):
                    return self._static("panel.html")
                if path == "/display":
                    return self._static("display.html")
                if path.startswith("/static/"):
                    return self._static(path[len("/static/"):])
                if path == "/favicon.ico":
                    return self._static("favicon.svg")
                if path == "/api/health":
                    return self._json({"ok": True, "uptime_s": round(time.time() - svc.t0, 1)})
                if path == "/api/state":
                    return self._json(svc.state())
                if path == "/api/config":
                    return self._json(svc.page_config())
                if path == "/api/log":
                    return self._json({"lines": svc.log_tail(qs.get("proc", ["sim"])[0])})
                if path == "/api/events":
                    return self._sse()
                if path in ("/cam.mjpg", "/robot.mjpg"):
                    return self._mjpeg(svc.relays[path[1:].split(".")[0]], float(qs.get("fps", ["30"])[0]))
                if path in ("/cam.jpg", "/robot.jpg"):
                    r = svc.relays[path[1:].split(".")[0]]
                    if r.jpg is None:
                        return self._bytes(503, b"no frame", "text/plain")
                    return self._bytes(200, r.jpg, "image/jpeg")
                m = re.fullmatch(r"/fallback/(\d+)(?:\.mp4)?", path)
                if m:
                    return self._video(int(m.group(1)))
                self._bytes(404, b"not found", "text/plain")
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            u = urlparse(self.path)
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}") if n else {}
                if not isinstance(body, dict):
                    raise ValueError("JSON object expected")
            except ValueError as e:
                return self._json({"ok": False, "msg": f"잘못된 요청: {e}"}, 400)
            if u.path == "/api/cmd":
                ok, msg = svc.handle_cmd(body)
                return self._json({"ok": ok, "msg": msg}, 200 if ok else 400)
            if u.path == "/api/set":
                ok, msg = svc.ctl.command("set", body)
                return self._json({"ok": ok, "msg": msg}, 200 if ok else 400)
            self._json({"ok": False, "msg": "not found"}, 404)

        def _sse(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            try:
                self.wfile.write(b"retry: 1000\n\n")
                n = 0
                while not svc.stopping.is_set():
                    st = svc.state(with_events=(n % 5 == 0))
                    self.wfile.write(b"data: " + json.dumps(st, ensure_ascii=False).encode() + b"\n\n")
                    self.wfile.flush()
                    n += 1
                    time.sleep(0.2)
            except (BrokenPipeError, ConnectionResetError, OSError):
                return

        def _mjpeg(self, relay: MjpegRelay, fps):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            min_dt = 1.0 / max(1.0, min(60.0, fps))
            last, t_last = -1, 0.0
            try:
                while not svc.stopping.is_set():
                    jpg, seq = relay.wait(last, 1.0)
                    if jpg is None or seq == last:
                        continue
                    now = time.time()
                    if now - t_last < min_dt:
                        time.sleep(min_dt - (now - t_last))
                        jpg, seq = relay.jpg, relay.seq
                    last, t_last = seq, time.time()
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError, OSError):
                return

        def _video(self, i):
            vids = svc.cfg["fallback_videos"]
            if not 0 <= i < len(vids) or not Path(vids[i]).is_file():
                return self._bytes(404, b"no such video", "text/plain")
            path = Path(vids[i])
            size = path.stat().st_size
            start, end = 0, size - 1
            rng = self.headers.get("Range")
            m = re.fullmatch(r"bytes=(\d*)-(\d*)", rng.strip()) if rng else None
            if m and (m.group(1) or m.group(2)):
                if m.group(1):
                    start = int(m.group(1))
                    end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
                else:
                    start = max(0, size - int(m.group(2)))
                if start > end or start >= size:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
            n = end - start + 1
            self.send_response(206 if m else 200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(n))
            self.send_header("Cache-Control", "max-age=3600")
            if m:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if self.command == "HEAD":
                return
            with open(path, "rb") as f:
                f.seek(start)
                left = n
                while left > 0:
                    chunk = f.read(min(262144, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)

    return Handler


# ---------------------------------------------------------------------------------------------------------
PORTS = (P.UI_HTTP_PORT, P.CAM_HTTP_PORT, P.ROBOT_HTTP_PORT, P.HUMAN_PORT, P.CTRL_PORT, P.SIM_STATE_PORT,
         P.SENDER_STATE_PORT)


def port_holders(port):
    """[(pid, command line)] listening on 127.0.0.1/any:port (lsof)."""
    try:
        out = subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"], capture_output=True,
                             text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    res = []
    for pid in {int(x) for x in out.split() if x.isdigit()}:
        cmd = subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
        res.append((pid, cmd))
    return res


def clear_stale_children(ports=PORTS[1:]):
    """Kill leftover k1live child processes (a previous service that died hard) that still hold our ports."""
    killed = []
    for port in ports:
        for pid, cmd in port_holders(port):
            if pid != os.getpid() and re.search(r"k1live\.(sender|sim_server|mock)", cmd):
                try:
                    os.kill(pid, signal.SIGKILL)
                    killed.append((port, pid, cmd))
                except (ProcessLookupError, PermissionError):
                    pass
    if killed:
        time.sleep(0.5)
    return killed


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
                st.flush()
            except (ValueError, OSError):
                pass
        return len(s)

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except (ValueError, OSError):
                pass


def main(argv=None):
    ap = argparse.ArgumentParser(description="K1 live demo service (MuJoCo simulation only)")
    ap.add_argument("--config", default=None, help="event settings YAML (default k1live/web/event.yaml)")
    ap.add_argument("--no-saved-settings", action="store_true", help="start from the selected profile, ignoring saved panel settings")
    ap.add_argument("--source", default=None, help="camera index or a video file (rehearsal); default: last camera")
    ap.add_argument("--backend", default=None, choices=P.BACKENDS)
    ap.add_argument("--loco-profile", default=None, choices=P.LOCO_PROFILES,
                    help="walking mode for this run (not saved; default: the panel's last choice / event.yaml)")
    ap.add_argument("--crop", default=None, help="person region x0,y0,x1,y1 for this run (not saved)")
    ap.add_argument("--mock", action="store_true", help="use k1live.mock processes instead of sender/sim_server")
    ap.add_argument("--mock-sender-args", default="")
    ap.add_argument("--mock-sim-args", default="")
    ap.add_argument("--host", default=P.HOST, help="HTTP bind address (default 127.0.0.1)")
    ap.add_argument("--port", type=int, default=P.UI_HTTP_PORT)
    ap.add_argument("--fallback", default=None, choices=["auto", "on", "off"])
    ap.add_argument("--idle-after", type=float, default=None)
    ap.add_argument("--start", action="store_true", help="start following right away (rehearsal / soak)")
    ap.add_argument("--auto", action="store_true", help="auto calibrate + auto start when someone shows up (soak)")
    ap.add_argument("--max-seconds", type=float, default=0.0, help="exit after this long (tests)")
    ap.add_argument("--session-dir", default=None)
    args = ap.parse_args(argv)

    src = args.source
    if src is not None:
        src = f"camera:{src}" if src.isdigit() else str(resolve(src))
    cfg = load_config(args.config, {"source": src, "backend": args.backend, "fallback": args.fallback,
                                    "loco_profile": args.loco_profile, "crop": args.crop,
                                    "idle_after_s": args.idle_after, "auto_calibrate": True if args.auto else None,
                                    "auto_start": True if args.auto else None}, read_saved=not args.no_saved_settings)
    sess = Path(args.session_dir) if args.session_dir else LOG_ROOT / time.strftime("%Y%m%d-%H%M%S")
    sess.mkdir(parents=True, exist_ok=True)
    log = open(sess / "service.log", "a", buffering=1, encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log)
    sys.stderr = Tee(sys.__stderr__, log)

    if port_holders(args.port):
        print(f"[service] 포트 {args.port} 이미 사용 중 — 서비스가 이미 실행 중인지 확인하세요 "
              f"(http://127.0.0.1:{args.port}/)", flush=True)
        return 2
    for port, pid, cmd in clear_stale_children():
        print(f"[service] 남아 있던 프로세스 종료: port {port} pid {pid} {cmd[:120]}", flush=True)

    svc = Service(cfg, sess, mock=args.mock, mock_sender_args=args.mock_sender_args,
                  mock_sim_args=args.mock_sim_args, host=args.host, port=args.port)
    done = threading.Event()

    def on_signal(signum, frame):
        done.set()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGHUP, on_signal)
    svc.start()
    print(f"[service] 운영 패널  http://127.0.0.1:{args.port}/", flush=True)
    print(f"[service] 관객 화면  http://127.0.0.1:{args.port}/display", flush=True)
    print(f"[service] 로그       {sess}", flush=True)
    if args.start:
        def auto_start():
            t_end = time.time() + 120
            while time.time() < t_end and not done.is_set():
                if svc.ctl.sim_fresh(time.time()):
                    svc.ctl.command("start", {})
                    return
                time.sleep(0.5)
        threading.Thread(target=auto_start, daemon=True).start()
    t_end = time.time() + args.max_seconds if args.max_seconds else None
    while not done.wait(0.5):
        if t_end and time.time() > t_end:
            break
    svc.shutdown()
    print("[service] 종료", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
