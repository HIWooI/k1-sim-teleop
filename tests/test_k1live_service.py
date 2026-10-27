"""K1 live service: controller decisions, supervisor, MJPEG relay, HTTP API (no robot, no camera, no MuJoCo).

    .venv/bin/python -m pytest tests/test_k1live_service.py -q

The controller tests drive time explicitly. The supervisor tests start tiny Python children. The HTTP/relay tests
bind free local ports (the ZMQ/HTTP port constants are patched), so they can run next to a live service.
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from k1live import protocol as P  # noqa: E402
from k1live import service as S  # noqa: E402
from k1live.supervisor import ProcSpec, Supervisor  # noqa: E402


# ---------------------------------------------------------------------------------------------------------
def free_port():
    s = socket.socket()
    s.bind((P.HOST, 0))
    p = s.getsockname()[1]
    s.close()
    return p


def sim_state(**kw):
    d = asdict(P.SimState(t=0.0, mode="live", backend="split", streaming=False, walk_enabled=True, estop=False,
                          fallen=False, resets=0, pelvis_z=0.78, tilt_deg=1.0, loco_cmd=[0, 0, 0],
                          human_present=True, human_age_ms=40.0, ctrl_hz=50.0, rt_factor=1.0, backends=["split"]))
    d.update(kw)
    return d


def snd_state(**kw):
    d = asdict(P.SenderState(t=0.0, cam_ok=True, source="camera:0", fps=30.0, present=True, calibrated=True,
                             pose_ms=20.0))
    d.update(kw)
    return d


class Rig:
    """Controller with recorded CTRL output and controllable frame ages."""

    def __init__(self, tmp_path, monkeypatch, **cfg_over):
        for env in ("K1LIVE_LOCO_PROFILE", "K1LIVE_CROP"):    # the defaults, not this shell's choice
            monkeypatch.delenv(env, raising=False)
        monkeypatch.setattr(S, "SETTINGS_PATH", tmp_path / "settings.json")
        monkeypatch.setattr(S, "LOG_ROOT", tmp_path)
        cfg = S.load_config(path=tmp_path / "none.yaml")
        cfg.update(cfg_over)
        cfg["source"] = cfg_over.get("source", "camera:0")
        self.sent, self.events, self.restarts = [], [], []
        self.ages = {"cam": 0.05, "robot": 0.05}
        self.ctl = S.Controller(cfg, send=lambda c, a: self.sent.append((c, dict(a))),
                                event=lambda k, **d: self.events.append((k, d)),
                                restart_proc=lambda n, r: self.restarts.append((n, r)),
                                frame_age=lambda n: self.ages[n])
        self.t = time.time()  # operator commands stamp wall-clock time; simulated time continues from it

    def feed(self, seconds, sim=None, snd=None, dt=0.1):
        """Advance time, delivering the given states every dt and ticking the controller."""
        end = self.t + seconds
        while self.t < end - 1e-9:
            self.t += dt
            if sim is not None:
                self.ctl.on_sim(sim_state(**sim), self.t)
            if snd is not None:
                self.ctl.on_snd(snd_state(**snd), self.t)
            self.ctl.tick(self.t)

    def cmds(self, name=None):
        return [c for c in self.sent if name is None or c[0] == name]

    def clear(self):
        self.sent.clear()
        self.events.clear()


@pytest.fixture
def rig(tmp_path, monkeypatch):
    return Rig(tmp_path, monkeypatch)


# ---------------------------------------------------------------------------------------------------------
def test_replay_full_settings_once_ready(rig):
    rig.ctl.on_sim(sim_state(note="loading"), rig.t)
    assert rig.sent == []                                  # still loading: wait
    rig.ctl.on_sim(sim_state(), rig.t + 0.1)
    sets = rig.cmds("set")
    assert len(sets) == 1
    a = sets[0][1]
    assert a["mode"] == "live" and a["backend"] == "split" and a["camera"] == 0 and "mirror" in a and "walk" in a
    assert rig.cmds("stop")                                # not running yet -> keep the sim stopped
    rig.clear()
    rig.ctl.on_sim(sim_state(), rig.t + 0.2)
    assert rig.sent == []                                  # no repeat while connected
    rig.ctl.on_sim(sim_state(), rig.t + 2.0)              # 1.8 s gap = reconnect
    assert rig.cmds("set")


def test_replay_after_process_restart_resumes_streaming(rig):
    rig.feed(0.5, sim={}, snd={})
    rig.ctl.command("start", {})
    rig.clear()
    rig.ctl.on_proc_event("sim", "start", {})
    rig.ctl.on_sim(sim_state(), rig.t + 0.1)
    assert rig.cmds("set") and rig.cmds("start")


def test_start_reconcile_retries_then_stops(rig):
    rig.feed(0.5, sim={}, snd={})
    rig.ctl.command("start", {})
    assert ("set", {"mode": "live"}) in rig.sent and rig.cmds("start")
    rig.clear()
    rig.feed(20.0, sim={"streaming": False}, snd={})     # the sim never confirms
    assert 1 <= len(rig.cmds("start")) <= 3
    rig.clear()
    rig.feed(5.0, sim={"streaming": True}, snd={})
    assert rig.cmds("start") == []


def test_auto_idle_then_operator_start(rig):
    rig.ctl.cfg["idle_after_s"] = 5.0
    rig.feed(0.5, sim={}, snd={})
    rig.ctl.command("start", {})
    rig.clear()
    rig.feed(4.0, sim={"streaming": True}, snd={"present": False})
    assert not rig.ctl.auto_idle
    rig.feed(1.5, sim={"streaming": True}, snd={"present": False})
    assert rig.ctl.auto_idle and not rig.ctl.d["running"]
    assert rig.cmds("stop") and ("set", {"mode": "idle"}) in rig.sent
    assert rig.ctl.snapshot(rig.t)["effective_mode"] == "idle"
    # somebody walks in: attention + welcome text, but nothing starts by itself
    rig.clear()
    rig.feed(3.0, sim={"mode": "idle"}, snd={"present": True})
    snap = rig.ctl.snapshot(rig.t)
    assert snap["attention"] and snap["audience_msg"] == rig.ctl.cfg["msg_welcome"]
    assert rig.cmds("start") == [] and rig.cmds("calibrate") == []
    rig.ctl.command("start", {})
    assert not rig.ctl.auto_idle and rig.ctl.d["running"] and rig.ctl.d["mode"] == "live"


def test_auto_calibrate_and_auto_start(rig):
    rig.ctl.cfg.update(idle_after_s=5.0, auto_calibrate=True, auto_start=True)
    rig.feed(0.5, sim={}, snd={})
    rig.ctl.command("calibrate", {})                         # the previous visitor's calibration ...
    rig.feed(2.0, sim={}, snd={"calibrated": True})
    assert rig.ctl.calib["state"] == "done"
    rig.ctl.command("start", {})
    rig.feed(6.0, sim={"streaming": True}, snd={"present": False})
    assert rig.ctl.auto_idle and rig.ctl.calib is None      # ... does not count for the next one
    rig.clear()
    rig.feed(2.5, sim={"mode": "idle"}, snd={"present": True, "calibrated": False})
    assert len(rig.cmds("calibrate")) == 1 and rig.cmds("start") == []
    rig.feed(1.0, sim={"mode": "idle"}, snd={"present": True, "calibrated": True})
    assert rig.cmds("start") and rig.ctl.d["running"] and not rig.ctl.auto_idle
    assert any(k == "auto_start" for k, _ in rig.events)


def test_fallback_hysteresis(rig):
    assert rig.ctl.view == "fallback"                      # nothing seen yet
    rig.feed(2.5, sim={}, snd={})
    assert rig.ctl.view == "fallback"                      # good for < live_after_s
    rig.feed(1.0, sim={}, snd={})
    assert rig.ctl.view == "live"
    rig.ages["cam"] = None                                 # camera stream stops
    rig.feed(1.5, sim={}, snd={})
    assert rig.ctl.view == "live"                          # short glitch tolerated
    rig.feed(1.0, sim={}, snd={})
    assert rig.ctl.view == "fallback" and "카메라" in rig.ctl.view_reason
    rig.ages["cam"] = 0.05
    rig.feed(1.0, sim={}, snd={"cam_ok": False})
    assert rig.ctl.view == "fallback"
    rig.feed(3.5, sim={}, snd={})
    assert rig.ctl.view == "live"
    assert rig.ctl.counters["fallback_switches"] == 1
    rig.feed(4.5, snd={})                                  # sim silent 2 s (stale) + 2 s -> fallback
    assert rig.ctl.view == "fallback" and "시뮬레이션" in rig.ctl.view_reason
    rig.ctl.command("set", {"fallback": "off"})
    rig.ctl.tick(rig.t)
    assert rig.ctl.view == "live"
    rig.ctl.command("set", {"fallback": "on"})
    rig.feed(0.2, sim={}, snd={})
    assert rig.ctl.view == "fallback"


def test_fall_watchdog_sends_one_reset(rig):
    rig.feed(0.5, sim={}, snd={})
    rig.clear()
    rig.feed(5.0, sim={"fallen": True}, snd={})
    assert rig.cmds("reset") == []                         # the sim resets itself first
    rig.feed(2.0, sim={"fallen": True}, snd={})
    assert len(rig.cmds("reset")) == 1
    assert rig.ctl.counters["falls"] == 1 and rig.ctl.counters["svc_resets"] == 1


def test_drive_is_clamped(rig):
    rig.ctl.command("set", {"max_speed": 0.4})
    rig.clear()
    rig.ctl.command("drive", {"vx": 2.0, "vy": -2.0, "wz": 5.0})
    (c, a), = rig.sent
    assert c == "drive" and a == {"vx": 0.4, "vy": -0.4, "wz": rig.ctl.cfg["turn_speed"]}
    rig.ctl.command("set", {"max_speed": 9})
    assert rig.ctl.d["max_speed"] == rig.ctl.cfg["speed_limit"]


def test_backend_needs_availability(rig, monkeypatch):
    monkeypatch.setattr(S, "sonic_available_locally", lambda cfg: False)
    ok, msg = rig.ctl.command("set", {"backend": "sonic"})
    assert not ok and "sonic" in msg
    rig.feed(0.3, sim={"backends": ["split", "sonic"]}, snd={})
    rig.clear()
    ok, _ = rig.ctl.command("set", {"backend": "sonic"})
    assert ok and ("set", {"backend": "sonic"}) in rig.sent
    # the sim never switches: resend a few times, then give up and show what really runs
    rig.feed(45.0, sim={"backends": ["split", "sonic"], "backend": "split"}, snd={})
    assert rig.ctl.d["backend"] == "split"
    assert any(k == "backend_failed" for k, _ in rig.events)
    assert 2 <= sum(1 for c, a in rig.sent if c == "set" and a.get("backend") == "sonic") <= 4  # 1 + 3 retries


def test_source_validation(rig, tmp_path):
    ok, msg = rig.ctl.command("set", {"source": str(tmp_path / "missing.mp4")})
    assert not ok
    vid = tmp_path / "clip.mp4"
    vid.write_bytes(b"x")
    rig.clear()
    assert rig.ctl.command("set", {"source": str(vid)})[0]
    assert ("set", {"video": str(vid)}) in rig.sent
    rig.clear()
    assert rig.ctl.command("set", {"source": "camera:1"})[0]
    assert ("set", {"camera": 1}) in rig.sent and rig.ctl.cfg["camera"] == 1
    saved = json.loads(S.SETTINGS_PATH.read_text())
    assert saved["camera"] == 1 and "source" not in saved   # a rehearsal video never becomes the event default


def test_estop_and_resume(rig):
    rig.feed(0.5, sim={}, snd={})
    rig.ctl.command("start", {})
    rig.ctl.command("estop", {})
    assert rig.ctl.d["estop"] and not rig.ctl.d["running"] and rig.cmds("estop")
    rig.clear()
    rig.feed(5.0, sim={"estop": False, "streaming": False}, snd={})
    assert rig.cmds("estop")                                # the sim lost it (restart): estop again
    rig.clear()
    rig.ctl.command("start", {})
    assert not rig.ctl.d["estop"] and rig.cmds("start")


def test_gesture_playlist_advances(rig):
    rig.feed(0.5, sim={}, snd={})
    rig.clear()
    rig.ctl.command("playlist", {"on": True})
    g = rig.ctl.cfg["gestures"]
    assert ("gesture", {"name": g[0]}) in rig.sent and rig.ctl.d["mode"] == "gesture"
    assert ("set", {"mode": "idle"}) in rig.sent            # gestures play on top of the sim's idle mode
    rig.feed(2.0, sim={"mode": "gesture", "gesture": g[0]}, snd={})
    rig.feed(1.0, sim={"mode": "gesture", "gesture": None}, snd={})
    assert ("gesture", {"name": g[1]}) in rig.sent
    ok, _ = rig.ctl.command("gesture", {"name": "nope"})
    assert not ok


def test_camera_watchdog_restarts_sender(rig):
    rig.ctl.cfg["cam_restart_s"] = 5.0
    rig.feed(6.0, sim={}, snd={"cam_ok": False})
    assert rig.restarts == [("sender", "카메라 프레임 없음")]
    rig.feed(6.0, sim={}, snd={"cam_ok": False})
    assert len(rig.restarts) == 1                           # rate-limited


def test_build_argv_and_event_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setenv("K1LIVE_EVENT_DIR", str(tmp_path / "ev"))
    cfg = S.load_config(path=tmp_path / "none.yaml")
    assert cfg["fallback_videos"][0].startswith(str(tmp_path / "ev"))
    assert cfg["source"] == "camera:0"
    cfg.update(source="camera:2", mirror=False)
    argv = S.build_argv(cfg["sender_cmd"], cfg)
    assert argv[1:] == ["-m", "k1live.sender", "--source", "2", "--no-mirror", "--arm-ik=dir"]
    cfg.update(source="/x/y.mp4", mirror=True)
    assert S.build_argv(cfg["sender_cmd"], cfg)[1:] == ["-m", "k1live.sender", "--source", "/x/y.mp4", "--loop",
                                                        "--arm-ik=dir"]
    assert S.build_argv(cfg["sim_cmd"], cfg, "/s")[1:] == ["-m", "k1live.sim_server", "--backend", "split",
                                                          "--no-autostart", "--idle-after", "20", "--log-dir", "/s/sim",
                                                          "--arm-mode=pd", "--arm-vmax=12", "--arm-amax=150"]


# ---------------------------------------------------------------------------------------------------------
def _wait(cond, timeout=10.0, dt=0.05):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(dt)
    return False


def test_supervisor_restarts_crashing_child(tmp_path):
    spec = ProcSpec("c", [sys.executable, "-c", "import sys; sys.exit(3)"], tmp_path / "c.log", grace_s=5,
                    backoff=(0.2, 0.4))
    ev = []
    sup = Supervisor([spec], on_event=lambda n, e, d: ev.append(e), poll_s=0.05)
    sup.start()
    try:
        assert _wait(lambda: sup.procs["c"].restarts >= 3)
        st = sup.status()["c"]
        assert st["last_exit"] == 3 and "exit 3" in st["last_reason"]
    finally:
        sup.shutdown()
    assert ev.count("fail") >= 3 and "start" in ev


def test_supervisor_stale_and_grace(tmp_path):
    spec = ProcSpec("h", [sys.executable, "-c", "import time; time.sleep(60)"], tmp_path / "h.log", grace_s=0.5,
                    stale_s=0.4, backoff=(0.1,))
    sup = Supervisor([spec], poll_s=0.05)
    sup.start()
    try:
        assert _wait(lambda: sup.procs["h"].restarts >= 1)        # never reported within the grace period
        assert "상태 없음" in sup.procs["h"].last_reason
        assert _wait(lambda: sup.procs["h"].state == "running")
        sup.beat("h")
        assert sup.status()["h"]["healthy"]
        n = sup.procs["h"].restarts
        assert _wait(lambda: sup.procs["h"].restarts > n)         # beats stop -> stale restart
        assert "끊김" in sup.procs["h"].last_reason
    finally:
        sup.shutdown()


def test_supervisor_replaces_frozen_child_quickly(tmp_path):
    spec = ProcSpec("z", [sys.executable, "-c", "import time; time.sleep(60)"], tmp_path / "z.log", stale_s=0.5,
                    backoff=(0.1,))
    sup = Supervisor([spec], poll_s=0.05)
    sup.start()
    try:
        assert _wait(lambda: sup.procs["z"].pid is not None)
        sup.beat("z")
        pid = sup.procs["z"].pid
        os.kill(pid, signal.SIGSTOP)                               # frozen: ignores SIGTERM until resumed
        t0 = time.time()
        assert _wait(lambda: sup.procs["z"].restarts >= 1, 5.0)
        assert time.time() - t0 < 0.5 + 1.0 + 0.6                  # stale + short kill wait, not the 3 s timeout
    finally:
        sup.shutdown()


def test_supervisor_kills_whole_group(tmp_path):
    marker = tmp_path / "grandchild.pid"
    code = ("import subprocess, sys, time\n"
            "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
            f"open({str(marker)!r}, 'w').write(str(p.pid))\n"
            "time.sleep(60)\n")
    sup = Supervisor([ProcSpec("g", [sys.executable, "-c", code], tmp_path / "g.log")], poll_s=0.05)
    sup.start()
    assert _wait(lambda: marker.exists() and marker.read_text())
    gpid = int(marker.read_text())
    sup.shutdown()

    def gone():
        try:
            os.kill(gpid, 0)
        except ProcessLookupError:
            return True
        # a zombie of our own test process does not count
        st = subprocess.run(["ps", "-o", "stat=", "-p", str(gpid)], capture_output=True, text=True).stdout
        return st.strip().startswith("Z") or not st.strip()

    assert _wait(gone, 5.0)


def test_supervisor_cleans_up_after_a_dead_leader(tmp_path):
    marker = tmp_path / "orphan.pid"
    code = ("import subprocess, sys, time\n"
            "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
            f"open({str(marker)!r}, 'w').write(str(p.pid))\n"
            "time.sleep(0.5)\n"
            "sys.exit(1)\n")                                       # leader crashes, its child would linger
    sup = Supervisor([ProcSpec("o", [sys.executable, "-c", code], tmp_path / "o.log", backoff=(30.0,))], poll_s=0.05)
    sup.start()
    try:
        assert _wait(lambda: sup.procs["o"].restarts >= 1)
        gpid = int(marker.read_text())

        def gone():
            try:
                os.kill(gpid, 0)
            except ProcessLookupError:
                return True
            st = subprocess.run(["ps", "-o", "stat=", "-p", str(gpid)], capture_output=True, text=True).stdout
            return st.strip().startswith("Z") or not st.strip()

        assert _wait(gone, 5.0)
    finally:
        sup.shutdown()


# ---------------------------------------------------------------------------------------------------------
def test_mjpeg_relay_roundtrip():
    from k1live.mock import MjpegServer
    port = free_port()
    srv = MjpegServer(port, "cam")
    relay = S.MjpegRelay("cam", port, "/cam.mjpg")
    try:
        img = np.zeros((48, 64, 3), np.uint8)
        for i in range(30):
            img[:] = i * 5
            srv.push(img)
            time.sleep(0.03)
            if relay.seq > 3:
                break
        assert _wait(lambda: relay.seq > 3)
        assert relay.jpg[:2] == b"\xff\xd8" and relay.jpg[-2:] == b"\xff\xd9"
        assert relay.age() < 1.0
    finally:
        relay.stop()
        srv.httpd.shutdown()


@pytest.fixture
def service(tmp_path, monkeypatch):
    """Service without child processes on private ports; tests publish fake states into it."""
    ports = {k: free_port() for k in ("UI_HTTP_PORT", "CAM_HTTP_PORT", "ROBOT_HTTP_PORT", "HUMAN_PORT",
                                       "CTRL_PORT", "SIM_STATE_PORT", "SENDER_STATE_PORT")}
    for k, v in ports.items():
        monkeypatch.setattr(P, k, v)
    monkeypatch.setattr(S, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(S, "LOG_ROOT", tmp_path)
    vid = tmp_path / "fb.mp4"
    vid.write_bytes(bytes(range(256)) * 40)
    cfg = S.load_config(path=tmp_path / "none.yaml")
    cfg["fallback_videos"] = [str(vid)]
    svc = S.Service(cfg, tmp_path / "sess", port=ports["UI_HTTP_PORT"], start_procs=False)
    svc.start()
    yield svc, ports
    svc.shutdown()


def _get(port, path, headers=None):
    req = urllib.request.Request(f"http://{P.HOST}:{port}{path}", headers=headers or {})
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status, dict(r.headers), r.read()


def _post(port, path, body):
    req = urllib.request.Request(f"http://{P.HOST}:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_http_pages_api_and_ctrl(service):
    import zmq
    svc, ports = service
    ui = ports["UI_HTTP_PORT"]
    for path in ("/", "/display", "/static/panel.js", "/static/display.js", "/static/panel.css",
                 "/static/display.css", "/favicon.ico"):
        code, _, body = _get(ui, path)
        assert code == 200 and body, path
    with pytest.raises(urllib.error.HTTPError):
        _get(ui, "/static/../service.py")
    cfgj = json.loads(_get(ui, "/api/config")[2])
    assert cfgj["gestures"] and cfgj["fallback"][0]["exists"]
    # a fake sim publishes state; a fake subscriber listens to CTRL
    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://{P.HOST}:{ports['SIM_STATE_PORT']}")
    sub = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://{P.HOST}:{ports['CTRL_PORT']}")
    sub.setsockopt(zmq.SUBSCRIBE, P.TOPIC_CTRL)
    got = []

    def pump(seconds):
        end = time.time() + seconds
        while time.time() < end:
            pub.send_multipart(P.encode(P.TOPIC_SIM_STATE, P.SimState(**{**sim_state(), "t": time.time()})))
            while True:
                try:
                    got.append(P.decode(sub.recv_multipart(zmq.NOBLOCK)))
                except zmq.Again:
                    break
            time.sleep(0.05)

    pump(1.0)
    assert any(m["cmd"] == "set" for m in got)              # re-sync on first state
    code, r = _post(ui, "/api/cmd", {"cmd": "start"})
    assert code == 200 and r["ok"]
    code, r = _post(ui, "/api/cmd", {"cmd": "drive", "vx": 3, "vy": 0, "wz": 0})
    pump(0.5)
    assert any(m["cmd"] == "start" for m in got)
    drv = [m for m in got if m["cmd"] == "drive"]
    assert drv and drv[0]["args"]["vx"] == svc.cfg["max_speed"]
    code, r = _post(ui, "/api/cmd", {"cmd": "bogus"})
    assert code == 400 and not r["ok"]
    code, r = _post(ui, "/api/set", {"walk": False})
    assert code == 200 and svc.ctl.d["walk"] is False
    st = json.loads(_get(ui, "/api/state")[2])
    assert st["sim"]["backend"] == "split" and st["desired"]["running"] and st["procs"]["sim"]["state"] == "stopped"
    # fallback video with ranges
    code, h, body = _get(ui, "/fallback/0.mp4", {"Range": "bytes=10-19"})
    assert code == 206 and body == bytes(range(10, 20)) and h["Content-Range"].endswith(f"/{256 * 40}")
    code, h, body = _get(ui, "/fallback/0.mp4")
    assert code == 200 and len(body) == 256 * 40
    # SSE delivers JSON snapshots
    with urllib.request.urlopen(f"http://{P.HOST}:{ui}/api/events", timeout=5) as r:
        first = b""
        while b"data: " not in first:
            first += r.readline()
        line = r.readline()
        assert json.loads(line.split(b"data: ", 1)[1])["view"] in ("live", "fallback") if line.startswith(
            b"data: ") else True
    pub.close(0)
    sub.close(0)
    lines = (svc.dir / "events.jsonl").read_text().splitlines()
    assert any(json.loads(x)["kind"] == "cmd" for x in lines)


def test_preflight_runs_without_camera(capsys):
    from k1live import preflight
    rc = preflight.main(["--no-camera", "--quick", "--mock"])
    out = capsys.readouterr().out
    assert "사전 점검" in out and "카메라" in out and "건너뜀" in out
    assert rc in (0, 1)


def test_cli_overrides_are_not_persisted(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(S, "LOG_ROOT", tmp_path)
    S.SETTINGS_PATH.write_text(json.dumps({"auto_start": False, "max_speed": 0.4}))
    cfg = S.load_config(path=tmp_path / "none.yaml", cli={"auto_start": True, "auto_calibrate": True,
                                                          "fallback": None})
    assert cfg["auto_start"] is True and cfg["max_speed"] == 0.4
    cfg["max_speed"] = 0.6                                   # an operator change is saved ...
    S.save_settings(cfg)
    saved = json.loads(S.SETTINGS_PATH.read_text())
    assert saved["max_speed"] == 0.6
    assert saved["auto_start"] is False and "auto_calibrate" not in saved  # ... the soak's --auto is not


def test_memory_guard_waits_until_nobody_is_followed(tmp_path, monkeypatch):
    for k in ("CTRL_PORT", "SIM_STATE_PORT", "SENDER_STATE_PORT", "CAM_HTTP_PORT", "ROBOT_HTTP_PORT"):
        monkeypatch.setattr(P, k, free_port())                    # the constructor binds CTRL
    monkeypatch.setattr(S, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(S, "LOG_ROOT", tmp_path)
    cfg = S.load_config(path=tmp_path / "none.yaml")
    svc = S.Service(cfg, tmp_path / "sess", port=free_port(), start_procs=False)
    restarted = []
    svc.sup.restart = lambda name, reason: restarted.append((name, reason))
    big = {"sender": {"state": "running", "rss_mb": 2000.0}, "sim": {"state": "running", "rss_mb": 900.0}}
    svc.ctl.d["running"] = True
    svc._memory_guard(big)
    assert restarted == []                                      # following a visitor: wait
    svc.ctl.d["running"] = False
    svc._memory_guard(big)
    svc._memory_guard(big)
    assert [n for n, _ in restarted] == ["sender"]              # once, and only the big one
    svc._ev_file.close()
    for r in svc.relays.values():
        r.stop()


# --------------------------------------------------------------------------------------------- presenter mode
def test_walk_mode_and_region_set_persist_and_replay(rig, tmp_path, monkeypatch):
    rig.feed(0.3, sim={}, snd={})
    rig.clear()
    ok, _ = rig.ctl.command("set", {"walk": True, "loco_profile": "presenter", "crop": [0.2, 0.0, 0.8, 0.85]})
    assert ok
    a = rig.cmds("set")[-1][1]
    assert a == {"walk": True, "loco_profile": "presenter", "crop": [0.2, 0.0, 0.8, 0.85]}
    assert rig.ctl.d["loco_profile"] == "presenter" and rig.ctl.d["crop"] == [0.2, 0.0, 0.8, 0.85]
    saved = json.loads(S.SETTINGS_PATH.read_text())
    assert saved["loco_profile"] == "presenter" and saved["crop"] == [0.2, 0.0, 0.8, 0.85]
    # bad values are refused and change nothing
    for bad in ({"loco_profile": "dancer"}, {"crop": [0.5, 0, 0.55, 1]}, {"crop": [0.9, 0, 0.1, 1]},
                {"crop": "x"}):
        ok, msg = rig.ctl.command("set", bad)
        assert not ok and msg, bad
    assert rig.ctl.d["loco_profile"] == "presenter" and rig.ctl.d["crop"] == [0.2, 0.0, 0.8, 0.85]
    # a restarted sender gets them back with the full settings
    rig.clear()
    rig.ctl.on_proc_event("sender", "start", {})
    rig.ctl.on_snd(snd_state(), rig.t + 0.1)
    full = rig.cmds("set")[-1][1]
    assert full["loco_profile"] == "presenter" and full["crop"] == [0.2, 0.0, 0.8, 0.85]
    # the next service start restores them
    cfg = S.load_config(path=tmp_path / "none.yaml")
    assert cfg["loco_profile"] == "presenter" and cfg["crop"] == [0.2, 0.0, 0.8, 0.85]
    # clearing the region sends None (whole frame) and is saved
    ok, _ = rig.ctl.command("set", {"crop": None})
    assert ok and rig.cmds("set")[-1][1] == {"crop": None}
    assert json.loads(S.SETTINGS_PATH.read_text())["crop"] is None


def test_presenter_mode_texts(rig):
    rig.feed(0.3, sim={}, snd={})
    c = rig.ctl.cfg
    assert rig.ctl.snapshot(rig.t)["guide"] == c["guide"]               # visitor
    rig.ctl.command("set", {"loco_profile": "presenter"})
    assert rig.ctl.snapshot(rig.t)["guide"] == c["guide_presenter"]
    rig.ctl.command("set", {"walk": False})
    assert rig.ctl.snapshot(rig.t)["guide"] == c["guide_no_walk"]
    # visitor mode greets someone who shows up while idle; presenter mode never puts that bubble over a presenter
    rig.ctl.command("set", {"walk": True, "loco_profile": "visitor"})
    rig.ctl.auto_idle, rig.ctl.present_since = True, rig.t - 2.0
    assert rig.ctl.audience_message(rig.t) == c["msg_welcome"]
    rig.ctl.command("set", {"loco_profile": "presenter"})
    assert rig.ctl.audience_message(rig.t) == ""


def test_profile_from_env_and_cli_is_used_not_saved(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(S, "LOG_ROOT", tmp_path)
    monkeypatch.setenv("K1LIVE_LOCO_PROFILE", "presenter")
    monkeypatch.setenv("K1LIVE_CROP", "0.1,0.0,0.9,0.8")
    cfg = S.load_config(path=tmp_path / "none.yaml")
    assert cfg["loco_profile"] == "presenter" and cfg["crop"] == [0.1, 0.0, 0.9, 0.8]
    S.save_settings(cfg)
    saved = json.loads(S.SETTINGS_PATH.read_text())
    assert "loco_profile" not in saved and "crop" not in saved
    monkeypatch.setenv("K1LIVE_LOCO_PROFILE", "dancer")                 # a typo falls back, never stops the show
    monkeypatch.setenv("K1LIVE_CROP", "nonsense")
    cfg = S.load_config(path=tmp_path / "none.yaml", cli={"loco_profile": None})
    assert cfg["loco_profile"] == "visitor" and cfg["crop"] is None
    cfg = S.load_config(path=tmp_path / "none.yaml", cli={"loco_profile": "presenter", "crop": "0.2,0,0.8,0.9"})
    assert cfg["loco_profile"] == "presenter" and cfg["crop"] == [0.2, 0.0, 0.8, 0.9]


def test_event_yaml_has_walking_mode_keys():
    import yaml

    cfg = yaml.safe_load((S.WEB / "event.yaml").read_text())
    assert cfg["loco_profile"] in P.LOCO_PROFILES and "crop" in cfg and cfg["guide_presenter"]


def test_http_set_walk_mode_reaches_ctrl(service):
    import zmq
    svc, ports = service
    ui = ports["UI_HTTP_PORT"]
    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://{P.HOST}:{ports['CTRL_PORT']}")
    sub.setsockopt(zmq.SUBSCRIBE, P.TOPIC_CTRL)
    time.sleep(0.3)                                                      # slow joiner
    code, r = _post(ui, "/api/set", {"walk": True, "loco_profile": "presenter", "crop": [0.25, 0.05, 0.75, 0.9]})
    assert code == 200 and r["ok"]
    got = []
    end = time.time() + 2.0
    while time.time() < end and not got:
        try:
            got.append(P.decode(sub.recv_multipart(zmq.NOBLOCK)))
        except zmq.Again:
            time.sleep(0.05)
    assert got and got[0]["cmd"] == "set"
    assert got[0]["args"]["loco_profile"] == "presenter" and got[0]["args"]["crop"] == [0.25, 0.05, 0.75, 0.9]
    st = json.loads(_get(ui, "/api/state")[2])
    assert st["desired"]["loco_profile"] == "presenter" and st["settings"]["crop"] == [0.25, 0.05, 0.75, 0.9]
    assert st["guide"] == svc.cfg["guide_presenter"]
    assert "guide_presenter" in json.loads(_get(ui, "/api/config")[2])
    code, r = _post(ui, "/api/set", {"loco_profile": "nope"})
    assert code == 400 and not r["ok"]
    for path in ("/", "/static/panel.js", "/static/panel.css"):
        body = _get(ui, path)[2].decode()
        assert ("cropbox" in body or "crop" in body), path
    sub.close(0)
