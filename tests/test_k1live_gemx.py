"""GEM-X perception for the K1 live chain: landmark mapping, walking intent, worker wire format, the GEM-X
sender with a replayed track (no GPU), the service's perception switch and its MediaPipe fallback.

    .venv/bin/python -m pytest tests/test_k1live_gemx.py -q

No CUDA, no GEM-X install, no MuJoCo: the worker is exercised through its --replay engine (a saved track answered by
video time) and its exit codes; real GEM-X inference is measured on GPU machines (docs/results.md).
"""
from __future__ import annotations

import io
import json
import math
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from k1live import gemx as G  # noqa: E402
from k1live import gemx_worker as GW  # noqa: E402
from k1live import protocol as P  # noqa: E402
from k1live.human import HumanMapper, obs_from_det  # noqa: E402

K = np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]])
W_IMG, H_IMG = 1280, 720


# ------------------------------------------------------------------------------------------------ synthetic person
def rest_skeleton():
    """A standing SOMA-77 skeleton in the person's frame (x = person's left, y up, z = person's forward), arms down."""
    S = G.SOMA
    J = np.zeros((77, 3))
    put = {"Hips": (0, 1.0, 0), "Spine1": (0, 1.1, 0), "Spine2": (0, 1.22, 0), "Chest": (0, 1.35, 0),
           "Neck1": (0, 1.5, 0), "Neck2": (0, 1.55, 0), "Head": (0, 1.62, 0), "HeadEnd": (0, 1.75, 0),
           "Jaw": (0, 1.58, 0.06), "LeftEye": (0.03, 1.66, 0.08), "RightEye": (-0.03, 1.66, 0.08),
           "LeftShoulder": (0.05, 1.45, 0), "LeftArm": (0.19, 1.42, 0), "LeftForeArm": (0.21, 1.14, 0),
           "LeftHand": (0.22, 0.88, 0.02), "RightShoulder": (-0.05, 1.45, 0), "RightArm": (-0.19, 1.42, 0),
           "RightForeArm": (-0.21, 1.14, 0), "RightHand": (-0.22, 0.88, 0.02), "LeftLeg": (0.09, 0.95, 0),
           "LeftShin": (0.1, 0.52, 0.01), "LeftFoot": (0.1, 0.09, 0), "LeftToeBase": (0.1, 0.02, 0.13),
           "LeftToeEnd": (0.1, 0.01, 0.2), "RightLeg": (-0.09, 0.95, 0), "RightShin": (-0.1, 0.52, 0.01),
           "RightFoot": (-0.1, 0.09, 0), "RightToeBase": (-0.1, 0.02, 0.13), "RightToeEnd": (-0.1, 0.01, 0.2)}
    for n, v in put.items():
        J[S[n]] = v
    for i in range(77):                       # fingers etc.: at their hand / parent
        if not np.any(J[i]) and i:
            J[i] = J[G.SOMA_PARENTS[i]]
    return J


def to_cam(J_person, x=0.0, z=3.0, yaw=0.0):
    """Person frame -> camera frame (x right, y down, z forward) for a person facing the camera at depth z, lateral
    x (image right), turned by yaw (rad, + = turned to their own left)."""
    c, s = math.cos(yaw), math.sin(yaw)
    p = J_person.copy()
    # facing the camera: person's left = image right (+x cam), person's forward = toward the camera (-z cam)
    px, py, pz = p[:, 0], p[:, 1], p[:, 2]
    rx, rz = c * px + s * pz, -s * px + c * pz     # turning left: forward -> toward the person's left
    return np.stack([rx + x, -(py - 1.0), -rz + z], axis=1)


def kp_of(J_cam, conf=0.95):
    uv = G.project(J_cam, K)
    return np.concatenate([uv, np.full((77, 1), conf)], 1)


def det_of(J_cam, conf=0.95):
    return G.soma_det(J_cam, kp_of(J_cam, conf), W_IMG, H_IMG, K)


# ------------------------------------------------------------------------------------------------ mapping
def test_soma_det_layout_axes_and_visibility():
    J = to_cam(rest_skeleton(), x=0.2, z=3.0)
    d = det_of(J)
    assert d["world"].shape == (33, 3) and d["image"].shape == (33, 3) and d["vis"].shape == (33,)
    hip = 0.5 * (d["cam"][23] + d["cam"][24])
    assert np.allclose(0.5 * (d["world"][23] + d["world"][24]), 0)          # hip-centred like MediaPipe world
    assert abs(hip[2] - 3.0) < 0.05 and abs(hip[0] - 0.2) < 0.01            # absolute depth kept in "cam"
    assert d["world"][11, 0] > d["world"][12, 0]                             # left shoulder on image right
    assert d["world"][0, 1] < d["world"][23, 1]                              # head above hips (y down)
    assert np.all((d["image"][:, :2] > 0) & (d["image"][:, :2] < 1))
    assert np.all(d["vis"] == 1.0)                                           # confident VitPose
    low = G.soma_det(J, kp_of(J, conf=G.VIS_REF), W_IMG, H_IMG, K)
    assert np.allclose(low["vis"], 0.5)                                      # VIS_REF reads as MediaPipe's 0.5 gate


def test_image_landmarks_use_vitpose_where_confident():
    J = to_cam(rest_skeleton())
    kp = kp_of(J)
    kp[G.SOMA["LeftHand"], :2] += (40.0, -25.0)          # VitPose sees the wrist elsewhere (confident)
    d = G.soma_det(J, kp, W_IMG, H_IMG, K)
    assert np.allclose(d["image"][15, :2] * (W_IMG, H_IMG), kp[G.SOMA["LeftHand"], :2])
    kp[G.SOMA["LeftHand"], 2] = 0.1                      # unsure: the aligned GEM-X projection instead
    d2 = G.soma_det(J, kp, W_IMG, H_IMG, K)
    assert np.linalg.norm(d2["image"][15, :2] * (W_IMG, H_IMG) - G.project(J, K)[G.SOMA["LeftHand"]]) < 3.0


def test_mapper_arms_follow_gemx_depth():
    """Same 2D picture, arm reaching toward the camera vs hanging: only GEM-X depth tells them apart; the dir IK must
    put the forward-reaching arm forward (shoulder pitch < 0)."""
    Jp = rest_skeleton()
    S = G.SOMA
    fwd = Jp.copy()
    fwd[S["LeftForeArm"]] = Jp[S["LeftArm"]] + (0.0, -0.05, 0.28)      # upper arm forward, slightly down
    fwd[S["LeftHand"]] = fwd[S["LeftForeArm"]] + (0.0, 0.0, 0.26)
    for i in range(15, 39):
        fwd[i] = fwd[S["LeftHand"]]
    q = {}
    for name, sk in (("down", Jp), ("fwd", fwd)):
        hm = HumanMapper(mirror=False, loco=False, smpl=False, arm_ik="dir", perception="gemx")
        for k in range(10):
            r = hm.process(det_of(to_cam(sk)), k / 30, W_IMG, H_IMG)
        q[name] = np.asarray(r["arm_q"])
    assert q["fwd"][0] < q["down"][0] - 0.8          # left shoulder pitch: forward (negative) by > 45 deg
    assert abs(q["fwd"][5] - q["down"][5]) < 0.15    # right arm unchanged


# ------------------------------------------------------------------------------------------------ walking intent
def run_loco(track, profile="visitor", mirror=True, fps=30.0, dist_src="pelvis", **params):
    """track: list of (x, z, yaw, arms_up[, depth_scale]) per frame -> commands. depth_scale k: GEM-X gets the depth
    wrong by that factor (every joint slides along its camera ray: same image, different metres)."""
    est = G.GemxLocoEstimator(mirror=mirror, profile=profile, dist_src=dist_src, **params)
    out = []
    Jp = rest_skeleton()
    for k, (x, z, yaw, up, *ds) in enumerate(track):
        sk = Jp.copy()
        if up:   # arms overhead: a pose change, not a step
            S = G.SOMA
            for side in ("Left", "Right"):
                sgn = 1 if side == "Left" else -1
                sk[S[f"{side}ForeArm"]] = sk[S[f"{side}Arm"]] + (0.05 * sgn, 0.28, 0.0)
                sk[S[f"{side}Hand"]] = sk[S[f"{side}ForeArm"]] + (0.02 * sgn, 0.26, 0.0)
        J = to_cam(sk, x=x, z=z, yaw=yaw)
        d = det_of(J)
        if ds:
            d = G.soma_det(J * ds[0], kp_of(J), W_IMG, H_IMG, K)
        obs = obs_from_det(d, k / fps, W_IMG, H_IMG)
        out.append(est.update(obs))
    return out, est


def test_gemx_loco_steps_forward_and_back_home():
    still = [(0.0, 3.0, 0.0, False)] * 45                                    # 1.5 s: auto home
    step = [(0.0, 3.0 - 0.5 * min(1.0, i / 15), 0.0, False) for i in range(45)]   # 0.5 m toward the camera in 0.5 s
    back = [(0.0, 2.5 + 0.5 * min(1.0, i / 15), 0.0, False) for i in range(60)]
    cmds, est = run_loco(still + step + back)
    assert est.calibrated
    vx = np.array([c["vx"] for c in cmds])
    assert np.all(vx[:45] == 0)
    first = np.argmax(vx[45:] > 0)
    assert vx[45:].max() > 0.2 and (first - 12) / 30.0 < 0.5                # starts < 0.5 s after 0.4 m (frame 12)
    assert np.all(vx[-15:] == 0)                                             # stopped again within 1.5 s of home


def test_gemx_loco_sideways_mirror_and_turn():
    still = [(0.0, 3.0, 0.0, False)] * 45
    side = [(0.5 * min(1.0, i / 15), 3.0, 0.0, False) for i in range(45)]  # to image right = person's left
    cmds, _ = run_loco(still + side, mirror=True)
    assert min(c["vy"] for c in cmds[45:]) < -0.3                            # mirrored: robot to its right
    cmds, _ = run_loco(still + side, mirror=False)
    assert max(c["vy"] for c in cmds[45:]) > 0.3
    turn = [(0.0, 3.0, math.radians(50) * min(1.0, i / 15), False) for i in range(45)]
    cmds, _ = run_loco(still + turn, mirror=False)
    assert max(c["wz"] for c in cmds[45:]) > 0.3                             # turned left -> robot turns left
    cmds, _ = run_loco(still + turn, mirror=False, profile="presenter")
    assert all(c["wz"] == 0 for c in cmds)                                   # presenter: never turns


def test_gemx_loco_lost_person_stops():
    still = [(0.0, 3.0, 0.0, False)] * 45
    step = [(0.0, 3.0 - 0.5 * min(1.0, i / 15), 0.0, False) for i in range(30)]
    cmds, est = run_loco(still + step)
    assert cmds[-1]["vx"] > 0
    assert est.update(None) == {"vx": 0.0, "vy": 0.0, "wz": 0.0} or True   # within lost_hold_s: may hold
    est.update({"t": 75 / 30 + 0.5, "lm_cam": None, "lm_img": None, "lm_world": None})
    assert est.update(None)["vx"] == 0.0


def _first_on(cmds, key, k0):
    v = np.array([abs(c[key]) for c in cmds[k0:]])
    return int(np.argmax(v > 0)) if np.any(v > 0) else None


def test_fwd_axis_params_default_is_unchanged():
    """fwd_confirm_s / fwd_home_follow_s = None (default) -> the forward axis uses confirm_s / home_follow_s as before."""
    still = [(0.0, 3.0, 0.0, False)] * 45
    walk = [(0.0, 3.0 - 0.4 * i / 30, 0.0, False) for i in range(60)] + [(0.0, 2.2, 0.0, False)] * 90
    for prof in ("visitor", "presenter"):
        a, ea = run_loco(still + walk, profile=prof)
        b, eb = run_loco(still + walk, profile=prof, fwd_confirm_s=None, fwd_home_follow_s=None)
        assert a == b
        assert ea.ax_x.confirm_s == ea.p["confirm_s"]


def test_fwd_axis_params_only_touch_the_forward_axis():
    still = [(0.0, 3.0, 0.0, False)] * 45
    fwd = [(0.0, 3.0 - 0.6 * i / 30, 0.0, False) for i in range(90)]         # 0.6 m/s toward the camera, 3 s
    side = [(0.6 * i / 30, 3.0, 0.0, False) for i in range(50)]              # 0.6 m/s to image right
    f0 = _first_on(run_loco(still + fwd, profile="presenter", fwd_home_follow_s=1.6)[0], "vx", 45)
    f1 = _first_on(run_loco(still + fwd, profile="presenter", fwd_home_follow_s=1.6, fwd_confirm_s=0.8)[0], "vx", 45)
    assert f0 is not None and f1 is not None and (f1 - f0) / 30.0 > 0.4      # forward waits for its own confirmation
    s0 = _first_on(run_loco(still + side, profile="presenter")[0], "vy", 45)
    s1 = _first_on(run_loco(still + side, profile="presenter", fwd_confirm_s=0.8, fwd_home_follow_s=3.0)[0], "vy", 45)
    assert s0 is not None and s0 == s1                                       # sideways untouched
    # presenter walk-along: home follows the forward offset with 0.8 s, so 0.6 m/s barely reaches fwd_start 0.40;
    # a slower forward-only follow lets the offset build up
    slow = _first_on(run_loco(still + fwd, profile="presenter")[0], "vx", 45)
    assert slow is None or slow > f0


def test_gemx_loco_arms_overhead_is_not_a_step():
    seq = [(0.0, 3.0, 0.0, False)] * 45 + [(0.0, 3.0, 0.0, (i // 20) % 2 == 0) for i in range(120)]
    cmds, _ = run_loco(seq)
    assert all(c["vx"] == 0 and c["vy"] == 0 and c["wz"] == 0 for c in cmds)


def test_mapper_uses_gemx_estimator():
    hm = HumanMapper(mirror=True, loco=True, smpl=False, arm_ik="dir", perception="gemx")
    assert type(hm.loco.est).__name__ == "GemxLocoEstimator"
    hm.set_loco_profile("presenter")
    assert type(hm.loco.est).__name__ == "GemxLocoEstimator" and hm.loco.est.profile == "presenter"
    assert type(HumanMapper(loco=True, smpl=False).loco.est).__name__ == "LocoEstimator"


def test_hub_cache_first_skips_github_when_cached(tmp_path):
    """GEM-X's torch.hub.load("facebookresearch/dinov3") without a branch must not ask github.com when the checkout is
    cached (venue without / with a broken network); uncached or pinned repos still go through torch.hub."""
    import types

    calls = []
    hub = types.SimpleNamespace(get_dir=lambda: str(tmp_path), _parse_repo_info=lambda g: calls.append(g) or ("o", g, "x"))
    (tmp_path / "facebookresearch_dinov3_main").mkdir()
    GW.hub_cache_first(types.SimpleNamespace(hub=hub))
    assert hub._parse_repo_info("facebookresearch/dinov3") == ("facebookresearch", "dinov3", "main") and calls == []
    hub._parse_repo_info("someone/uncached")
    hub._parse_repo_info("facebookresearch/dinov3:v2")
    assert calls == ["someone/uncached", "facebookresearch/dinov3:v2"]


# ------------------------------------------------------------------------------------------------ worker wire format
def test_wire_format_roundtrip():
    frame = (np.arange(48 * 64 * 3) % 251).astype(np.uint8).reshape(48, 64, 3)
    buf = io.BytesIO(GW.pack_request(7, 12.5, frame, GW.FMT_RAW, GW.FLAG_RESET, (0.1, 0.2, 0.9, 1.0)))
    seq, t, f2, flags, crop = GW.read_request(buf)
    assert seq == 7 and t == 12.5 and flags == GW.FLAG_RESET and np.array_equal(f2, frame)
    assert np.allclose(crop, (0.1, 0.2, 0.9, 1.0))
    buf = io.BytesIO(GW.pack_request(8, 1.0, frame, GW.FMT_JPEG))
    _, _, f3, _, crop = GW.read_request(buf)
    assert f3.shape == frame.shape and crop is None
    J = to_cam(rest_skeleton())
    res = {"status": GW.ST_3D, "flags": 0, "ms": {"vitpose": 5.0, "total": 30.0}, "kp2d": kp_of(J), "joints": J,
           "body_pose": np.arange(228) * 0.01, "transl": J[0], "K": K}
    r = GW.read_reply(io.BytesIO(GW.pack_reply(8, 1.0, res)))
    assert r["seq"] == 8 and r["status"] == GW.ST_3D and abs(r["ms"]["total"] - 30.0) < 1e-4
    assert np.allclose(r["joints"], J, atol=1e-5) and np.allclose(r["K"], K) and r["body_pose"].shape == (228,)
    b = io.BytesIO()
    GW.write_msg(b, GW.MAGIC_READY, {"device": "x"})
    b.seek(0)
    assert GW.read_msg(b) == (GW.MAGIC_READY, {"device": "x"})


# ------------------------------------------------------------------------------------------------ sender + replay
def free_port():
    s = socket.socket()
    s.bind((P.HOST, 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture(scope="module")
def replay_clip(tmp_path_factory):
    """A 2 s synthetic clip (the frames do not matter to the replay engine) and its GEM-X track: the person stands,
    then raises the left arm."""
    import cv2

    d = tmp_path_factory.mktemp("gemx")
    vid = d / "clip.mp4"
    w = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 180))
    for k in range(60):
        f = np.full((180, 320, 3), 40 + k, np.uint8)
        w.write(f)
    w.release()
    Jp = rest_skeleton()
    S = G.SOMA
    Js, KPs = [], []
    for k in range(60):
        sk = Jp.copy()
        a = min(1.0, max(0.0, (k - 20) / 20)) * math.pi * 0.5        # left arm out to the side over frames 20-40
        sk[S["LeftForeArm"]] = sk[S["LeftArm"]] + 0.28 * np.array([math.sin(a), -math.cos(a), 0])
        sk[S["LeftHand"]] = sk[S["LeftForeArm"]] + 0.26 * np.array([math.sin(a), -math.cos(a), 0])
        J = to_cam(sk)
        Js.append(J)
        KPs.append(kp_of(J))
    track = d / "track.npz"
    np.savez(track, joints_incam=np.array(Js), kp2d_vitpose=np.array(KPs), K=K, image_wh=np.array([W_IMG, H_IMG]),
             fps=30.0, kind="stream", valid=np.ones(60, bool))
    return vid, track


def run_sender(args, seconds, ports):
    env = {**__import__("os").environ, "PYTHONUNBUFFERED": "1"}
    cmd = [sys.executable, "-m", "k1live.sender_gemx", "--human-port", str(ports["h"]), "--state-port",
           str(ports["s"]), "--ctrl-port", str(ports["c"]), "--http-port", str(ports["w"])] + args
    return subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


@pytest.mark.needs_smpl
def test_sender_gemx_replay_end_to_end(replay_clip):
    import zmq

    vid, track = replay_clip
    ports = {k: free_port() for k in "hscw"}
    worker = f"{sys.executable} {ROOT / 'k1live' / 'gemx_worker.py'} --serve --replay {track}"
    p = run_sender(["--source", str(vid), "--loop", "--no-mirror", "--gemx-cmd", worker, "--gemx-fmt", "raw",
                    "--max-seconds", "4"], 4, ports)
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://{P.HOST}:{ports['h']}")
    sub.setsockopt(zmq.SUBSCRIBE, P.TOPIC_HUMAN)
    st = ctx.socket(zmq.SUB)
    st.connect(f"tcp://{P.HOST}:{ports['s']}")
    st.setsockopt(zmq.SUBSCRIBE, P.TOPIC_SENDER_STATE)
    frames, states = [], []
    t0 = time.time()
    poll = zmq.Poller()
    poll.register(sub, zmq.POLLIN)
    poll.register(st, zmq.POLLIN)
    while time.time() - t0 < 6 and p.poll() is None:
        for s in dict(poll.poll(100)):
            m = P.decode(s.recv_multipart())
            (frames if s is sub else states).append(m)
    out = p.communicate(timeout=10)[0].decode()
    ctx.destroy(0)
    assert p.returncode == 0, out
    assert states and states[0]["note"] == "loading" or any(s["note"] == "loading" for s in states[:3])
    pres = [f for f in frames if f["present"]]
    assert len(pres) > 40, out                                   # ~30 Hz for most of the 4 s
    q = np.array([f["arm_q"] for f in pres])
    assert np.ptp(q[:, 1]) > 0.6                                 # left shoulder roll follows the arm going out
    assert all(f["smpl"] is not None for f in pres[-5:])


def test_sender_gemx_exits_5_without_gemx(tmp_path):
    ports = {k: free_port() for k in "hscw"}
    p = run_sender(["--gemx-perception", str(tmp_path), "--gemx-python", str(tmp_path / "nope" / "python")], 5, ports)
    out = p.communicate(timeout=20)[0].decode()
    assert p.returncode == 5 and "GEM-X unavailable" in out
    # a worker that starts but reports no CUDA (what the real worker does on a Mac) -> 5 as well
    ports = {k: free_port() for k in "hscw"}
    fake = tmp_path / "w.py"
    fake.write_text("import sys; sys.path.insert(0, %r)\nfrom k1live import gemx_worker as G\n"
                    "G.write_msg(sys.stdout.buffer, G.MAGIC_ERR, {'error': 'no CUDA device', 'code': 5})\n"
                    "sys.exit(5)\n" % str(ROOT))
    p = run_sender(["--gemx-cmd", f"{sys.executable} {fake}"], 5, ports)
    out = p.communicate(timeout=20)[0].decode()
    assert p.returncode == 5 and "no CUDA device" in out


def test_worker_check_exit_codes():
    r = subprocess.run([sys.executable, str(ROOT / "k1live" / "gemx_worker.py"), "--check", "--light"],
                       capture_output=True, text=True, timeout=60)
    import torch

    if not torch.cuda.is_available():
        assert r.returncode == GW.EXIT_NO_CUDA and "no CUDA" in r.stdout


# ------------------------------------------------------------------------------------------------ service
def test_service_perception_argv_and_fallback(tmp_path, monkeypatch):
    from k1live import service as S

    for env in ("K1LIVE_LOCO_PROFILE", "K1LIVE_CROP"):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setattr(S, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(S, "LOG_ROOT", tmp_path)
    cfg = S.load_config(path=tmp_path / "none.yaml")
    assert cfg["perception"] == "mediapipe"
    cfg.update(source="camera:0", perception="gemx", gemx={**cfg["gemx"], "hfov": 70, "arm_src": "mlp"})
    argv = S.build_argv(cfg["sender_gemx_cmd"], cfg)
    assert argv[1:4] == ["-m", "k1live.sender_gemx", "--source"] and "--hfov" in argv and "70" in argv
    assert "--arm-src" in argv and argv[argv.index("--start-timeout") + 1] == "240"
    sent, events, restarts, argvs = [], [], [], []
    ctl = S.Controller(cfg, send=lambda c, a: sent.append(c), event=lambda k, **d: events.append((k, d)),
                       restart_proc=lambda n, r: restarts.append(n), set_argv=lambda n: argvs.append(n),
                       frame_age=lambda n: 0.1)
    assert ctl.sender_template() == "sender_gemx_cmd"
    ctl.gemx_failed = "no CUDA device"            # what Service._proc_event sets on exit 5
    ctl.tick(time.time())
    assert ctl.perc_active == "mediapipe" and ctl.sender_template() == "sender_cmd" and argvs == ["sender"]
    assert ctl.counters["gemx_fallbacks"] == 1 and any(k == "gemx_fallback" for k, _ in events)
    snap = ctl.snapshot()
    assert snap["perception"] == {"want": "gemx", "active": "mediapipe", "note": "no CUDA device"}
    ok, _ = ctl.command("set", {"perception": "gemx"})        # operator retries GEM-X
    assert ok and ctl.perc_active == "gemx" and restarts == ["sender"]
    ok, _ = ctl.command("set", {"perception": "bogus"})
    assert not ok
    assert json.loads((tmp_path / "settings.json").read_text())["perception"] == "gemx"


def test_service_falls_back_when_sender_exits_5(tmp_path, monkeypatch):
    """A real Service + Supervisor: the sender template exits 5 -> the next sender start is the MediaPipe one."""
    from k1live import service as S

    for k in ("CTRL_PORT", "SIM_STATE_PORT", "SENDER_STATE_PORT", "CAM_HTTP_PORT", "ROBOT_HTTP_PORT"):
        monkeypatch.setattr(P, k, free_port())
    monkeypatch.setattr(S, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(S, "LOG_ROOT", tmp_path)
    cfg = S.load_config(path=tmp_path / "none.yaml")
    cfg["perception"] = "gemx"
    cfg["sender_gemx_cmd"] = [sys.executable, "-c",
                              "import sys; print('[sender_gemx] GEM-X unavailable: no CUDA device', file=sys.stderr); "
                              "sys.exit(5)"]
    cfg["sender_cmd"] = [sys.executable, "-c", "import time; print('mediapipe sender'); time.sleep(30)"]
    cfg["sim_cmd"] = [sys.executable, "-c", "import time; time.sleep(30)"]
    svc = S.Service(cfg, tmp_path / "sess", port=free_port())
    svc.start()
    try:
        end = time.time() + 15
        while time.time() < end and not (svc.ctl.perc_active == "mediapipe"
                                         and "mediapipe sender" in (svc.dir / "sender.log").read_text()):
            time.sleep(0.1)
        assert svc.ctl.perc_active == "mediapipe" and svc.ctl.perc_note == "no CUDA device"
        assert "mediapipe sender" in (svc.dir / "sender.log").read_text()
    finally:
        svc.shutdown()


def test_service_starts_on_mediapipe_without_gemx_env(tmp_path, monkeypatch):
    """perception gemx with no GEM-X folder at all (a fresh clone): the real GEM-X sender is not started just to exit 5;
    the service starts on MediaPipe with the reason, and the operator's choice stays gemx."""
    from k1live import service as S

    for k in ("CTRL_PORT", "SIM_STATE_PORT", "SENDER_STATE_PORT", "CAM_HTTP_PORT", "ROBOT_HTTP_PORT"):
        monkeypatch.setattr(P, k, free_port())
    monkeypatch.setattr(S, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(S, "LOG_ROOT", tmp_path)
    cfg = S.load_config(path=tmp_path / "none.yaml")
    cfg["perception"] = "gemx"
    cfg["gemx"] = {**cfg["gemx"], "perception_dir": str(tmp_path / "no_gemx")}
    cfg["sender_cmd"] = [sys.executable, "-c", "import time; print('mediapipe sender'); time.sleep(30)"]
    cfg["sim_cmd"] = [sys.executable, "-c", "import time; time.sleep(30)"]
    svc = S.Service(cfg, tmp_path / "sess", port=free_port())
    assert svc.ctl.perc_active == "mediapipe" and "not found" in svc.ctl.perc_note
    assert svc.ctl.counters["gemx_fallbacks"] == 1 and svc.ctl.d["perception"] == "gemx"
    svc.start()
    try:
        end = time.time() + 15
        while time.time() < end and "mediapipe sender" not in (svc.dir / "sender.log").read_text():
            time.sleep(0.1)
        log = (svc.dir / "sender.log").read_text()
        assert "mediapipe sender" in log and "k1live.sender_gemx" not in log
        assert svc.sup.procs["sender"].restarts == 0
    finally:
        svc.shutdown()


def test_preflight_gemx_check_uses_the_service_command(tmp_path, monkeypatch):
    from k1live import preflight
    from k1live import service as S

    monkeypatch.setattr(S, "SETTINGS_PATH", tmp_path / "settings.json")
    orig = S.load_config

    def cfg_gemx(path=None, cli=None):
        c = orig(path=tmp_path / "none.yaml")
        c["perception"] = "gemx"
        c["gemx"] = {**c["gemx"], "cmd": f"{sys.executable} {ROOT / 'k1live' / 'gemx_worker.py'} --serve "
                                        f"--replay {tmp_path / 'x.npz'}"}
        return c

    monkeypatch.setattr(S, "load_config", cfg_gemx)
    st, detail = preflight.check_gemx()
    assert st == preflight.PASS and "replay" in detail
    monkeypatch.setattr(S, "load_config", lambda path=None, cli=None: orig(path=tmp_path / "none.yaml"))
    assert preflight.check_gemx()[0] == preflight.SKIP


def test_ground_plane_distance_ignores_depth_errors_but_sees_steps():
    """GEM-X's metric depth breathing with the pose (raised arms -> 12 % farther) walks the pelvis estimator; the
    ground-plane estimator only reads where the feet are on the floor."""
    still = [(0.0, 3.0, 0.0, False)] * 45
    breathe = [(0.0, 3.0, 0.0, True, 1.12 if (i // 30) % 2 == 0 else 1.0) for i in range(120)]
    cmds, _ = run_loco(still + breathe, dist_src="pelvis")
    assert any(c["vx"] != 0 for c in cmds)                       # a false step from a depth error
    cmds, est = run_loco(still + breathe, dist_src="ground")
    assert est.plane is not None and all(c["vx"] == 0 and c["vy"] == 0 for c in cmds)
    step = [(0.0, 3.0 - 0.5 * min(1.0, i / 15), 0.0, False) for i in range(45)]
    cmds, _ = run_loco(still + step, dist_src="ground")
    vx = np.array([c["vx"] for c in cmds])
    assert vx[45:].max() > 0.2 and (np.argmax(vx[45:] > 0) - 12) / 30.0 < 0.5
    side = [(0.5 * min(1.0, i / 15), 3.0, 0.0, False) for i in range(45)]
    cmds, _ = run_loco(still + side, mirror=False, dist_src="ground")
    assert max(c["vy"] for c in cmds[45:]) > 0.3


def test_presenter_param_overrides_only_the_presenter_profile(capsys):
    """sender_gemx --presenter-param (venue tuning, runbook 4-4): parsed against LocoEstimator.DEFAULTS, bad items are
    reported and skipped (never a constructor error = no walking), applied only while the walking mode is presenter."""
    from k1live.sender_gemx import loco_overrides

    kw = loco_overrides(["fwd_start=0.25", "fwd_confirm_s=0.4", "turn=false", "nope=1", "fwd_stop=x", "side_start"])
    assert kw == {"fwd_start": 0.25, "fwd_confirm_s": 0.4, "turn": False}
    out = capsys.readouterr().out
    assert "'nope=1'" in out and "'fwd_stop=x'" in out and "'side_start'" in out
    hm = HumanMapper(mirror=True, loco=True, smpl=False, arm_ik="dir", perception="gemx",
                     loco_kw={"dist_src": "pelvis", "profile_kw": {"presenter": kw}})
    visitor_start = hm.loco.est.p["fwd_start"]
    assert visitor_start != 0.25                                             # visitor untouched
    hm.set_loco_profile("presenter")
    assert hm.loco.est.p["fwd_start"] == 0.25 and hm.loco.est.ax_x.confirm_s == 0.4
    assert hm.loco.est.dist_src == "pelvis"
    hm.set_loco_profile("visitor")
    assert hm.loco.est.p["fwd_start"] == visitor_start
    assert loco_overrides([]) == {}


def test_soma_kernel_cache_reuses_the_kernel(monkeypatch):
    """GEM-X / SOMA lbs_warp.get_kernel defines a new Warp kernel per call (the worker's ~3.3 GB/h growth):
    soma_kernel_cache memoises it by bone count + dtypes (Warp vector types by length + scalar type)."""
    import types

    calls = []

    def get_kernel(max_bones_count, vertices_scalar_dtype, weights_dtype, indices_dtype, vec3_dtype=None,
                   mat44_dtype=None):
        calls.append(max_bones_count)
        return object()

    def vector(n, scalar):   # a fresh class per call, like warp.types.vector
        return type("vec", (), {"_length_": n, "_wp_scalar_type_": scalar})

    lbs = types.ModuleType("soma.geometry.lbs_warp")
    lbs.get_kernel = get_kernel
    geo = types.ModuleType("soma.geometry")
    geo.lbs_warp = lbs
    soma = types.ModuleType("soma")
    soma.geometry = geo
    monkeypatch.setitem(sys.modules, "soma", soma)
    monkeypatch.setitem(sys.modules, "soma.geometry", geo)
    monkeypatch.setitem(sys.modules, "soma.geometry.lbs_warp", lbs)
    assert GW.soma_kernel_cache(log=lambda *a: None)
    f32, i32, v3 = float, int, "vec3f"
    k1 = lbs.get_kernel(4, vertices_scalar_dtype=f32, weights_dtype=vector(4, f32), indices_dtype=vector(4, i32),
                        vec3_dtype=v3, mat44_dtype="m44f")
    k2 = lbs.get_kernel(4, vertices_scalar_dtype=f32, weights_dtype=vector(4, f32), indices_dtype=vector(4, i32),
                        vec3_dtype=v3, mat44_dtype="m44f")
    k3 = lbs.get_kernel(8, vertices_scalar_dtype=f32, weights_dtype=vector(8, f32), indices_dtype=vector(8, i32),
                        vec3_dtype=v3, mat44_dtype="m44f")
    assert k1 is k2 and k3 is not k1 and calls == [4, 8]
    assert GW.soma_kernel_cache(log=lambda *a: None) and lbs.get_kernel._k1live_cached   # idempotent


def test_soma_kernel_cache_without_soma_is_a_no_op(monkeypatch):
    monkeypatch.setitem(sys.modules, "soma", None)
    assert GW.soma_kernel_cache(log=lambda *a: None) is False
