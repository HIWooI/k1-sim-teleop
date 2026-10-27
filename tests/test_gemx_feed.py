"""scripts/gemx_feed.py: GEM-X worker dumps replayed through the live sender_gemx mapping (docs/results.md).

    .venv/bin/python -m pytest tests/test_gemx_feed.py -q

Synthetic every-frame dump (the standing SOMA skeleton of tests/test_k1live_gemx.py raising its left arm); no GPU, no
MuJoCo (the SimServer replay is exercised on the GPU box).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts", ROOT / "tests"):
    sys.path.insert(0, str(p))

import gemx_feed as GF  # noqa: E402
from test_k1live_gemx import K, H_IMG, W_IMG, kp_of, rest_skeleton, to_cam  # noqa: E402

from k1live import gemx as G  # noqa: E402


@pytest.fixture(scope="module")
def dump(tmp_path_factory):
    """90 frames at 30 fps, every frame (no --realtime), 3D from frame 16 (the window), left arm out over 30-60."""
    S = G.SOMA
    Js, KPs = [], []
    for k in range(90):
        sk = rest_skeleton()
        a = min(1.0, max(0.0, (k - 30) / 30)) * math.pi * 0.5
        sk[S["LeftForeArm"]] = sk[S["LeftArm"]] + 0.28 * np.array([math.sin(a), -math.cos(a), 0])
        sk[S["LeftHand"]] = sk[S["LeftForeArm"]] + 0.26 * np.array([math.sin(a), -math.cos(a), 0])
        J = to_cam(sk)
        Js.append(J)
        KPs.append(kp_of(J))
    valid = np.arange(90) >= 16
    J = np.where(valid[:, None, None], np.array(Js), np.nan)
    p = tmp_path_factory.mktemp("feed") / "clip.npz"
    np.savez(p, joints_incam=J, kp2d_vitpose=np.array(KPs), valid=valid, source_frame_index=np.arange(90),
             t_done=np.arange(90) * 0.1, K=K, image_wh=np.array([W_IMG, H_IMG]), fps=30.0, kind="stream",
             meta=json.dumps({"realtime": False, "frames": 90}))
    return p


def test_retime_keeps_every_frame_with_live_latency(dump):
    rows = GF.load_rows(dump)
    assert not rows.realtime and rows.n_video == 90 and rows.seconds == pytest.approx(3.0)
    rt = rows.retime()
    assert rt.realtime and len(rt.k) == 90
    lat = rt.t_done - rt.k / rt.fps
    assert lat.min() >= 0.034 - 1e-9 and lat.max() <= 0.038 + 1e-9
    assert rt.arrive(0) == pytest.approx(rt.t_done[0] + GF.MAP_MS / 1e3)
    assert np.allclose(rows.retime(seed=3).t_done, rows.retime(seed=3).t_done)     # deterministic per seed


@pytest.mark.needs_smpl
def test_sender_frames_follow_the_arm_and_stream_smpl(dump):
    rows = GF.load_rows(dump).retime()
    fr = GF.sender_frames(rows, mirror=False, profile="presenter")
    assert len(fr) == 90
    assert not any(f["present"] for f in fr[:16])                  # no 3D before the window is full
    pres = [f for f in fr if f["present"]]
    assert len(pres) >= 70 and all(f["smpl"] is not None for f in pres)
    assert all(f["_t_arrive"] > f["t_capture"] for f in fr)
    q = np.array([f["arm_q"] for f in pres])
    assert np.ptp(q[:, 1]) > 0.6                                    # left shoulder roll: the arm goes out
    st = GF.stream_stats(fr)
    assert st["frames"] == 90 and 0.7 < st["present_frac"] < 0.85 and 39 < st["latency_ms_p50"] < 45


@pytest.mark.needs_smpl
def test_segment_window(dump):
    rows = GF.load_rows(dump).retime()
    fr = GF.sender_frames(rows, t0=1.0, t1=2.0, mirror=False)
    assert len(fr) == 30 and fr[0]["_k"] == 30 and fr[0]["t_capture"] == pytest.approx(0.0)


@pytest.mark.needs_smpl
def test_playlist_cuts_and_switches(dump):
    rows = GF.load_rows(dump).retime()
    frames, timeline, total = GF.playlist_frames([("a", rows), ("b", rows)], minutes=0.25, repeat=2,
                                                 mirror=False, loco=False)
    assert total == pytest.approx(15.0)
    assert [s["clip"] for s in timeline] == ["a", "b", "a"]          # 2 x 3 s per clip, cycling
    assert [s["t"] for s in timeline] == pytest.approx([0.0, 6.0, 12.0])
    t = np.array([f["t_capture"] for f in frames])
    assert np.all(np.diff(t) > 0) and t[-1] < 15.0
    # every cut / switch restarts the GEM-X window: nobody for its first 16 frames
    starts = [int(np.searchsorted(t, s - 1e-9)) for s in (0.0, 3.0, 6.0, 9.0, 12.0)]
    for i in starts:
        assert not frames[i]["present"]
