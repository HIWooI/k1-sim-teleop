"""GEM-X-like SMPL input augmentation of k1s_bones2 (overlay .../mdp/smpl_gemx_aug.py, docs/training.md).

    .venv/bin/python -m pytest tests/test_smpl_gemx_aug.py -q

Plain torch (no Isaac Lab): the generator reproduces its configured statistics (bias / slow error / time constant per
bone group and root axis, dropout rate and length, camera rate), keeps bone lengths, holds the pose in a dropout and
cross-fades back, leaves disabled envs bit-identical, and the three Isaac Lab terms equal the vendor formulas on the
clean stream. The fitted parameters (scripts/gemx_aug_stats.py fit) are the ones in the training config and their
calibration reproduces the live GEM-X target.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sonic_k1_train" / "data"))
import smpl_aug_stats as S  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "smpl_gemx_aug", ROOT / "sonic_k1_train/overlay/gear_sonic/envs/manager_env/mdp/smpl_gemx_aug.py")
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)

CFG_YAML = ROOT / "sonic_k1_train/overlay/gear_sonic/config/exp/manager/universal_token/all_modes/sonic_k1_small_la4_bones2.yaml"
PARAMS = ROOT / "sonic_k1_train/data/gemx_aug/gemx_aug_params.yaml"
TARGET = ROOT / "sonic_k1_train/data/gemx_aug/gemx_aug_target.json"


def rest_pose():
    """Standing SMPL-24 (root-local, pelvis at 0): the converter's idle frame, rounded."""
    return np.array([[0, 0, 0], [-.026, .058, -.093], [-.021, -.063, -.104], [-.028, -.003, .11], [-.035, .113, -.472],
                     [-.038, -.108, -.466], [-.034, .007, .242], [-.067, .069, -.875], [-.058, -.092, -.877],
                     [-.005, -.005, .294], [.051, .117, -.933], [.061, -.131, -.935], [-.037, -.017, .459],
                     [-.012, .042, .379], [-.019, -.052, .378], [-.016, .008, .62], [-.028, .161, .437],
                     [-.031, -.155, .432], [-.028, .198, .172], [-.031, -.193, .16], [.025, .233, -.073],
                     [.021, -.227, -.082], [.106, .221, -.138], [.102, -.213, -.146]], np.float32)


def quiet(**over):
    """One profile, no error, no dropouts, a camera frame every tick, no smoothing / low-pass (identity stream)."""
    z = {g: 0.0 for g in A.BONE_GROUPS}
    pr = {"weight": 1.0, "bone_bias_deg": dict(z), "bone_slow_deg": dict(z), "bone_fast_deg": dict(z),
          "root_bias_deg": [0.0] * 3, "root_slow_deg": [0.0] * 3, "root_fast_deg": [0.0] * 3, "gap_per_min": 0.0}
    cfg = {"prob": 1.0, "camera_hz": [50.0, 50.0], "smooth_tau": 1e-6, "lowpass_tau": dict(z), "root_lowpass_tau": 0.0,
           "noise_smooth_tau": 0.0, "profiles": {"p": pr}}
    for k, v in over.items():
        if k in A.DEFAULT["profiles"]["full"]:
            pr[k] = v
        else:
            cfg[k] = v
    return cfg


def stream(N, F, T, moving=False, seed=0):
    """Clean windows (T, N, F, 24, 3) / (T, N, F, 4) / joint pos (T, N, F, 23): standing, optionally the left arm
    swinging and the root yawing."""
    J0 = torch.tensor(rest_pose())
    t = torch.arange(T + F, dtype=torch.float32) * 0.02
    J = J0.expand(T + F, 24, 3).clone()
    q = torch.zeros(T + F, 4)
    q[:, 0] = 1.0
    if moving:
        a = 0.6 * torch.sin(2 * math.pi * 0.5 * t)
        for j in (18, 20, 22):                              # rotate the left arm about the shoulder (x axis)
            v = J0[j] - J0[16]
            c, s = torch.cos(a), torch.sin(a)
            J[:, j, 0] = J0[16, 0] + v[0]
            J[:, j, 1] = J0[16, 1] + c * v[1] - s * v[2]
            J[:, j, 2] = J0[16, 2] + s * v[1] + c * v[2]
        yaw = 0.3 * torch.sin(2 * math.pi * 0.2 * t)
        q = torch.stack((torch.cos(yaw / 2), 0 * yaw, 0 * yaw, torch.sin(yaw / 2)), -1)
    jp = torch.sin(t)[:, None].expand(T + F, 23).clone()
    idx = torch.arange(T)[:, None] + torch.arange(F)[None]
    return (J[idx][:, None].expand(T, N, F, 24, 3), q[idx][:, None].expand(T, N, F, 4),
            jp[idx][:, None].expand(T, N, F, 23))


def run(cfg, N=64, T=400, F=4, moving=False, seed=0, record=False):
    g = torch.Generator().manual_seed(seed)
    aug = A.SmplGemxAug(N, F, cfg, generator=g)
    Js, qs, jps = stream(N, F, T, moving)
    out, rec = [], {"cam_J": [], "cam_q": [], "new": [], "gap": [], "clean_J": [], "clean_q": []}
    for k in range(T):
        o = aug.step(Js[k], qs[k], jps[k], torch.full((N,), k > 0))
        out.append(o)
        if record:
            rec["cam_J"].append(aug.cam_J.clone())
            rec["cam_q"].append(aug.cam_q.clone())
            rec["new"].append(aug.new_cam.clone())
            rec["gap"].append(aug.in_gap.clone())
            rec["clean_J"].append(Js[k][:, -1].clone())
            rec["clean_q"].append(qs[k][:, -1].clone())
    return aug, out, (Js, qs, jps), {k: torch.stack(v) for k, v in rec.items()} if record else None


# ------------------------------------------------------------------------------------------------ mechanics
def test_disabled_envs_are_bit_identical():
    aug, out, (Js, qs, jps), _ = run(dict(quiet(), prob=0.0, camera_hz=[24.0, 30.0], smooth_tau=0.03), T=60,
                                     moving=True)
    for k, (Jo, qo, jpo) in enumerate(out):
        assert torch.equal(Jo, Js[k]) and torch.equal(qo, qs[k]) and torch.equal(jpo, jps[k])


def test_quiet_stream_is_the_clean_window_and_newest_wrist():
    aug, out, (Js, qs, jps), _ = run(quiet(), T=40, moving=True)
    for k in range(5, 40):
        Jo, qo, jpo = out[k]
        assert torch.allclose(Jo, Js[k], atol=1e-5)                 # ring buffer aligned with the clean window
        dot = (qo * qs[k]).sum(-1).abs()
        assert torch.allclose(dot, torch.ones_like(dot), atol=1e-6)
        assert torch.allclose(jpo, jps[k][:, -1:].expand_as(jpo))   # live: every frame carries the newest wrist
    cfg = quiet(wrist_newest=False)
    _, out2, (_, _, jps2), _ = run(cfg, T=10)
    assert torch.allclose(out2[9][2], jps2[9])


def test_profiles_block_replaces_the_default_set():
    c = A.merge_cfg({"profiles": {"only": {"weight": 1.0, "gap_per_min": 3.0}}})
    assert list(c["profiles"]) == ["only"]
    assert c["profiles"]["only"]["gap_per_min"] == 3.0
    assert c["profiles"]["only"]["tau_slow"] == A.DEFAULT["profiles"]["full"]["tau_slow"]   # filled from the template
    assert set(A.merge_cfg({"prob": 0.5})["profiles"]) == set(A.DEFAULT["profiles"])


def test_bone_lengths_are_kept():
    cfg = quiet(bone_bias_deg={g: 10.0 for g in A.BONE_GROUPS}, bone_slow_deg={g: 8.0 for g in A.BONE_GROUPS},
                root_bias_deg=[5.0, 5.0, 5.0], root_slow_deg=[3.0, 3.0, 3.0])
    aug, out, (Js, _, _), rec = run(cfg, T=50, moving=True, record=True)
    _, l0 = A.bone_angles(rec["clean_J"])
    _, l1 = A.bone_angles(rec["cam_J"])
    assert (l0 - l1).abs().max() < 1e-5


def test_error_statistics_match_the_configuration():
    """Static pose: bias RMS / slow-noise RMS (across each bone) / slow time constant / root axes as configured."""
    bias = {g: 4.0 + i for i, g in enumerate(A.BONE_GROUPS)}
    slow = {g: 3.0 + 0.5 * i for i, g in enumerate(A.BONE_GROUPS)}
    tau = 0.4
    cfg = quiet(bone_bias_deg=bias, bone_slow_deg=slow, tau_slow=tau, root_bias_deg=[2.0, 3.0, 1.0],
                root_slow_deg=[1.5, 2.5, 3.5], root_tau_slow=tau)
    N, T = 512, 600
    aug, _, _, rec = run(cfg, N=N, T=T, record=True)
    sl = slice(1, None)
    clips = [{"res": S.residual(rec["cam_J"][sl, i].numpy(), rec["cam_q"][sl, i].numpy(),
                                rec["clean_J"][sl, i].numpy(), rec["clean_q"][sl, i].numpy(), remove_yaw=False),
              "dt": 0.02} for i in range(N)]
    sm = S.summarize(clips, 0.02, max_lag_s=1.0)
    # the clip mean of the slow OU part adds to the bias: Var(mean) ~ s^2 * 2 tau / T for T >> tau
    leak = 2 * tau / (T * 0.02)
    for g in A.BONE_GROUPS:
        exp_b = math.sqrt(bias[g] ** 2 + leak * slow[g] ** 2)
        exp_n = slow[g] * math.sqrt(1 - leak)
        assert sm["bone_bias_deg"][g] == pytest.approx(exp_b, rel=0.08), g
        assert sm["bone_noise_deg"][g] == pytest.approx(exp_n, rel=0.08), g
    for k, (b, s) in enumerate(zip([2.0, 3.0, 1.0], [1.5, 2.5, 3.5])):
        assert sm["root_bias_deg"][k] == pytest.approx(math.sqrt(b * b + leak * s * s), rel=0.1)
        assert sm["root_noise_deg"][k] == pytest.approx(s * math.sqrt(1 - leak), rel=0.1)
    # mean-removed ACF of an OU process over a finite window decays a little faster than exp(-l / tau)
    assert sm["bone_tau_slow"] == pytest.approx(tau, rel=0.3)
    assert sm["bone_acf"]["0.40"] == pytest.approx(math.exp(-1), abs=0.1)


def test_slow_error_is_smooth():
    """noise_smooth_tau: the applied slow error keeps (almost) its size but loses the OU path's 20 ms roughness."""
    cfg = quiet(bone_slow_deg={g: 10.0 for g in A.BONE_GROUPS}, tau_slow=3.0)
    # per-tick increment of the applied error (3D rotation vector, per-component sd 10 / sqrt(2) deg):
    # raw OU sqrt(3) sd sqrt(1 - rho^2), smoothed much smaller
    def incr(seed, smooth_tau):
        aug = A.SmplGemxAug(64, 4, dict(cfg, noise_smooth_tau=smooth_tau), generator=torch.Generator().manual_seed(seed))
        Js, qs, jps = stream(64, 4, 200)
        e = []
        for k in range(200):
            aug.step(Js[k], qs[k], jps[k], torch.full((64,), k > 0))
            e.append((aug.n_slow_s if smooth_tau > 0 else aug.n_slow).clone())
        e = torch.stack(e)
        return float(torch.rad2deg((e[1:] - e[:-1]).norm(dim=-1)).pow(2).mean().sqrt()), float(torch.rad2deg(e.norm(dim=-1)).pow(2).mean().sqrt())
    d_raw, s_raw = incr(0, 0.0)
    d_sm, s_sm = incr(0, 0.3)
    assert d_raw == pytest.approx(10.0 / math.sqrt(2) * math.sqrt(3) * math.sqrt(1 - math.exp(-2 * 0.02 / 3.0)), rel=0.1)
    assert d_sm < 0.2 * d_raw and s_sm > 0.85 * s_raw


def test_dropouts_rate_length_and_hold():
    cfg = quiet(gap_per_min=30.0, gap_logn_mu=math.log(0.12), gap_logn_sd=0.6, gap_min_s=0.02)
    N, T = 256, 1500
    aug, out, _, rec = run(cfg, N=N, T=T, moving=True, record=True)
    gap = rec["gap"].numpy()
    gl = [S.gaps(~gap[:, i], 0.02) for i in range(N)]
    tracked = float((~gap).sum()) * 0.02
    gs = S.gap_summary(gl, tracked, hold_s=0.5)
    assert gs["per_min"] == pytest.approx(30.0, rel=0.1)
    L = np.exp(math.log(0.12) + 0.6 * np.random.default_rng(0).standard_normal(200000))
    L = np.clip(np.round(np.clip(L, 0.02, 0.5) / 0.02), 1, 25) * 0.02
    assert gs["len_s_p25_p50_p90"][1] == pytest.approx(float(np.median(L)), rel=0.15)
    assert max(max(x) for x in gl if x) <= 0.5 + 1e-9                  # longer losses are the backend's idle fade
    # in a dropout the camera pose is frozen (the backend holds the last delivered pose)
    cam = rec["cam_J"].numpy()
    held = gap[1:] & gap[:-1]
    assert np.abs(cam[1:][held] - cam[:-1][held]).max() == 0.0


def test_redetection_cross_fades_instead_of_jumping():
    """The pose changes while the person is lost: held, then the PresenceGate cross-fade (smoothstep over quick_s)."""
    cfg = quiet(gap_per_min=0.0, smooth_tau=0.03, quick_s=0.25)
    N, F = 4, 4
    aug = A.SmplGemxAug(N, F, cfg, generator=torch.Generator().manual_seed(0))
    Ja, q, jp = (x[0] for x in stream(N, F, 1))
    Jb = Ja.clone()
    Jb[..., [18, 20, 22], 2] += 0.3                    # left arm raised by 30 cm
    newest = []
    for k in range(160):
        if k == 60:                                   # a 0.4 s loss on every env
            aug.gap_left[:] = 20
        J = Ja if k < 70 else Jb                      # the person moves during the loss
        Jo, _, _ = aug.step(J, q, jp, torch.full((N,), k > 0))
        newest.append(Jo[:, -1, 20, 2].clone())
    W = torch.stack(newest)[:, 0]
    assert (W[60:80] - Ja[0, -1, 20, 2]).abs().max() < 1e-3     # held at the last delivered pose
    steps = (W[81:110] - W[80:109]).abs()
    assert steps.max() < 0.3 * (1.5 / 12.5) * 1.3              # <= 30 cm x smoothstep peak slope (1.5 / quick ticks)
    assert steps.max() > 0.01                                           # but it does move back
    assert (W[110] - Jb[0, -1, 20, 2]).abs() < 2e-3                     # tracking again after the fade


def test_camera_rate():
    cfg = quiet(camera_hz=[24.0, 30.0])
    aug, _, _, rec = run(cfg, N=512, T=500, record=True)
    frac = rec["new"][1:].float().mean().item()
    assert frac == pytest.approx(27.0 / 50.0, rel=0.03)


def test_restart_redraws_episode_errors():
    cfg = quiet(bone_bias_deg={g: 10.0 for g in A.BONE_GROUPS})
    N, F = 16, 4
    aug = A.SmplGemxAug(N, F, cfg, generator=torch.Generator().manual_seed(0))
    Js, qs, jps = stream(N, F, 10)
    for k in range(5):
        aug.step(Js[k], qs[k], jps[k], torch.full((N,), k > 0))
    b0 = aug.b_bone.clone()
    cont = torch.ones(N, dtype=torch.bool)
    cont[:4] = False
    Jo, _, _ = aug.step(Js[5], qs[5], jps[5], cont)
    assert not torch.allclose(aug.b_bone[:4], b0[:4]) and torch.equal(aug.b_bone[4:], b0[4:])
    d_clean, _ = A.bone_angles(Js[5][:4])
    d_out, _ = A.bone_angles(Jo[:4])
    assert (d_clean - d_out).norm(dim=-1).mean() > 0.05       # restarted envs read their corrupted window at once


# ------------------------------------------------------------------------------------------------ Isaac Lab terms
def _stub_env(N, F, seed=0):
    """A ManagerBasedRLEnv-shaped stub around a TrackingCommand-shaped stub (random poses, unit quats)."""
    g = torch.Generator().manual_seed(seed)
    qn = lambda *s: (lambda q: q / q.norm(dim=-1, keepdim=True))(torch.randn(*s, 4, generator=g))
    q = qn(N, F)
    J_local = torch.randn(N, F, 24, 3, generator=g) * 0.3
    J_w = A.quat_apply(q[..., None, :], J_local)
    cmd = SimpleNamespace(smpl_num_future_frames=F, smpl_joints_multi_future=J_w, smpl_root_quat_w_multi_future=q,
                          joint_pos_multi_future_for_smpl=torch.randn(N, F * 23, generator=g),
                          robot_anchor_quat_w=qn(N), motion_ids=torch.arange(N), time_steps=torch.zeros(N, dtype=torch.long),
                          motion_start_time_steps=torch.zeros(N, dtype=torch.long))
    env = SimpleNamespace(num_envs=N, device="cpu", common_step_counter=0, episode_length_buf=torch.zeros(N),
                          command_manager=SimpleNamespace(get_term=lambda name: cmd))
    return env, cmd, J_local


def test_isaac_terms_equal_the_vendor_formulas_on_the_clean_stream():
    from scipy.spatial.transform import Rotation as R

    N, F = 6, 4
    env, cmd, J_local = _stub_env(N, F)
    aug = {"prob": 0.0}
    J = A.smpl_joints_multi_future_local_gemx(env, "motion", non_flatten=True, aug=aug)
    assert J.shape == (N, F, 72)
    assert torch.allclose(J, J_local.reshape(N, F, 72), atol=1e-5)
    rot = A.smpl_root_ori_b_mf_gemx(env, "motion", non_flatten=True, aug=aug)
    assert rot.shape == (N, F, 6)
    xyzw = lambda q: q[..., [1, 2, 3, 0]].numpy().reshape(-1, 4)
    anchor = np.repeat(xyzw(cmd.robot_anchor_quat_w), F, 0)
    M = (R.from_quat(anchor).inv() * R.from_quat(xyzw(cmd.smpl_root_quat_w_multi_future))).as_matrix()
    assert np.allclose(rot.numpy().reshape(-1, 6), M[:, :, :2].reshape(-1, 6), atol=1e-5)   # vendor: mat[..., :2]
    w = A.joint_pos_multi_future_wrist_for_smpl_gemx(env, "motion", joints_idx=[21, 22], aug=aug)
    assert torch.equal(w, cmd.joint_pos_multi_future_for_smpl.view(N, F, -1)[..., [21, 22]])
    # one advance per env step: a second call in the same step returns the same window
    st = env._smpl_gemx_aug
    k0 = st["key"]
    A.smpl_joints_multi_future_local_gemx(env, "motion", non_flatten=True, aug=aug)
    assert env._smpl_gemx_aug["key"] == k0


def test_isaac_terms_continue_the_stream_across_steps():
    N, F = 4, 4
    env, cmd, _ = _stub_env(N, F)
    aug = {"prob": 1.0, "profiles": {"p": quiet()["profiles"]["p"]}, "camera_hz": [50.0, 50.0], "smooth_tau": 1e-6,
           "lowpass_tau": {g: 0.0 for g in A.BONE_GROUPS}, "root_lowpass_tau": 0.0}
    A.smpl_joints_multi_future_local_gemx(env, "motion", non_flatten=True, aug=aug)
    st = env._smpl_gemx_aug
    env.common_step_counter, env.episode_length_buf = 1, torch.ones(N)
    cmd.time_steps = cmd.time_steps + 1
    A.smpl_joints_multi_future_local_gemx(env, "motion", non_flatten=True, aug=aug)
    assert st["aug"].fade_k.min() >= 1 and st["key"][0] == 1            # advanced (continuous), not restarted
    cmd.motion_ids = cmd.motion_ids + 1                                 # resampled motion -> restart
    env.common_step_counter = 2
    cmd.time_steps = cmd.time_steps + 1
    A.smpl_joints_multi_future_local_gemx(env, "motion", non_flatten=True, aug=aug)
    assert int(st["aug"].fade_k.max()) == 10 ** 6


# ------------------------------------------------------------------------------------------------ fitted parameters
def test_training_config_carries_the_fitted_parameters():
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load(CFG_YAML.read_text())
    fitted = yaml.safe_load(PARAMS.read_text())["aug"]
    assert cfg["smpl_gemx_aug"] == fitted
    tok = cfg["manager_env"]["observations"]["tokenizer"]
    for term, fn in (("smpl_joints_multi_future_local_nonflat", "smpl_joints_multi_future_local_gemx"),
                     ("smpl_root_ori_b_multi_future", "smpl_root_ori_b_mf_gemx"),
                     ("joint_pos_multi_future_wrist_for_smpl", "joint_pos_multi_future_wrist_for_smpl_gemx")):
        assert tok[term]["func"].endswith(":" + fn) and tok[term]["params"]["aug"] == "${smpl_gemx_aug}"
        assert callable(getattr(A, fn))
    rw = cfg["manager_env"]["rewards"]["motion_5point_local_pos"]      # walk4's feet reward
    assert rw["weight"] == 1.0 and rw["params"]["std"] == 0.06
    assert cfg["algo"]["config"]["num_learning_iterations"] == 25000
    c = A.merge_cfg(fitted)
    assert set(c["profiles"]) == {"full", "waist_up"}


def test_calibration_reproduces_the_live_target():
    """The fit's final check (augmented training SMPL of the same 8 clips at the live capture times, same estimator):
    legs / pelvis / spine / head within 25 % (or 1.2 deg) of the live GEM-X numbers, time constants within 30 %."""
    T = json.loads(TARGET.read_text())["calibration"]
    for prof, h in T.items():
        tg, fin = h["target_arm_scaled"], h["final"]
        for key in ("bone_noise_deg",):
            for g in ("hip", "thigh", "shin", "foot", "spine", "neck_head"):
                assert abs(fin[key][g] - tg[key][g]) <= max(0.25 * tg[key][g], 1.2), (prof, key, g)
        for g in ("thigh", "shin", "foot", "neck_head"):
            assert abs(fin["bone_bias_deg"][g] - tg["bone_bias_deg"][g]) <= max(0.25 * tg["bone_bias_deg"][g], 1.2), (prof, g)
        for k in range(3):
            assert abs(fin["root_noise_deg"][k] - tg["root_noise_deg"][k]) <= max(0.25 * tg["root_noise_deg"][k], 1.2)
        assert fin["bone_tau_slow"] == pytest.approx(tg["bone_tau_slow"], rel=0.3)
        assert fin["root_tau_slow"] == pytest.approx(tg["root_tau_slow"], rel=0.3)
