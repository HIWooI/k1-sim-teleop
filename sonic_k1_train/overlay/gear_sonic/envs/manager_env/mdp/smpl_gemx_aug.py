"""GEM-X-like corruption of the K1 SONIC SMPL encoder input (k1s_bones2, docs/training.md).

The BONES SMPL encoder (k1s_bones 20000) is trained on clean mocap SMPL; at the event the stream comes from the GEM-X
real-time path (SOMA-77 -> MP_FROM_SOMA -> mp2smpl, k1live/sender_gemx.py) through the sonic_human backend. What the
encoder sees live differs from the training SMPL of the same capture in a few measurable ways, each
modelled here with the parameters fitted from every-frame GEM-X real-time-worker dumps (scripts/gemx_aug_stats.py):

  pose error     every bone direction is off by a small rotation (mp2smpl re-lays directions on fixed SMPL bone
                 lengths, so the error is angular): a per-episode bias + a slow and a fast Ornstein-Uhlenbeck part;
                 the slow part is smoothed once more (noise_smooth_tau): GEM-X's error is smooth, a raw OU path is
                 rough at 20 ms (1 deg per tick at 13 deg / 5 s) and would add jitter the live stream does not have
  pelvis wobble  the root frame (hip line + up) rotates against the body: q_live = q_train * dR, local joints
                 rotate by dR^-1 (the body itself stays where it is) - large on waist-up clips (legs not visible)
  causal smoothing  GEM-X's causal temporal window: the live pose is a lagging low-pass of the motion (high-frequency
                 bone jitter 0.4-0.7 deg live vs 1-3 deg in BONES walking; best lag vs the offline GEM-X -140 ms):
                 a first-order low-pass on every bone direction and the root (lowpass_tau per bone group)
  camera rate    a new pose only every 1/camera_hz (24-30 fps), held in between, then the backend's 30 ms smoothing
  dropouts       GEM-X loses the person (or JumpGate rejects a frame): the backend's PresenceGate holds the last pose
                 (<= hold_s), and on re-detection cross-fades back over quick_s (smoothstep); GEM-X's own error is
                 re-drawn after a re-detection (its causal window restarts)
  wrist          the live sender sends one K1 wrist roll per frame: all n future frames carry the newest value

Only the SMPL encoder's observation terms change (smpl_joints_multi_future_local, smpl_root_ori_b_mf,
joint_pos_multi_future_select_joints_for_smpl): the robot reference, rewards, terminations and the critic keep the
clean motion. The n future frames are a ring buffer of the corrupted stream (frame n-1 = newest), exactly the
backend's 50 Hz buffer; it restarts from the clean window whenever the motion cursor jumps (reset / resample).

Config (params `aug` of the three terms, sonic_k1_small_la4_bones2.yaml): see DEFAULT. Disabled envs (1 - prob, drawn
per episode) see the clean window - bit-identical to the vendor terms.
The core (SmplGemxAug) is plain torch so it is unit-tested without Isaac Lab (tests/test_smpl_gemx_aug.py).
"""
from __future__ import annotations

import copy
import math
from typing import Optional

import torch

# SMPL-24 parents (sonic.smpl.parents24 / mp2smpl.parents): bone b = joint b+1 from its parent
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
N_J = 24

# Bone groups (index = child joint) for the fitted angular errors; values are per group, expanded per bone.
BONE_GROUPS = {
    "hip": [1, 2],               # pelvis -> hips (the hip line)
    "thigh": [4, 5],
    "shin": [7, 8],
    "foot": [10, 11],
    "spine": [3, 6, 9],
    "neck_head": [12, 15],
    "collar": [13, 14],
    "upper_arm": [16, 17, 18, 19],   # collar -> shoulder, shoulder -> elbow
    "forearm": [20, 21],
    "hand": [22, 23],
}

DEFAULT = {
    "enable": True,
    "prob": 0.8,                 # share of episodes that see the GEM-X stream model (the rest: clean reference)
    "dt": 0.02,
    "camera_hz": [24.0, 30.0],   # per-episode camera / worker rate (U); new pose when the phase wraps
    "smooth_tau": 0.03,          # sonic_human SMOOTH_TAU
    "hold_s": 0.5,               # PresenceGate hold_s (longer losses fade to idle -> robot encoder: not modelled)
    "quick_s": 0.25,             # PresenceGate quick_s (HOLD -> TRACK cross-fade)
    "wrist_newest": True,        # all future frames carry the newest wrist roll (sonic_human.smpl_motion)
    "lowpass_tau": {"hip": 0.15, "thigh": 0.15, "shin": 0.15, "foot": 0.15, "spine": 0.1, "neck_head": 0.1,
                    "collar": 0.1, "upper_arm": 0.1, "forearm": 0.1, "hand": 0.1},   # s, 0 = off
    "root_lowpass_tau": 0.1,
    "noise_smooth_tau": 0.3,     # s, first-order smoothing of the slow OU error (bones and root)
    "profiles": {
        # name: weight (share of augmented episodes) + error model; angles in degrees, times in seconds
        "full": {
            "weight": 0.75,
            "bone_bias_deg": {"hip": 4.0, "thigh": 4.0, "shin": 4.0, "foot": 6.0, "spine": 3.0, "neck_head": 4.0,
                              "collar": 3.0, "upper_arm": 5.0, "forearm": 6.0, "hand": 8.0},
            "bone_slow_deg": {"hip": 3.0, "thigh": 3.0, "shin": 3.0, "foot": 5.0, "spine": 2.0, "neck_head": 3.0,
                              "collar": 2.0, "upper_arm": 4.0, "forearm": 5.0, "hand": 6.0},
            "bone_fast_deg": {"hip": 1.0, "thigh": 1.0, "shin": 1.0, "foot": 2.0, "spine": 0.5, "neck_head": 1.0,
                              "collar": 0.5, "upper_arm": 1.0, "forearm": 1.5, "hand": 2.0},
            "tau_slow": 0.6,
            "tau_fast": 0.05,
            "root_bias_deg": [2.0, 2.0, 2.0],    # dR as rotation vector in the root frame: roll, pitch, yaw
            "root_slow_deg": [2.0, 2.0, 2.0],
            "root_fast_deg": [0.5, 0.5, 0.5],
            "root_tau_slow": 0.6,
            "root_tau_fast": 0.05,
            "gap_per_min": 2.0,                  # dropouts per minute of tracking
            "gap_logn_mu": -2.0,                 # log(gap seconds) ~ N(mu, sd), clipped to [gap_min_s, hold_s]
            "gap_logn_sd": 0.8,
            "gap_min_s": 0.02,
            "redetect_resample": 1.0,            # P(slow error re-drawn at a re-detection)
        },
        "waist_up": {
            "weight": 0.25,
            "bone_bias_deg": {"hip": 8.0, "thigh": 10.0, "shin": 10.0, "foot": 12.0, "spine": 4.0, "neck_head": 5.0,
                              "collar": 4.0, "upper_arm": 6.0, "forearm": 8.0, "hand": 10.0},
            "bone_slow_deg": {"hip": 6.0, "thigh": 8.0, "shin": 8.0, "foot": 10.0, "spine": 3.0, "neck_head": 4.0,
                              "collar": 3.0, "upper_arm": 5.0, "forearm": 6.0, "hand": 8.0},
            "bone_fast_deg": {"hip": 2.0, "thigh": 2.0, "shin": 2.0, "foot": 3.0, "spine": 1.0, "neck_head": 1.0,
                              "collar": 1.0, "upper_arm": 1.5, "forearm": 2.0, "hand": 3.0},
            "tau_slow": 0.6,
            "tau_fast": 0.05,
            "root_bias_deg": [6.0, 8.0, 3.0],
            "root_slow_deg": [4.0, 6.0, 3.0],
            "root_fast_deg": [1.0, 1.0, 1.0],
            "root_tau_slow": 0.6,
            "root_tau_fast": 0.05,
            "gap_per_min": 10.0,
            "gap_logn_mu": -1.6,
            "gap_logn_sd": 0.9,
            "gap_min_s": 0.02,
            "redetect_resample": 1.0,
        },
    },
}


def merge_cfg(over: Optional[dict]) -> dict:
    """DEFAULT updated by `over` (nested dicts merged). A `profiles` block in `over` is the profile set: each profile it
    names is its DEFAULT namesake (or DEFAULT "full") updated by the given keys; DEFAULT profiles it omits are dropped."""
    cfg = copy.deepcopy(DEFAULT)

    def upd(dst, src):
        for k, v in src.items():
            v = _plain(v)
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                upd(dst[k], v)
            else:
                dst[k] = v

    over = _plain(over) if over else {}
    profiles = over.pop("profiles", None) if over else None
    upd(cfg, over)
    if profiles:
        out = {}
        for name, pr in profiles.items():
            base = copy.deepcopy(DEFAULT["profiles"].get(name, DEFAULT["profiles"]["full"]))
            upd(base, pr)
            out[name] = base
        cfg["profiles"] = out
    return cfg


def _plain(v):
    """OmegaConf containers (hydra params) -> python dict / list."""
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf

        if isinstance(v, (DictConfig, ListConfig)):
            return OmegaConf.to_container(v, resolve=True)
    except ImportError:
        pass
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    return v


# ------------------------------------------------------------------------------------------------ quaternions (wxyz)
def quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw * bw - ax * bx - ay * by - az * bz,
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw), -1)


def quat_conj(q):
    return torch.cat((q[..., :1], -q[..., 1:]), -1)


def quat_apply(q, v):
    """Rotate v (..., 3) by unit q (..., 4) (broadcast)."""
    w, xyz = q[..., :1], q[..., 1:]
    t = 2.0 * torch.cross(xyz.expand_as(v), v, dim=-1)
    return v + w * t + torch.cross(xyz.expand_as(v), t, dim=-1)


def quat_from_rotvec(r):
    ang = torch.linalg.norm(r, dim=-1, keepdim=True)
    half = 0.5 * ang
    k = torch.where(ang > 1e-8, torch.sin(half) / ang.clamp_min(1e-8), 0.5 - ang * ang / 48.0)
    return torch.cat((torch.cos(half), r * k), -1)


def quat_to_rotvec(q):
    q = torch.where(q[..., :1] < 0, -q, q)
    s = torch.linalg.norm(q[..., 1:], dim=-1, keepdim=True)
    ang = 2.0 * torch.atan2(s, q[..., :1])
    k = torch.where(s > 1e-8, ang / s.clamp_min(1e-8), 2.0 / q[..., :1].clamp_min(1e-8))
    return q[..., 1:] * k


def quat_normalize(q):
    return q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(1e-9)


def nlerp(a, b, w):
    """Normalised lerp along the short arc (w broadcast over the last dim)."""
    b = torch.where((a * b).sum(-1, keepdim=True) < 0, -b, b)
    return quat_normalize(a + w * (b - a))


def matrix_from_quat(q):
    w, x, y, z = q.unbind(-1)
    two_s = 2.0 / (q * q).sum(-1)
    o = torch.stack((1 - two_s * (y * y + z * z), two_s * (x * y - z * w), two_s * (x * z + y * w),
                     two_s * (x * y + z * w), 1 - two_s * (x * x + z * z), two_s * (y * z - x * w),
                     two_s * (x * z - y * w), two_s * (y * z + x * w), 1 - two_s * (x * x + y * y)), -1)
    return o.reshape(q.shape[:-1] + (3, 3))


def rotate_dirs(d, r):
    """Rotate unit directions d (..., 3) by rotation vectors r (..., 3) (Rodrigues)."""
    ang = torch.linalg.norm(r, dim=-1, keepdim=True)
    k = r / ang.clamp_min(1e-9)
    c, s = torch.cos(ang), torch.sin(ang)
    return d * c + torch.cross(k, d, dim=-1) * s + k * (k * d).sum(-1, keepdim=True) * (1 - c)


def smoothstep(x):
    x = x.clamp(0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def bone_angles(J, parents=PARENTS):
    """(..., 24, 3) joints -> (..., 23) unit bone directions (..., 23, 3) and lengths (..., 23, 1)."""
    child = torch.arange(1, N_J, device=J.device)
    par = torch.tensor(parents[1:], device=J.device)
    v = J[..., child, :] - J[..., par, :]
    ln = torch.linalg.norm(v, dim=-1, keepdim=True)
    return v / ln.clamp_min(1e-9), ln


def rechain(J0, dirs, lens, parents=PARENTS):
    """Joints from the pelvis J0 (..., 3) along bone directions / lengths (child order is topological)."""
    out = [J0]
    for i in range(1, N_J):
        out.append(out[parents[i]] + dirs[..., i - 1, :] * lens[..., i - 1, :])
    return torch.stack(out, -2)


# ------------------------------------------------------------------------------------------------ the augmenter
class SmplGemxAug:
    """Per-env GEM-X stream model on the SMPL encoder's n future frames.

    step(J, q, jp, cont): J (N, F, 24, 3) clean root-local joints, q (N, F, 4) clean root quats (Z-up world, wxyz),
    jp (N, F, D) the robot joint positions the wrist term reads, cont (N,) bool = the window moved on by exactly one
    frame since the last call (else the env restarts from the clean window). Returns the corrupted (J, q, jp)."""

    def __init__(self, num_envs: int, num_frames: int, cfg: Optional[dict] = None, device="cpu",
                 generator: Optional[torch.Generator] = None):
        self.cfg = merge_cfg(cfg)
        c = self.cfg
        self.N, self.F, self.device = int(num_envs), int(num_frames), torch.device(device)
        self.g = generator
        self.dt = float(c["dt"])
        names = list(c["profiles"])
        self.profile_names = names
        P = len(names)
        dev = self.device

        def per_bone(d):
            out = torch.zeros(N_J - 1)
            for grp, joints in BONE_GROUPS.items():
                for j in joints:
                    out[j - 1] = math.radians(float(d.get(grp, 0.0)))
            return out

        pr = [c["profiles"][n] for n in names]
        w = torch.tensor([float(p.get("weight", 1.0)) for p in pr])
        self.p_weight = (w / w.sum()).to(dev)
        self.p_bias = torch.stack([per_bone(p["bone_bias_deg"]) for p in pr]).to(dev)       # (P, 23)
        self.p_slow = torch.stack([per_bone(p["bone_slow_deg"]) for p in pr]).to(dev)
        self.p_fast = torch.stack([per_bone(p["bone_fast_deg"]) for p in pr]).to(dev)
        self.p_tau_s = torch.tensor([float(p["tau_slow"]) for p in pr], device=dev)
        self.p_tau_f = torch.tensor([float(p["tau_fast"]) for p in pr], device=dev)
        rad = lambda k: torch.tensor([[math.radians(float(x)) for x in p[k]] for p in pr], device=dev)  # (P, 3)
        self.p_rbias, self.p_rslow, self.p_rfast = rad("root_bias_deg"), rad("root_slow_deg"), rad("root_fast_deg")
        self.p_rtau_s = torch.tensor([float(p["root_tau_slow"]) for p in pr], device=dev)
        self.p_rtau_f = torch.tensor([float(p["root_tau_fast"]) for p in pr], device=dev)
        self.p_gap_rate = torch.tensor([float(p["gap_per_min"]) / 60.0 for p in pr], device=dev)    # per s
        self.p_gap_mu = torch.tensor([float(p["gap_logn_mu"]) for p in pr], device=dev)
        self.p_gap_sd = torch.tensor([float(p["gap_logn_sd"]) for p in pr], device=dev)
        self.p_gap_min = torch.tensor([float(p.get("gap_min_s", self.dt)) for p in pr], device=dev)
        self.p_resample = torch.tensor([float(p.get("redetect_resample", 1.0)) for p in pr], device=dev)
        self.hold_ticks = max(1, int(round(float(c["hold_s"]) / self.dt)))
        self.quick_ticks = max(1, int(round(float(c["quick_s"]) / self.dt)))
        self.alpha = min(1.0, self.dt / max(float(c["smooth_tau"]), 1e-6))
        hz = c["camera_hz"]
        self.hz_lo, self.hz_hi = (float(hz), float(hz)) if isinstance(hz, (int, float)) else (float(hz[0]), float(hz[1]))
        self.enabled = bool(c["enable"])
        lp = c.get("lowpass_tau") or {}
        tau_b = torch.zeros(N_J - 1)
        for grp, joints in BONE_GROUPS.items():
            for j in joints:
                tau_b[j - 1] = float(lp.get(grp, 0.0))
        self.lp_beta = torch.where(tau_b > 0, (self.dt / tau_b.clamp_min(1e-6)).clamp(max=1.0), torch.ones_like(tau_b))
        self.lp_beta = self.lp_beta.to(dev)[None, :, None]                                  # (1, 23, 1)
        rt = float(c.get("root_lowpass_tau") or 0.0)
        self.lp_beta_root = min(1.0, self.dt / rt) if rt > 0 else 1.0
        ns = float(c.get("noise_smooth_tau") or 0.0)
        self.ns_beta = min(1.0, self.dt / ns) if ns > 0 else 1.0
        self._alloc()

    # ---------------------------------------------------------------------------------------- state
    def _alloc(self):
        N, F, dev = self.N, self.F, self.device
        z = lambda *s: torch.zeros(*s, device=dev)
        self.active = torch.zeros(N, dtype=torch.bool, device=dev)
        self.prof = torch.zeros(N, dtype=torch.long, device=dev)
        self.b_bone, self.n_slow, self.n_fast = z(N, N_J - 1, 3), z(N, N_J - 1, 3), z(N, N_J - 1, 3)
        self.b_root, self.r_slow, self.r_fast = z(N, 3), z(N, 3), z(N, 3)
        self.n_slow_s, self.r_slow_s = z(N, N_J - 1, 3), z(N, 3)   # the smoothed slow parts (what is applied)
        self.hz = z(N)
        self.phase = z(N)
        self.gap_left = torch.zeros(N, dtype=torch.long, device=dev)
        self.fade_k = torch.full((N,), 10 ** 6, dtype=torch.long, device=dev)   # ticks since the last re-detection
        self.lp_d, self.lp_q = z(N, N_J - 1, 3), z(N, 4)        # causal low-pass of the clean world bone dirs / root
        self.cam_J, self.cam_q = z(N, N_J, 3), z(N, 4)          # newest camera pose (held between camera frames)
        self.cam_jp = None
        self.from_J, self.from_q = z(N, N_J, 3), z(N, 4)        # gate output when the re-detection started
        self.from_jp = None
        self.cur_J, self.cur_q = z(N, N_J, 3), z(N, 4)          # smoothed output (newest buffer frame)
        self.cur_jp = None
        self.buf_J, self.buf_q, self.buf_jp = z(N, self.F, N_J, 3), z(N, self.F, 4), None
        # bookkeeping for the statistics / tests
        self.in_gap = torch.zeros(N, dtype=torch.bool, device=dev)
        self.new_cam = torch.zeros(N, dtype=torch.bool, device=dev)
        self.n_gaps = torch.zeros(N, dtype=torch.long, device=dev)
        self.started = False

    def _randn(self, *s):
        return torch.randn(*s, device=self.device, generator=self.g)

    def _rand(self, *s):
        return torch.rand(*s, device=self.device, generator=self.g)

    def _sigmas(self, ids):
        p = self.prof[ids]
        # per-component sd of an isotropic 3D rotation vector whose part across the bone has RMS angle = the fitted value
        k = 1.0 / math.sqrt(2.0)
        return (self.p_bias[p] * k, self.p_slow[p] * k, self.p_fast[p] * k,
                self.p_rbias[p], self.p_rslow[p], self.p_rfast[p])

    def _restart(self, ids, J, q, jp):
        """Envs `ids` start a new stream: new episode draws, OU parts from their stationary law, buffer = clean."""
        n = len(ids)
        if n == 0:
            return
        self.active[ids] = self._rand(n) < float(self.cfg["prob"]) if self.enabled else False
        self.prof[ids] = torch.multinomial(self.p_weight, n, replacement=True, generator=self.g)
        sb, ss, sf, rb, rs, rf = self._sigmas(ids)
        self.b_bone[ids] = self._randn(n, N_J - 1, 3) * sb[..., None]
        self.n_slow[ids] = self._randn(n, N_J - 1, 3) * ss[..., None]
        self.n_fast[ids] = self._randn(n, N_J - 1, 3) * sf[..., None]
        self.b_root[ids] = self._randn(n, 3) * rb
        self.r_slow[ids] = self._randn(n, 3) * rs
        self.r_fast[ids] = self._randn(n, 3) * rf
        self.n_slow_s[ids], self.r_slow_s[ids] = self.n_slow[ids], self.r_slow[ids]
        self.hz[ids] = self.hz_lo + (self.hz_hi - self.hz_lo) * self._rand(n)
        self.phase[ids] = self._rand(n)
        self.gap_left[ids] = 0
        self.fade_k[ids] = 10 ** 6
        self.in_gap[ids] = False
        a = self.active[ids]
        self.lp_d[ids] = quat_apply(q[:, -1, None, :], bone_angles(J[:, -1])[0])
        self.lp_q[ids] = q[:, -1]
        Jw, qw = J.clone(), q.clone()
        for f in range(self.F):                     # the whole window with the episode's error state
            Jf, qf = self._corrupt(ids, J[:, f], q[:, f])
            Jw[:, f] = torch.where(a[:, None, None], Jf, J[:, f])
            qw[:, f] = torch.where(a[:, None], qf, q[:, f])
        self.cam_J[ids], self.cam_q[ids] = Jw[:, -1], qw[:, -1]
        self.cur_J[ids], self.cur_q[ids] = Jw[:, -1], qw[:, -1]
        self.from_J[ids], self.from_q[ids] = Jw[:, -1], qw[:, -1]
        self.buf_J[ids], self.buf_q[ids] = Jw, qw
        if jp is not None:
            self.cam_jp[ids] = jp[:, -1]
            self.cur_jp[ids] = jp[:, -1]
            self.from_jp[ids] = jp[:, -1]
            self.buf_jp[ids] = jp

    def _corrupt(self, ids, J, q, d=None):
        """One clean frame (n, 24, 3) local / (n, 4) world -> the GEM-X-like frame with the current error state
        (d: bone directions to use instead of J's, e.g. the low-passed ones; bone lengths always from J)."""
        d0, ln = bone_angles(J)
        d = d0 if d is None else d
        d = rotate_dirs(d, self.b_bone[ids] + self.n_slow_s[ids] + self.n_fast[ids])
        Jb = rechain(J[:, 0], d, ln)
        dq = quat_from_rotvec(self.b_root[ids] + self.r_slow_s[ids] + self.r_fast[ids])   # root frame
        Jl = quat_apply(quat_conj(dq)[:, None, :], Jb - Jb[:, :1]) + Jb[:, :1]
        return Jl, quat_normalize(quat_mul(q, dq))

    def _ou(self, x, sd, tau):
        """Exact OU step over dt: x (n, ..., 3), sd (n, ...) stationary per-component sd, tau (n,) s."""
        rho = torch.exp(-self.dt / tau.clamp_min(1e-4)).view((-1,) + (1,) * (x.dim() - 1))
        sd = sd.view(sd.shape + (1,) * (x.dim() - sd.dim()))
        return rho * x + torch.sqrt(1 - rho * rho) * sd * self._randn(*x.shape)

    # ---------------------------------------------------------------------------------------- one step
    @torch.no_grad()
    def step(self, J, q, jp=None, cont=None):
        N = self.N
        if jp is not None and self.cam_jp is None:
            D = jp.shape[-1]
            z = lambda *s: torch.zeros(*s, device=self.device)
            self.cam_jp, self.from_jp, self.cur_jp, self.buf_jp = z(N, D), z(N, D), z(N, D), z(N, self.F, D)
        if cont is None or not self.started:
            cont = torch.zeros(N, dtype=torch.bool, device=self.device)
        self.started = True
        rs = torch.nonzero(~cont).flatten()
        self._restart(rs, J[rs], q[rs], None if jp is None else jp[rs])
        ids = torch.nonzero(cont & self.active).flatten()
        if len(ids):
            self._advance(ids, J[ids, -1], q[ids, -1], None if jp is None else jp[ids, -1])
        # augmented envs read the buffer (restarted ones: their corrupted window), the others the clean window
        clean = ~self.active
        Jo = torch.where(clean[:, None, None, None], J, self.buf_J)
        qo = torch.where(clean[:, None, None], q, self.buf_q)
        if jp is None:
            return Jo, qo, None
        jpo = torch.where(clean[:, None, None], jp, self.buf_jp)
        if self.cfg.get("wrist_newest", True):
            jpo = torch.where(self.active[:, None, None], self.buf_jp[:, -1:].expand_as(jpo), jpo)
        return Jo, qo, jpo

    def _advance(self, ids, J, q, jp):
        n = len(ids)
        p = self.prof[ids]
        _, ss, sf, _, rs, rf = self._sigmas(ids)
        # error processes (continuous time, 50 Hz)
        self.n_slow[ids] = self._ou(self.n_slow[ids], ss, self.p_tau_s[p])
        self.n_fast[ids] = self._ou(self.n_fast[ids], sf, self.p_tau_f[p])
        self.r_slow[ids] = self._ou(self.r_slow[ids], rs, self.p_rtau_s[p])
        self.r_fast[ids] = self._ou(self.r_fast[ids], rf, self.p_rtau_f[p])
        b = self.ns_beta
        self.n_slow_s[ids] = self.n_slow_s[ids] + b * (self.n_slow[ids] - self.n_slow_s[ids])
        self.r_slow_s[ids] = self.r_slow_s[ids] + b * (self.r_slow[ids] - self.r_slow_s[ids])
        # GEM-X's causal smoothing of the motion itself: the body (bone directions in the world-aligned frame) and the
        # root frame lag together, so a lagging but otherwise exact pose reads as a lag, not as a bone error
        lpw = self.lp_d[ids] + self.lp_beta * (quat_apply(q[:, None, :], bone_angles(J)[0]) - self.lp_d[ids])
        lpw = lpw / torch.linalg.norm(lpw, dim=-1, keepdim=True).clamp_min(1e-9)
        self.lp_d[ids] = lpw
        lq = nlerp(self.lp_q[ids], q, torch.full((n, 1), self.lp_beta_root, device=self.device))
        self.lp_q[ids] = lq
        lp = quat_apply(quat_conj(lq)[:, None, :], lpw)                    # low-passed dirs in the low-passed root
        # dropouts: start (Poisson while tracked and not re-fading), count down
        gl = self.gap_left[ids]
        tracking = gl == 0
        start = tracking & (self._rand(n) < self.p_gap_rate[p] * self.dt)
        L = torch.exp(self.p_gap_mu[p] + self.p_gap_sd[p] * self._randn(n))
        L = torch.minimum(torch.maximum(L, self.p_gap_min[p]), torch.full_like(L, self.hold_ticks * self.dt))
        Lt = torch.clamp(torch.round(L / self.dt).long(), 1, self.hold_ticks)
        gl = torch.where(start, Lt, gl)
        in_gap = gl > 0
        gl_next = torch.clamp(gl - 1, min=0)
        self.gap_left[ids] = gl_next
        self.n_gaps[ids] += start.long()
        # camera frames (only while the person is delivered)
        ph = self.phase[ids] + self.hz[ids] * self.dt
        new = ph >= 1.0
        self.phase[ids] = ph - new.float()
        new = new & ~in_gap
        was_gap = self.in_gap[ids] & ~in_gap        # first delivered tick after a gap: a fresh detection
        new = new | was_gap
        sel = ids[new]
        if len(sel):
            Jc, qc = self._corrupt(sel, J[new], lq[new], lp[new])
            self.cam_J[sel], self.cam_q[sel] = Jc, qc
            if jp is not None:
                self.cam_jp[sel] = jp[new]
        # re-detection: GEM-X restarts its window (slow error re-drawn), the gate cross-fades from what it streamed
        rd = ids[was_gap]
        if len(rd):
            pr = self.p_resample[self.prof[rd]]
            m = self._rand(len(rd)) < pr
            if m.any():
                r2 = rd[m]
                _, ss2, _, _, rs2, _ = self._sigmas(r2)
                self.n_slow[r2] = self._randn(len(r2), N_J - 1, 3) * ss2[..., None]
                self.r_slow[r2] = self._randn(len(r2), 3) * rs2
                self.n_slow_s[r2], self.r_slow_s[r2] = self.n_slow[r2], self.r_slow[r2]
                Jc, qc = self._corrupt(r2, J[was_gap][m], lq[was_gap][m], lp[was_gap][m])
                self.cam_J[r2], self.cam_q[r2] = Jc, qc
            self.fade_k[rd] = 0
        self.in_gap[ids] = in_gap
        self.new_cam[:] = False
        self.new_cam[sel] = True
        # PresenceGate: HOLD -> target = the held camera pose (cam_* is not updated in a gap); TRACK after a gap:
        # cross-fade from the output at the switch over quick_s
        fk = self.fade_k[ids]
        w = smoothstep(fk.float() / self.quick_ticks)
        tgt_J = self.from_J[ids] + w[:, None, None] * (self.cam_J[ids] - self.from_J[ids])
        tgt_q = nlerp(self.from_q[ids], self.cam_q[ids], w[:, None])
        # first-order smoothing (sonic_human SMOOTH_TAU)
        a = self.alpha
        cur_J = self.cur_J[ids] + a * (tgt_J - self.cur_J[ids])
        cur_q = nlerp(self.cur_q[ids], tgt_q, torch.full((n, 1), a, device=self.device))
        self.cur_J[ids], self.cur_q[ids] = cur_J, cur_q
        # while tracking without a fade, "from" follows the gate output (a later re-detection fades from here)
        steady = (fk >= self.quick_ticks) | in_gap
        st = ids[steady]
        self.from_J[st] = torch.where(in_gap[steady][:, None, None], self.cam_J[st], tgt_J[steady])
        self.from_q[st] = torch.where(in_gap[steady][:, None], self.cam_q[st], tgt_q[steady])
        self.fade_k[ids] = torch.where(in_gap, fk, fk + 1)
        self.buf_J[ids] = torch.cat((self.buf_J[ids, 1:], cur_J[:, None]), 1)
        self.buf_q[ids] = torch.cat((self.buf_q[ids, 1:], cur_q[:, None]), 1)
        if jp is not None:
            tgt_jp = self.from_jp[ids] + w[:, None] * (self.cam_jp[ids] - self.from_jp[ids])
            cur_jp = self.cur_jp[ids] + a * (tgt_jp - self.cur_jp[ids])
            self.cur_jp[ids] = cur_jp
            self.from_jp[st] = torch.where(in_gap[steady][:, None], self.cam_jp[st], tgt_jp[steady])
            self.buf_jp[ids] = torch.cat((self.buf_jp[ids, 1:], cur_jp[:, None]), 1)


# ------------------------------------------------------------------------------------------------ Isaac Lab terms
def _state(env, command_name: str, aug) -> dict:
    """The env's augmenter, advanced once per env step (all three terms read the same corrupted window)."""
    cmd = env.command_manager.get_term(command_name)
    st = getattr(env, "_smpl_gemx_aug", None)
    # one advance per env step: common_step_counter + the motion cursor (an env.reset() between steps re-advances)
    key = (int(getattr(env, "common_step_counter", 0)), int(cmd.time_steps.sum()), int(cmd.motion_ids.sum()))
    if st is not None and st["key"] == key:
        return st
    N, F = env.num_envs, cmd.smpl_num_future_frames
    if st is None:
        st = {"aug": SmplGemxAug(N, F, aug, device=env.device), "key": None, "prev_mid": None, "prev_t0": None}
        env._smpl_gemx_aug = st
        a = st["aug"]
        print(f"[smpl_gemx_aug] {N} envs x {F} frames, enabled={a.enabled} prob={a.cfg['prob']} profiles="
              f"{dict(zip(a.profile_names, [round(float(w), 2) for w in a.p_weight]))} hold={a.hold_ticks} ticks "
              f"quick={a.quick_ticks} ticks camera={a.hz_lo}-{a.hz_hi} Hz", flush=True)
    J_w = cmd.smpl_joints_multi_future                                     # (N, F, 24, 3), world-rotated
    q = cmd.smpl_root_quat_w_multi_future                                  # (N, F, 4) Z-up, base removed
    J = quat_apply(quat_conj(q)[..., None, :], J_w)                        # = vendor smpl_joints_multi_future_local
    jp = cmd.joint_pos_multi_future_for_smpl.view(N, F, -1)
    mid = cmd.motion_ids.clone()
    t0 = (cmd.time_steps + cmd.motion_start_time_steps).clone()
    if st["prev_mid"] is None:
        cont = torch.zeros(N, dtype=torch.bool, device=env.device)
    else:
        cont = (mid == st["prev_mid"]) & (t0 == st["prev_t0"] + 1)
        elb = getattr(env, "episode_length_buf", None)
        if elb is not None:
            cont &= elb > 0
    Jo, qo, jpo = st["aug"].step(J, q, jp, cont)
    st.update(J=Jo, q=qo, jp=jpo, key=key, prev_mid=mid, prev_t0=t0)
    return st


def smpl_joints_multi_future_local_gemx(env, command_name: str, non_flatten=False, joints_idx=None, aug=None):
    """vendor smpl_joints_multi_future_local on the GEM-X-like stream."""
    st = _state(env, command_name, aug)
    J = st["J"]
    if joints_idx is not None:
        J = J[..., joints_idx, :]
    cmd = env.command_manager.get_term(command_name)
    if non_flatten:
        return J.reshape(env.num_envs, cmd.smpl_num_future_frames, -1)
    return J.reshape(env.num_envs, -1)


def smpl_root_ori_b_mf_gemx(env, command_name: str, non_flatten=True, aug=None):
    """vendor smpl_root_ori_b_mf (quat_inv(robot anchor) * smpl root, 6D = first two columns) on the stream."""
    st = _state(env, command_name, aug)
    cmd = env.command_manager.get_term(command_name)
    q = st["q"]
    F = q.shape[1]
    anchor = cmd.robot_anchor_quat_w.view(env.num_envs, 1, 4).expand(-1, F, -1)
    mat = matrix_from_quat(quat_mul(quat_conj(anchor), q))
    rot = mat[..., :2].reshape(env.num_envs, -1)
    if non_flatten:
        return rot.reshape(env.num_envs, cmd.smpl_num_future_frames, -1)
    return rot


def joint_pos_multi_future_wrist_for_smpl_gemx(env, command_name: str, joints_idx: list, aug=None):
    """vendor joint_pos_multi_future_select_joints_for_smpl on the stream (live: newest wrist roll in every frame)."""
    st = _state(env, command_name, aug)
    return st["jp"][..., joints_idx]
