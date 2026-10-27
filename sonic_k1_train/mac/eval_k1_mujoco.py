"""K1 SONIC checkpoint (ONNX exported on the GPU server) in the Mac MuJoCo loop: stand, then event gestures.

    .venv/bin/python sonic_k1_train/mac/eval_k1_mujoco.py --export models/k1/<run>/exported --step 4000 \
        --motions <built k1_motions dir> --stand 20 --clips bow wave_hello welcome_open_arms clap point_guide \
        --video videos/k1_sonic_<run>.mp4 --json logs/k1_sonic_<run>.json

Uses the robot-parameterised deploy port (sonic.deploy.SonicController + sonic.sim.G1Sim with
sonic.robots.get_robot("k1"): 200 Hz MuJoCo PD with the K1 training gains / armature / torque limits,
INIT ramp on the elastic band, 50 Hz policy). Only the observation layout is specific to our export:
gear_sonic's export_universal_token_encoders_as_onnx takes [encoder_index | tokenizer obs in TokenizerCfg order]
(not the released deploy's encoder_mode_4 layout; the tokenizer block itself starts with the 3-dim encoder one-hot),
and the decoder takes [token | proprioception history].
prepare_variant() writes that layout as an observation_config.yaml; this script adds the two registry terms
the released layout does not have (encoder_index, and the 4 x 20 ms aliases of the lookahead variant).

Scenario per run: INIT 3 s on the band (--start band: hanging from the 1.0 m G1 anchor, ~20 cm drop at release;
--start ground: feet already on the floor) -> CONTROL -> band released after 2 s -> stand on the default-pose
reference for --stand s -> each clip in turn (2 s standing between clips) -> 3 s standing.
Metrics per segment: fall (pelvis z < 0.45 m or tilt > 45 deg), min pelvis z, max tilt, mean |q - q_ref|
(deg; all / legs / arms), root orientation error after the deploy heading alignment.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from sonic import rot  # noqa: E402
from sonic.deploy import SonicController  # noqa: E402
from sonic.motion import load_robot_pkl, standing_motion  # noqa: E402
from sonic.params import CONTROL_DT  # noqa: E402
from sonic.robots import get_robot  # noqa: E402
from sonic.runner import EpisodeLog, control_tick, place_on_ground  # noqa: E402
from sonic.sim import G1Sim, Recorder  # noqa: E402

FALL_Z, FALL_TILT = 0.45, 45.0
HIS = ["his_base_angular_velocity_10frame_step1", "his_body_joint_positions_10frame_step1",
       "his_body_joint_velocities_10frame_step1", "his_last_actions_10frame_step1", "his_gravity_dir_10frame_step1"]


def obs_config(n_future: int, step: int) -> dict:
    """Deploy observation_config for a gear_sonic export of sonic_k1_small(_la4)."""
    sfx = "" if (n_future, step) == (10, 5) else f"_{n_future}f{step}"
    # exporter input = [scalar encoder index | tokenizer obs]; the tokenizer block starts with the 3-dim one-hot
    enc = ["encoder_index", "encoder_index_onehot", f"command_multi_future_nonflat{sfx}", "motion_anchor_ori_b",
           f"motion_anchor_ori_b_mf_nonflat{sfx}", f"command_multi_future_lower_body{sfx}",
           "vr_3point_local_target", "vr_3point_local_orn_target", "smpl_joints_multi_future_local_nonflat",
           "smpl_root_ori_b_multi_future", "joint_pos_multi_future_wrist_for_smpl"]
    return {
        "observations": [{"name": "token_state"}] + [{"name": n} for n in HIS],
        "encoder": {
            "dimension": 64,
            "encoder_observations": [{"name": n} for n in enc],
            "encoder_modes": [{"name": "g1", "mode_id": 0, "required_observations": enc[:3] + enc[4:5]}],
        },
        "k1": {"num_future_frames": n_future, "step": step},
    }


def prepare_variant(export_dir: Path, step_id: int, out: Path, n_future: int, step: int) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    stem = f"model_step_{step_id:06d}"
    shutil.copy(export_dir / f"{stem}_encoder.onnx", out / "model_encoder.onnx")
    shutil.copy(export_dir / f"{stem}_decoder.onnx", out / "model_decoder.onnx")
    (out / "observation_config.yaml").write_text(yaml.safe_dump(obs_config(n_future, step), sort_keys=False))
    return out


class K1Controller(SonicController):
    """SonicController + the registry terms of our export layout (encoder_index, n-frame aliases)."""

    def _build_registry(self):
        R = super()._build_registry()
        R["encoder_index"] = (1, lambda: np.zeros(1))  # encoders_to_iterate = [g1, teleop, smpl] -> g1 = 0
        R["encoder_index_onehot"] = (3, lambda: np.array([1.0, 0.0, 0.0]))
        k1cfg = {}
        cfgp = getattr(self, "_cfg_path", None)
        if cfgp is not None:
            k1cfg = yaml.safe_load(Path(cfgp).read_text()).get("k1", {})
        n, s = k1cfg.get("num_future_frames", 10), k1cfg.get("step", 5)
        if (n, s) != (10, 5):
            sfx = f"_{n}f{s}"
            N = self.robot.num_dof
            L = list(self.robot.lower_body_il)
            R[f"command_multi_future_nonflat{sfx}"] = (
                2 * n * N, lambda: np.concatenate([self._motion_joint_pos(n, s), self._motion_joint_vel(n, s)]))
            R[f"motion_anchor_ori_b_mf_nonflat{sfx}"] = (6 * n, lambda: self._anchor_orientation(n, s))
            R[f"command_multi_future_lower_body{sfx}"] = (
                2 * n * len(L), lambda: np.concatenate([self._motion_joint_pos(n, s, L),
                                                        self._motion_joint_vel(n, s, L)]))
        return R

    @classmethod
    def load(cls, vdir: Path, robot):
        obj = cls.__new__(cls)
        obj._cfg_path = vdir / "observation_config.yaml"
        cls.__init__(obj, vdir.name, models_dir=vdir.parent, robot=robot)
        return obj


class Ghost:
    """Kinematic reference on a second MjData, rendered with the same tracking camera."""

    def __init__(self, sim, robot, **cam):
        import mujoco

        self.mj, self.sim, self.robot = mujoco, sim, robot
        self.data = mujoco.MjData(sim.model)
        self.rec = Recorder(sim, **cam)
        self.rec.sim = type("S", (), {"data": self.data})()

    def render(self, motion, f, adh, xy):
        d = self.data
        d.qpos[:] = 0
        d.qpos[0:2] = xy
        d.qpos[2] = motion.body_pos[f, 0, 2]
        d.qpos[3:7] = rot.quat_mul(adh, motion.body_quat[f, 0])
        d.qpos[self.sim.qadr] = motion.joint_pos[f][self.robot.isaaclab_to_mujoco]
        self.mj.mj_forward(self.sim.model, d)
        self.rec.renderer.update_scene(d, camera=self.rec.cam)
        return self.rec.renderer.render().copy()


def caption(img, text):
    import cv2

    img = img.copy()
    cv2.putText(img, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def seg_metrics(log: EpisodeLog, robot, a, b):
    A = log.arrays()
    sl = slice(a, b)
    if b - a < 5:
        return {}
    q, r = A["q_isaac"][sl], A["ref_isaac"][sl]
    err = np.degrees(np.abs(q - r))
    names = robot.isaaclab_joint_names
    legs = [i for i, n in enumerate(names) if any(t in n for t in ("hip", "knee", "ankle"))]
    arms = [i for i, n in enumerate(names) if any(t in n for t in ("shoulder", "elbow", "wrist"))]
    dq = rot.quat_mul(rot.quat_conj(A["base_quat"][sl]), A["ref_root_quat"][sl])
    ang = np.degrees(2 * np.arccos(np.clip(np.abs(dq[:, 0]), 0, 1)))
    z = A["pelvis_z"][sl]
    # robot lagging the reference by L ticks (same as the baseline report): arms error vs L = 0..10
    lag_err = [float(np.degrees(np.abs(q[L:, arms] - r[:len(r) - L, arms])).mean()) for L in range(0, 11)]
    best = int(np.argmin(lag_err))
    worst = int(np.argmax(err.mean(0)))
    out_worst = {"worst_joint": names[worst].replace("_joint", ""), "worst_joint_deg": round(float(err[:, worst].mean()), 1)}
    return {**out_worst,"joint_err_deg": round(float(err.mean()), 2), "legs_deg": round(float(err[:, legs].mean()), 2),
            "arms_best_lag_deg": round(lag_err[best], 2), "best_lag_frames": best,
            "arms_deg": round(float(err[:, arms].mean()), 2), "arms_p95_deg": round(float(np.percentile(err[:, arms], 95)), 2),
            "root_ori_err_deg": round(float(ang.mean()), 2), "min_pelvis_z": round(float(z.min()), 3)}


def run(vdir: Path, clips: list, motions_dir: Path, stand_s: float, video: str, fps: int, ghost: bool, start: str = "band"):
    robot = get_robot("k1")
    sim = G1Sim(robot=robot, elastic_band=True)
    if start == "ground":  # default pose, feet on the floor, band 3 cm above -> no drop at release
        place_on_ground(sim)
    ctrl = K1Controller.load(vdir, robot)
    stand = standing_motion(n_frames=50, robot=robot)
    ctrl.set_motion(stand, play=False)
    cam = dict(width=640, height=480, distance=2.4, azimuth=150.0, elevation=-10.0)
    rec = Recorder(sim, **cam) if video else None
    gh = Ghost(sim, robot, **cam) if (video and ghost) else None
    every = max(1, int(round(1.0 / (fps * CONTROL_DT))))
    frames, log = [], EpisodeLog()

    # plan: (label, motion or None for standing, seconds for standing)
    plan = [("stand", None, stand_s)]
    for c in clips:
        plan.append((c.stem.replace("gest_", ""), load_robot_pkl(c, robot=robot), None))
        plan.append(("stand", None, 2.0))
    plan[-1] = ("stand", None, 3.0)

    t_ctrl0 = t_drop = None
    segs, cur, seg_start, seg_t0 = [], -1, 0, 0.0
    fell_at, max_tilt, tick = None, 0.0, 0
    while True:
        if ctrl.state == "WAIT_FOR_CONTROL":
            ctrl.start_control()
            t_ctrl0 = sim.time
        if t_ctrl0 is not None and t_drop is None and sim.time - t_ctrl0 >= 2.0:
            sim.set_band(False)
            t_drop = sim.time
        if t_drop is not None and sim.time - t_drop >= 1.0:
            # advance the plan
            label, mot, dur = plan[cur] if cur >= 0 else (None, None, 0)
            done = cur < 0 or (mot is None and sim.time - seg_t0 >= dur) or (mot is not None and ctrl.motion_completed)
            if done:
                if cur >= 0:
                    segs.append((label, seg_start, len(log.t), sim.time - seg_t0))
                cur += 1
                if cur >= len(plan):
                    break
                label, mot, dur = plan[cur]
                ctrl.motion_completed = False
                ctrl.set_motion(mot if mot is not None else stand, play=mot is not None)
                seg_start, seg_t0 = len(log.t), sim.time
            z, tilt = sim.pelvis_height(), sim.tilt_deg()
            max_tilt = max(max_tilt, tilt)
            if fell_at is None and (z < FALL_Z or tilt > FALL_TILT):
                fell_at = (round(sim.time, 2), plan[cur][0])
                print(f"FALL at t={sim.time:.2f} during {plan[cur][0]} (z={z:.3f}, tilt={tilt:.0f})")
        f_used = ctrl.current_frame
        control_tick(ctrl, sim, log)
        if rec is not None and tick % every == 0:
            label = plan[cur][0] if cur >= 0 else "init"
            img = caption(rec.grab(), f"K1 SONIC (ours)  {label}  t={sim.time:5.1f}s")
            if gh is not None:
                m = ctrl.current_motion
                g = gh.render(m, min(f_used, m.timesteps - 1), ctrl.apply_delta_heading(), sim.data.qpos[0:2])
                img = np.concatenate([caption(g, "reference (kinematic)"), img], axis=1)
            frames.append(img)
        tick += 1
        if sim.time > 600:
            break

    out = {"model": str(vdir), "fell": fell_at is not None, "fell_at": fell_at, "max_tilt_deg": round(max_tilt, 1),
           "control_s": round(sim.time - (t_ctrl0 or 0), 1), "segments": []}
    for label, a, b, dur in segs:
        out["segments"].append({"clip": label, "seconds": round(dur, 1), **seg_metrics(log, robot, a, b)})
    A = log.arrays()
    out["policy_tick_ms_mean"] = round(float(np.mean(A["tick_ms"])), 2)
    if video:
        import imageio

        Path(video).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(video, frames, fps=fps, quality=7, macro_block_size=1)
        out["video"] = video
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant-dir", type=Path, help="prepared dir (model_encoder/decoder.onnx + observation_config)")
    ap.add_argument("--export", type=Path, help="gear_sonic exported/ dir to prepare from")
    ap.add_argument("--step", type=int, help="checkpoint step of the export")
    ap.add_argument("--future", default="10x5", help="10x5 (sonic_k1_small) or 4x1 (sonic_k1_small_la4)")
    ap.add_argument("--motions", type=Path, required=True, help="built k1_motions dir (robot/ heldout/)")
    ap.add_argument("--clips", nargs="*", default=["bow", "wave_hello", "welcome_open_arms", "clap", "point_guide"])
    ap.add_argument("--stance", default="sonic", choices=["sonic", "loco"])
    ap.add_argument("--speed", default="1.00")
    ap.add_argument("--stand", type=float, default=20.0)
    ap.add_argument("--start", default="band", choices=["band", "ground"],
                    help="band: hang on the elastic band (anchor 1.0 m = G1 value -> K1 drops ~20 cm at release); "
                         "ground: feet on the floor first (sonic.runner.place_on_ground)")
    ap.add_argument("--video", default="")
    ap.add_argument("--ghost", action="store_true")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    vdir = a.variant_dir
    if a.export is not None:
        n, s = (int(x) for x in a.future.split("x"))
        vdir = prepare_variant(a.export, a.step, vdir or (a.export.parent / f"deploy_{a.step:06d}"), n, s)
    clips = []
    for c in a.clips:
        p = Path(c)
        if not p.exists():
            hits = sorted(a.motions.rglob(f"gest_{c}_{a.stance}_x{a.speed}.pkl")) or sorted(a.motions.rglob(f"{c}.pkl"))
            if not hits:
                sys.exit(f"clip {c} not found under {a.motions}")
            p = hits[0]
        clips.append(p)
    res = run(vdir, clips, a.motions, a.stand, a.video, a.fps, a.ghost, a.start)
    res["start"] = a.start
    print(json.dumps(res, indent=1))
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
