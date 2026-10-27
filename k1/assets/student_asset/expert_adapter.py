# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""One interface over the frozen K1 expert policies.

Every deployed expert is an ONNX file plus a ``sim2real.yaml`` that fully
describes its deployable contract: joint order, observation terms and history
depth, action scale/offset, and PD gains. The experts disagree on all of these.
In particular ``walk_jhl_0715_v1`` lists ``policy_joints`` in the robot's
anatomical order while every mimic expert lists them in the Isaac Lab
breadth-first order, so raw action index ``i`` means a different joint for each.
Comparing two experts' actions without going through this adapter is a silent
mis-mapping, not an error.

The adapter therefore exposes everything in one canonical order -- the runtime's
``robot_joint_order`` from ``k1_config.yaml`` -- while running the network on its
own native ordering, unchanged. Loading and inference are byte-identical to what
``PolicyRuntime`` does; ``test_expert_adapter.py`` pins that.

The transition manager and the transition training environment talk only to this
class. Neither should ever read a ``sim2real.yaml`` itself.
"""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

# Terms the runtime computes from the reference motion rather than from sensors.
# A policy carrying any of these needs a reference clock to be stepped.
_REFERENCE_TERMS = ("motion_command", "motion_joint_pos", "motion_joint_vel", "motion_anchor_ori_b")


def _load_robot_joint_order(k1_config_path: Path) -> list[str]:
    return yaml.safe_load(k1_config_path.read_text())["robot_joint_order"]


@dataclass(frozen=True)
class ObservationTerm:
    """One entry of a policy's observation vector, in YAML order."""

    name: str
    size: int
    history_length: int
    scale: np.ndarray | None
    clip: tuple[float, float] | None

    @property
    def width(self) -> int:
        return self.size * self.history_length

    @property
    def is_reference(self) -> bool:
        return self.name in _REFERENCE_TERMS


class ExpertAdapter:
    """A single frozen expert, addressed in canonical joint order.

    Construct with :meth:`from_asset_dir`. The expert's parameters are never
    touched -- there is no training path through this class at all, which is how
    the "experts stay frozen" requirement is enforced structurally rather than by
    an optimizer filter.
    """

    def __init__(
        self,
        name: str,
        asset_dir: Path,
        config: dict,
        robot_joint_order: list[str],
        session=None,
    ):
        self.name = name
        self.asset_dir = asset_dir
        self.config = config
        self.robot_joint_order = list(robot_joint_order)
        self._session = session

        self.policy_joints: list[str] = list(config["policy_joints"])
        self.step_dt: float = float(config["step_dt"])

        missing = set(self.policy_joints) - set(self.robot_joint_order)
        if missing:
            raise ValueError(f"{name}: policy joints absent from robot_joint_order: {sorted(missing)}")

        # policy index -> canonical index, and its inverse where it exists.
        self._policy_to_canonical = np.array(
            [self.robot_joint_order.index(joint) for joint in self.policy_joints], dtype=np.int64
        )
        self._canonical_to_policy = np.full(len(self.robot_joint_order), -1, dtype=np.int64)
        self._canonical_to_policy[self._policy_to_canonical] = np.arange(len(self.policy_joints))

        action = config["actions"]["joint_pos"]
        self._action_scale = np.asarray(action["scale"], dtype=np.float64)
        self._action_offset = np.asarray(action["offset"], dtype=np.float64)
        self._action_clip = action.get("clip")

        props = config["joint_properties"]
        self._kp = np.array([props[j]["stiffness"] for j in self.policy_joints], dtype=np.float64)
        self._kd = np.array([props[j]["damping"] for j in self.policy_joints], dtype=np.float64)
        self._default_position = np.array(
            [props[j]["default_position"] for j in self.policy_joints], dtype=np.float64
        )

        self.observation_terms = self._parse_observation_terms(config["observations"])
        self.observation_size = sum(term.width for term in self.observation_terms)

        # obs history, one deque per term, newest last -- the layout
        # ObservationTermCfg::append_to produces.
        self._history: dict[str, deque] = {}
        self._last_action = np.zeros(len(self.policy_joints))
        self.reset_history(np.zeros(self.observation_size))

    # -- construction ----------------------------------------------------------

    @classmethod
    def from_asset_dir(cls, asset_dir, k1_config_path=None, load_session: bool = True) -> "ExpertAdapter":
        """Load an expert from an ``assets/k1/...`` directory.

        Args:
            asset_dir: directory holding ``params/sim2real.yaml`` and ``exported/policy.onnx``.
            k1_config_path: runtime config supplying ``robot_joint_order``. Defaults to
                the ``k1_config.yaml`` that sits alongside the assets tree.
            load_session: set False to inspect the contract without onnxruntime.
        """
        asset_dir = Path(asset_dir)
        config = yaml.safe_load((asset_dir / "params/sim2real.yaml").read_text())

        if k1_config_path is None:
            k1_config_path = _default_k1_config_path(asset_dir)
        robot_joint_order = _load_robot_joint_order(Path(k1_config_path))

        session = None
        if load_session:
            import onnxruntime  # imported lazily; the contract alone needs no runtime

            session = onnxruntime.InferenceSession(
                str(asset_dir / "exported/policy.onnx"), providers=["CPUExecutionProvider"]
            )

        adapter = cls(asset_dir.name, asset_dir, config, robot_joint_order, session)
        if session is not None:
            expected = session.get_inputs()[0].shape[-1]
            if int(expected) != adapter.observation_size:
                raise ValueError(
                    f"{adapter.name}: ONNX expects {expected} observations but sim2real.yaml "
                    f"describes {adapter.observation_size}"
                )
        return adapter

    def _parse_observation_terms(self, obs_config: dict) -> list[ObservationTerm]:
        terms = []
        n_joints = len(self.policy_joints)
        for name, term in obs_config.items():
            if name in ("scale_first", "use_gym_history"):
                continue
            scale = term.get("scale")
            # The exporter writes one scale per element, so its length is the term's
            # true width -- authoritative, and the same thing the runtime validates
            # against in ObservationManager::initialize_term_history. Fall back to the
            # name table only when the scale is a bare scalar.
            size = (
                len(scale)
                if isinstance(scale, list) and len(scale) > 1
                else _term_size(name, n_joints)
            )
            clip = term.get("clip")
            terms.append(
                ObservationTerm(
                    name=name,
                    size=size,
                    history_length=int(term.get("history_length", 1)),
                    scale=(
                        np.full(size, scale[0], dtype=np.float64)
                        if isinstance(scale, list) and len(scale) == 1 and size > 1
                        else (np.asarray(scale, dtype=np.float64) if scale is not None else None)
                    ),
                    clip=(tuple(clip) if clip else None),
                )
            )
        return terms

    # -- identity --------------------------------------------------------------

    @property
    def checkpoint_sha256(self) -> str:
        digest = hashlib.sha256()
        with (self.asset_dir / "exported/policy.onnx").open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @property
    def history_mode(self) -> str:
        depth = max(term.history_length for term in self.observation_terms)
        return f"obs_stack_{depth}"

    @property
    def uses_reference_motion(self) -> bool:
        return any(term.is_reference for term in self.observation_terms)

    @property
    def is_cyclic(self) -> bool:
        """Locomotion is a steerable gait with no clip end; mimic experts play a clip."""
        return not self.uses_reference_motion

    # -- joint-order translation ----------------------------------------------

    def to_canonical(self, policy_ordered, fill: float = 0.0) -> np.ndarray:
        """Scatter a policy-ordered per-joint vector into canonical order."""
        out = np.full(len(self.robot_joint_order), fill, dtype=np.float64)
        out[self._policy_to_canonical] = np.asarray(policy_ordered, dtype=np.float64)
        return out

    def from_canonical(self, canonical) -> np.ndarray:
        """Gather the joints this policy controls out of a canonical vector."""
        return np.asarray(canonical, dtype=np.float64)[self._policy_to_canonical]

    @property
    def controlled_canonical_indices(self) -> np.ndarray:
        return self._policy_to_canonical.copy()

    # -- gains and torque ------------------------------------------------------

    def gain_table(self, canonical: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """This policy's deployed ``(Kp, Kd)``.

        In canonical order, joints the policy does not control report NaN rather
        than 0: the runtime leaves those slots at the *previous* behavior's gains,
        so a zero would be a lie that silently produces a limp joint in a ramp.
        """
        if not canonical:
            return self._kp.copy(), self._kd.copy()
        return self.to_canonical(self._kp, np.nan), self.to_canonical(self._kd, np.nan)

    def commanded_torque(self, q_des, q, dq, gains=None, canonical: bool = True) -> np.ndarray:
        """Torque the impedance controller would apply for this ``q_des``.

        ``gains`` overrides the policy's own table -- pass the ramped gains for the
        tick actually being evaluated, which is the whole point of the tau-level
        continuity gate.
        """
        from cyclo_lab.transition.gain_ramp import commanded_torque as _torque

        kp, kd = self.gain_table(canonical=canonical) if gains is None else gains
        return _torque(q_des, q, dq, kp, kd)

    # -- action ----------------------------------------------------------------

    def decode_action(self, raw_action, canonical: bool = True) -> np.ndarray:
        """Raw network output -> joint position target.

        ``q_des = raw * scale + offset``, then the optional post-clip. This is
        ``ActionPipeline::process`` exactly; note the ``raw_clip`` field present in
        some yamls is *not* applied by the runtime and so is not applied here.
        """
        raw = np.asarray(raw_action, dtype=np.float64)
        q_des = raw * self._action_scale + self._action_offset
        if self._action_clip:
            lows = np.array([c[0] for c in self._action_clip])
            highs = np.array([c[1] for c in self._action_clip])
            q_des = np.clip(q_des, lows, highs)
        return self.to_canonical(q_des) if canonical else q_des

    def encode_action(self, q_des, canonical: bool = True) -> np.ndarray:
        """Inverse of :meth:`decode_action`: the raw action that would command ``q_des``.

        This is the equivalent-action seeding rule for a target expert that cannot
        be warmed by live shadow inference:
        ``last_action = (q_now - offset_target) / scale_target``.
        """
        q = self.from_canonical(q_des) if canonical else np.asarray(q_des, dtype=np.float64)
        return (q - self._action_offset) / self._action_scale

    @property
    def default_pose(self) -> np.ndarray:
        """Per-joint default position in canonical order (the action offset's basis)."""
        return self.to_canonical(self._default_position)

    # -- history and previous action ------------------------------------------

    def reset_history(self, flat_observation) -> None:
        """Seed every history slot with one observation.

        This reproduces ``ObservationManager::reset``, which is the runtime's
        current behaviour and the source of the observation-OOD half of the
        handoff torque step. It is kept so the hard-switch baseline can be
        reproduced faithfully -- warm entry uses :meth:`push_observation` instead.
        """
        flat = np.asarray(flat_observation, dtype=np.float64)
        cursor = 0
        for term in self.observation_terms:
            single = flat[cursor : cursor + term.size]
            cursor += term.width
            self._history[term.name] = deque(
                [single.copy() for _ in range(term.history_length)], maxlen=term.history_length
            )

    def push_observation(self, term_values: dict) -> None:
        """Append one real timestep to the history, oldest dropped.

        Called once per tick during BRIDGE live-shadow inference so the target
        expert enters with a history of the trajectory that actually happened
        rather than five copies of one frame.
        """
        for term in self.observation_terms:
            if term.name not in term_values:
                raise KeyError(f"{self.name}: missing observation term '{term.name}'")
            values = np.asarray(term_values[term.name], dtype=np.float64)
            if values.size != term.size:
                raise ValueError(
                    f"{self.name}: term '{term.name}' expects {term.size} values, got {values.size}"
                )
            self._history[term.name].append(values)

    def get_history(self) -> dict:
        return {name: [frame.copy() for frame in frames] for name, frames in self._history.items()}

    def set_history(self, history: dict) -> None:
        for term in self.observation_terms:
            frames = history[term.name]
            if len(frames) != term.history_length:
                raise ValueError(
                    f"{self.name}: term '{term.name}' expects {term.history_length} frames, got {len(frames)}"
                )
            self._history[term.name] = deque(
                [np.asarray(f, dtype=np.float64) for f in frames], maxlen=term.history_length
            )

    def get_last_action(self, canonical: bool = True) -> np.ndarray:
        return self.to_canonical(self._last_action) if canonical else self._last_action.copy()

    def set_last_action(self, raw_action, canonical: bool = True) -> None:
        self._last_action = (
            self.from_canonical(raw_action) if canonical else np.asarray(raw_action, dtype=np.float64)
        )

    def seed_last_action_from_pose(self, q_canonical) -> np.ndarray:
        """Equivalent-action seeding: set ``last_action`` to whatever would hold ``q``.

        The fallback for a target expert that cannot be shadow-inferred. Zero-filling
        instead -- what the runtime does today -- claims the previous action was
        "hold the default pose", which is wrong by the full offset difference.
        """
        seeded = self.encode_action(q_canonical)
        self._last_action = seeded
        return seeded

    # -- observation assembly and inference ------------------------------------

    def build_observation(self) -> np.ndarray:
        """Flatten the current history into the ONNX input vector.

        Scale then clip per term, matching ``ObservationTermCfg`` with the default
        ``scale_first=false``... note the runtime clips *before* scaling unless the
        yaml says otherwise, and none of the K1 yamls set a clip, so the two orders
        are indistinguishable here. Kept explicit so a future clipped term is right.
        """
        chunks = []
        for term in self.observation_terms:
            for frame in self._history[term.name]:
                values = frame
                if term.clip is not None:
                    values = np.clip(values, term.clip[0], term.clip[1])
                if term.scale is not None:
                    values = values * term.scale
                chunks.append(values)
        return np.concatenate(chunks).astype(np.float32)

    def act(self, observation=None, commit: bool = True) -> np.ndarray:
        """Run inference and return the raw action in policy order.

        Args:
            observation: flat input vector. Defaults to the assembled history.
            commit: when False this is a dry run -- ``last_action`` is left alone, so
                the target expert's first action can be inspected for continuity
                without disturbing the state it will actually enter with.
        """
        if self._session is None:
            raise RuntimeError(f"{self.name}: loaded without an ONNX session")
        obs = self.build_observation() if observation is None else np.asarray(observation, dtype=np.float32)
        input_name = self._session.get_inputs()[0].name
        raw = self._session.run(None, {input_name: obs.reshape(1, -1)})[0].reshape(-1)
        if commit:
            self._last_action = raw.astype(np.float64)
        return raw.astype(np.float64)

    def dry_run(self, observation=None) -> dict:
        """Inspect the action this expert would take without applying it."""
        raw = self.act(observation, commit=False)
        return {"raw_action": raw, "q_des": self.decode_action(raw)}


def _term_size(name: str, n_joints: int) -> int:
    """Observation term widths, from the runtime's registered term functions."""
    if name in ("base_ang_vel", "projected_gravity", "velocity_commands"):
        return 3
    if name == "gait_phase":
        return 2
    if name == "motion_anchor_ori_b":
        return 6
    if name == "motion_command":
        return 2 * n_joints
    return n_joints


def _default_k1_config_path(asset_dir: Path) -> Path:
    for parent in asset_dir.parents:
        candidate = parent / "config/k1_config.yaml"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no k1_config.yaml above {asset_dir}; pass k1_config_path")
