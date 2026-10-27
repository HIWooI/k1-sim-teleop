"""Portable NumPy/SciPy K1 kinematics + warm-started CPU IK for MediaPipe targets.

Copied unchanged (apart from this header) from the author's earlier K1 RGB-teleop pipeline
(`perception/cpu_kinematics.py`). Original docstring:


The production retargeter uses JAX and PyRoki from a private environment.  This module
intentionally has no dependency on that stack: it reads the vendored URDF directly,
performs forward kinematics, and solves a warm-started position IK problem on CPU.
Meshes are not needed because the portable demo renders a stick model from link frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares


TARGET_NAMES = (
    "Hips", "Chest",
    "LeftArm", "LeftForeArm", "LeftHand",
    "RightArm", "RightForeArm", "RightHand",
    "LeftLeg", "LeftShin", "LeftFoot",
    "RightLeg", "RightShin", "RightFoot",
)

TARGET_LINKS = (
    "pelvis", "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_roll_rubber_hand",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_roll_rubber_hand",
    "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
)

TARGET_PARENTS = np.array((-1, 0, 1, 2, 3, 1, 5, 6, 0, 8, 9, 0, 11, 12), dtype=np.int32)
TARGET_BASE_WEIGHTS = np.array(
    (30.0, 0.7, 1.5, 1.0, 2.0, 1.5, 1.0, 2.0, 1.5, 1.0, 30.0, 1.5, 1.0, 30.0),
    dtype=np.float64,
)


def _floats(value: str | None, default: tuple[float, ...]) -> np.ndarray:
    if not value:
        return np.asarray(default, dtype=np.float64)
    return np.fromstring(value, sep=" ", dtype=np.float64)


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    x, y, z = axis
    c, s, one_c = np.cos(angle), np.sin(angle), 1.0 - np.cos(angle)
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )


def _transform(rotation: np.ndarray | None = None, translation: np.ndarray | None = None) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    if rotation is not None:
        out[:3, :3] = rotation
    if translation is not None:
        out[:3, 3] = translation
    return out


@dataclass(frozen=True)
class Joint:
    name: str
    kind: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float
    q_index: int


class UrdfKinematics:
    """Small URDF tree reader supporting fixed and revolute joints."""

    def __init__(self, urdf_path: str | Path):
        root = ET.parse(urdf_path).getroot()
        links = [node.attrib["name"] for node in root.findall("link")]
        raw: list[Joint] = []
        child_links: set[str] = set()
        q_names: list[str] = []
        lower: list[float] = []
        upper: list[float] = []

        for node in root.findall("joint"):
            kind = node.attrib.get("type", "fixed")
            parent = node.find("parent").attrib["link"]
            child = node.find("child").attrib["link"]
            child_links.add(child)
            origin_node = node.find("origin")
            xyz = _floats(None if origin_node is None else origin_node.attrib.get("xyz"), (0, 0, 0))
            rpy = _floats(None if origin_node is None else origin_node.attrib.get("rpy"), (0, 0, 0))
            origin = _transform(_rpy_matrix(rpy), xyz)
            axis_node = node.find("axis")
            axis = _floats(None if axis_node is None else axis_node.attrib.get("xyz"), (1, 0, 0))
            if kind in {"revolute", "continuous"}:
                q_index = len(q_names)
                q_names.append(node.attrib["name"])
                limit = node.find("limit")
                lo = -np.pi if kind == "continuous" else float(limit.attrib["lower"])
                hi = np.pi if kind == "continuous" else float(limit.attrib["upper"])
                lower.append(lo)
                upper.append(hi)
            else:
                q_index, lo, hi = -1, 0.0, 0.0
            raw.append(Joint(node.attrib["name"], kind, parent, child, origin, axis, lo, hi, q_index))

        roots = [link for link in links if link not in child_links]
        if len(roots) != 1:
            raise ValueError(f"expected one URDF root link, found {roots}")
        self.root_link = roots[0]
        self.joint_names = tuple(q_names)
        self.lower = np.asarray(lower, dtype=np.float64)
        self.upper = np.asarray(upper, dtype=np.float64)

        by_parent: dict[str, list[Joint]] = {}
        for joint in raw:
            by_parent.setdefault(joint.parent, []).append(joint)
        ordered: list[Joint] = []

        def walk(link: str) -> None:
            for joint in by_parent.get(link, ()):
                ordered.append(joint)
                walk(joint.child)

        walk(self.root_link)
        if len(ordered) != len(raw):
            raise ValueError("URDF contains a disconnected joint tree")
        self.joints = tuple(ordered)
        self.link_names = tuple(links)
        self.edges = tuple((joint.parent, joint.child) for joint in ordered)

        # A target link is affected by every actuated joint along its root path.
        parent_joint = {joint.child: joint for joint in ordered}
        self.target_ancestors: list[tuple[int, ...]] = []
        for link in TARGET_LINKS:
            ancestors: list[int] = []
            cursor = link
            while cursor in parent_joint:
                joint = parent_joint[cursor]
                if joint.q_index >= 0:
                    ancestors.append(joint.q_index)
                cursor = joint.parent
            self.target_ancestors.append(tuple(reversed(ancestors)))

    @property
    def ndof(self) -> int:
        return len(self.joint_names)

    def forward(
        self, q: np.ndarray
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (self.ndof,):
            raise ValueError(f"q must have shape ({self.ndof},), got {q.shape}")
        link_tf = {self.root_link: np.eye(4, dtype=np.float64)}
        joint_origins = np.zeros((self.ndof, 3), dtype=np.float64)
        joint_axes = np.zeros((self.ndof, 3), dtype=np.float64)
        for joint in self.joints:
            before_rotation = link_tf[joint.parent] @ joint.origin
            child_tf = before_rotation
            if joint.q_index >= 0:
                joint_origins[joint.q_index] = before_rotation[:3, 3]
                joint_axes[joint.q_index] = before_rotation[:3, :3] @ joint.axis
                child_tf = before_rotation @ _transform(_axis_angle(joint.axis, q[joint.q_index]))
            link_tf[joint.child] = child_tf
        positions = {name: tf[:3, 3].copy() for name, tf in link_tf.items()}
        return positions, joint_origins, joint_axes


class K1CpuRetargeter:
    """Warm-started CPU position IK for the 14 MediaPipe target points."""

    def __init__(self, urdf_path: str | Path, max_nfev: int = 8, smoothing: float = 0.35):
        self.robot = UrdfKinematics(urdf_path)
        if self.robot.ndof != 23:
            raise ValueError(f"K1 model should have 23 DOF, found {self.robot.ndof}")
        self.q = np.zeros(self.robot.ndof, dtype=np.float64)
        self.max_nfev = max_nfev
        self.smoothing = smoothing
        rest, _, _ = self.robot.forward(self.q)
        self.rest_targets = np.stack([rest[name] for name in TARGET_LINKS])
        # torso_link shares the pelvis origin in the K1 URDF.  For retargeting, "Chest"
        # is a virtual point between the two shoulder roots; without this point the
        # shoulder chain is rebuilt horizontally from the hips and the solver sees a
        # collapsed torso.
        self.rest_targets[1] = (
            self.rest_targets[TARGET_NAMES.index("LeftArm")]
            + self.rest_targets[TARGET_NAMES.index("RightArm")]
        ) / 2.0
        self.bone_lengths = np.zeros(len(TARGET_LINKS), dtype=np.float64)
        for i, parent in enumerate(TARGET_PARENTS):
            if parent >= 0:
                self.bone_lengths[i] = np.linalg.norm(self.rest_targets[i] - self.rest_targets[parent])
        self._weights = TARGET_BASE_WEIGHTS.copy()
        self.last_cost = 0.0
        self.last_nfev = 0

    def scale_human_targets(self, camera_points: np.ndarray) -> np.ndarray:
        """Map MediaPipe camera coordinates onto K1 bone lengths and axes."""
        points = np.asarray(camera_points, dtype=np.float64)
        if points.shape != (len(TARGET_LINKS), 3):
            raise ValueError(f"camera_points must be (14, 3), got {points.shape}")
        # MediaPipe world landmarks use x toward the person's left in a front-facing
        # camera, y down, and z away from the viewer.  K1 uses x forward, y left, z up.
        # Negating depth makes forward motion positive; keeping MediaPipe x makes a
        # person's left limb drive the robot's left limb in the preview.
        world = np.stack((-points[:, 2], points[:, 0], -points[:, 1]), axis=-1)
        out = np.zeros_like(world)
        for i, parent in enumerate(TARGET_PARENTS):
            if parent < 0:
                continue
            direction = world[i] - world[parent]
            norm = np.linalg.norm(direction)
            if norm < 1e-6 or self.bone_lengths[i] < 1e-8:
                out[i] = out[parent]
            else:
                out[i] = out[parent] + direction * (self.bone_lengths[i] / norm)
        return out

    @staticmethod
    def visibility_weights(visibility: np.ndarray) -> np.ndarray:
        visibility = np.asarray(visibility, dtype=np.float64)
        confidence = np.clip((visibility - 0.55) / 0.45, 0.0, 1.0)
        # Keep the root weakly active so an uncertain frame remains numerically stable.
        confidence[0] = max(confidence[0], 0.25)
        return TARGET_BASE_WEIGHTS * confidence

    def solve(self, targets: np.ndarray, visibility: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        targets = np.asarray(targets, dtype=np.float64)
        weights = self.visibility_weights(visibility)
        sqrt_w = np.sqrt(weights)[:, None]
        regularization = 0.035

        def evaluate(q: np.ndarray, with_jacobian: bool = False):
            positions, origins, axes = self.robot.forward(q)
            current = np.stack([positions[name] for name in TARGET_LINKS])
            residual = ((current - targets) * sqrt_w).ravel()
            residual = np.concatenate((residual, regularization * q))
            if not with_jacobian:
                return residual
            jac = np.zeros((len(TARGET_LINKS) * 3 + self.robot.ndof, self.robot.ndof))
            for link_i, ancestors in enumerate(self.robot.target_ancestors):
                for q_i in ancestors:
                    derivative = np.cross(axes[q_i], current[link_i] - origins[q_i])
                    jac[link_i * 3 : link_i * 3 + 3, q_i] = derivative * sqrt_w[link_i, 0]
            jac[-self.robot.ndof :, :] = np.eye(self.robot.ndof) * regularization
            return jac

        # Starting exactly on a one-sided bound (knees and shoulder roll are neutral at
        # zero) can leave scipy's reflective trust region with a microscopic first step.
        # Move only the optimizer's initial value just inside; the reported pose may
        # still settle exactly on the physical limit.
        margin = np.minimum(1e-4, np.maximum((self.robot.upper - self.robot.lower) * 1e-3, 0.0))
        initial = np.clip(self.q, self.robot.lower + margin, self.robot.upper - margin)
        result = least_squares(
            evaluate,
            initial,
            jac=lambda q: evaluate(q, with_jacobian=True),
            bounds=(self.robot.lower, self.robot.upper),
            max_nfev=self.max_nfev,
            method="trf",
            ftol=1e-4,
            xtol=1e-4,
            gtol=1e-4,
        )
        q_new = result.x
        self.q = self.q * (1.0 - self.smoothing) + q_new * self.smoothing
        self.q = np.clip(self.q, self.robot.lower, self.robot.upper)
        self.last_cost = float(result.cost)
        self.last_nfev = int(result.nfev)
        positions, _, _ = self.robot.forward(self.q)
        return self.q.copy(), positions


def target_rms(positions: dict[str, np.ndarray], targets: np.ndarray) -> float:
    current = np.stack([positions[name] for name in TARGET_LINKS])
    return float(np.sqrt(np.mean(np.square(current - targets))))
