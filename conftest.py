"""Shared pytest setup: tests that need downloaded or staged assets are skipped, with the reason, when the assets are
missing (a fresh clone before scripts/setup.sh and sonic_k1/scripts/stage_assets.py, docs/setup.md)."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
SMPL_REST = (ROOT / "models" / "human_joints_info.npz",
             ROOT / "vendor" / "GR00T-WholeBodyControl" / "gear_sonic" / "data" / "human" / "human_joints_info.pkl")
K1_MESHES = ROOT / "sonic_k1" / "overlay" / "gear_sonic" / "data" / "assets" / "robot_description" / "urdf" / "k1" / "meshes"

NEEDS = {
    "needs_smpl": (lambda: any(p.exists() for p in SMPL_REST),
                   "SMPL rest skeleton (human_joints_info) missing: run scripts/setup.sh"),
    "needs_k1_meshes": (lambda: any(K1_MESHES.glob("*.stl")),
                        "K1 meshes missing: run sonic_k1/scripts/stage_assets.py"),
}


def pytest_configure(config):
    for name, (_, why) in NEEDS.items():
        config.addinivalue_line("markers", f"{name}: skipped when the asset is absent ({why})")


def pytest_collection_modifyitems(config, items):
    for name, (have, why) in NEEDS.items():
        if have():
            continue
        for item in items:
            if name in item.keywords:
                item.add_marker(pytest.mark.skip(reason=why))
