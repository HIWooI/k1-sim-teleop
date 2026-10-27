"""Service arm path setting (panel 설정 › 팔 경로): improved arm path (sender --arm-ik dir, sim_server
--arm-mode pd --arm-vmax 12 --arm-amax 150, docs/architecture.md) by default, classic = the previous chain. No child
processes, no MuJoCo.

    .venv/bin/python -m pytest -q tests/test_k1live_arm_path.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from k1live import service as S  # noqa: E402


@pytest.fixture
def cfg_env(tmp_path, monkeypatch):
    for env in ("K1LIVE_LOCO_PROFILE", "K1LIVE_CROP", "K1LIVE_ARM_IK", "K1LIVE_ARM_MODE", "K1LIVE_ARM_VMAX",
                "K1LIVE_ARM_AMAX"):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setattr(S, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(S, "LOG_ROOT", tmp_path)
    return tmp_path


def _argv(cfg):
    return S.build_argv(cfg["sender_cmd"], cfg), S.build_argv(cfg["sim_cmd"], cfg, "/s")


def test_default_is_improved(cfg_env):
    cfg = S.load_config(path=cfg_env / "none.yaml")
    assert cfg["arm_path"] == "improved"
    assert S.arm_settings(cfg) == {"arm_ik": "dir", "arm_mode": "pd", "arm_vmax": 12.0, "arm_amax": 150.0}
    snd, sim = _argv(cfg)
    assert snd[-1] == "--arm-ik=dir"
    assert sim[-3:] == ["--arm-mode=pd", "--arm-vmax=12", "--arm-amax=150"]


def test_event_yaml(cfg_env):
    cfg = S.load_config()          # the shipped k1live/web/event.yaml (no saved panel settings)
    assert cfg["arm_path"] == "improved"
    assert S.arm_settings(cfg) == {"arm_ik": "dir", "arm_mode": "pd", "arm_vmax": 12, "arm_amax": 150}
    assert cfg["backend"] in ("hybrid", "split")    # split only when the hybrid files are missing
    if S.hybrid_available_locally():
        assert cfg["backend"] == "hybrid"


def test_classic_and_bad_value(cfg_env):
    cfg = S.load_config(path=cfg_env / "none.yaml", cli={"arm_path": "classic"})
    snd, sim = _argv(cfg)
    assert snd[-1] == "--arm-ik=shape18"
    assert sim[-3:] == ["--arm-mode=student", "--arm-vmax=6", "--arm-amax=80"]
    assert S.load_config(path=cfg_env / "none.yaml", cli={"arm_path": "fast"})["arm_path"] == "improved"


def test_children_parse_the_arguments(cfg_env):
    from k1live import sender, sim_server

    cfg = S.load_config(path=cfg_env / "none.yaml")
    snd, sim = _argv(cfg)
    a = sender.parse_args(snd[3:])
    assert a.arm_ik == "dir"
    b = sim_server.parse_args(sim[3:])
    assert (b.arm_mode, b.arm_vmax, b.arm_amax) == ("pd", 12.0, 150.0)
    cfg["arm_path"] = "classic"
    snd, sim = _argv(cfg)
    assert sender.parse_args(snd[3:]).arm_ik == "shape18"
    b = sim_server.parse_args(sim[3:])
    assert (b.arm_mode, b.arm_vmax, b.arm_amax) == ("student", 6.0, 80.0)


def test_panel_switch_restarts_and_persists(cfg_env):
    cfg = S.load_config(path=cfg_env / "none.yaml")
    cfg["source"] = "camera:0"
    restarts, argv_set = [], []
    ctl = S.Controller(cfg, send=lambda c, a: None, event=lambda k, **d: None,
                       restart_proc=lambda n, r: restarts.append(n), set_argv=lambda n: argv_set.append(n))
    ok, _ = ctl.command("set", {"arm_path": "classic"})
    assert ok
    assert sorted(restarts) == ["sender", "sim"] and sorted(argv_set) == ["sender", "sim"]
    assert json.loads((cfg_env / "settings.json").read_text())["arm_path"] == "classic"
    assert S.load_config(path=cfg_env / "none.yaml")["arm_path"] == "classic"     # saved choice wins next start
    restarts.clear()
    ok, _ = ctl.command("set", {"arm_path": "classic"})                          # same value: no restart
    assert ok and restarts == []
    ok, msg = ctl.command("set", {"arm_path": "fast"})
    assert not ok and "팔 경로" in msg
    assert ctl.snapshot()["arm"]["arm_ik"] == "shape18"


def test_missing_hybrid_variant_falls_back(cfg_env, monkeypatch):
    """event.yaml hybrid_variant pointing at a missing folder: the default checkpoint, not split."""
    monkeypatch.delenv("K1LIVE_HYBRID_VARIANT", raising=False)
    cfg = S.load_config(path=cfg_env / "none.yaml", cli={"hybrid_variant": "models/k1/nope/deploy/x"})
    assert cfg["hybrid_variant"] is None
    assert "K1LIVE_HYBRID_VARIANT" not in S.child_env(cfg)
