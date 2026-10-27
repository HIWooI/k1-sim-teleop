#!/usr/bin/env python3
# Portions derived from NVIDIA GR00T-WholeBodyControl (GEAR-SONIC), Copyright (c) NVIDIA CORPORATION & AFFILIATES,
# licensed under the Apache License 2.0 (https://github.com/NVlabs/GR00T-WholeBodyControl). Modified for the K1 port.
"""Upstream gear_sonic/scripts/run_sim_loop.py, plus keys typed in *this terminal* — for SSH / tmux use.

Upstream only releases the elastic band from the MuJoCo viewer's key callback ('9' in the window).
Over SSH there may be no window to type into (or no display at all), so this wrapper reads single keys
from its own terminal and forwards them to the same ElasticBand object:

    9      toggle the elastic band (drop the robot / lift it again)  == '9' in the MuJoCo window
    7 / 8  band shorter / longer                                     == '7' / '8' in the MuJoCo window

  --headless   no viewer window (ENABLE_ONSCREEN=False), e.g. no X display on the box
Every other argument goes to run_sim_loop.py's tyro CLI unchanged. Run it like run_sim_loop.py:
    cd $SONIC_ROOT && .venv_sim/bin/python <bundle>/linux/sim_loop_keys.py [--headless] [tyro args]
"""
from __future__ import annotations

import atexit
import importlib.util
import os
import sys
import threading
from pathlib import Path

BANDS = []


def _install_band_hook():
    from gear_sonic.utils.mujoco_sim import unitree_sdk2py_bridge as bridge

    orig = bridge.ElasticBand.__init__

    def init(self, *a, **k):
        orig(self, *a, **k)
        BANDS.append(self)

    bridge.ElasticBand.__init__ = init     # base_sim imported the class object, so patching it reaches DefaultEnv


def _key_loop():
    fd = sys.stdin.fileno()
    if not os.isatty(fd):
        print("[sim_loop_keys] stdin is not a terminal; terminal keys disabled", flush=True)
        return
    import termios
    import tty

    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)                       # single keys, Ctrl+C still raises SIGINT
    atexit.register(termios.tcsetattr, fd, termios.TCSADRAIN, old)
    while True:
        try:
            ch = os.read(fd, 1).decode(errors="ignore")
        except OSError:
            return
        if not ch:
            return
        if not BANDS:
            if ch in "789":
                print("[sim_loop_keys] no elastic band (ENABLE_ELASTIC_BAND off?)", flush=True)
            continue
        band = BANDS[-1]
        if ch == "9":
            band.handle_keyboard_button("9")   # prints "ElasticBand enable: ..."
        elif ch == "7":
            band.length -= 0.1
            print(f"ElasticBand length: {band.length:+.1f}", flush=True)
        elif ch == "8":
            band.length += 0.1
            print(f"ElasticBand length: {band.length:+.1f}", flush=True)


def main():
    argv = sys.argv[1:]
    headless = "--headless" in argv
    argv = [a for a in argv if a != "--headless"]
    script = Path.cwd() / "gear_sonic/scripts/run_sim_loop.py"
    if not script.exists():
        sys.exit(f"run from the SONIC repo root (no {script})")
    spec = importlib.util.spec_from_file_location("run_sim_loop", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)            # defines main/ArgsConfig; its __main__ block does not run
    import tyro

    config = tyro.cli(mod.ArgsConfig, args=argv)
    if headless:
        config.enable_onscreen = False      # -> ENABLE_ONSCREEN in the WBC yaml dict
    _install_band_hook()
    threading.Thread(target=_key_loop, daemon=True).start()
    print("[sim_loop_keys] terminal keys: 9 = drop/lift robot (elastic band), 7/8 = band length, Ctrl+C = quit"
          + ("  [headless]" if headless else ""), flush=True)
    mod.main(config)


if __name__ == "__main__":
    main()
