#!/usr/bin/env python3
"""Open a recorded raw demonstration in MuJoCo for visual inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import mujoco.viewer
import numpy as np

from stage4_bimanual.bimanual import reset_scene
from stage4_bimanual.sim import MuJoCoSim


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay a raw skill demonstration in the MuJoCo viewer.")
    parser.add_argument("episode", type=Path, help="Episode directory created by record_skill_demos.py")
    parser.add_argument("--speed", type=float, default=1.0, help="1 = recorded speed; 2 = twice as fast")
    args = parser.parse_args()
    manifest = json.loads((args.episode / "manifest.json").read_text(encoding="utf-8"))
    trajectory = np.load(args.episode / "trajectory.npz")
    sim = reset_scene(int(manifest["seed"]))
    if not isinstance(sim, MuJoCoSim):
        raise RuntimeError("MuJoCo is required for replay")
    dt = 1 / float(manifest["fps_requested"]) / args.speed
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        viewer.cam.lookat[:] = [-0.04, 0.00, 0.74]
        viewer.cam.distance = 1.52
        viewer.cam.elevation = -58.0
        viewer.cam.azimuth = 145.0
        for action in trajectory["action"]:
            if not viewer.is_running():
                break
            sim.data.ctrl[:12] = action
            steps = max(1, round(dt / sim.model.opt.timestep))
            for _ in range(steps):
                mujoco.mj_step(sim.model, sim.data)
            viewer.sync()
            time.sleep(dt)
        print("Replay finished. Inspect the final scene; close the viewer when done.")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.03)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
