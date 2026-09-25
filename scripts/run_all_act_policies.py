#!/usr/bin/env python3
"""Run every trained LeRobot ACT skill policy one after another in MuJoCo.

Loads the per-skill checkpoints from assets/models/lerobot_outputs and drives the
same scene through open_drawer -> pick_plate -> place_plate -> pick_mug ->
pick_bottle -> pour_water. The policies only output joint targets, so grasps use
the same contact-gated welds as the scripted primitives: a weld attaches only
when both jaws touch the object, and detaches when the gripper opens again.

Usage:
    # Watch all skills in the MuJoCo viewer, chained in one scene:
    python scripts/run_all_act_policies.py --view

    # Test each skill on its own (scripted prerequisites, like the training demos):
    python scripts/run_all_act_policies.py --view --isolated

    # Only some skills, a specific checkpoint step, and a video:
    python scripts/run_all_act_policies.py --skills open_drawer pick_plate --checkpoint-step 002500 --record outputs/act_all.mp4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import mujoco
import numpy as np

from stage3_policy.learned.inference import select_motor_policy
from stage4_bimanual.bimanual import reset_scene
from stage4_bimanual.primitives import BaseManipulationPrimitive
from stage4_bimanual.trajectory import TrajectoryExecutor
from stage4_bimanual.primitives import (
    OpenDrawerPrimitive,
    PickBottlePrimitive,
    PickMugPrimitive,
    PickPlatePrimitive,
)

TRAIN_DIR = ROOT / "assets" / "models" / "lerobot_outputs" / "lerobot_outputs" / "train"
SUBSTEPS = 20  # 20 * 0.002 s = 0.04 s per policy step (25 Hz, matches the demos)

# skill: (instruction used in training, default policy steps ~1.2x mean demo length)
SKILLS = {
    "open_drawer": ("Open the top drawer with arm A.", 220),
    "pick_plate": ("Pick up the plate from the open drawer with arm A.", 130),
    "place_plate": ("Place the plate at the center of the table with arm A.", 280),
    "pick_mug": ("Pick up and hold the mug with arm B.", 215),
    "pick_bottle": ("Pick up the water bottle by its body with arm A.", 150),
    "pour_water": ("Hold the mug with arm B and pour water from the bottle with arm A.", 750),
}

# Welds each skill may attach/detach: (weld name, gripper ctrl index)
SKILL_WELDS = {
    "open_drawer": [("weld_drawer", 5)],
    "pick_plate": [("weld_plate", 5)],
    "place_plate": [("weld_plate", 5)],
    "pick_mug": [("weld_mug", 11)],
    "pick_bottle": [("weld_bottle", 5)],
    "pour_water": [("weld_mug", 11), ("weld_bottle", 5)],
}

# Scripted setup for --isolated mode (mirrors scripts/record_skill_demos.py)
PREREQUISITES = {
    "open_drawer": [],
    "pick_plate": [OpenDrawerPrimitive],
    "place_plate": [OpenDrawerPrimitive, PickPlatePrimitive],
    "pick_mug": [],
    "pick_bottle": [PickMugPrimitive],
    "pour_water": [PickMugPrimitive, PickBottlePrimitive],
}

RELEASE_MARGIN = 0.20  # rad the gripper must open past its grasp angle to release (plate: 1.36 -> 1.60)
FULLY_OPEN = 1.55     # GRIPPER_OPEN is 1.60


class _WeldHelper(BaseManipulationPrimitive):
    """Reuses the primitives' contact checks and weld attach/detach."""

    def execute(self) -> bool:  # pragma: no cover - not a skill
        return True


def find_checkpoint(skill: str, step: str | None) -> Path | None:
    ckpt_root = TRAIN_DIR / f"act_{skill}" / "checkpoints"
    if not ckpt_root.is_dir():
        return None
    candidates = sorted(p for p in ckpt_root.iterdir() if p.is_dir())
    if step:
        candidates = [p for p in candidates if p.name == step]
    usable = [p / "pretrained_model" for p in candidates if (p / "pretrained_model" / "model.safetensors").is_file()]
    return usable[-1] if usable else None


def configure_replanning(motor_policy, n_action_steps: int | None, ensemble_coeff: float | None) -> None:
    """Shorten ACT's open-loop horizon; the checkpoints were trained to execute 100 steps blind."""
    act = motor_policy._policy
    if ensemble_coeff is not None:
        from lerobot.policies.act.modeling_act import ACTTemporalEnsembler

        act.config.temporal_ensemble_coeff = ensemble_coeff
        act.config.n_action_steps = 1
        act.temporal_ensembler = ACTTemporalEnsembler(ensemble_coeff, act.config.chunk_size)
    elif n_action_steps is not None:
        act.config.n_action_steps = n_action_steps
    act.reset()


class WeldTracker:
    def __init__(self, helper: _WeldHelper):
        self.helper = helper
        self.grasp_ctrl: dict[str, float] = {}

    def update(self, skill: str, ctrl: np.ndarray) -> list[str]:
        events = []
        for weld, slot in SKILL_WELDS[skill]:
            g = float(ctrl[slot])
            if not self.helper.weld_active(weld):
                _, b1, _ = self.helper._weld_bodies(weld)
                touching = self.helper.gripper_contact_bodies(weld)
                if b1 in touching and any(b != b1 for b in touching):
                    if self.helper.attach_weld(weld, require_both_jaws=True):
                        self.grasp_ctrl[weld] = g
                        events.append(f"attach {weld}")
                        if weld == "weld_plate":
                            self.helper.detach_weld("weld_plate_drawer")
            elif g > min(self.grasp_ctrl.get(weld, g) + RELEASE_MARGIN, FULLY_OPEN):
                self.helper.detach_weld(weld)
                self.grasp_ctrl.pop(weld, None)
                events.append(f"detach {weld}")
        return events


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all trained ACT skill policies in sequence in MuJoCo")
    parser.add_argument("--skills", nargs="+", choices=list(SKILLS), default=list(SKILLS))
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--steps", type=int, default=None, help="policy steps per skill (default: per-skill)")
    parser.add_argument("--checkpoint-step", default=None, help="e.g. 002500; default = latest with weights")
    parser.add_argument("--isolated", action="store_true", help="fresh scene + scripted prerequisites per skill")
    parser.add_argument("--view", action="store_true", help="open the MuJoCo 3D viewer")
    parser.add_argument("--realtime", type=float, default=1.0, help="viewer speed factor (1.0 = real time)")
    parser.add_argument("--record", default=None, help="save an annotated overhead MP4")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-action-steps", type=int, default=None,
                        help="re-plan every N steps instead of the trained 100 (e.g. 10)")
    parser.add_argument("--temporal-ensemble", type=float, default=None, metavar="COEFF",
                        help="ACT temporal ensembling, re-plans every step (paper uses 0.01)")
    args = parser.parse_args()

    policies = {}
    for skill in args.skills:
        ckpt = find_checkpoint(skill, args.checkpoint_step)
        if ckpt is None:
            print(f"[skip] {skill}: no checkpoint with model.safetensors")
            continue
        sel = select_motor_policy(ckpt, backend="pytorch", device=args.device, load=True)
        if sel.policy is None:
            print(f"[skip] {skill}: {sel.reason}")
            continue
        print(f"[load] {skill}: {ckpt.relative_to(ROOT)}")
        configure_replanning(sel.policy, args.n_action_steps, args.temporal_ensemble)
        policies[skill] = sel.policy
    if not policies:
        print("No policies loaded.")
        return 1

    import mujoco.viewer as mj_viewer

    sim = viewer = renderer = helper = tracker = None
    frames: list[np.ndarray] = []

    def new_scene():
        nonlocal sim, viewer, renderer, helper, tracker
        if viewer is not None:
            viewer.close()
        sim = reset_scene(args.seed)
        renderer = mujoco.Renderer(sim.model, height=480, width=640)
        helper = _WeldHelper(TrajectoryExecutor(sim.model, sim.data), sim)
        tracker = WeldTracker(helper)
        if args.view:
            viewer = mj_viewer.launch_passive(sim.model, sim.data)

    new_scene()
    results = {}
    for skill, policy in policies.items():
        instruction, default_steps = SKILLS[skill]
        steps = args.steps or default_steps

        if args.isolated:
            new_scene()
            executor = TrajectoryExecutor(sim.model, sim.data)
            for prereq in PREREQUISITES[skill]:
                print(f"  [setup] scripted {prereq.__name__}")
                if not prereq(executor, sim).execute():
                    print(f"  [setup] {prereq.__name__} failed")
                if viewer is not None:
                    viewer.sync()
            tracker.grasp_ctrl = {
                w: float(sim.data.ctrl[s]) for w, s in SKILL_WELDS[skill] if helper.weld_active(w)
            }

        print(f"\n=== {skill}: '{instruction}' ({steps} steps)")
        policy.reset()
        for i in range(steps):
            if viewer is not None and not viewer.is_running():
                print("Viewer closed; stopping.")
                return 0
            t0 = time.perf_counter()
            state = np.asarray(sim.data.qpos[36:48], dtype=np.float32).tolist()
            renderer.update_scene(sim.data, camera="overhead_cam")
            rgb = renderer.render().copy()

            action = policy.act(state, rgb, instruction)
            sim.data.ctrl[:12] = np.asarray(action, dtype=np.float64)
            for _ in range(SUBSTEPS):
                mujoco.mj_step(sim.model, sim.data)
            for ev in tracker.update(skill, sim.data.ctrl):
                print(f"  step {i+1:4d}: {ev}")

            if args.record:
                cv2.putText(rgb, f"{skill}  {i+1}/{steps}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                frames.append(rgb)
            if viewer is not None:
                viewer.sync()
                dt = SUBSTEPS * sim.model.opt.timestep / max(args.realtime, 1e-3)
                time.sleep(max(0.0, dt - (time.perf_counter() - t0)))

        active = [w for w, _ in SKILL_WELDS[skill] if helper.weld_active(w)]
        results[skill] = active
        print(f"  done. welds held at end: {active or 'none'}")

    print("\nSummary (welds held at end of each skill):")
    for skill, active in results.items():
        print(f"  {skill:12s} {active or '-'}")

    if args.record and frames:
        out = Path(args.record)
        out.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (640, 480))
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"Video saved to {out.resolve()}")

    if viewer is not None:
        print("Close the viewer window to exit.")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
