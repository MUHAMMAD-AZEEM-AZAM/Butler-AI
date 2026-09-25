"""Build a 100% LeRobot 0.6.1 (v3.0) compliant dataset for one atomic manipulation skill.

Generalized from the open_drawer-only version: any of the six primitives in
stage4_bimanual/primitives.py can be recorded. Prerequisite primitives (e.g. the
drawer must be open before a plate can be picked) are executed in the same real
scene but not recorded, exactly as scripts/record_skill_demos.py does for its
raw-bundle collector -- this script instead writes directly into LeRobotDataset
v3.0 format in one pass (no separate convert_demos.py step), which is what was
actually used to build data/butler_demos/open_drawer/.

    python scripts/build_clean_dataset.py --skill open_drawer
    python scripts/build_clean_dataset.py --skill pour_water --train-count 6 --val-count 2
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from stage3_policy.learned import schema
from stage3_policy.learned.validate_dataset import validate_dataset
from stage4_bimanual.bimanual import reset_scene
from stage4_bimanual.primitives import (
    OpenDrawerPrimitive,
    PickBottlePrimitive,
    PickMugPrimitive,
    PickPlatePrimitive,
    PlacePlatePrimitive,
    PourWaterPrimitive,
)
from stage4_bimanual.trajectory import TrajectoryExecutor

# --skill (this script's directory-level name) -> (primitive class, task instruction, arm, object)
SKILLS = {
    "open_drawer": (OpenDrawerPrimitive, "Open the top drawer with arm A.", "A", "top_drawer"),
    "pick_plate": (PickPlatePrimitive, "Pick up the plate from the open drawer with arm A.", "A", "plate"),
    "place_plate": (PlacePlatePrimitive, "Place the plate at the center of the table with arm A.", "A", "plate"),
    "pick_mug": (PickMugPrimitive, "Pick up and hold the mug with arm B.", "B", "mug"),
    "pick_bottle": (PickBottlePrimitive, "Pick up the water bottle by its body with arm A.", "A", "water_bottle"),
    "pour_water": (PourWaterPrimitive, "Hold the mug with arm B and pour water from the bottle with arm A.", "A", "water_bottle"),
}

# schema.SKILLS is the coarser 4-category taxonomy ("one ACT policy per skill" -
# pick/place/pour each cover multiple objects). validate_dataset.py requires the
# butler_episodes.json "skill" field to be one of these, not our 6 directory names.
GENERIC_SKILL = {
    "open_drawer": "open_drawer",
    "pick_plate": "pick",
    "place_plate": "place",
    "pick_mug": "pick",
    "pick_bottle": "pick",
    "pour_water": "pour",
}

# Same dependency chain as scripts/record_skill_demos.py -- executed unrecorded
# in the same real scene so the recorded skill starts from a realistic state.
PREREQUISITES = {
    "open_drawer": [],
    "pick_plate": [OpenDrawerPrimitive],
    "place_plate": [OpenDrawerPrimitive, PickPlatePrimitive],
    "pick_mug": [],
    "pick_bottle": [PickMugPrimitive],
    "pour_water": [PickMugPrimitive, PickBottlePrimitive],
}


class FastDemonstrationExecutor(TrajectoryExecutor):
    """Samples 12-DoF state, actions and overhead camera at 25 Hz."""

    def __init__(self, model, data, fps: int = 25):
        super().__init__(model, data)
        self.sample_interval = max(1, round(1 / (fps * model.opt.timestep)))
        self.renderer = mujoco.Renderer(model, height=480, width=640)
        self.states: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.frames: list[np.ndarray] = []
        self._substeps = 0

    def capture(self) -> None:
        self.states.append(np.asarray(self.data.qpos[36:48], dtype=np.float32).copy())
        self.actions.append(np.asarray(self.data.ctrl[:12], dtype=np.float32).copy())
        self.renderer.update_scene(self.data, camera="overhead_cam")
        self.frames.append(self.renderer.render().copy())

    def interpolate(self, target_ctrl, steps: int = 60) -> None:
        start = np.copy(self.data.ctrl)
        target = np.asarray(target_ctrl, dtype=np.float64)
        for step in range(steps):
            alpha = 0.5 * (1.0 - np.cos(np.pi * (step + 1) / steps))
            self.data.ctrl[:] = start + alpha * (target - start)
            mujoco.mj_step(self.model, self.data)
            self._substeps += 1
            if self._substeps % self.sample_interval == 0:
                self.capture()
        self.data.ctrl[:] = target
        for _ in range(min(25, max(10, steps // 3))):
            mujoco.mj_step(self.model, self.data)
            self._substeps += 1
            if self._substeps % self.sample_interval == 0:
                self.capture()


def _run_prerequisites(skill: str, executor: FastDemonstrationExecutor, sim) -> None:
    """Execute (but do not record) any primitives the target skill depends on."""
    for primitive_class in PREREQUISITES[skill]:
        if not primitive_class(executor, sim).execute():
            raise RuntimeError(f"prerequisite {primitive_class.__name__} failed")
    # Prerequisite frames/states are only there to reach a realistic start state;
    # the recorded episode must start clean at the target skill's beginning.
    executor.states.clear()
    executor.actions.clear()
    executor.frames.clear()
    executor.capture()


def build_dataset(
    skill: str,
    out_dir: Path,
    train_seeds: list[int],
    val_seeds: list[int],
    jitter: float = 1.0,
) -> None:
    primitive_class, task_desc, arm, obj = SKILLS[skill]

    if out_dir.exists():
        shutil.rmtree(out_dir)

    print(f"Initializing LeRobotDataset v3.0 at {out_dir} for skill '{skill}'...", flush=True)
    dataset = LeRobotDataset.create(
        repo_id="local/butler_demos",
        fps=schema.FPS,
        features=schema.lerobot_features(use_videos=True),
        root=out_dir,
        robot_type=schema.ROBOT_TYPE,
        use_videos=True,
    )

    all_jobs = [("train", s) for s in train_seeds] + [("val", s) for s in val_seeds]
    episodes_meta = []

    for ep_idx, (split, seed) in enumerate(all_jobs):
        t_start = time.time()
        sim = reset_scene(seed, trajectory_jitter=jitter)
        executor = FastDemonstrationExecutor(sim.model, sim.data, fps=schema.FPS)
        try:
            _run_prerequisites(skill, executor, sim)
            primitive = primitive_class(executor, sim)
            success = primitive.execute()
        except Exception as exc:
            print(f"[{ep_idx+1}/{len(all_jobs)}] Seed {seed} ({split}) ERROR: {exc}", flush=True)
            continue
        n_frames = len(executor.frames)

        if not success or n_frames < 5:
            print(f"[{ep_idx+1}/{len(all_jobs)}] Seed {seed} ({split}) FAILED (success={success}, frames={n_frames})", flush=True)
            continue

        for i in range(n_frames):
            is_last = (i == n_frames - 1)
            dataset.add_frame({
                schema.STATE_KEY: executor.states[i],
                schema.ACTION_KEY: executor.actions[i],
                schema.IMAGE_KEY: executor.frames[i],
                schema.DONE_KEY: np.array([is_last]),
                schema.SUCCESS_KEY: np.array([is_last and success]),
                "task": task_desc,
            })
        dataset.save_episode()

        episodes_meta.append({
            "episode_index": ep_idx,
            "seed": seed,
            "skill": GENERIC_SKILL[skill],
            "primitive": skill,
            "arm": arm,
            "object": obj,
            "instruction": task_desc,
            "split": split,
            "success": bool(success),
            "physics_only": True,
        })
        elapsed = time.time() - t_start
        print(f"[{ep_idx+1}/{len(all_jobs)}] Seed {seed} ({split}): {n_frames} frames in {elapsed:.1f}s (ACCEPTED)", flush=True)

    print("Finalizing video encoders and metadata...", flush=True)
    dataset.finalize()

    butler_meta_file = out_dir / schema.EPISODE_METADATA_FILE
    butler_meta_file.parent.mkdir(parents=True, exist_ok=True)
    butler_meta_file.write_text(json.dumps({"episodes": episodes_meta}, indent=2), encoding="utf-8")
    print(f"Saved {butler_meta_file}", flush=True)

    print("\nRunning validator...", flush=True)
    report = validate_dataset(out_dir, min_successful_episodes=1)
    print(report.format())


def main():
    parser = argparse.ArgumentParser(description="Generate a LeRobot v3.0 dataset for one atomic skill.")
    parser.add_argument("--skill", required=True, choices=sorted(SKILLS))
    parser.add_argument("--out", type=Path, default=None, help="default: data/butler_demos/<skill>")
    parser.add_argument("--train-count", type=int, default=6)
    parser.add_argument("--val-count", type=int, default=2)
    parser.add_argument("--jitter", type=float, default=1.0)
    parser.add_argument("--train-seed-start", type=int, default=100)
    parser.add_argument("--val-seed-start", type=int, default=200)
    args = parser.parse_args()

    out_dir = args.out or (ROOT / "data" / "butler_demos" / args.skill)
    train_seeds = [args.train_seed_start + i for i in range(args.train_count)]
    val_seeds = [args.val_seed_start + i for i in range(args.val_count)]
    build_dataset(args.skill, out_dir, train_seeds, val_seeds, jitter=args.jitter)


if __name__ == "__main__":
    main()
