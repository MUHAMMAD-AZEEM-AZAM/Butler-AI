#!/usr/bin/env python3
"""Convert raw bimanual_raw_demo_v1 bundles into a LeRobotDataset using the installed API.

    python -m stage3_policy.learned.convert_demos \
        --train data/smoke_demos/train/open_drawer \
        --val   data/smoke_demos/val/open_drawer \
        --out   data/butler_demos/open_drawer

Nothing is invented. Every field comes from the recorder's own manifest.json and
trajectory.npz, except:
  * `split`  - assigned by which --train/--val directory the episode came from.
  * `arm`    - looked up in SKILL_ARM below, which mirrors the arm named in the
               recorder's own task string (e.g. "Open the top drawer with arm A.").
Both are labelling decisions about data we recorded, not fabricated observations.

The dataset fps is read from the recording's real timestamps and must match
schema.FPS exactly; a mismatch aborts rather than relabelling the data.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from stage3_policy.learned import schema

# The arm each skill uses, per the recorder's task strings in scripts/record_skill_demos.py.
SKILL_ARM = {
    "open_drawer": "A",
    "pick_plate": "A",
    "place_plate": "A",
    "pick_mug": "B",
    "pick_bottle": "A",
    "pour_water": "A",
}


def _episodes(root: Path) -> list[Path]:
    return sorted(p for p in root.glob("episode_*") if (p / "manifest.json").is_file())


def _check_fps(manifest: dict, timestamps: np.ndarray, label: str) -> None:
    """The recording's real rate must equal schema.FPS. Never relabel."""
    deltas = np.diff(timestamps)
    if len(deltas) == 0:
        raise SystemExit(f"{label}: episode has fewer than 2 frames")
    if not np.allclose(deltas, deltas[0], atol=1e-9):
        raise SystemExit(
            f"{label}: timestamps are not uniformly spaced "
            f"(min {deltas.min():.6f}, max {deltas.max():.6f}); LeRobot requires "
            "timestamp == frame_index / fps"
        )
    measured = 1.0 / float(deltas[0])
    if abs(measured - schema.FPS) > 1e-6:
        raise SystemExit(
            f"{label}: recorded at {measured:.6f} fps but schema.FPS is {schema.FPS}. "
            "Re-record at the schema rate or change schema.FPS deliberately - "
            "this converter will not relabel the frame rate."
        )


def convert(train_dir: Path, val_dir: Path | None, out: Path, repo_id: str) -> int:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    groups: list[tuple[str, Path]] = [("train", train_dir)]
    if val_dir is not None:
        groups.append(("val", val_dir))

    if out.exists():
        raise SystemExit(f"{out} already exists; remove it or choose another --out")

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=schema.FPS,
        features=schema.lerobot_features(use_videos=True),
        root=out,
        robot_type=schema.ROBOT_TYPE,
        use_videos=True,
    )

    episode_meta: list[dict] = []
    episode_index = 0
    for split, root in groups:
        for ep in _episodes(root):
            manifest = json.loads((ep / "manifest.json").read_text(encoding="utf-8"))
            bundle = np.load(ep / "trajectory.npz")
            states = bundle["observation_state"].astype(np.float32)
            actions = bundle["action"].astype(np.float32)
            timestamps = bundle["timestamp"]
            label = f"{split}/{ep.name}"

            if manifest.get("accepted_for_training") is not True:
                print(f"  skipping {label}: accepted_for_training is not true")
                continue
            _check_fps(manifest, timestamps, label)

            frames_dir = ep / "images" / "overhead"
            images = sorted(frames_dir.glob("*.png"))
            if not (len(images) == len(states) == len(actions)):
                raise SystemExit(
                    f"{label}: {len(images)} overhead images, {len(states)} states, "
                    f"{len(actions)} actions - they must match one-to-one"
                )

            success = bool(manifest["primitive_returned_success"])
            task = manifest["task"]
            for i, image_path in enumerate(images):
                image = np.asarray(Image.open(image_path).convert("RGB"))
                if image.shape != schema.IMAGE_SHAPE:
                    raise SystemExit(f"{label} frame {i}: image {image.shape}, expected {schema.IMAGE_SHAPE}")
                last = i == len(images) - 1
                dataset.add_frame(
                    {
                        schema.STATE_KEY: states[i],
                        schema.ACTION_KEY: actions[i],
                        schema.IMAGE_KEY: image,
                        schema.DONE_KEY: np.array([last]),
                        schema.SUCCESS_KEY: np.array([last and success]),
                        "task": task,
                    }
                )
            dataset.save_episode()

            skill = manifest["skill"]
            episode_meta.append(
                {
                    "episode_index": episode_index,
                    "seed": int(manifest["seed"]),
                    "skill": skill,
                    "arm": SKILL_ARM[skill],
                    "object": manifest.get("object", skill),
                    "instruction": task,
                    "split": split,
                    "success": success,
                    "physics_only": bool(manifest["physics_only"]),
                }
            )
            print(f"  episode {episode_index:3d}  {label}  seed {manifest['seed']}  {len(states)} frames  split={split}")
            episode_index += 1

    if not episode_meta:
        shutil.rmtree(out, ignore_errors=True)
        raise SystemExit("no accepted episodes were converted")

    meta_path = out / schema.EPISODE_METADATA_FILE
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps({"episodes": episode_meta}, indent=2), encoding="utf-8")
    print(f"\nwrote {episode_index} episodes to {out}")
    print(f"wrote {meta_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--val", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--repo-id", default="local/butler_demos")
    args = parser.parse_args(argv)
    return convert(args.train, args.val, args.out, args.repo_id)


if __name__ == "__main__":
    sys.exit(main())
