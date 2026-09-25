"""Build, check and (only when everything is ready) run a bounded LeRobot ACT training job.

    python -m stage3_policy.learned.train --dataset-root data/butler_demos/pick
    python -m stage3_policy.learned.train --dataset-root data/butler_demos/pick --run

The default is a dry run: print the lerobot-train command and the preflight
results. --run refuses unless lerobot is installed and the dataset passes every
validation check (including frame checks). The default 200-step CPU run is a
plumbing smoke test only; its checkpoint is not evidence of policy quality.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from stage3_policy.learned.validate_dataset import validate_dataset


@dataclass(frozen=True)
class TrainSettings:
    dataset_root: Path
    repo_id: str = "local/butler_demos"  # a label for the local dataset; nothing is downloaded or uploaded
    job_name: str = "act_butler_smoke"
    output_dir: Path = Path("outputs/train/act_butler_smoke")
    device: str = "cpu"
    steps: int = 200
    batch_size: int = 8
    save_freq: int = 200
    seed: int = 1000


def build_train_command(settings: TrainSettings) -> list[str]:
    """lerobot-train flags as defined by lerobot 0.6.1 TrainPipelineConfig / PreTrainedConfig."""
    return [
        "lerobot-train",
        f"--dataset.repo_id={settings.repo_id}",
        f"--dataset.root={settings.dataset_root}",
        "--policy.type=act",
        f"--policy.device={settings.device}",
        "--policy.push_to_hub=false",  # PreTrainedConfig defaults to pushing to the Hugging Face Hub
        f"--output_dir={settings.output_dir}",
        f"--job_name={settings.job_name}",
        f"--steps={settings.steps}",
        f"--batch_size={settings.batch_size}",
        f"--save_freq={settings.save_freq}",
        f"--seed={settings.seed}",
        "--wandb.enable=false",
    ]


def preflight(settings: TrainSettings) -> list[str]:
    """Every reason the job must not start yet (empty list = ready)."""
    problems: list[str] = []
    if importlib.util.find_spec("lerobot") is None:
        problems.append("lerobot is not installed in this interpreter (lerobot 0.6.1 needs Python >= 3.12; see stage3_policy/README.md)")
    elif shutil.which("lerobot-train") is None:
        problems.append("the lerobot-train command is not on PATH")
    report = validate_dataset(settings.dataset_root)
    problems.extend(f"dataset: {error}" for error in report.errors)
    problems.extend(f"dataset check not verified: {item}" for item in report.skipped)
    if settings.output_dir.exists():
        problems.append(f"{settings.output_dir} already exists; choose a new --job-name so earlier results are not mixed in")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m stage3_policy.learned.train", description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--job-name", default=TrainSettings.job_name)
    parser.add_argument("--device", default=TrainSettings.device, help="cpu, cuda or xpu - check your install first")
    parser.add_argument("--steps", type=int, default=TrainSettings.steps)
    parser.add_argument("--batch-size", type=int, default=TrainSettings.batch_size)
    parser.add_argument("--run", action="store_true", help="actually start training (refused unless preflight passes)")
    args = parser.parse_args(argv)

    settings = TrainSettings(
        dataset_root=args.dataset_root,
        job_name=args.job_name,
        output_dir=Path("outputs") / "train" / args.job_name,
        device=args.device,
        steps=args.steps,
        batch_size=args.batch_size,
        save_freq=args.steps,
    )
    command = build_train_command(settings)
    print("command:", subprocess.list2cmdline(command))
    problems = preflight(settings)
    print("preflight:", "ready" if not problems else "NOT READY")
    for problem in problems:
        print(f"  - {problem}")

    if not args.run:
        return 0
    if problems:
        print("refusing to start training until every preflight problem is fixed")
        return 2
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
