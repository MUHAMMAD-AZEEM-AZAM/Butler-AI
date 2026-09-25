"""Validate a local LeRobotDataset v3.0 directory against the Butler demonstration schema.

    python -m stage3_policy.learned.validate_dataset --root data/butler_demos/pick

Metadata checks need only the standard library. Frame-level checks (timestamps,
finite and in-range joint values) read data/*.parquet with pyarrow; without
pyarrow they are reported as SKIPPED, never as passed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from stage3_policy.learned import schema

MIN_SUCCESSFUL_EPISODES_PER_SKILL = 50  # ACT docs: "often achieves high success rates with just 50 demonstrations"
TIMESTAMP_TOLERANCE_S = 1e-4  # lerobot 0.6.1 default tolerance_s
JOINT_LIMIT_TOLERANCE_RAD = 1e-3


@dataclass
class ValidationReport:
    root: Path
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def format(self) -> str:
        if self.errors:
            status = "INVALID"
        elif self.skipped:
            status = "metadata valid, some checks SKIPPED (not verified)"
        else:
            status = "VALID"
        lines = [f"dataset: {self.root}", f"status: {status}"]
        for title, items in (("error", self.errors), ("warning", self.warnings), ("skipped", self.skipped)):
            lines.extend(f"  {title}: {item}" for item in items)
        lines.extend(f"  {key}: {value}" for key, value in self.stats.items())
        return "\n".join(lines)


def validate_dataset(
    root: str | Path,
    *,
    min_successful_episodes: int = MIN_SUCCESSFUL_EPISODES_PER_SKILL,
    check_frames: bool = True,
) -> ValidationReport:
    report = ValidationReport(root=Path(root))
    info = _read_json(report, "meta/info.json")
    if info is not None:
        _check_info(info, report)
    episodes = _read_json(report, schema.EPISODE_METADATA_FILE)
    if info is not None and episodes is not None:
        _check_episode_metadata(info, episodes, report, min_successful_episodes)

    if not check_frames:
        report.skipped.append("frame-level checks were disabled by the caller")
    elif report.errors:
        report.skipped.append("frame-level checks were not run because the metadata is invalid")
    else:
        columns = _load_frame_columns(report.root)
        if columns is None:
            report.skipped.append("frame-level checks (timestamps, finite and in-range joints): pyarrow is not installed")
        elif not columns:
            report.errors.append("no data/*.parquet files found")
        else:
            report.errors.extend(check_frame_columns(columns, fps=info["fps"], total_episodes=info.get("total_episodes")))
    return report


def check_frame_columns(columns: Mapping[str, Sequence[Any]], *, fps: float, total_episodes: int | None) -> list[str]:
    """Check frame rows given as plain column lists (as read from data/*.parquet)."""
    required = ("episode_index", "frame_index", "timestamp", schema.STATE_KEY, schema.ACTION_KEY)
    missing = [name for name in required if name not in columns]
    if missing:
        return [f"data parquet is missing columns {missing}"]
    if len({len(columns[name]) for name in required}) != 1:
        return ["data parquet columns have different lengths"]

    errors: list[str] = []
    by_episode: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for episode, frame, timestamp in zip(columns["episode_index"], columns["frame_index"], columns["timestamp"]):
        by_episode[int(episode)].append((int(frame), float(timestamp)))
    if total_episodes is not None and len(by_episode) != total_episodes:
        errors.append(f"data has {len(by_episode)} episodes but meta/info.json says {total_episodes}")
    for episode, frames in sorted(by_episode.items()):
        frames.sort()
        if [frame for frame, _ in frames] != list(range(len(frames))):
            errors.append(f"episode {episode}: frame_index is not 0..{len(frames) - 1} without gaps")
            continue
        for frame, timestamp in frames:
            if abs(timestamp - frame / fps) > TIMESTAMP_TOLERANCE_S:
                errors.append(f"episode {episode} frame {frame}: timestamp {timestamp} != frame_index / fps = {frame / fps:.6f}")
                break

    limits = [schema.JOINT_LIMITS_RAD[name.split("_", 1)[1]] for name in schema.MOTOR_NAMES]
    for key in (schema.STATE_KEY, schema.ACTION_KEY):
        invalid, first = 0, None
        for row, vector in enumerate(columns[key]):
            problem = _vector_problem(list(vector), limits)
            if problem:
                invalid += 1
                first = first or f"row {row}: {problem}"
        if invalid:
            errors.append(f"{key}: {invalid} invalid row(s); first at {first}")
    return errors


def _vector_problem(values: list[float], limits: list[tuple[float, float]]) -> str | None:
    if len(values) != len(schema.MOTOR_NAMES):
        return f"{len(values)} values instead of {len(schema.MOTOR_NAMES)}"
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
        return "non-finite value"
    for name, value, (low, high) in zip(schema.MOTOR_NAMES, values, limits):
        if value < low - JOINT_LIMIT_TOLERANCE_RAD or value > high + JOINT_LIMIT_TOLERANCE_RAD:
            return f"{name}={value:.4f} rad outside [{low}, {high}]"
    return None


def _read_json(report: ValidationReport, relative: str) -> Any | None:
    path = report.root / relative
    if not path.is_file():
        report.errors.append(f"missing {relative}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        report.errors.append(f"cannot read {relative}: {exc}")
        return None


def _check_info(info: Any, report: ValidationReport) -> None:
    if not isinstance(info, Mapping):
        report.errors.append("meta/info.json must be a JSON object")
        return
    if info.get("codebase_version") != schema.CODEBASE_VERSION:
        report.errors.append(f"codebase_version is {info.get('codebase_version')!r}, expected {schema.CODEBASE_VERSION!r}")
    if info.get("fps") != schema.FPS:
        report.errors.append(
            f"fps is {info.get('fps')!r}; the schema records one frame every {schema.PHYSICS_STEPS_PER_FRAME} "
            f"physics steps of {schema.PHYSICS_TIMESTEP_S} s = {schema.FPS} fps"
        )
    if info.get("robot_type") != schema.ROBOT_TYPE:
        report.warnings.append(f"robot_type is {info.get('robot_type')!r}, expected {schema.ROBOT_TYPE!r}")

    features = info.get("features")
    if not isinstance(features, Mapping):
        report.errors.append("meta/info.json has no features mapping")
        return
    for key in schema.DEFAULT_FEATURE_KEYS:
        if key not in features:
            report.errors.append(f"missing default LeRobot feature {key!r}")
    for key, expected in schema.lerobot_features().items():
        actual = features.get(key)
        if not isinstance(actual, Mapping):
            report.errors.append(f"missing feature {key!r}")
            continue
        allowed = {"video", "image"} if key == schema.IMAGE_KEY else {expected["dtype"]}
        if actual.get("dtype") not in allowed:
            report.errors.append(f"{key}: dtype {actual.get('dtype')!r}, expected one of {sorted(allowed)}")
        if list(actual.get("shape") or []) != list(expected["shape"]):
            report.errors.append(f"{key}: shape {actual.get('shape')!r}, expected {list(expected['shape'])}")
        if key in (schema.STATE_KEY, schema.ACTION_KEY) and list(actual.get("names") or []) != expected["names"]:
            report.errors.append(f"{key}: names must be the MuJoCo actuator order {expected['names']}")

    total = info.get("total_episodes")
    if not isinstance(total, int) or isinstance(total, bool) or total < 1:
        report.errors.append(f"total_episodes must be a positive integer, got {total!r}")
    report.stats["total_episodes"] = total
    report.stats["total_frames"] = info.get("total_frames")


def _check_episode_metadata(info: Mapping[str, Any], meta: Any, report: ValidationReport, min_successful: int) -> None:
    entries = meta.get("episodes") if isinstance(meta, Mapping) else None
    if not isinstance(entries, list):
        report.errors.append(f"{schema.EPISODE_METADATA_FILE} must contain an 'episodes' list")
        return
    total = info.get("total_episodes")
    indices = sorted(e.get("episode_index") for e in entries if isinstance(e, Mapping) and isinstance(e.get("episode_index"), int))
    if isinstance(total, int) and indices != list(range(total)):
        report.errors.append(f"{schema.EPISODE_METADATA_FILE}: episode_index values must be exactly 0..{total - 1}")

    per_skill, successful, splits = Counter(), Counter(), Counter()
    seeds_by_split: dict[str, set[int]] = defaultdict(set)
    for entry in entries:
        if not isinstance(entry, Mapping):
            report.errors.append(f"{schema.EPISODE_METADATA_FILE}: every episode entry must be an object")
            continue
        label = f"episode {entry.get('episode_index')}"
        seed, skill, split, success = entry.get("seed"), entry.get("skill"), entry.get("split"), entry.get("success")
        if not isinstance(seed, int) or isinstance(seed, bool):
            report.errors.append(f"{label}: seed must be an integer")
        elif seed in schema.EVAL_SEEDS:
            report.errors.append(f"{label}: seed {seed} is an evaluation seed (configs/default.yaml); demonstrations must not use it")
        if skill not in schema.SKILLS:
            report.errors.append(f"{label}: skill {skill!r} not in {list(schema.SKILLS)}")
        if entry.get("arm") not in ("A", "B"):
            report.errors.append(f"{label}: arm must be 'A' or 'B'")
        if not isinstance(entry.get("instruction"), str) or not entry["instruction"].strip():
            report.errors.append(f"{label}: instruction must be a non-empty string")
        if split not in ("train", "val"):
            report.errors.append(f"{label}: split must be 'train' or 'val'")
        if not isinstance(success, bool):
            report.errors.append(f"{label}: success must be true or false")
        if entry.get("physics_only") is not True:
            report.errors.append(
                f"{label}: physics_only must be true. Episodes where objects are moved by writing qpos directly "
                "(teleporting) do not show how the commanded actions move objects and cannot train a motor policy"
            )
        per_skill[skill] += 1
        splits[split] += 1
        if isinstance(seed, int) and split in ("train", "val"):
            seeds_by_split[split].add(seed)
        if success is True:
            successful[skill] += 1

    for seed in sorted(seeds_by_split["train"] & seeds_by_split["val"]):
        report.errors.append(f"seed {seed} appears in both train and val splits; held-out evaluation must use unseen seeds")
    for skill in sorted(s for s in per_skill if s in schema.SKILLS):
        if successful[skill] < min_successful:
            report.warnings.append(
                f"skill {skill!r} has {successful[skill]} successful episodes; ACT docs suggest about {min_successful} as a starting point"
            )
    if splits["val"] == 0:
        report.warnings.append("no 'val' episodes: policy quality cannot be measured on held-out seeds")
    report.stats.update(
        episodes_per_skill=dict(per_skill), successful_per_skill=dict(successful), episodes_per_split=dict(splits)
    )


def _load_frame_columns(root: Path) -> dict[str, list[Any]] | None:
    """Read the columns check_frame_columns needs; None if pyarrow is unavailable, {} if there are no files."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        return None
    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        return {}
    wanted = ["episode_index", "frame_index", "timestamp", schema.STATE_KEY, schema.ACTION_KEY]
    tables = [pq.read_table(path, columns=[c for c in wanted if c in pq.read_schema(path).names]) for path in files]
    table = pa.concat_tables(tables)
    return {name: table.column(name).to_pylist() for name in table.column_names}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m stage3_policy.learned.validate_dataset", description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True, help="local LeRobotDataset directory (contains meta/ and data/)")
    parser.add_argument("--min-successful-episodes", type=int, default=MIN_SUCCESSFUL_EPISODES_PER_SKILL)
    args = parser.parse_args(argv)
    report = validate_dataset(args.root, min_successful_episodes=args.min_successful_episodes)
    print(report.format())
    return 0 if report.ok and not report.skipped else 1


if __name__ == "__main__":
    sys.exit(main())
