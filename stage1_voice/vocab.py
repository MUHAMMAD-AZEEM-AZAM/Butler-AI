"""Allowed vocabulary for stage 1: loaded from configs/default.yaml (scene section).

The action list comes from common.types.ActionType — never duplicated here.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from common.types import ActionType, Task

from stage1_voice.errors import UnknownObjectError

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


@dataclass(frozen=True)
class Vocabulary:
    """Allowed names for Task fields. Everything except actions comes from the config."""

    objects: tuple[str, ...]
    drawers: tuple[str, ...]
    destinations: tuple[str, ...]
    arms: tuple[str, ...]
    actions: tuple[str, ...]  # from the ActionType enum


def _load_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_vocab(config_path: Path = CONFIG_PATH) -> Vocabulary:
    """Load the scene vocabulary from the shared config."""
    scene = _load_config(config_path)["scene"]
    return Vocabulary(
        objects=tuple(scene["objects"]),
        drawers=tuple(scene["drawers"]),
        destinations=tuple(scene["destinations"]),
        arms=tuple(str(a) for a in scene["arms"]),
        actions=tuple(action.value for action in ActionType),
    )


def load_voice_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the voice section (model id, max_retries) from the shared config."""
    return _load_config(config_path)["voice"]


def validate_task_vocabulary(task: Task, vocab: Vocabulary) -> None:
    """Check every name in the Task against the allowed vocabulary.

    Raises UnknownObjectError listing all out-of-vocabulary names.
    """
    problems: list[str] = []
    for step in task.steps:
        if step.target is not None and step.target not in vocab.drawers:
            problems.append(f"step {step.id}: target '{step.target}' not in drawers {list(vocab.drawers)}")
        if step.object is not None and step.object not in vocab.objects:
            problems.append(f"step {step.id}: object '{step.object}' not in objects {list(vocab.objects)}")
        if step.destination is not None and step.destination not in vocab.destinations:
            problems.append(
                f"step {step.id}: destination '{step.destination}' not in destinations {list(vocab.destinations)}"
            )
        if step.source is not None and step.source not in vocab.objects:
            problems.append(f"step {step.id}: source '{step.source}' not in objects {list(vocab.objects)}")
        if step.into is not None and step.into not in vocab.objects:
            problems.append(f"step {step.id}: into '{step.into}' not in objects {list(vocab.objects)}")
        if step.arm not in vocab.arms:
            problems.append(f"step {step.id}: arm '{step.arm}' not in arms {list(vocab.arms)}")
        if step.requires_hold is not None:
            hold = step.requires_hold
            if hold.arm not in vocab.arms:
                problems.append(f"step {step.id}: requires_hold.arm '{hold.arm}' not in arms {list(vocab.arms)}")
            if hold.object not in vocab.objects:
                problems.append(
                    f"step {step.id}: requires_hold.object '{hold.object}' not in objects {list(vocab.objects)}"
                )
    if problems:
        raise UnknownObjectError("; ".join(problems))
