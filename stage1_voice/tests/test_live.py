"""Live tests against the real Claude API. Run with: pytest -m live (needs ANTHROPIC_API_KEY)."""

import os

import pytest

from common import EXAMPLE_COMMAND
from common.types import ActionType, Task, TaskStep
from stage1_voice.voice import parse_text

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"),
]


@pytest.fixture(autouse=True)
def no_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOICE_STUB", raising=False)


def _steps(task: Task) -> list[TaskStep]:
    return sorted(task.steps, key=lambda s: s.id)


def _step_for(task: Task, action: ActionType, obj: str) -> TaskStep:
    matches = [s for s in task.steps if s.action == action and s.object == obj]
    assert matches, f"no {action.value} step for {obj} in {task.steps}"
    return matches[0]


def test_official_command() -> None:
    task = parse_text(EXAMPLE_COMMAND)
    steps = _steps(task)
    assert len(steps) == 6
    assert [s.action for s in steps] == [
        ActionType.OPEN_DRAWER,
        ActionType.PICK,
        ActionType.PLACE,
        ActionType.PICK,
        ActionType.PICK,
        ActionType.POUR,
    ]
    assert _step_for(task, ActionType.PICK, "plate").arm == "A"
    assert _step_for(task, ActionType.PICK, "mug").arm == "B"
    pour = steps[-1]
    assert pour.arm == "A"
    assert pour.source == "water_bottle"
    assert pour.into == "mug"
    # pour implies an explicit preceding pick of the source with the same arm
    source_pick = _step_for(task, ActionType.PICK, "water_bottle")
    assert source_pick.arm == pour.arm
    assert source_pick.id in pour.depends_on


def test_official_command_arms_swapped() -> None:
    command = (
        "Open the top drawer, pick up the plate with arm B, place it on the table, "
        "pick up the mug with arm A, pour water into the mug with arm B."
    )
    task = parse_text(command)
    steps = _steps(task)
    assert len(steps) == 6
    assert [s.action for s in steps] == [
        ActionType.OPEN_DRAWER,
        ActionType.PICK,
        ActionType.PLACE,
        ActionType.PICK,
        ActionType.PICK,
        ActionType.POUR,
    ]
    assert _step_for(task, ActionType.PICK, "plate").arm == "B"
    assert _step_for(task, ActionType.PICK, "mug").arm == "A"
    assert steps[-1].arm == "B"
    assert _step_for(task, ActionType.PICK, "water_bottle").arm == "B"


def test_different_object_subset() -> None:
    task = parse_text("Pick up the spoon with arm A and place it on the table.")
    steps = _steps(task)
    assert [s.action for s in steps] == [ActionType.PICK, ActionType.PLACE]
    assert all(s.arm == "A" for s in steps)
    assert all(s.object == "spoon" for s in steps)
    assert steps[1].destination == "table"
