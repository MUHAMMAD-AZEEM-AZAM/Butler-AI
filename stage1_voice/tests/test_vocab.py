"""Direct tests for the vocabulary loader and validator."""

import pytest

from common.types import ActionType, HoldRequirement, Task, TaskStep
from stage1_voice.errors import UnknownObjectError
from stage1_voice.vocab import load_vocab, load_voice_config, validate_task_vocabulary
from stage1_voice.voice import _example_task


def _task(step: TaskStep) -> Task:
    return Task(command="test", steps=[step])


def test_load_vocab_matches_config_and_enum() -> None:
    vocab = load_vocab()
    assert vocab.objects == ("plate", "mug", "water_bottle", "spoon", "fork")
    assert vocab.drawers == ("top_drawer",)
    assert vocab.destinations == ("table",)
    assert vocab.arms == ("A", "B")
    assert vocab.actions == tuple(a.value for a in ActionType)


def test_load_voice_config() -> None:
    cfg = load_voice_config()
    assert cfg["model"].startswith("claude-")
    assert cfg["max_retries"] == 1


def test_example_task_passes_validation() -> None:
    validate_task_vocabulary(_example_task("official command"), load_vocab())


def test_unknown_object_rejected() -> None:
    step = TaskStep(id=1, action=ActionType.PICK, arm="A", object="teapot")
    with pytest.raises(UnknownObjectError, match="teapot"):
        validate_task_vocabulary(_task(step), load_vocab())


def test_unknown_drawer_target_rejected() -> None:
    step = TaskStep(id=1, action=ActionType.OPEN_DRAWER, arm="A", target="bottom_drawer")
    with pytest.raises(UnknownObjectError, match="bottom_drawer"):
        validate_task_vocabulary(_task(step), load_vocab())


def test_unknown_destination_rejected() -> None:
    step = TaskStep(id=1, action=ActionType.PLACE, arm="A", object="plate", destination="shelf")
    with pytest.raises(UnknownObjectError, match="shelf"):
        validate_task_vocabulary(_task(step), load_vocab())


def test_unknown_pour_source_rejected() -> None:
    step = TaskStep(id=1, action=ActionType.POUR, arm="A", source="juice_carton", into="mug")
    with pytest.raises(UnknownObjectError, match="juice_carton"):
        validate_task_vocabulary(_task(step), load_vocab())


def test_unknown_hold_object_rejected() -> None:
    step = TaskStep(
        id=1,
        action=ActionType.POUR,
        arm="A",
        source="water_bottle",
        into="mug",
        requires_hold=HoldRequirement(arm="B", object="glass"),
    )
    with pytest.raises(UnknownObjectError, match="glass"):
        validate_task_vocabulary(_task(step), load_vocab())
