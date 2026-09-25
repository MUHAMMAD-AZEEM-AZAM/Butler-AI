"""Unit tests for parse_text — the anthropic client is faked, no API calls."""

import json
import logging

import pytest

from common.types import ActionType
from stage1_voice.errors import UnknownObjectError, ParseError
from stage1_voice.voice import parse_text

COMMAND = "put the plate on the table"

VALID_TASK_JSON = json.dumps(
    {
        "command": COMMAND,
        "steps": [
            {"id": 1, "action": "pick", "arm": "A", "object": "plate"},
            {
                "id": 2,
                "action": "place",
                "arm": "A",
                "object": "plate",
                "destination": "table",
                "depends_on": [1],
            },
        ],
        "constraints": [],
    }
)

UNKNOWN_OBJECT_TASK_JSON = json.dumps(
    {
        "command": COMMAND,
        "steps": [{"id": 1, "action": "pick", "arm": "A", "object": "teapot"}],
        "constraints": [],
    }
)


def test_valid_json_parses(fake_anthropic) -> None:
    api = fake_anthropic(VALID_TASK_JSON)
    task = parse_text(COMMAND)
    assert len(api.calls) == 1
    assert task.command == COMMAND
    assert [s.action for s in task.steps] == [ActionType.PICK, ActionType.PLACE]
    assert task.steps[1].depends_on == [1]


def test_fenced_json_still_parses(fake_anthropic) -> None:
    fake_anthropic(f"```json\n{VALID_TASK_JSON}\n```")
    task = parse_text(COMMAND)
    assert [s.object for s in task.steps] == ["plate", "plate"]


def test_invalid_json_then_valid_retries(fake_anthropic) -> None:
    api = fake_anthropic("this is not json {", VALID_TASK_JSON)
    task = parse_text(COMMAND)
    assert len(api.calls) == 2
    assert len(task.steps) == 2
    # The retry conversation must carry the failed output and the error message.
    retry_messages = api.calls[1]["messages"]
    assert retry_messages[1]["role"] == "assistant"
    assert retry_messages[1]["content"] == "this is not json {"
    assert "failed validation" in retry_messages[2]["content"]


def test_unknown_object_raises_after_retry(fake_anthropic) -> None:
    api = fake_anthropic(UNKNOWN_OBJECT_TASK_JSON, UNKNOWN_OBJECT_TASK_JSON)
    with pytest.raises(UnknownObjectError, match="teapot"):
        parse_text(COMMAND)
    assert len(api.calls) == 2


def test_persistent_bad_json_raises_parse_error(fake_anthropic) -> None:
    api = fake_anthropic("nope", "still nope")
    with pytest.raises(ParseError):
        parse_text(COMMAND)
    assert len(api.calls) == 2


def test_stub_mode_without_key(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("VOICE_STUB", raising=False)
    with caplog.at_level(logging.WARNING):
        task = parse_text(COMMAND)
    assert task.command == COMMAND
    assert len(task.steps) == 6
    assert any("STUB mode" in record.message for record in caplog.records)


def test_stub_mode_forced_by_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("VOICE_STUB", "1")
    task = parse_text(COMMAND)
    assert len(task.steps) == 6


def test_pour_implies_pick_of_source_with_same_arm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every pour must be preceded by an explicit pick of its source by the same arm."""
    monkeypatch.setenv("VOICE_STUB", "1")
    task = parse_text(COMMAND)
    pours = [s for s in task.steps if s.action == ActionType.POUR]
    assert pours, "stub task must contain a pour step"
    for pour in pours:
        picks = [
            s
            for s in task.steps
            if s.action == ActionType.PICK
            and s.object == pour.source
            and s.arm == pour.arm
            and s.id < pour.id
        ]
        assert picks, f"no preceding pick of {pour.source} with arm {pour.arm}"
        assert any(p.id in pour.depends_on for p in picks)


def test_system_prompt_states_pour_pick_rule(fake_anthropic) -> None:
    """The rule must reach the model: pour implies a same-arm pick of the source."""
    api = fake_anthropic(VALID_TASK_JSON)
    parse_text(COMMAND)
    system = api.calls[0]["system"]
    assert "pour" in system.lower()
    assert "pick step for the source object with the same arm" in system
