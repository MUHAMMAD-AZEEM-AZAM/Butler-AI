"""Shared fixtures: a fake `anthropic` module so unit tests never touch the API."""

import sys
import types
from typing import Any, Callable

import pytest


class FakeContentBlock:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class FakeResponse:
    def __init__(self, text: str) -> None:
        self.content = [FakeContentBlock(text)]


class FakeMessagesAPI:
    """Returns scripted response texts in order and records every create() call."""

    def __init__(self, responses: tuple[str, ...]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(kwargs)
        return FakeResponse(self._responses.pop(0))


@pytest.fixture
def fake_anthropic(monkeypatch: pytest.MonkeyPatch) -> Callable[..., FakeMessagesAPI]:
    """Install a fake `anthropic` module scripted with the given response texts."""

    def _install(*responses: str) -> FakeMessagesAPI:
        api = FakeMessagesAPI(responses)
        module = types.ModuleType("anthropic")

        class Anthropic:
            def __init__(self, **kwargs: Any) -> None:
                self.messages = api

        module.Anthropic = Anthropic  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "anthropic", module)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.delenv("VOICE_STUB", raising=False)
        return api

    return _install
