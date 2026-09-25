"""Planner exceptions: every refusal to plan is explicit and actionable.

All inherit from PlanningError so a caller (e.g. common/pipeline.py) can report
a planning failure without also swallowing unrelated programming bugs.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from common.types import Action


class PlanningError(Exception):
    """The planner could not produce a trustworthy action list."""

    def __init__(self, message: str, *, step_id: int | None = None, issues: Iterable[str] = ()) -> None:
        self.step_id = step_id
        self.issues: tuple[str, ...] = tuple(issues)
        if self.issues:
            message = message + "\n  - " + "\n  - ".join(self.issues)
        super().__init__(message)


class ConfigError(PlanningError):
    """stage3_policy/planner_config.yaml is malformed or internally inconsistent."""


class TaskValidationError(PlanningError):
    """The Task itself is malformed: step IDs, dependencies, required fields or names."""


class UnsupportedError(PlanningError):
    """The Task needs an action, constraint or semantics the current contracts cannot express."""


class SceneError(PlanningError):
    """The observation is missing, or has invalid, information the plan needs."""


class PreconditionError(PlanningError):
    """The predicted state at a step makes it impossible (e.g. placing an object no arm holds)."""


class ObservationRequired(PlanningError):
    """Only a prefix of the task can be planned now.

    Execute `executable_actions`, observe the scene again, then re-plan the
    `pending_step_ids` with `plan_detailed(..., completed_step_ids=...)`.
    """

    def __init__(
        self,
        message: str,
        *,
        step_id: int | None,
        executable_actions: Sequence[Action],
        pending_step_ids: Sequence[int],
    ) -> None:
        super().__init__(message, step_id=step_id)
        self.executable_actions: tuple[Action, ...] = tuple(executable_actions)
        self.pending_step_ids: tuple[int, ...] = tuple(pending_step_ids)
