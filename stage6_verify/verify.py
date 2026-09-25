"""Stage 6: task-aware postcondition checks for simulation recovery.

Two families of checks, both kept deliberately:

* **Requirement checks** (from the original Stage 6 implementation): every object
  the task refers to must be observable, and every drawer the task opens must
  actually be open.
* **Placement checks** (added later, aligned with stage4_bimanual.constants):
  objects placed on the table must land at their destination, within
  PLACEMENT_TOLERANCE_M.

``expected_positions`` stays optional because the shared Task contract does not
carry destination coordinates; callers that know them can pass them in and get a
tighter 3-D check via ``position_within_tolerance``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from common.types import ActionType, SceneState, Task, VerifyResult

# Where placed objects belong. plate mirrors stage4_bimanual.constants.PLATE_TABLE_XY.
TABLE_DESTINATIONS = {"plate": (0.06, 0.00), "mug": (0.12, 0.20)}

# xy tolerance for "did the placement land on its slot".
PLACEMENT_TOLERANCE_M = 0.045

# tighter 3-D tolerance used for explicitly supplied expected_positions.
POSITION_TOLERANCE_M = 0.01


def position_within_tolerance(
    actual: Sequence[float],
    expected: Sequence[float],
    tolerance_m: float = POSITION_TOLERANCE_M,
) -> bool:
    """Whether two positions of equal dimension lie within ``tolerance_m`` (Euclidean)."""
    if len(actual) != len(expected):
        return False
    return math.dist(tuple(actual), tuple(expected)) <= tolerance_m


def _xy_close(actual: tuple[float, float, float], expected: tuple[float, float]) -> bool:
    return math.dist(actual[:2], expected) <= PLACEMENT_TOLERANCE_M


def _required_objects(task: Task) -> set[str]:
    """Every object the task names, in any role."""
    required: set[str] = set()
    for step in task.steps:
        for name in (step.object, step.source, step.into):
            if name:
                required.add(name)
    return required


def verify(
    scene_after: SceneState,
    task: Task,
    expected_positions: Mapping[str, Sequence[float]] | None = None,
    tolerance_m: float = POSITION_TOLERANCE_M,
) -> VerifyResult:
    """Check observable postconditions implied by ``task``.

    Fluid transfer is not represented in the current MuJoCo scene, so a pour is
    reported as a pose-only proxy rather than falsely asserting water level.
    """
    if not task.steps:
        return VerifyResult(ok=True, replan=False, details="no task steps to verify")

    failures: list[str] = []
    notes: list[str] = []

    missing = sorted(n for n in _required_objects(task) if n not in scene_after.objects)
    if missing:
        failures.append(f"missing objects: {', '.join(missing)}")

    for step in task.steps:
        if step.action is ActionType.OPEN_DRAWER:
            drawer = step.target or "top_drawer"
            actual_state = scene_after.drawers.get(drawer)
            if actual_state != "open":
                failures.append(f"{drawer} is {actual_state or 'missing'}, expected open")
        elif step.action is ActionType.PLACE and step.object in TABLE_DESTINATIONS:
            actual = scene_after.objects.get(step.object)
            expected = TABLE_DESTINATIONS[step.object]
            if actual is None or not _xy_close(actual, expected):
                failures.append(f"{step.object} is not at its table destination")
        elif step.action is ActionType.POUR:
            mug = scene_after.objects.get(step.into or "mug")
            if mug is None:
                failures.append("mug is not observable after pour")
            else:
                notes.append("pour checked as pose-only proxy (no fluid model)")

    if expected_positions:
        for name, expected in expected_positions.items():
            actual = scene_after.objects.get(name)
            if actual is None:
                continue  # already reported by the missing-object check when required
            if not position_within_tolerance(actual, expected, tolerance_m):
                failures.append(f"{name} is outside the {tolerance_m:.3f} m tolerance")

    if failures:
        return VerifyResult(ok=False, replan=True, details="; ".join(failures))
    return VerifyResult(
        ok=True, replan=False, details="; ".join(notes) or "observable postconditions satisfied"
    )
