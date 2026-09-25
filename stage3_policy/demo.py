"""Readable Stage 3 planner demo on SYNTHETIC inputs.

    python -m stage3_policy                      # every example
    python -m stage3_policy --example missing    # one example

Every Task and SceneState here is a hand-written fixture in MuJoCo world
coordinates (metres). The output shows planner decisions only: nothing is
simulated, executed or verified. The exit code is 0 when every example behaves
as documented (including the examples that are supposed to be refused).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

from common.types import ActionType, HoldRequirement, SceneState, Task, TaskStep

from stage3_policy.errors import PlanningError
from stage3_policy.planner import PlanResult, plan_detailed

BANNER = (
    "SYNTHETIC INPUTS: every Task and SceneState below is a hand-written fixture in MuJoCo world\n"
    "coordinates (metres, table top z = 0.70). This prints planner decisions only; nothing is\n"
    "simulated, executed or verified, and no output here shows that the robot works."
)

# Body positions of the unrandomized scene in assets/bimanual_scene.xml.
NOMINAL_MUG = (0.06, 0.18, 0.70)
NOMINAL_BOTTLE = (0.12, -0.04, 0.70)
NOMINAL_SPOON = (0.18, 0.08, 0.70)
NOMINAL_FORK = (0.18, 0.02, 0.70)

_P, _S = ActionType.PICK, TaskStep


def one_shot_example() -> tuple[Task, SceneState]:
    """Drawer already observed open, everything visible: the whole task plans in one call."""
    task = Task(
        command="Put the plate on the table with arm A; with arm B move the mug to the table and pour water into it.",
        constraints=["keep_glasses_away_from_edge"],
        steps=[
            _S(id=1, action=_P, arm="A", object="plate"),
            _S(id=2, action=ActionType.PLACE, arm="A", object="plate", destination="table", depends_on=[1]),
            _S(id=3, action=_P, arm="B", object="mug"),
            _S(id=4, action=ActionType.PLACE, arm="B", object="mug", destination="table", depends_on=[3]),
            _S(id=5, action=_P, arm="B", object="water_bottle", depends_on=[4]),
            _S(id=6, action=ActionType.POUR, arm="B", source="water_bottle", into="mug", depends_on=[5]),
        ],
    )
    scene = SceneState(
        objects={
            "plate": (0.08, -0.22, 0.72),  # on the open drawer tray
            "mug": NOMINAL_MUG,
            "water_bottle": NOMINAL_BOTTLE,
            "spoon": (0.40, 0.10, 0.70),  # utensils moved aside in this fixture
            "fork": (0.40, -0.02, 0.70),
        },
        drawers={"top_drawer": "open"},
    )
    return task, scene


def staged_example() -> tuple[Task, list[tuple[str, SceneState]]]:
    """The challenge command, with the bottle pick the Stage 1 stub example leaves out."""
    task = Task(
        command=(
            "Open the top drawer, pick up the plate with arm A, place it on the table, pick up the mug with "
            "arm B, pick up the water bottle with arm A, pour water into the mug with arm A."
        ),
        constraints=["keep_glasses_away_from_edge"],
        steps=[
            _S(id=1, action=ActionType.OPEN_DRAWER, arm="A", target="top_drawer"),
            _S(id=2, action=_P, arm="A", object="plate", depends_on=[1]),
            _S(id=3, action=ActionType.PLACE, arm="A", object="plate", destination="table", depends_on=[2]),
            _S(id=4, action=_P, arm="B", object="mug"),
            _S(id=5, action=_P, arm="A", object="water_bottle", depends_on=[3]),
            _S(
                id=6,
                action=ActionType.POUR,
                arm="A",
                source="water_bottle",
                into="mug",
                depends_on=[4, 5],
                requires_hold=HoldRequirement(arm="B", object="mug"),
            ),
        ],
    )
    utensils = {"spoon": NOMINAL_SPOON, "fork": NOMINAL_FORK}
    observations = [
        (
            "initial observation: drawer closed; the overhead camera cannot see the plate inside it",
            SceneState(objects={"mug": NOMINAL_MUG, "water_bottle": NOMINAL_BOTTLE, **utensils}, drawers={"top_drawer": "closed"}),
        ),
        (
            "invented re-observation after step 1: drawer open, plate on the pulled-out tray",
            SceneState(
                objects={"plate": (0.08, -0.22, 0.72), "mug": NOMINAL_MUG, "water_bottle": NOMINAL_BOTTLE, **utensils},
                drawers={"top_drawer": "open"},
            ),
        ),
        (
            "invented re-observation after steps 2-5: arm B holds the mug up, arm A holds the bottle",
            SceneState(
                objects={
                    "plate": (-0.04, 0.02, 0.70),
                    "mug": (0.04, 0.08, 0.82),
                    "water_bottle": (-0.02, -0.08, 0.86),
                    **utensils,
                },
                drawers={"top_drawer": "open"},
            ),
        ),
    ]
    return task, observations


def missing_object_example() -> tuple[Task, SceneState]:
    task = Task(
        command="Pick up the mug with arm B and place it on the table.",
        steps=[
            _S(id=1, action=_P, arm="B", object="mug"),
            _S(id=2, action=ActionType.PLACE, arm="B", object="mug", destination="table", depends_on=[1]),
        ],
    )
    scene = SceneState(objects={"water_bottle": NOMINAL_BOTTLE}, drawers={"top_drawer": "open"})
    return task, scene


def stage1_stub_example() -> tuple[Task, SceneState]:
    """The fixed Task stage1_voice returns in stub mode (copied from stage1_voice/voice.py)."""
    task = Task(
        command="Open the top drawer, pick up the plate with arm A, place it on the table, pick up the mug with arm B, pour water into the mug with arm A.",
        constraints=["keep_glasses_away_from_edge"],
        steps=[
            _S(id=1, action=ActionType.OPEN_DRAWER, arm="A", target="top_drawer"),
            _S(id=2, action=_P, arm="A", object="plate", depends_on=[1]),
            _S(id=3, action=ActionType.PLACE, arm="A", object="plate", destination="table", depends_on=[2]),
            _S(id=4, action=_P, arm="B", object="mug"),
            _S(
                id=5,
                action=ActionType.POUR,
                arm="A",
                source="water_bottle",
                into="mug",
                depends_on=[4],
                requires_hold=HoldRequirement(arm="B", object="mug"),
            ),
        ],
    )
    scene = SceneState(
        objects={"plate": (0.22, -0.22, 0.72), "mug": NOMINAL_MUG, "water_bottle": NOMINAL_BOTTLE, "spoon": NOMINAL_SPOON, "fork": NOMINAL_FORK},
        drawers={"top_drawer": "closed"},
    )
    return task, scene


# --------------------------------------------------------------------------- printing


def _print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def _print_inputs(task: Task, scene: SceneState, completed: Sequence[int] = ()) -> None:
    print(f"Command: {task.command}")
    if task.constraints:
        print(f"Constraints: {task.constraints}")
    print("Steps:")
    for step in task.steps:
        fields = {k: v for k, v in step.model_dump(exclude={"id", "action", "arm", "depends_on"}).items() if v}
        deps = f" after {step.depends_on}" if step.depends_on else ""
        done = "  [reported complete]" if step.id in completed else ""
        print(f"  {step.id}. {step.action.value:<11} arm {step.arm} {fields}{deps}{done}")
    print(f"Scene drawers: {dict(scene.drawers)}")
    for name, (x, y, z) in scene.objects.items():
        print(f"  {name:<13} ({x:+.3f}, {y:+.3f}, {z:+.3f})")


def _print_result(result: PlanResult) -> None:
    print(f"Planned actions ({result.mode}):")
    if result.actions:
        print("  step  action       arm  object         target_pose x, y, z (m)       grip  approach (m)")
        for a in result.actions:
            x, y, z = a.target_pose
            print(
                f"  {a.step_id:<5} {a.action.value:<12} {a.arm:<4} {a.object or '-':<14} "
                f"({x:+.3f}, {y:+.3f}, {z:+.3f})    {a.grip_force:.2f}  {a.approach_height:.2f}"
            )
    else:
        print("  (none)")
    if not result.complete:
        print(f"NOT COMPLETE: steps {list(result.pending_step_ids)} still pending.")
        print(f"  Blocked at step {result.blocked_step_id}: {result.blocked_reason}")
    for note in result.notes:
        print(f"  note: {note}")
    for warning in result.warnings:
        print(f"  WARNING: {warning}")
    held = ", ".join(f"{arm}: {obj or 'empty'}" for arm, obj in result.predicted_holdings.items())
    print(f"Predicted holdings after all steps (assumes every action succeeds): {held}")


def _attempt(task: Task, scene: SceneState, completed: Sequence[int] = ()) -> PlanResult | PlanningError:
    try:
        return plan_detailed(task, scene, completed_step_ids=completed)
    except PlanningError as exc:
        print(f"PLANNER REFUSED ({type(exc).__name__}):\n  {exc}")
        return exc


# --------------------------------------------------------------------------- examples


def run_one_shot() -> bool:
    _print_header("Example 1: one-shot bimanual plan")
    task, scene = one_shot_example()
    _print_inputs(task, scene)
    result = _attempt(task, scene)
    if isinstance(result, PlanningError):
        return False
    _print_result(result)
    return result.complete


def run_staged() -> bool:
    _print_header("Example 2: staged observe-act planning for the challenge command")
    task, observations = staged_example()
    completed: list[int] = []
    for index, (label, scene) in enumerate(observations, start=1):
        print(f"\n--- plan call {index}: {label}")
        if completed:
            print(f"    (assumes the executor reported steps {completed} complete)")
        _print_inputs(task, scene, completed)
        result = _attempt(task, scene, completed)
        if isinstance(result, PlanningError):
            return False
        _print_result(result)
        completed.extend(a.step_id for a in result.actions)
        if result.complete:
            return index == len(observations)
    return False


def run_missing() -> bool:
    _print_header("Example 3: missing object -> explicit failure (no (0, 0, 0) fallback)")
    task, scene = missing_object_example()
    _print_inputs(task, scene)
    return isinstance(_attempt(task, scene), PlanningError)


def run_stage1_stub() -> bool:
    _print_header("Example 4: Stage 1 stub Task (no bottle pick) -> refused before anything moves")
    task, scene = stage1_stub_example()
    _print_inputs(task, scene)
    return isinstance(_attempt(task, scene), PlanningError)


EXAMPLES: dict[str, Callable[[], bool]] = {
    "one-shot": run_one_shot,
    "staged": run_staged,
    "missing": run_missing,
    "stage1-stub": run_stage1_stub,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m stage3_policy", description=__doc__.splitlines()[0])
    parser.add_argument("--example", choices=["all", *EXAMPLES], default="all")
    args = parser.parse_args(argv)

    print(BANNER)
    names = list(EXAMPLES) if args.example == "all" else [args.example]
    outcomes = {name: EXAMPLES[name]() for name in names}

    print("\n" + "-" * 78)
    for name, ok in outcomes.items():
        print(f"{name:<12} {'behaved as documented' if ok else 'UNEXPECTED BEHAVIOUR'}")
    return 0 if all(outcomes.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
