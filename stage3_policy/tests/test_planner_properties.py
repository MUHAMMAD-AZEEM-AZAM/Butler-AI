"""Cross-cutting guarantees: determinism, no input mutation, dependency warnings, runnable demo."""

from common.types import ActionType as T, HoldRequirement, SceneState, Task
from stage3_policy import demo
from stage3_policy.planner import plan_detailed
from stage3_policy.tests.conftest import step

TASK = Task(
    command="plate, mug hold, pour",
    constraints=["keep_glasses_away_from_edge"],
    steps=[
        step(1, T.PICK, "A", object="plate"),
        step(2, T.PLACE, "A", object="plate", destination="table", depends_on=[1]),
        step(3, T.PICK, "B", object="mug"),
        step(4, T.PICK, "A", object="water_bottle", depends_on=[2]),
        step(5, T.POUR, "A", source="water_bottle", into="mug", depends_on=[3, 4], requires_hold=HoldRequirement(arm="B", object="mug")),
    ],
)


def _summary(result):
    return (result.actions, result.complete, result.pending_step_ids, result.blocked_reason, result.notes, result.warnings, dict(result.predicted_holdings))


def test_planning_is_deterministic_and_ignores_dict_insertion_order(cfg, scene):
    observed = scene()
    reversed_scene = SceneState(objects=dict(reversed(list(observed.objects.items()))), drawers=dict(observed.drawers))

    first = plan_detailed(TASK, observed, config=cfg)
    assert _summary(first) == _summary(plan_detailed(TASK, observed, config=cfg))
    assert _summary(first) == _summary(plan_detailed(TASK, reversed_scene, config=cfg))


def test_inputs_are_not_mutated(cfg, scene):
    observed = scene()
    task_before, scene_before = TASK.model_dump(), observed.model_dump()
    completed = [1]

    plan_detailed(TASK, observed, config=cfg)
    plan_detailed(TASK, scene(plate=(0.0, 0.0, 0.70)), completed_step_ids=completed, config=cfg)

    assert TASK.model_dump() == task_before
    assert observed.model_dump() == scene_before
    assert completed == [1]


def test_prerequisite_met_only_by_listing_order_is_flagged(cfg, scene):
    task = Task(command="no depends_on", steps=[step(1, T.PICK, "A", object="plate"), step(2, T.PLACE, "A", object="plate", destination="table")])
    result = plan_detailed(task, scene(), config=cfg)
    assert result.complete
    assert any("relies on step 1" in w and "depends_on" in w for w in result.warnings)


def test_demo_examples_behave_as_documented(capsys):
    assert demo.main([]) == 0
    output = capsys.readouterr().out
    assert "SYNTHETIC INPUTS" in output
    assert "PLANNER REFUSED (SceneError)" in output
    assert "PLANNER REFUSED (PreconditionError)" in output
    assert "UNEXPECTED" not in output
