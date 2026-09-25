"""Tasks and scenes the planner must refuse, each with an explicit, actionable error."""

import math

import pytest

from common import EXAMPLE_COMMAND
from common.types import ActionType as T, HoldRequirement, SceneState, Task
from stage3_policy.config import parse_planner_config
from stage3_policy.errors import PreconditionError, SceneError, TaskValidationError, UnsupportedError
from stage3_policy.planner import plan_detailed
from stage3_policy.tests.conftest import config_dict, step


def _messages(exc: pytest.ExceptionInfo) -> str:
    return str(exc.value)


# --------------------------------------------------------------------------- task structure


def test_duplicate_unknown_and_self_dependencies_are_all_reported(cfg, scene):
    task = Task(
        command="broken ids",
        steps=[
            step(1, T.PICK, "A", object="plate", depends_on=[7]),
            step(1, T.PICK, "B", object="mug"),
            step(2, T.PICK, "B", object="water_bottle", depends_on=[2]),
        ],
    )
    with pytest.raises(TaskValidationError) as caught:
        plan_detailed(task, scene(), config=cfg)
    text = _messages(caught)
    assert "duplicate step ids [1]" in text
    assert "unknown step 7" in text
    assert "depends on itself" in text


def test_dependency_cycle_is_rejected(cfg, scene):
    task = Task(
        command="cycle",
        steps=[step(1, T.PICK, "A", object="plate", depends_on=[2]), step(2, T.PICK, "B", object="mug", depends_on=[1])],
    )
    with pytest.raises(TaskValidationError, match="dependency cycle"):
        plan_detailed(task, scene(), config=cfg)


def test_steps_run_after_their_dependencies_otherwise_in_listed_order(cfg, scene):
    out_of_order = Task(
        command="place listed first",
        steps=[step(2, T.PLACE, "A", object="plate", destination="table", depends_on=[1]), step(1, T.PICK, "A", object="plate")],
    )
    result = plan_detailed(out_of_order, scene(), config=cfg)
    assert [a.step_id for a in result.actions] == [1, 2]
    assert any("out of dependency order" in w for w in result.warnings)

    independent = Task(command="independent", steps=[step(3, T.PICK, "B", object="mug"), step(1, T.PICK, "A", object="plate")])
    assert [a.step_id for a in plan_detailed(independent, scene(), config=cfg).actions] == [3, 1]


@pytest.mark.parametrize(
    ("bad_step", "expected"),
    [
        (step(1, T.PLACE, "A", object="plate"), "missing required field 'destination'"),
        (step(1, T.PICK, "A", object="plate", destination="table"), "not used by this action"),
        (step(1, T.PICK, "A", object="glass"), "no grasp configuration"),
        (step(1, T.OPEN_DRAWER, "A", target="bottom_drawer"), "unknown drawer"),
        (step(1, T.POUR, "A", source="water_bottle", into="mug", requires_hold=HoldRequirement(arm="A", object="mug")), "must be the other arm"),
    ],
)
def test_malformed_steps_are_rejected(cfg, scene, bad_step, expected):
    with pytest.raises(TaskValidationError, match=expected):
        plan_detailed(Task(command="bad", steps=[bad_step]), scene(), config=cfg)


def test_unmapped_object_from_stage1_is_not_planned_partially(cfg, scene):
    task = Task(command="pick the teapot", steps=[step(1, T.PICK, "B", object="mug")], constraints=["unknown_object:teapot"])
    with pytest.raises(TaskValidationError, match="teapot"):
        plan_detailed(task, scene(), config=cfg)


def test_completed_steps_must_exist_and_include_their_dependencies(cfg, scene):
    task = Task(
        command="move plate",
        steps=[step(1, T.PICK, "A", object="plate"), step(2, T.PLACE, "A", object="plate", destination="table", depends_on=[1])],
    )
    with pytest.raises(TaskValidationError, match="not steps of this task"):
        plan_detailed(task, scene(), completed_step_ids=[9], config=cfg)
    with pytest.raises(TaskValidationError, match="dependencies \\[1\\] are not"):
        plan_detailed(task, scene(), completed_step_ids=[2], config=cfg)


# --------------------------------------------------------------------------- unsupported semantics


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        (Task(command="handoff", steps=[step(1, T.HANDOFF, "A", object="plate")]), "handoff is not supported"),
        (Task(command="close", steps=[step(1, T.CLOSE_DRAWER, "A", target="top_drawer")]), "close_drawer is not supported"),
        (Task(command="spoon", steps=[step(1, T.PICK, "B", object="spoon")]), "needs its yaw"),
        (Task(command="gentle", steps=[step(1, T.PICK, "B", object="mug")], constraints=["pour_slowly"]), "'pour_slowly' is not supported"),
    ],
)
def test_unsupported_actions_and_constraints_are_rejected_not_ignored(cfg, scene, task, expected):
    with pytest.raises(UnsupportedError, match=expected):
        plan_detailed(task, scene(), config=cfg)


# --------------------------------------------------------------------------- observation problems


def test_missing_detection_is_an_error_not_the_origin(cfg, scene):
    task = Task(command="pick mug", steps=[step(1, T.PICK, "B", object="mug")])
    with pytest.raises(SceneError, match="'mug' is not in the observed scene") as caught:
        plan_detailed(task, scene(drawer="open", mug=None), config=cfg)
    assert "drawer" not in _messages(caught)

    with pytest.raises(SceneError, match="top_drawer"):
        plan_detailed(task, scene(drawer="closed", mug=None), config=cfg)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_coordinates_are_rejected(cfg, scene, bad):
    task = Task(command="pick mug", steps=[step(1, T.PICK, "B", object="mug")])
    with pytest.raises(SceneError, match="non-finite"):
        plan_detailed(task, scene(mug=(bad, 0.25, 0.70)), config=cfg)


def test_coordinates_from_another_frame_are_rejected(cfg, scene):
    # e.g. a table-corner frame whose table top is z = 0
    task = Task(command="pick mug", steps=[step(1, T.PICK, "B", object="mug")])
    with pytest.raises(SceneError, match="frame"):
        plan_detailed(task, scene(mug=(0.20, 0.15, 0.05)), config=cfg)


def test_unused_object_outside_the_workspace_only_warns(cfg, scene):
    task = Task(command="pick mug", steps=[step(1, T.PICK, "B", object="mug")])
    result = plan_detailed(task, scene(spoon=(0.0, 0.0, 0.0)), config=cfg)
    assert result.complete
    assert any("'spoon'" in w and "ignored" in w for w in result.warnings)


def test_drawer_state_that_was_not_observed_is_not_assumed_closed(cfg, scene):
    task = Task(command="open", steps=[step(1, T.OPEN_DRAWER, "A", target="top_drawer")])
    with pytest.raises(SceneError, match="will not assume it is closed"):
        plan_detailed(task, scene(drawer=None), config=cfg)


# --------------------------------------------------------------------------- impossible sequences


def test_object_inside_a_closed_drawer_needs_an_open_step(cfg, scene):
    task = Task(command="pick plate", steps=[step(1, T.PICK, "A", object="plate")])
    with pytest.raises(PreconditionError, match="inside 'top_drawer'.*add an open_drawer step"):
        plan_detailed(task, scene(drawer="closed", plate=(0.25, -0.25, 0.72)), config=cfg)


def test_opening_an_observed_open_drawer_is_rejected(cfg, scene):
    task = Task(command="open", steps=[step(1, T.OPEN_DRAWER, "A", target="top_drawer")])
    with pytest.raises(PreconditionError, match="observed open.*completed_step_ids"):
        plan_detailed(task, scene(drawer="open"), config=cfg)


def test_completed_open_step_contradicted_by_the_observation(cfg, scene):
    task = Task(
        command="plate",
        steps=[step(1, T.OPEN_DRAWER, "A", target="top_drawer"), step(2, T.PICK, "A", object="plate", depends_on=[1])],
    )
    with pytest.raises(SceneError, match="reported complete"):
        plan_detailed(task, scene(drawer="closed"), completed_step_ids=[1], config=cfg)


def test_arm_cannot_pick_while_holding_something(cfg, scene):
    task = Task(command="two picks", steps=[step(1, T.PICK, "A", object="plate"), step(2, T.PICK, "A", object="mug")])
    with pytest.raises(PreconditionError, match="still holding 'plate'") as caught:
        plan_detailed(task, scene(), config=cfg)
    assert caught.value.step_id == 2


def test_both_arms_cannot_hold_the_same_object(cfg, scene):
    task = Task(command="tug of war", steps=[step(1, T.PICK, "A", object="mug"), step(2, T.PICK, "B", object="mug")])
    with pytest.raises(PreconditionError, match="already held by arm A"):
        plan_detailed(task, scene(), config=cfg)


def test_placing_an_object_the_arm_does_not_hold_is_rejected(cfg, scene):
    task = Task(command="place", steps=[step(1, T.PLACE, "A", object="plate", destination="table")])
    with pytest.raises(PreconditionError, match="not holding 'plate'"):
        plan_detailed(task, scene(), config=cfg)


def test_pouring_without_picking_the_bottle_is_rejected(cfg, scene):
    task = Task(
        command="pour",
        steps=[
            step(1, T.PICK, "B", object="mug"),
            step(2, T.POUR, "A", source="water_bottle", into="mug", depends_on=[1], requires_hold=HoldRequirement(arm="B", object="mug")),
        ],
    )
    with pytest.raises(PreconditionError, match="must already hold 'water_bottle'"):
        plan_detailed(task, scene(), config=cfg)


def test_pouring_without_the_declared_mug_support_is_rejected(cfg, scene):
    task = Task(
        command="pour",
        steps=[
            step(1, T.PICK, "A", object="water_bottle"),
            step(2, T.POUR, "A", source="water_bottle", into="mug", depends_on=[1], requires_hold=HoldRequirement(arm="B", object="mug")),
        ],
    )
    with pytest.raises(PreconditionError, match="requires arm B to hold 'mug'"):
        plan_detailed(task, scene(), config=cfg)


def test_no_free_slot_lists_why_each_slot_was_rejected(cfg, scene):
    task = Task(
        command="move plate",
        steps=[step(1, T.PICK, "A", object="plate"), step(2, T.PLACE, "A", object="plate", destination="table", depends_on=[1])],
    )
    crowded = scene(water_bottle=(0.0, 0.0, 0.70), mug=(0.0, 0.15, 0.70))
    with pytest.raises(PreconditionError) as caught:
        plan_detailed(task, crowded, config=cfg)
    assert len(caught.value.issues) == 2
    assert "slot 0" in caught.value.issues[0] and "slot 1" in caught.value.issues[1]


def test_edge_constraint_uses_object_extent_and_can_block_a_slot(cfg, scene):
    move_mug = [step(1, T.PICK, "B", object="mug"), step(2, T.PLACE, "B", object="mug", destination="table", depends_on=[1])]
    first_slot_taken = scene(water_bottle=(0.15, 0.15, 0.70))

    # slot 1 centre (0.25, 0.30) + radius 0.04 + base margin 0.05 = 0.39 <= 0.40: allowed
    unconstrained = plan_detailed(Task(command="mug", steps=move_mug), first_slot_taken, config=cfg)
    assert tuple(unconstrained.actions[1].target_pose) == pytest.approx((0.25, 0.30, 0.755))

    # with the 0.12 m glasses margin: 0.30 + 0.04 + 0.12 = 0.46 > 0.40: rejected
    constrained = Task(command="mug", steps=move_mug, constraints=["keep_glasses_away_from_edge"])
    with pytest.raises(PreconditionError) as caught:
        plan_detailed(constrained, first_slot_taken, config=cfg)
    assert "keep_glasses_away_from_edge" in caught.value.issues[1]


def test_stage1_stub_example_is_staged_not_refused(monkeypatch):
    """Integration with Stage 1 real stub Task.

    CHANGED with the staged-planning integration: the original test asserted a
    PreconditionError because the stub poured without ever picking up the
    bottle. Stage 1 has since implemented CONTRACT_PROPOSAL.md P4 (the stub now
    picks the water_bottle with the pouring arm, 6 steps), so the task is valid
    and plan_detailed returns a staged partial plan instead of refusing.
    """
    monkeypatch.setenv("VOICE_STUB", "1")
    from stage1_voice import parse_text

    task = parse_text(EXAMPLE_COMMAND)
    nominal_mujoco_scene = SceneState(
        objects={
            "plate": (0.22, -0.22, 0.72),
            "mug": (0.06, 0.18, 0.70),
            "water_bottle": (0.12, -0.04, 0.70),
            "spoon": (0.18, 0.08, 0.70),
            "fork": (0.18, 0.02, 0.70),
        },
        drawers={"top_drawer": "closed"},
    )
    result = plan_detailed(task, nominal_mujoco_scene)
    # the plate sits inside the closed drawer, so only open_drawer is plannable now
    assert [a.step_id for a in result.actions] == [1]
    assert not result.complete
    assert result.pending_step_ids == (2, 3, 4, 5, 6)
    assert result.blocked_step_id == 2


def test_config_changes_are_not_needed_for_refusals_to_be_explicit():
    """A config with no drawers still rejects drawer steps by name instead of guessing."""
    data = config_dict()
    data["drawers"] = {}
    no_drawers = parse_planner_config(data)
    task = Task(command="open", steps=[step(1, T.OPEN_DRAWER, "A", target="top_drawer")])
    with pytest.raises(TaskValidationError, match="unknown drawer"):
        plan_detailed(task, SceneState(objects={}, drawers={"top_drawer": "closed"}), config=no_drawers)
