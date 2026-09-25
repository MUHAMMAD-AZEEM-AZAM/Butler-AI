"""Valid tasks: resolved poses, per-object grasp settings, destinations, drawers, staged re-planning."""

import pytest

from common.types import Action, ActionType as T, HoldRequirement, SceneState, Task
from stage3_policy import plan
from stage3_policy.errors import ObservationRequired
from stage3_policy.planner import plan_detailed
from stage3_policy.tests.conftest import step


def _pose(action: Action) -> tuple:
    return tuple(action.target_pose)


def test_bimanual_task_resolves_every_step_to_one_action(cfg, scene):
    task = Task(
        command="plate to table with A; B moves the mug to the table and pours into it",
        steps=[
            step(1, T.PICK, "A", object="plate"),
            step(2, T.PLACE, "A", object="plate", destination="table", depends_on=[1]),
            step(3, T.PICK, "B", object="mug"),
            step(4, T.PLACE, "B", object="mug", destination="table", depends_on=[3]),
            step(5, T.PICK, "B", object="water_bottle", depends_on=[4]),
            step(6, T.POUR, "B", source="water_bottle", into="mug", depends_on=[5]),
        ],
    )
    result = plan_detailed(task, scene(), config=cfg)

    assert result.complete and result.pending_step_ids == ()
    assert [(a.step_id, a.action, a.arm, a.object) for a in result.actions] == [
        (1, T.PICK, "A", "plate"),
        (2, T.PLACE, "A", "plate"),
        (3, T.PICK, "B", "mug"),
        (4, T.PLACE, "B", "mug"),
        (5, T.PICK, "B", "water_bottle"),
        (6, T.POUR, "B", "water_bottle"),
    ]
    poses = [_pose(a) for a in result.actions]
    # pick = observed base + grasp offset; place = first free slot at surface + release height + offset;
    # pour = receiver base + receiver height + clearance above the rim.
    assert poses[0] == pytest.approx((-0.34, 0.00, 0.72))
    assert poses[1] == pytest.approx((-0.04, 0.00, 0.725))
    assert poses[2] == pytest.approx((-0.30, 0.25, 0.75))
    assert poses[3] == pytest.approx((0.15, 0.15, 0.755))
    assert poses[4] == pytest.approx((0.30, 0.20, 0.78))
    assert poses[5] == pytest.approx((0.15, 0.15, 0.85))
    assert dict(result.predicted_holdings) == {"A": None, "B": "water_bottle"}
    assert result.mode == "rule_based"


def test_grasp_settings_come_from_each_objects_configuration(cfg, scene):
    task = Task(command="grab both", steps=[step(1, T.PICK, "B", object="mug"), step(2, T.PICK, "A", object="water_bottle")])
    mug, bottle = plan_detailed(task, scene(), config=cfg).actions

    assert (mug.grip_force, mug.approach_height) == (0.5, 0.10)
    assert (bottle.grip_force, bottle.approach_height) == (0.3, 0.12)
    assert all(0.0 < a.grip_force <= 1.0 for a in (mug, bottle))


def test_pick_target_follows_the_current_observation(cfg, scene):
    task = Task(command="pick mug", steps=[step(1, T.PICK, "B", object="mug")])

    before = plan_detailed(task, scene(mug=(-0.30, 0.25, 0.70)), config=cfg).actions[0]
    after = plan_detailed(task, scene(mug=(-0.10, 0.30, 0.70)), config=cfg).actions[0]

    assert _pose(before) == pytest.approx((-0.30, 0.25, 0.75))
    assert _pose(after) == pytest.approx((-0.10, 0.30, 0.75))


def test_place_targets_the_destination_slot_not_the_objects_current_position(cfg, scene):
    task = Task(
        command="move mug",
        steps=[step(1, T.PICK, "B", object="mug"), step(2, T.PLACE, "B", object="mug", destination="table", depends_on=[1])],
    )
    place = plan_detailed(task, scene(), config=cfg).actions[1]

    assert _pose(place) == pytest.approx((0.15, 0.15, 0.755))
    assert _pose(place)[:2] != pytest.approx((-0.30, 0.25))


def test_place_skips_an_occupied_slot_and_records_the_choice(cfg, scene):
    task = Task(
        command="move plate",
        steps=[step(1, T.PICK, "A", object="plate"), step(2, T.PLACE, "A", object="plate", destination="table", depends_on=[1])],
    )
    result = plan_detailed(task, scene(water_bottle=(0.0, 0.0, 0.70)), config=cfg)

    assert _pose(result.actions[1]) == pytest.approx((-0.04, 0.15, 0.725))
    assert any("slot 1" in note and "water_bottle" in note for note in result.notes)


def test_public_plan_returns_a_list_and_empty_task_plans_nothing():
    assert plan(Task(command="nothing", steps=[]), SceneState(objects={}, drawers={})) == []


def test_opening_a_drawer_returns_the_prefix_and_asks_for_reobservation(cfg, scene):
    task = Task(
        command="plate from drawer",
        steps=[
            step(1, T.OPEN_DRAWER, "A", target="top_drawer"),
            step(2, T.PICK, "A", object="plate", depends_on=[1]),
            step(3, T.PLACE, "A", object="plate", destination="table", depends_on=[2]),
        ],
    )
    inside_closed_drawer = scene(drawer="closed", plate=(0.25, -0.25, 0.72))
    result = plan_detailed(task, inside_closed_drawer, config=cfg)

    assert not result.complete
    assert [a.step_id for a in result.actions] == [1]
    opening = result.actions[0]
    assert (opening.object, _pose(opening), opening.grip_force, opening.approach_height) == (
        "top_drawer",
        pytest.approx((0.10, -0.25, 0.73)),
        0.8,
        0.06,
    )
    assert result.pending_step_ids == (2, 3)
    assert result.blocked_step_id == 2
    assert "re-observe" in result.blocked_reason


def test_public_plan_refuses_a_partial_plan_but_attaches_the_prefix(scene):
    # shipped config: plate inside the closed top drawer of the MuJoCo scene
    task = Task(
        command="plate from drawer",
        steps=[step(1, T.OPEN_DRAWER, "A", target="top_drawer"), step(2, T.PICK, "A", object="plate", depends_on=[1])],
    )
    observed = SceneState(objects={"plate": (0.22, -0.22, 0.72)}, drawers={"top_drawer": "closed"})

    with pytest.raises(ObservationRequired) as caught:
        plan(task, observed)
    assert [a.step_id for a in caught.value.executable_actions] == [1]
    assert caught.value.pending_step_ids == (2,)


def test_object_hidden_by_a_drawer_opened_in_this_plan_is_not_invented(cfg, scene):
    task = Task(
        command="plate from drawer",
        steps=[step(1, T.OPEN_DRAWER, "A", target="top_drawer"), step(2, T.PICK, "A", object="plate", depends_on=[1])],
    )
    result = plan_detailed(task, scene(drawer="closed", plate=None), config=cfg)

    assert [a.step_id for a in result.actions] == [1]
    assert result.blocked_step_id == 2 and "not observed" in result.blocked_reason


def test_replanning_with_completed_steps_uses_the_fresh_observation(cfg, scene):
    task = Task(
        command="plate from drawer",
        steps=[
            step(1, T.OPEN_DRAWER, "A", target="top_drawer"),
            step(2, T.PICK, "A", object="plate", depends_on=[1]),
            step(3, T.PLACE, "A", object="plate", destination="table", depends_on=[2]),
        ],
    )
    after_opening = scene(drawer="open", plate=(0.05, -0.25, 0.72))
    result = plan_detailed(task, after_opening, completed_step_ids=[1], config=cfg)

    assert result.complete
    assert [a.step_id for a in result.actions] == [2, 3]
    assert _pose(result.actions[0]) == pytest.approx((0.01, -0.25, 0.74))


def test_pour_into_a_mug_held_after_a_planned_pick_needs_reobservation(cfg, scene):
    task = Task(
        command="hold and pour",
        steps=[
            step(1, T.PICK, "B", object="mug"),
            step(2, T.PICK, "A", object="water_bottle"),
            step(3, T.POUR, "A", source="water_bottle", into="mug", depends_on=[1, 2], requires_hold=HoldRequirement(arm="B", object="mug")),
        ],
    )
    result = plan_detailed(task, scene(), config=cfg)

    assert [a.step_id for a in result.actions] == [1, 2]
    assert result.pending_step_ids == (3,)
    assert "held by arm B" in result.blocked_reason


def test_pour_with_holds_from_completed_steps_targets_the_observed_mug(cfg, scene):
    task = Task(
        command="hold and pour",
        steps=[
            step(1, T.PICK, "B", object="mug"),
            step(2, T.PICK, "A", object="water_bottle"),
            step(3, T.POUR, "A", source="water_bottle", into="mug", depends_on=[1, 2], requires_hold=HoldRequirement(arm="B", object="mug")),
        ],
    )
    lifted = scene(mug=(0.0, 0.10, 0.85), water_bottle=(-0.10, -0.10, 0.90))
    result = plan_detailed(task, lifted, completed_step_ids=[1, 2], config=cfg)

    assert result.complete
    (pour,) = result.actions
    assert (pour.step_id, pour.arm, pour.object) == (3, "A", "water_bottle")
    assert _pose(pour) == pytest.approx((0.0, 0.10, 1.0))
    assert dict(result.predicted_holdings) == {"A": "water_bottle", "B": "mug"}
