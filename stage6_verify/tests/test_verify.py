from common.types import ActionType, SceneState, Task, TaskStep
from stage6_verify import verify
from stage6_verify.verify import position_within_tolerance


def official_task() -> Task:
    return Task(
        command="official dinner task",
        steps=[
            TaskStep(id=1, action=ActionType.OPEN_DRAWER, arm="A", target="top_drawer"),
            TaskStep(id=2, action=ActionType.PICK, arm="A", object="plate"),
            TaskStep(id=3, action=ActionType.PLACE, arm="A", object="plate", destination="table"),
            TaskStep(id=4, action=ActionType.PICK, arm="B", object="mug"),
            TaskStep(
                id=5,
                action=ActionType.POUR,
                arm="A",
                source="water_bottle",
                into="mug",
            ),
        ],
    )


def valid_scene() -> SceneState:
    """A scene that genuinely satisfies the task, destinations included.

    plate and mug sit on their configured table destinations
    (stage6_verify.verify.TABLE_DESTINATIONS, which mirrors
    stage4_bimanual.constants.PLATE_TABLE_XY), so the placement check passes for
    the right reason rather than because it is not exercised.
    """
    return SceneState(
        objects={
            "plate": (0.06, 0.00, 0.72),
            "mug": (0.12, 0.20, 0.72),
            "water_bottle": (0.12, -0.04, 0.70),
        },
        drawers={"top_drawer": "open"},
    )


def test_valid_scene_passes():
    result = verify(valid_scene(), official_task())

    assert result.ok is True
    assert result.replan is False


def test_closed_drawer_requests_replan():
    scene = valid_scene().model_copy(update={"drawers": {"top_drawer": "closed"}})

    result = verify(scene, official_task())

    assert result.ok is False
    assert result.replan is True
    assert "expected open" in result.details


def test_missing_referenced_object_requests_replan():
    scene = valid_scene().model_copy(
        update={"objects": {"plate": (0.20, 0.10, 0.72), "mug": (0.30, 0.10, 0.72)}}
    )

    result = verify(scene, official_task())

    assert result.ok is False
    assert result.replan is True
    assert "water_bottle" in result.details


def test_position_tolerance_is_enforced():
    result = verify(
        valid_scene(),
        official_task(),
        expected_positions={"plate": (0.30, 0.10, 0.72)},
    )

    assert result.ok is False
    assert result.replan is True
    assert "plate" in result.details


def test_position_helper_accepts_one_centimeter_error():
    assert position_within_tolerance((0.0, 0.0, 0.0), (0.006, 0.008, 0.0))
    assert not position_within_tolerance((0.0, 0.0, 0.0), (0.011, 0.0, 0.0))


def test_empty_task_passes():
    result = verify(SceneState(objects={}, drawers={}), Task(command="", steps=[]))

    assert result.ok is True
    assert result.replan is False
