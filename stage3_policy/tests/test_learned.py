"""Learned-policy plumbing: schema vs MuJoCo XML, dataset validation, fallback selection, training preflight.

Needs no torch, lerobot, pyarrow, dataset or checkpoint. Nothing here trains or
runs a policy; these tests check the code that would, and that it refuses to
pretend when prerequisites are missing.
"""

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from stage3_policy.learned import inference, schema, train
from stage3_policy.learned import validate_dataset as vd

ROOT = Path(__file__).resolve().parents[2]
SO101 = ROOT / "assets" / "SO-ARM100" / "Simulation" / "SO101"


# --------------------------------------------------------------------------- schema vs simulator


def test_motor_names_order_and_limits_match_the_mujoco_actuators():
    names, limits = [], {}
    for arm in ("a", "b"):
        for actuator in ET.parse(SO101 / f"so101_arm_{arm}.xml").find("actuator"):
            names.append(actuator.get("name"))
            joint = actuator.get("name").split("_", 1)[1]
            limits.setdefault(joint, set()).add(tuple(float(v) for v in actuator.get("ctrlrange").split()))

    assert tuple(names) == schema.MOTOR_NAMES
    assert limits == {joint: {bounds} for joint, bounds in schema.JOINT_LIMITS_RAD.items()}


def test_frame_rate_is_a_whole_number_of_physics_steps_and_camera_exists():
    scene = ET.parse(ROOT / "assets" / "bimanual_scene.xml")
    timestep = float(scene.find("option").get("timestep"))

    assert timestep == schema.PHYSICS_TIMESTEP_S
    assert schema.FPS * schema.PHYSICS_STEPS_PER_FRAME * timestep == pytest.approx(1.0)
    assert any(camera.get("name") == schema.CAMERA_NAME for camera in scene.iter("camera"))


def test_features_cover_state_action_camera_and_labels():
    features = schema.lerobot_features()
    assert features[schema.STATE_KEY]["shape"] == features[schema.ACTION_KEY]["shape"] == (12,)
    assert features[schema.IMAGE_KEY]["dtype"] == "video"
    assert schema.lerobot_features(use_videos=False)[schema.IMAGE_KEY]["dtype"] == "image"
    assert {schema.SUCCESS_KEY, schema.DONE_KEY} <= features.keys()


# --------------------------------------------------------------------------- dataset validation (synthetic metadata)


def _episode(index: int, seed: int, split: str = "train", **changes) -> dict:
    entry = {
        "episode_index": index,
        "seed": seed,
        "skill": "pick",
        "arm": "B",
        "object": "mug",
        "instruction": "pick up the mug with arm B",
        "split": split,
        "success": True,
        "physics_only": True,
    }
    entry.update(changes)
    return entry


def write_dataset(root: Path, *, info_changes=None, feature_changes=None, episodes=None) -> Path:
    features = {key: {"dtype": "int64", "shape": [1], "names": None} for key in schema.DEFAULT_FEATURE_KEYS}
    features["timestamp"]["dtype"] = "float32"
    features.update({key: {**spec, "shape": list(spec["shape"])} for key, spec in schema.lerobot_features().items()})
    for key, change in (feature_changes or {}).items():
        if change is None:
            features.pop(key)
        else:
            features[key] = {**features[key], **change}
    info = {"codebase_version": "v3.0", "fps": schema.FPS, "robot_type": schema.ROBOT_TYPE, "total_episodes": 2, "total_frames": 6, "features": features}
    info.update(info_changes or {})
    episodes = episodes if episodes is not None else [_episode(0, 100), _episode(1, 101, split="val")]
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (root / "meta" / "butler_episodes.json").write_text(json.dumps({"episodes": episodes}), encoding="utf-8")
    return root


@pytest.fixture
def no_pyarrow(monkeypatch):
    monkeypatch.setattr(vd, "_load_frame_columns", lambda root: None)


def test_valid_metadata_passes_but_unrun_frame_checks_are_not_called_valid(tmp_path, no_pyarrow, capsys):
    root = write_dataset(tmp_path / "demos")
    report = vd.validate_dataset(root, min_successful_episodes=1)

    assert report.ok
    assert report.skipped and "pyarrow" in report.skipped[0]
    assert "SKIPPED" in report.format()
    assert vd.main(["--root", str(root), "--min-successful-episodes", "1"]) == 1  # not verified -> non-zero


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"info_changes": {"fps": 30}}, "fps is 30"),
        ({"feature_changes": {schema.ACTION_KEY: None}}, "missing feature 'action'"),
        ({"feature_changes": {schema.STATE_KEY: {"names": list(reversed(schema.MOTOR_NAMES))}}}, "actuator order"),
        ({"feature_changes": {schema.IMAGE_KEY: {"shape": [240, 320, 3]}}}, "shape"),
        ({"episodes": [_episode(0, 3), _episode(1, 101, split="val")]}, "evaluation seed"),
        ({"episodes": [_episode(0, 100, physics_only=False), _episode(1, 101, split="val")]}, "teleporting"),
        ({"episodes": [_episode(0, 100), _episode(1, 100, split="val")]}, "both train and val"),
        ({"episodes": [_episode(0, 100)]}, "episode_index values"),
    ],
)
def test_metadata_problems_are_reported(tmp_path, no_pyarrow, changes, expected):
    report = vd.validate_dataset(write_dataset(tmp_path / "demos", **changes), min_successful_episodes=1)
    assert not report.ok
    assert any(expected in error for error in report.errors), report.errors


def test_missing_dataset_is_an_error(tmp_path):
    report = vd.validate_dataset(tmp_path / "nothing_here")
    assert "missing meta/info.json" in report.errors


def test_too_few_successful_demonstrations_warn(tmp_path, no_pyarrow):
    report = vd.validate_dataset(write_dataset(tmp_path / "demos"))
    assert report.ok
    assert any("successful episodes" in warning for warning in report.warnings)


def _columns(episodes: int = 2, frames: int = 3) -> dict:
    rows = [(e, f) for e in range(episodes) for f in range(frames)]
    return {
        "episode_index": [e for e, _ in rows],
        "frame_index": [f for _, f in rows],
        "timestamp": [f / schema.FPS for _, f in rows],
        schema.STATE_KEY: [[0.0] * 12 for _ in rows],
        schema.ACTION_KEY: [[0.0] * 12 for _ in rows],
    }


def test_well_formed_frames_pass():
    assert vd.check_frame_columns(_columns(), fps=schema.FPS, total_episodes=2) == []


@pytest.mark.parametrize(
    ("corrupt", "expected"),
    [
        (lambda c: c["timestamp"].__setitem__(2, 0.5), "timestamp 0.5"),
        (lambda c: c["frame_index"].__setitem__(1, 5), "without gaps"),
        (lambda c: c[schema.ACTION_KEY][0].__setitem__(0, math.nan), "non-finite"),
        (lambda c: c[schema.ACTION_KEY][4].__setitem__(5, 2.5), "a_gripper=2.5000"),
        (lambda c: c[schema.STATE_KEY].__setitem__(3, [0.0] * 11), "11 values"),
    ],
)
def test_corrupt_frames_are_reported(corrupt, expected):
    columns = _columns()
    corrupt(columns)
    errors = vd.check_frame_columns(columns, fps=schema.FPS, total_episodes=2)
    assert any(expected in error for error in errors), errors


def test_episode_count_must_match_metadata():
    errors = vd.check_frame_columns(_columns(episodes=1), fps=schema.FPS, total_episodes=2)
    assert any("1 episodes" in error for error in errors)


# --------------------------------------------------------------------------- policy selection / fallback


def test_no_configured_checkpoint_uses_the_scripted_fallback():
    selection = inference.select_motor_policy(None)
    assert (selection.mode, selection.policy) == (inference.FALLBACK, None)
    assert "no ACT checkpoint" in selection.reason


def test_a_path_in_a_config_is_not_a_trained_model(tmp_path):
    selection = inference.select_motor_policy(tmp_path / "assets" / "models" / "policy.safetensors")
    assert selection.mode == inference.FALLBACK
    assert "does not exist" in selection.reason


def test_incomplete_checkpoint_directory_is_reported(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    selection = inference.select_motor_policy(tmp_path)
    assert selection.mode == inference.FALLBACK
    assert "model.safetensors" in selection.reason


def _fake_checkpoint(directory: Path) -> Path:
    for name in inference.REQUIRED_CHECKPOINT_FILES:
        (directory / name).write_text("placeholder", encoding="utf-8")
    return directory


def test_missing_ml_dependencies_are_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(inference.importlib.util, "find_spec", lambda name: None)
    selection = inference.select_motor_policy(_fake_checkpoint(tmp_path))
    assert selection.mode == inference.FALLBACK
    assert "lerobot" in selection.reason


def test_a_checkpoint_that_fails_to_load_falls_back_with_the_error(tmp_path, monkeypatch):
    monkeypatch.setattr(inference.importlib.util, "find_spec", lambda name: object())

    def broken(directory, device="cpu"):
        raise RuntimeError("placeholder weights")

    monkeypatch.setattr(inference.ActMotorPolicy, "load", broken)
    selection = inference.select_motor_policy(_fake_checkpoint(tmp_path))
    assert selection.mode == inference.FALLBACK
    assert "placeholder weights" in selection.reason

    ready = inference.select_motor_policy(tmp_path, load=False)
    assert ready.mode == inference.LEARNED and ready.policy is None


def test_configured_checkpoint_is_read_from_the_shared_config(tmp_path):
    config = tmp_path / "default.yaml"
    config.write_text("models:\n  policy_checkpoint: outputs/train/act_pick/checkpoints/last/pretrained_model\n", encoding="utf-8")
    assert inference.configured_checkpoint(config) == "outputs/train/act_pick/checkpoints/last/pretrained_model"
    assert inference.configured_checkpoint(tmp_path / "missing.yaml") is None


# --------------------------------------------------------------------------- training command / preflight


def test_training_command_is_local_act_without_uploads(tmp_path):
    command = train.build_train_command(train.TrainSettings(dataset_root=tmp_path / "demos"))
    assert command[0] == "lerobot-train"
    for flag in ("--policy.type=act", "--policy.push_to_hub=false", "--wandb.enable=false", "--policy.device=cpu", "--steps=200"):
        assert flag in command
    assert f"--dataset.root={tmp_path / 'demos'}" in command


def test_run_is_refused_until_preflight_passes(tmp_path, monkeypatch, capsys):
    def must_not_run(*args, **kwargs):
        raise AssertionError("training must not start")

    monkeypatch.setattr(train.subprocess, "run", must_not_run)
    missing = str(tmp_path / "no_dataset")

    assert train.main(["--dataset-root", missing, "--job-name", "pytest_refusal", "--run"]) == 2
    output = capsys.readouterr().out
    assert "NOT READY" in output and "missing meta/info.json" in output

    assert train.main(["--dataset-root", missing, "--job-name", "pytest_refusal"]) == 0  # dry run starts nothing
