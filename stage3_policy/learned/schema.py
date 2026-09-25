"""Demonstration schema for LeRobot ACT training.

STATUS: PROPOSED TO STAGE 4 (Muhammad), NOT AGREED. It mirrors the actuator
interface of the origin/stage4-bimanual branch (12 position actuators:
ctrl[0:6] arm A, ctrl[6:12] arm B) and the overhead camera in
assets/bimanual_scene.xml. stage3_policy/tests/test_learned.py checks the names,
order, limits and timestep against the XML.

A symbolic Action (one pose per step) is NOT a LeRobot action. The learned
action is a 12-D vector of absolute joint position targets written to
MjData.ctrl at every recorded frame.
"""

from __future__ import annotations

from typing import Any

LEROBOT_VERSION = "0.6.1"  # APIs read from this release's source and docs
CODEBASE_VERSION = "v3.0"  # LeRobotDataset format written by lerobot 0.6.1
ROBOT_TYPE = "butler_bimanual_so101_sim"

PHYSICS_TIMESTEP_S = 0.002  # <option timestep> in assets/bimanual_scene.xml
PHYSICS_STEPS_PER_FRAME = 20
FPS = 25  # 1 / (0.002 s * 20): an integer number of physics steps per frame keeps timestamps exact

JOINTS_PER_ARM = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
MOTOR_NAMES = tuple(f"{arm}_{joint}" for arm in ("a", "b") for joint in JOINTS_PER_ARM)  # MuJoCo actuator order

# ctrlrange of each position actuator, radians (identical for both arms).
JOINT_LIMITS_RAD: dict[str, tuple[float, float]] = {
    "shoulder_pan": (-1.91986, 1.91986),
    "shoulder_lift": (-1.74533, 1.74533),
    "elbow_flex": (-1.69, 1.69),
    "wrist_flex": (-1.65806, 1.65806),
    "wrist_roll": (-2.74385, 2.84121),
    "gripper": (-0.17453, 1.74533),
}

STATE_KEY = "observation.state"  # lerobot.utils.constants.OBS_STATE: measured joint positions (qpos), rad
ACTION_KEY = "action"  # lerobot.utils.constants.ACTION: commanded joint position targets (ctrl), rad
CAMERA_NAME = "overhead_cam"
IMAGE_KEY = "observation.images.overhead"  # OBS_IMAGES + camera; ACT needs at least one image or env state
IMAGE_SHAPE = (480, 640, 3)  # height, width, channels of the Stage 4 renderer
SUCCESS_KEY = "next.success"  # True on the final frame of a successful episode
DONE_KEY = "next.done"  # lerobot.utils.constants.DONE: True on the final frame of every episode

SKILLS = ("open_drawer", "pick", "place", "pour")  # one ACT policy per skill; language is not an ACT input
EVAL_SEEDS = tuple(range(10))  # configs/default.yaml evaluation seeds: never used for demonstrations
EPISODE_METADATA_FILE = "meta/butler_episodes.json"  # Butler-specific per-episode labels (see README)
DEFAULT_FEATURE_KEYS = ("timestamp", "frame_index", "episode_index", "index", "task_index")  # added by LeRobot


def lerobot_features(use_videos: bool = True) -> dict[str, dict[str, Any]]:
    """The `features` argument for LeRobotDataset.create(repo_id, fps=FPS, features=..., robot_type=ROBOT_TYPE)."""
    return {
        STATE_KEY: {"dtype": "float32", "shape": (len(MOTOR_NAMES),), "names": list(MOTOR_NAMES)},
        ACTION_KEY: {"dtype": "float32", "shape": (len(MOTOR_NAMES),), "names": list(MOTOR_NAMES)},
        IMAGE_KEY: {"dtype": "video" if use_videos else "image", "shape": IMAGE_SHAPE, "names": ["height", "width", "channels"]},
        SUCCESS_KEY: {"dtype": "bool", "shape": (1,), "names": None},
        DONE_KEY: {"dtype": "bool", "shape": (1,), "names": None},
    }
