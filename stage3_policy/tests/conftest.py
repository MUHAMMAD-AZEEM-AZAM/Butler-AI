"""Shared fixtures for Stage 3 planner tests.

Most tests use TEST_CONFIG: a small config with round numbers, so expected poses
can be worked out by hand and do not change when the shipped heuristics are tuned.
All scenes are synthetic, in a MuJoCo-world-like frame (table top z = 0.70).
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import pytest

from common.types import ActionType, SceneState, TaskStep
from stage3_policy.config import PlannerConfig, parse_planner_config

TEST_CONFIG: dict[str, Any] = {
    "frame": {"name": "test_world", "workspace": {"x": [-0.5, 0.5], "y": [-0.4, 0.4], "z": [0.66, 1.2]}},
    "table": {
        "x": [-0.5, 0.5],
        "y": [-0.4, 0.4],
        "surface_z": 0.70,
        "edge_margin_m": 0.05,
        "place_clearance_m": 0.02,
        "release_height_m": 0.005,
        "on_table_tolerance_m": 0.03,
    },
    "keepout_zones": {"drawer_zone": {"x": [0.10, 0.40], "y": [-0.40, -0.10]}},
    "arms": {"A": {"base_xy": [-0.2, -0.2], "max_reach_m": 0.8}, "B": {"base_xy": [-0.2, 0.2], "max_reach_m": 0.8}},
    "objects": {
        "plate": {"footprint_radius_m": 0.05, "height_m": 0.01, "grasp_offset_m": [-0.04, 0.0, 0.02], "grip_force": 0.6, "approach_height_m": 0.08},
        "mug": {"footprint_radius_m": 0.04, "height_m": 0.10, "grasp_offset_m": [0.0, 0.0, 0.05], "grip_force": 0.5, "approach_height_m": 0.10},
        "water_bottle": {"footprint_radius_m": 0.03, "height_m": 0.16, "grasp_offset_m": [0.0, 0.0, 0.08], "grip_force": 0.3, "approach_height_m": 0.12},
        "spoon": {
            "footprint_radius_m": 0.08,
            "height_m": 0.03,
            "grasp_offset_m": [0.0, 0.0, 0.005],
            "grip_force": 0.35,
            "approach_height_m": 0.06,
            "requires_orientation": True,
        },
    },
    "drawers": {
        "top_drawer": {
            "footprint": {"x": [0.15, 0.35], "y": [-0.35, -0.15]},
            "interior_top_z": 0.76,
            "handle_grasp_point_closed": [0.10, -0.25, 0.73],
            "grip_force": 0.8,
            "approach_height_m": 0.06,
        }
    },
    "destinations": {
        "table": {"slots": {"plate": [[0.0, 0.0], [0.0, 0.15]], "mug": [[0.15, 0.15], [0.25, 0.30]], "water_bottle": [[0.20, 0.0]]}}
    },
    "pour": {"sources": ["water_bottle"], "receivers": ["mug"], "clearance_above_rim_m": 0.05},
    "constraints": {
        "keep_glasses_away_from_edge": {"kind": "edge_margin", "applies_to": ["mug", "water_bottle"], "edge_margin_m": 0.12}
    },
}

# Synthetic observation: every object on the table, away from the configured slots.
BASE_OBJECTS = {
    "plate": (-0.30, 0.00, 0.70),
    "mug": (-0.30, 0.25, 0.70),
    "water_bottle": (0.30, 0.20, 0.70),
    "spoon": (-0.35, -0.30, 0.70),
}


def config_dict() -> dict[str, Any]:
    return copy.deepcopy(TEST_CONFIG)


@pytest.fixture
def cfg() -> PlannerConfig:
    return parse_planner_config(config_dict(), source="TEST_CONFIG")


@pytest.fixture
def scene() -> Callable[..., SceneState]:
    """scene(drawer="closed", plate=None, mug=(x, y, z), ...): BASE_OBJECTS with overrides; None removes an object."""

    def build(drawer: str | None = "closed", **objects: tuple[float, float, float] | None) -> SceneState:
        merged = {**BASE_OBJECTS, **objects}
        return SceneState(
            objects={name: xyz for name, xyz in merged.items() if xyz is not None},
            drawers={} if drawer is None else {"top_drawer": drawer},
        )

    return build


def step(step_id: int, action: ActionType, arm: str, **fields: Any) -> TaskStep:
    return TaskStep(id=step_id, action=action, arm=arm, **fields)
