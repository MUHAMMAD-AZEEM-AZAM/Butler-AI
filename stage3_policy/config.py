"""Load and validate the Stage 3 planner configuration (planner_config.yaml).

Validation collects every problem at once and raises ConfigError, so a bad
value is reported instead of silently producing wrong poses.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from stage3_policy.errors import ConfigError
from stage3_policy.geometry import Rect, circle_rect_gap, circle_within_rect

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "planner_config.yaml"
ARM_IDS = ("A", "B")
MAX_APPROACH_HEIGHT_M = 0.30

Point2 = tuple[float, float]
Point3 = tuple[float, float, float]


@dataclass(frozen=True)
class ArmSpec:
    base_xy: Point2
    max_reach_m: float


@dataclass(frozen=True)
class ObjectSpec:
    footprint_radius_m: float
    height_m: float
    grasp_offset_m: Point3
    grip_force: float
    approach_height_m: float
    requires_orientation: bool


@dataclass(frozen=True)
class DrawerSpec:
    footprint: Rect
    interior_top_z: float
    handle_grasp_point_closed: Point3
    grip_force: float
    approach_height_m: float


@dataclass(frozen=True)
class EdgeMarginConstraint:
    applies_to: frozenset[str]
    edge_margin_m: float


@dataclass(frozen=True)
class PourSpec:
    sources: frozenset[str]
    receivers: frozenset[str]
    clearance_above_rim_m: float


@dataclass(frozen=True)
class PlannerConfig:
    source: str
    frame_name: str
    workspace_xy: Rect
    workspace_z: tuple[float, float]
    table: Rect
    surface_z: float
    edge_margin_m: float
    place_clearance_m: float
    release_height_m: float
    on_table_tolerance_m: float
    keepout_zones: Mapping[str, Rect]
    arms: Mapping[str, ArmSpec]
    objects: Mapping[str, ObjectSpec]
    drawers: Mapping[str, DrawerSpec]
    slots: Mapping[str, Mapping[str, tuple[Point2, ...]]]  # destination -> object -> candidate centres
    pour: PourSpec
    constraints: Mapping[str, EdgeMarginConstraint]


class _Reader:
    """Typed accessors that record problems instead of stopping at the first one."""

    def __init__(self) -> None:
        self.issues: list[str] = []

    def mapping(self, value: Any, path: str) -> Mapping[str, Any]:
        if isinstance(value, Mapping):
            return value
        self.issues.append(f"{path}: expected a mapping, got {value!r}")
        return {}

    def number(
        self,
        value: Any,
        path: str,
        *,
        low: float | None = None,
        high: float | None = None,
        positive: bool = False,
    ) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            self.issues.append(f"{path}: expected a number, got {value!r}")
            return math.nan
        number = float(value)
        if not math.isfinite(number):
            self.issues.append(f"{path}: must be finite, got {value!r}")
        elif positive and number <= 0:
            self.issues.append(f"{path}: must be > 0, got {number}")
        elif low is not None and number < low:
            self.issues.append(f"{path}: must be >= {low}, got {number}")
        elif high is not None and number > high:
            self.issues.append(f"{path}: must be <= {high}, got {number}")
        return number

    def vector(self, value: Any, path: str, length: int) -> tuple[float, ...]:
        if not isinstance(value, (list, tuple)) or len(value) != length:
            self.issues.append(f"{path}: expected a list of {length} numbers, got {value!r}")
            return (math.nan,) * length
        return tuple(self.number(v, f"{path}[{i}]") for i, v in enumerate(value))

    def interval(self, value: Any, path: str) -> tuple[float, float]:
        low, high = self.vector(value, path, 2)
        if low >= high:
            self.issues.append(f"{path}: lower bound must be below upper bound, got {value!r}")
        return low, high

    def rect(self, value: Any, path: str) -> Rect:
        section = self.mapping(value, path)
        x_min, x_max = self.interval(section.get("x"), f"{path}.x")
        y_min, y_max = self.interval(section.get("y"), f"{path}.y")
        return Rect(x_min, x_max, y_min, y_max)

    def flag(self, value: Any, path: str) -> bool:
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        self.issues.append(f"{path}: expected true or false, got {value!r}")
        return False

    def object_names(self, value: Any, path: str, known: Mapping[str, Any]) -> frozenset[str]:
        if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) for v in value):
            self.issues.append(f"{path}: expected a list of object names, got {value!r}")
            return frozenset()
        for name in value:
            if name not in known:
                self.issues.append(f"{path}: unknown object {name!r}")
        return frozenset(value)


def parse_planner_config(data: Any, source: str = "<dict>") -> PlannerConfig:
    """Build a validated PlannerConfig from already-loaded YAML data."""
    r = _Reader()
    root = r.mapping(data, "config")

    frame = r.mapping(root.get("frame"), "frame")
    frame_name = frame.get("name")
    if not isinstance(frame_name, str) or not frame_name:
        r.issues.append(f"frame.name: expected a non-empty string, got {frame_name!r}")
        frame_name = "?"
    workspace = r.mapping(frame.get("workspace"), "frame.workspace")
    workspace_xy = r.rect(workspace, "frame.workspace")
    workspace_z = r.interval(workspace.get("z"), "frame.workspace.z")

    table_cfg = r.mapping(root.get("table"), "table")
    table = r.rect(table_cfg, "table")
    surface_z = r.number(table_cfg.get("surface_z"), "table.surface_z")
    edge_margin = r.number(table_cfg.get("edge_margin_m"), "table.edge_margin_m", low=0.0)
    clearance = r.number(table_cfg.get("place_clearance_m"), "table.place_clearance_m", low=0.0)
    release = r.number(table_cfg.get("release_height_m"), "table.release_height_m", low=0.0)
    on_table = r.number(table_cfg.get("on_table_tolerance_m"), "table.on_table_tolerance_m", positive=True)

    keepouts = {
        name: r.rect(zone, f"keepout_zones.{name}")
        for name, zone in r.mapping(root.get("keepout_zones", {}), "keepout_zones").items()
    }

    arms_cfg = r.mapping(root.get("arms"), "arms")
    if set(arms_cfg) != set(ARM_IDS):
        r.issues.append(f"arms: expected exactly {list(ARM_IDS)}, got {sorted(map(str, arms_cfg))}")
    arms = {}
    for arm_id, spec in arms_cfg.items():
        path = f"arms.{arm_id}"
        spec = r.mapping(spec, path)
        arms[str(arm_id)] = ArmSpec(
            base_xy=r.vector(spec.get("base_xy"), f"{path}.base_xy", 2),
            max_reach_m=r.number(spec.get("max_reach_m"), f"{path}.max_reach_m", positive=True),
        )

    objects = {}
    for name, spec in r.mapping(root.get("objects"), "objects").items():
        path = f"objects.{name}"
        spec = r.mapping(spec, path)
        objects[name] = ObjectSpec(
            footprint_radius_m=r.number(spec.get("footprint_radius_m"), f"{path}.footprint_radius_m", positive=True),
            height_m=r.number(spec.get("height_m"), f"{path}.height_m", positive=True),
            grasp_offset_m=r.vector(spec.get("grasp_offset_m"), f"{path}.grasp_offset_m", 3),
            grip_force=r.number(spec.get("grip_force"), f"{path}.grip_force", positive=True, high=1.0),
            approach_height_m=r.number(
                spec.get("approach_height_m"), f"{path}.approach_height_m", positive=True, high=MAX_APPROACH_HEIGHT_M
            ),
            requires_orientation=r.flag(spec.get("requires_orientation"), f"{path}.requires_orientation"),
        )

    drawers = {}
    for name, spec in r.mapping(root.get("drawers", {}), "drawers").items():
        path = f"drawers.{name}"
        spec = r.mapping(spec, path)
        drawers[name] = DrawerSpec(
            footprint=r.rect(spec.get("footprint"), f"{path}.footprint"),
            interior_top_z=r.number(spec.get("interior_top_z"), f"{path}.interior_top_z"),
            handle_grasp_point_closed=r.vector(
                spec.get("handle_grasp_point_closed"), f"{path}.handle_grasp_point_closed", 3
            ),
            grip_force=r.number(spec.get("grip_force"), f"{path}.grip_force", positive=True, high=1.0),
            approach_height_m=r.number(
                spec.get("approach_height_m"), f"{path}.approach_height_m", positive=True, high=MAX_APPROACH_HEIGHT_M
            ),
        )

    slots: dict[str, Mapping[str, tuple[Point2, ...]]] = {}
    for destination, dest_cfg in r.mapping(root.get("destinations"), "destinations").items():
        dest_path = f"destinations.{destination}"
        dest_slots = r.mapping(r.mapping(dest_cfg, dest_path).get("slots"), f"{dest_path}.slots")
        per_object = {}
        for name, candidates in dest_slots.items():
            path = f"{dest_path}.slots.{name}"
            if name not in objects:
                r.issues.append(f"{path}: unknown object {name!r}")
                continue
            if not isinstance(candidates, (list, tuple)) or not candidates:
                r.issues.append(f"{path}: expected a non-empty list of [x, y] slots, got {candidates!r}")
                continue
            per_object[name] = tuple(r.vector(c, f"{path}[{i}]", 2) for i, c in enumerate(candidates))
        slots[destination] = MappingProxyType(per_object)

    pour_cfg = r.mapping(root.get("pour"), "pour")
    pour = PourSpec(
        sources=r.object_names(pour_cfg.get("sources"), "pour.sources", objects),
        receivers=r.object_names(pour_cfg.get("receivers"), "pour.receivers", objects),
        clearance_above_rim_m=r.number(pour_cfg.get("clearance_above_rim_m"), "pour.clearance_above_rim_m", low=0.0),
    )

    constraints = {}
    for name, spec in r.mapping(root.get("constraints", {}), "constraints").items():
        path = f"constraints.{name}"
        spec = r.mapping(spec, path)
        if spec.get("kind") != "edge_margin":
            r.issues.append(f"{path}.kind: only 'edge_margin' is implemented, got {spec.get('kind')!r}")
        constraints[name] = EdgeMarginConstraint(
            applies_to=r.object_names(spec.get("applies_to"), f"{path}.applies_to", objects),
            edge_margin_m=r.number(spec.get("edge_margin_m"), f"{path}.edge_margin_m", low=0.0),
        )

    # Geometric consistency of the slots, only once every number parsed cleanly.
    if not r.issues:
        for destination, per_object in slots.items():
            for name, candidates in per_object.items():
                radius = objects[name].footprint_radius_m
                for i, (x, y) in enumerate(candidates):
                    path = f"destinations.{destination}.slots.{name}[{i}]"
                    if not circle_within_rect(x, y, radius, table, edge_margin):
                        r.issues.append(f"{path}: footprint (radius {radius} m) violates the {edge_margin} m edge margin")
                    for zone_name, zone in keepouts.items():
                        if circle_rect_gap(x, y, radius, zone) < clearance:
                            r.issues.append(f"{path}: footprint is within {clearance} m of keep-out zone {zone_name!r}")

    if r.issues:
        raise ConfigError(f"invalid planner config {source}", issues=r.issues)

    return PlannerConfig(
        source=source,
        frame_name=frame_name,
        workspace_xy=workspace_xy,
        workspace_z=workspace_z,
        table=table,
        surface_z=surface_z,
        edge_margin_m=edge_margin,
        place_clearance_m=clearance,
        release_height_m=release,
        on_table_tolerance_m=on_table,
        keepout_zones=MappingProxyType(keepouts),
        arms=MappingProxyType(arms),
        objects=MappingProxyType(objects),
        drawers=MappingProxyType(drawers),
        slots=MappingProxyType(slots),
        pour=pour,
        constraints=MappingProxyType(constraints),
    )


def load_planner_config(path: str | Path | None = None) -> PlannerConfig:
    """Load and validate a planner config file (default: the shipped planner_config.yaml)."""
    if path is None:
        return _load_default()
    return _load_file(Path(path))


@lru_cache(maxsize=1)
def _load_default() -> PlannerConfig:
    return _load_file(DEFAULT_CONFIG_PATH)


def _load_file(path: Path) -> PlannerConfig:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read planner config {path}: {exc}") from exc
    return parse_planner_config(data, source=str(path))
