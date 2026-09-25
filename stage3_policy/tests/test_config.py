"""The shipped planner config: loads, covers the shared vocabulary, and matches the MuJoCo scene XML."""

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

from stage3_policy.config import load_planner_config, parse_planner_config
from stage3_policy.errors import ConfigError
from stage3_policy.tests.conftest import config_dict

ROOT = Path(__file__).resolve().parents[2]
SCENE = ET.parse(ROOT / "assets" / "bimanual_scene.xml")


def _element(tree: ET.ElementTree, tag: str, name: str) -> ET.Element:
    return next(e for e in tree.iter(tag) if e.get("name") == name)


def _vec(element: ET.Element, attribute: str) -> tuple[float, ...]:
    return tuple(float(v) for v in element.get(attribute).split())


def test_shipped_config_covers_the_shared_scene_vocabulary():
    cfg = load_planner_config()
    vocabulary = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))["scene"]

    assert set(vocabulary["objects"]) <= set(cfg.objects)
    assert set(vocabulary["drawers"]) <= set(cfg.drawers)
    assert set(vocabulary["destinations"]) <= set(cfg.slots)
    assert {str(a) for a in vocabulary["arms"]} == set(cfg.arms)


def test_table_geometry_matches_the_scene_xml():
    cfg = load_planner_config()
    surface = _element(SCENE, "geom", "table_surface")
    (sx, sy, sz), (px, py, pz) = _vec(surface, "size"), _vec(surface, "pos")

    assert cfg.surface_z == pytest.approx(pz + sz)
    assert (cfg.table.x_min, cfg.table.x_max, cfg.table.y_min, cfg.table.y_max) == pytest.approx((px - sx, px + sx, py - sy, py + sy))


def test_drawer_geometry_matches_the_scene_xml():
    cfg = load_planner_config()
    drawer = cfg.drawers["top_drawer"]
    unit = _vec(_element(SCENE, "body", "drawer_unit"), "pos")
    tray = _vec(_element(SCENE, "body", "sliding_tray"), "pos")
    handle = _vec(_element(SCENE, "site", "drawer_handle_site"), "pos")

    expected_handle = tuple(u + t + h for u, t, h in zip(unit, tray, handle))
    assert drawer.handle_grasp_point_closed == pytest.approx(expected_handle)

    bx, by, _ = _vec(_element(SCENE, "geom", "cabinet_bottom"), "size")
    footprint = drawer.footprint
    assert (footprint.x_min, footprint.x_max, footprint.y_min, footprint.y_max) == pytest.approx(
        (unit[0] - bx, unit[0] + bx, unit[1] - by, unit[1] + by)
    )

    roof = _element(SCENE, "geom", "cabinet_top")
    assert drawer.interior_top_z == pytest.approx(unit[2] + _vec(roof, "pos")[2] - _vec(roof, "size")[2])

    # The keep-out zone must cover the fully opened tray and handle (slide joint range, -x axis).
    slide_max = _vec(_element(SCENE, "joint", "drawer_slide"), "range")[1]
    handle_radius = _vec(_element(SCENE, "geom", "drawer_handle"), "size")[0]
    swept = cfg.keepout_zones["top_drawer_swept"]
    assert swept.x_min <= unit[0] + handle[0] - slide_max - handle_radius
    assert swept.x_max >= footprint.x_max and swept.y_min <= footprint.y_min and swept.y_max >= footprint.y_max


def test_arm_bases_match_the_arm_xml():
    cfg = load_planner_config()
    for arm in ("a", "b"):
        tree = ET.parse(ROOT / "assets" / "SO-ARM100" / "Simulation" / "SO101" / f"so101_arm_{arm}.xml")
        base = _vec(_element(tree, "body", f"{arm}_base"), "pos")
        spec = cfg.arms[arm.upper()]
        assert spec.base_xy == pytest.approx(base[:2])
        assert cfg.keepout_zones[f"arm_{arm}_base"].contains(*spec.base_xy)


def _cylinder_extent(geoms: list[str]) -> tuple[float, float]:
    """Footprint radius and total height of a compound object made of cylinders.

    An object may be several geoms stacked along +z (the water bottle is a wide
    body with a narrow neck on top). The footprint is the widest radius and the
    height spans from the lowest to the highest surface, so a single geom must
    not be treated as the whole object.
    """
    radius = 0.0
    low, high = math.inf, -math.inf
    for name in geoms:
        element = _element(SCENE, "geom", name)
        r, half_height = _vec(element, "size")
        centre_z = _vec(element, "pos")[2]
        radius = max(radius, r)
        low = min(low, centre_z - half_height)
        high = max(high, centre_z + half_height)
    return radius, high - low


# Objects whose collision shape is one or more cylinders, listed bottom-up.
# water_bottle is compound: bottle_body (z 0.000-0.100) plus bottle_geom, the
# narrow neck (z 0.100-0.160).
OBJECT_GEOMS = {
    "plate": ["plate_geom"],
    "mug": ["mug_geom"],
    "water_bottle": ["bottle_body", "bottle_geom"],
}

# The geom a body grasp closes on. For the bottle that is the wide body, which is
# what stage4_bimanual.PickBottlePrimitive actually grips - not the neck.
GRASP_GEOM = {"mug": "mug_geom", "water_bottle": "bottle_body"}


@pytest.mark.parametrize("name", sorted(OBJECT_GEOMS))
def test_object_extents_cover_the_collision_geometry(name):
    spec = load_planner_config().objects[name]
    radius, height = _cylinder_extent(OBJECT_GEOMS[name])

    assert spec.footprint_radius_m >= radius
    assert spec.height_m == pytest.approx(height)


@pytest.mark.parametrize("name", sorted(GRASP_GEOM))
def test_body_grasp_height_is_inside_the_grasped_geom(name):
    """The planner's grasp point must lie within the geom the gripper closes on."""
    spec = load_planner_config().objects[name]
    element = _element(SCENE, "geom", GRASP_GEOM[name])
    centre_z = _vec(element, "pos")[2]
    half_height = _vec(element, "size")[1]
    grasp_z = spec.grasp_offset_m[2]

    assert centre_z - half_height <= grasp_z <= centre_z + half_height, (
        f"{name}: grasp offset z={grasp_z} is outside {GRASP_GEOM[name]} "
        f"(z {centre_z - half_height:.3f}..{centre_z + half_height:.3f})"
    )


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda c: c["objects"]["mug"].update(grip_force=1.5), "grip_force: must be <= 1.0"),
        (lambda c: c["objects"]["mug"].update(grip_force=0), "grip_force: must be > 0"),
        (lambda c: c["objects"]["plate"].update(grasp_offset_m=[0.0, math.nan, 0.0]), "must be finite"),
        (lambda c: c["objects"]["plate"].update(approach_height_m=0.5), "approach_height_m: must be <= 0.3"),
        (lambda c: c["destinations"]["table"]["slots"].update(mug=[[0.45, 0.0]]), "edge margin"),
        (lambda c: c["destinations"]["table"]["slots"].update(mug=[[0.25, -0.20]]), "keep-out zone 'drawer_zone'"),
        (lambda c: c["pour"].update(sources=["teapot"]), "unknown object 'teapot'"),
        (lambda c: c["constraints"]["keep_glasses_away_from_edge"].update(kind="max_speed"), "only 'edge_margin'"),
        (lambda c: c["arms"].pop("B"), "expected exactly ['A', 'B']"),
    ],
)
def test_invalid_configs_are_rejected_with_the_offending_key(mutate, expected):
    data = config_dict()
    mutate(data)
    with pytest.raises(ConfigError) as caught:
        parse_planner_config(data)
    assert any(expected in issue for issue in caught.value.issues), caught.value.issues


def test_every_config_problem_is_reported_at_once():
    data = config_dict()
    data["objects"]["mug"]["grip_force"] = 2
    data["objects"]["plate"]["height_m"] = -1
    with pytest.raises(ConfigError) as caught:
        parse_planner_config(data)
    assert len(caught.value.issues) == 2
