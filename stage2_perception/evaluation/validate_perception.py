"""Validate marker detection, homography mapping, and drawer state."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Dict, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage2_perception.perception.scene_pipeline import ScenePipeline


IMAGE_SIZE = (640, 480)
IMAGE_CORNERS = [(80.0, 60.0), (560.0, 60.0), (560.0, 420.0), (80.0, 420.0)]
TABLE_CORNERS = [(0.0, 0.0), (0.8, 0.0), (0.8, 0.6), (0.0, 0.6)]
GROUND_TRUTH: Dict[str, Tuple[float, float]] = {
    "plate": (0.20, 0.18),
    "mug": (0.60, 0.18),
    "drawer_handle": (0.20, 0.42),
    "water_bottle": (0.60, 0.42),
}


def _table_to_pixel(point: Tuple[float, float]) -> Tuple[int, int]:
    x, y = point
    pixel_x = IMAGE_CORNERS[0][0] + (x / TABLE_CORNERS[1][0]) * (
        IMAGE_CORNERS[1][0] - IMAGE_CORNERS[0][0]
    )
    pixel_y = IMAGE_CORNERS[0][1] + (y / TABLE_CORNERS[2][1]) * (
        IMAGE_CORNERS[2][1] - IMAGE_CORNERS[0][1]
    )
    return round(pixel_x), round(pixel_y)


def create_marker_scene(path: Path, drawer_state: str = "closed") -> None:
    """Create a deterministic four-marker image with known table positions."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    image = np.full((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), 245, dtype=np.uint8)
    cv2.rectangle(image, (80, 60), (560, 420), (170, 120, 80), thickness=-1)

    positions = dict(GROUND_TRUTH)
    if drawer_state == "open":
        positions["drawer_handle"] = (0.36, 0.42)
    elif drawer_state != "closed":
        raise ValueError("drawer_state must be 'open' or 'closed'")

    for marker_id, name in ((0, "plate"), (1, "mug"), (2, "drawer_handle"), (3, "water_bottle")):
        center_x, center_y = _table_to_pixel(positions[name])
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, 70)
        top_left = (center_x - 35, center_y - 35)
        image[top_left[1] : top_left[1] + 70, top_left[0] : top_left[0] + 70] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write marker scene to {path}")


def _distance(first: Tuple[float, float], second: Tuple[float, float]) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def validate_image(
    image_path: Path,
    output_path: Path | None = None,
    drawer_state: str = "closed",
) -> dict:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)

    pipeline = ScenePipeline()
    pipeline.compute_homography_from_points(IMAGE_CORNERS, TABLE_CORNERS)
    scene = pipeline.detect_from_image(image)

    detected = {item.name: (item.x, item.y) for item in scene.objects}
    if scene.drawer.detected:
        detected["drawer_handle"] = (scene.drawer.center_x, scene.drawer.center_y)

    errors = {}
    expected_positions = dict(GROUND_TRUTH)
    if drawer_state == "open":
        expected_positions["drawer_handle"] = (0.36, 0.42)

    for name, expected in expected_positions.items():
        actual = detected.get(name)
        errors[name] = {
            "expected": expected,
            "detected": actual,
            "error_m": None if actual is None else _distance(actual, expected),
            "pass": actual is not None and _distance(actual, expected) <= 0.01,
        }

    report = {
        "image": str(image_path),
        "stable": scene.stable,
        "warnings": scene.warnings,
        "drawer_state": None
        if not scene.drawer.detected
        else ("open" if scene.drawer.open_fraction else "closed"),
        "errors": errors,
        "pass": all(item["pass"] for item in errors.values()),
        "scene": scene.to_dict(),
    }
    if output_path is not None:
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--drawer-state", choices=("open", "closed"), default="closed")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--write-scene", type=Path, default=Path("stage2_perception/evaluation/multi_marker_scene.png"))
    args = parser.parse_args()

    image_path = args.image or args.write_scene
    if args.image is None:
        create_marker_scene(image_path, args.drawer_state)

    report = validate_image(image_path, args.output, args.drawer_state)
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())