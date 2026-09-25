from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .aruco_detector import ArucoDetector
from .scene_state import DrawerState, ObjectDetection, SceneState
from .tabletop_transform import TabletopTransform


DEFAULT_MARKER_MAP = {
    0: "plate",
    1: "mug",
    2: "drawer_handle",
    3: "water_bottle",
}


class ScenePipeline:
    """Planner-facing perception interface for the challenge scene.

    This class provides a simple, stable API for the rest of the team to consume:
    - detect objects and drawer markers from an image
    - map pixel locations to tabletop coordinates using a homography
    - produce a structured scene state for the robot or policy layer
    """

    def __init__(
        self,
        marker_map: Optional[Dict[int, str]] = None,
        homography_matrix: Optional[np.ndarray] = None,
        drawer_open_threshold: float = 0.28,
    ):
        self.marker_map = {**DEFAULT_MARKER_MAP, **(marker_map or {})}
        self.detector = ArucoDetector()
        self.transform = TabletopTransform(homography_matrix)
        self.drawer_open_threshold = drawer_open_threshold

    def set_homography(self, homography_matrix: np.ndarray) -> None:
        self.transform.homography_matrix = homography_matrix

    def compute_homography_from_points(
        self,
        src_points: Sequence[Tuple[float, float]],
        dst_points: Sequence[Tuple[float, float]],
    ) -> None:
        self.transform = TabletopTransform.from_points(src_points, dst_points)

    def detect_from_image(self, image, object_names: Optional[Iterable[str]] = None) -> SceneState:
        corners, ids = self.detector.detect_markers(image)
        scene = SceneState(objects=[], drawer=DrawerState(), stable=True, warnings=[])

        if not self.detector.available:
            scene.warnings.append("OpenCV not installed; using synthetic scene state for test/demo mode.")
            return self._synthetic_scene(object_names)

        if ids is None:
            scene.warnings.append("No markers detected.")
            scene.stable = False
            return scene

        seen_ids = []
        for marker_idx, marker_id in enumerate(ids.flatten().tolist()):
            seen_ids.append(marker_id)
            label = self.marker_map.get(marker_id, f"object_{marker_id}")
            if object_names is not None and label not in object_names:
                continue

            pixel_center = corners[marker_idx][0].mean(axis=0)
            if self.transform.homography_matrix is not None:
                x, y = self.transform.project_point((float(pixel_center[0]), float(pixel_center[1])))
            else:
                x, y = float(pixel_center[0]), float(pixel_center[1])

            if label == "drawer_handle":
                scene.drawer.detected = True
                scene.drawer.handle_xy = (float(pixel_center[0]), float(pixel_center[1]))
                scene.drawer.center_x = x
                scene.drawer.center_y = y
                scene.drawer.open_fraction = 1.0 if x >= self.drawer_open_threshold else 0.0
                scene.drawer.pose = {
                    "x": x,
                    "y": y,
                    "state": "open" if scene.drawer.open_fraction else "closed",
                }
            else:
                scene.objects.append(
                    ObjectDetection(
                        name=label,
                        x=x,
                        y=y,
                        z=0.0,
                        confidence=0.9,
                        pixel_center=(float(pixel_center[0]), float(pixel_center[1])),
                        class_name=label,
                    )
                )

        if not scene.objects:
            scene.warnings.append("No task objects detected.")
            scene.stable = False

        if not scene.drawer.detected:
            scene.warnings.append("Drawer handle not detected.")
        return scene

    def detect_scene(self, image=None, object_names: Optional[Iterable[str]] = None) -> SceneState:
        """Convenience wrapper for the whole pipeline.

        If no image is provided, this returns a synthetic plausible scene for testing.
        """
        if image is None:
            return self._synthetic_scene(object_names)
        return self.detect_from_image(image, object_names)

    def _synthetic_scene(self, object_names: Optional[Iterable[str]] = None) -> SceneState:
        names = set(object_names) if object_names is not None else {"plate", "mug", "drawer_handle"}
        objects = []
        if "plate" in names:
            objects.append(ObjectDetection(name="plate", x=0.25, y=0.55, z=0.0, confidence=0.95))
        if "mug" in names:
            objects.append(ObjectDetection(name="mug", x=0.68, y=0.25, z=0.0, confidence=0.94))
        drawer = DrawerState(detected=True, center_x=0.50, center_y=0.20, open_fraction=0.4, handle_xy=(0.60, 0.22), pose={"x": 0.5, "y": 0.2})
        return SceneState(objects=objects, drawer=drawer, stable=True, warnings=[])


def build_demo_scene() -> SceneState:
    pipeline = ScenePipeline()
    return pipeline.detect_scene(object_names=["plate", "mug", "drawer_handle"])
