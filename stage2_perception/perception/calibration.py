from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np


class CameraCalibration:
    """Handles a simple homography calibration flow for tabletop mapping.

    The process is:
    1. define calibration corners in image space
    2. define their corresponding real-world tabletop coordinates
    3. estimate a homography
    4. project image points into table coordinates
    """

    def __init__(self, image_corners: Sequence[Tuple[float, float]] | None = None, table_corners: Sequence[Tuple[float, float]] | None = None):
        self.image_corners = list(image_corners) if image_corners is not None else []
        self.table_corners = list(table_corners) if table_corners is not None else []
        self.homography = None

    def set_points(self, image_corners: Sequence[Tuple[float, float]], table_corners: Sequence[Tuple[float, float]]) -> None:
        if len(image_corners) != len(table_corners):
            raise ValueError("Image and table corner counts must match.")
        if len(image_corners) < 4:
            raise ValueError("Need at least 4 calibration points.")

        self.image_corners = list(image_corners)
        self.table_corners = list(table_corners)
        self.homography = self._compute_homography(self.image_corners, self.table_corners)

    @staticmethod
    def _compute_homography(src_points: Sequence[Tuple[float, float]], dst_points: Sequence[Tuple[float, float]]):
        try:
            import cv2
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("OpenCV is required for calibration.") from exc

        src = np.asarray(src_points, dtype=np.float32)
        dst = np.asarray(dst_points, dtype=np.float32)
        return cv2.findHomography(src, dst)[0]

    def project_point(self, image_xy: Tuple[float, float]) -> Tuple[float, float]:
        if self.homography is None:
            raise ValueError("Calibration has not been set with enough points.")

        try:
            import cv2
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("OpenCV is required for projection.") from exc

        px = np.array([image_xy[0], image_xy[1], 1.0], dtype=np.float64)
        transformed = self.homography @ px
        w = transformed[2]
        if abs(w) < 1e-8:
            raise ValueError("Degenerate homography. Calibration likely failed.")
        x = transformed[0] / w
        y = transformed[1] / w
        return float(x), float(y)

    def project_points(self, image_points: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
        return [self.project_point(point) for point in image_points]


def example_calibration() -> CameraCalibration:
    # Example calibration points: image corners to tabletop coordinates in meters.
    image_corners = [
        (100.0, 100.0),
        (500.0, 100.0),
        (500.0, 400.0),
        (100.0, 400.0),
    ]
    table_corners = [
        (0.0, 0.0),
        (0.8, 0.0),
        (0.8, 0.6),
        (0.0, 0.6),
    ]

    calibration = CameraCalibration()
    calibration.set_points(image_corners, table_corners)
    return calibration
