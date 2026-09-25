from __future__ import annotations

from typing import Iterable, Tuple
import numpy as np


class TabletopTransform:
    """Simple homography-based transform from image space to tabletop space.

    This is a clean abstraction for a baseline marker-based calibration. It can be
    replaced or extended with a more robust perception stack later.
    """

    def __init__(self, homography_matrix: np.ndarray | None = None):
        self.homography_matrix = homography_matrix

    @staticmethod
    def from_points(src_points: Iterable[Tuple[float, float]], dst_points: Iterable[Tuple[float, float]]) -> "TabletopTransform":
        src = np.asarray(list(src_points), dtype=np.float32)
        dst = np.asarray(list(dst_points), dtype=np.float32)

        if len(src) < 4 or len(dst) < 4:
            raise ValueError("Need at least 4 point correspondences to estimate a homography.")

        if src.shape[0] != dst.shape[0]:
            raise ValueError("Source and destination point counts must match.")

        H, _ = cv2.findHomography(src, dst)
        return TabletopTransform(H)

    def project_point(self, image_xy: Tuple[float, float]) -> Tuple[float, float]:
        if self.homography_matrix is None:
            raise ValueError("Homography not initialized. Provide calibration points first.")

        point = np.array([image_xy[0], image_xy[1], 1.0], dtype=np.float64)
        transformed = self.homography_matrix @ point
        w = transformed[2]
        if abs(w) < 1e-8:
            raise ValueError("Degenerate homography encountered.")
        x = transformed[0] / w
        y = transformed[1] / w
        return float(x), float(y)

    def project_points(self, image_points):
        return [self.project_point(p) for p in image_points]


# Import is kept local to avoid unnecessary dependency issues in environments without OpenCV.
try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None
