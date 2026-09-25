from __future__ import annotations

from typing import Dict, List, Optional, Tuple

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


class ArucoDetector:
    """Detector wrapper for ArUco-based object and drawer localization.

    This is the baseline detection strategy requested in the challenge plan.
    """

    def __init__(self, dictionary_type: str = "DICT_ARUCO_ORIGINAL"):
        self.available = cv2 is not None
        self.dictionary_type = dictionary_type
        self.dictionary = None
        self.parameters = None

        if not self.available:
            return

        self.dictionary = cv2.aruco.getPredefinedDictionary(self._map_dictionary_name(dictionary_type))
        self.parameters = cv2.aruco.DetectorParameters()

    @staticmethod
    def _map_dictionary_name(name: str):
        mapping = {
            "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
            "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
            "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
            "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
        }
        if name not in mapping:
            raise ValueError(f"Unsupported ArUco dictionary: {name}")
        return mapping[name]

    def detect_markers(self, image):
        if not self.available:
            return [], None

        if hasattr(cv2.aruco, "ArucoDetector"):
            detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
            corners, ids, _ = detector.detectMarkers(image)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(image, self.dictionary, parameters=self.parameters)
        return corners, ids

    def detect_object_centers(self, image) -> List[Tuple[int, int]]:
        if not self.available:
            return []

        corners, ids = self.detect_markers(image)
        centers: List[Tuple[int, int]] = []

        if ids is None:
            return centers

        for corner in corners:
            points = corner[0]
            x = float(points[:, 0].mean())
            y = float(points[:, 1].mean())
            centers.append((x, y))

        return centers

    def estimate_drawer_state(self, image) -> Dict[str, object]:
        if not self.available:
            return {"detected": False, "open_fraction": 0.0, "reason": "OpenCV not installed."}

        corners, ids = self.detect_markers(image)
        if ids is None:
            return {"detected": False, "open_fraction": 0.0}

        # Placeholder logic: the drawer state will be refined once the actual scene geometry is known.
        return {
            "detected": True,
            "open_fraction": 0.5,
            "marker_count": len(ids),
        }
