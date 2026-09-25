from __future__ import annotations

import cv2
import numpy as np


def generate_aruco_marker(marker_id: int = 0, size: int = 200):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = np.zeros((size, size), dtype=np.uint8)
    image = cv2.aruco.generateImageMarker(dictionary, marker_id, size)
    return image


def main() -> None:
    marker = generate_aruco_marker(marker_id=0, size=300)
    cv2.imwrite("aruco_marker_0.png", marker)
    print("Saved aruco_marker_0.png")


if __name__ == "__main__":
    main()
