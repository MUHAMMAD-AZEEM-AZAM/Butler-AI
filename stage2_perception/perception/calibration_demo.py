from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from perception.calibration import example_calibration
else:
    from .calibration import example_calibration


def main() -> None:
    calibration = example_calibration()
    image_point = (250.0, 250.0)
    table_point = calibration.project_point(image_point)

    print("Camera frame -> tabletop frame calibration example")
    print("================================================")
    print(f"Image point (camera frame): {image_point}")
    print(f"Projected tabletop point: {table_point}")
    print(
        "This demonstrates the key mapping used by the project: "
        "pixel coordinates in the camera view are transformed into planar table coordinates."
    )


if __name__ == "__main__":
    main()
