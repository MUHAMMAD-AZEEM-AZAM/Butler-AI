from __future__ import annotations

import json

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from perception.scene_pipeline import build_demo_scene
else:
    from .scene_pipeline import build_demo_scene


def main() -> None:
    scene = build_demo_scene()
    summary = scene.to_dict()

    print("\nTabletop scene summary")
    print("====================")
    print(f"Stable: {summary['stable']}")
    print(f"Warnings: {summary['warnings'] or 'none'}")

    if summary["drawer"]["detected"]:
        print(f"Drawer handle detected at image position: {summary['drawer']['handle_xy']}")

    if summary["objects"]:
        print("Objects:")
        for obj in summary["objects"]:
            print(f"  - {obj['name']}: x={obj['x']:.3f}, y={obj['y']:.3f}, confidence={obj['confidence']:.2f}")
    else:
        print("Objects: none detected")

    print("\nFull structured scene payload:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
