from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validate_perception import create_marker_scene, validate_image


def run_seed(seed: int, source: str = "synthetic") -> Dict[str, Any]:
    if source == "mujoco":
        from stage4_bimanual.bimanual import get_camera_frame, reset_scene

        image = get_camera_frame(reset_scene(seed))
        if image is None:
            return {"seed": seed, "success": False, "reason": "MuJoCo camera frame unavailable."}
        import cv2

        image_path = ROOT / "stage2_perception" / "evaluation" / f"mujoco_seed_{seed}.png"
        cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    else:
        image_path = ROOT / "stage2_perception" / "evaluation" / f"seed_{seed}_marker_scene.png"
        create_marker_scene(image_path, "open" if seed % 2 else "closed")

    report = validate_image(image_path, drawer_state="open" if seed % 2 else "closed")
    return {
        "seed": seed,
        "success": report["pass"],
        "drawer_state": report["drawer_state"],
        "errors": report["errors"],
        "warnings": report["warnings"],
    }


def run_suite(num_seeds: int = 10, source: str = "synthetic") -> List[Dict[str, Any]]:
    return [run_seed(seed, source) for seed in range(num_seeds)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run randomized evaluation seeds for the scene pipeline.")
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--source", choices=("synthetic", "mujoco"), default="synthetic")
    args = parser.parse_args()

    results = run_suite(args.num_seeds, args.source)
    success_count = sum(1 for result in results if result["success"])
    report = {
        "source": args.source,
        "num_seeds": args.num_seeds,
        "success_count": success_count,
        "success_rate": success_count / args.num_seeds if args.num_seeds else 0.0,
        "results": results,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
