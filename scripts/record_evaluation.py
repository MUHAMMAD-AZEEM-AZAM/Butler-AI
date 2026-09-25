#!/usr/bin/env python3
"""Record video of the ACTUAL evaluated pipeline (the same runs scripts/evaluate.py scores).

Unlike scripts/record_simulation.py, which builds its own primitive sequence, this drives
common.pipeline.run_once — so the footage is of the evaluated run, not a lookalike.

Recording is behaviour-neutral by construction: RecordingTrajectoryExecutor only overrides
TrajectoryExecutor._on_step(), a hook called after each mj_step. It renders and never
touches model, data or ctrl, so actions, physics stepping, seeds, success checks and retry
behaviour are unchanged.

    python scripts/record_evaluation.py --seeds 10 --out outputs/eval10_video
    python scripts/record_evaluation.py --seeds 1 --compare   # seed 0 only, verify neutrality
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw
except ImportError as err:  # pragma: no cover
    sys.exit(f"Missing dependency: {err}. Install with: pip install mujoco numpy Pillow")

from common import EXAMPLE_COMMAND
from common.pipeline import run_once
from stage4_bimanual.trajectory import TrajectoryExecutor

LABEL = (
    "Scripted controller; simulator ground-truth perception; preset "
    "instruction; pouring assessed through a pose proxy, with no fluid model."
)
WIDTH, HEIGHT, FPS = 640, 480, 30


class RecordingTrajectoryExecutor(TrajectoryExecutor):
    """TrajectoryExecutor plus frame capture. Physics is inherited, never reimplemented."""

    renderer: "mujoco.Renderer | None" = None
    camera: "mujoco.MjvCamera | None" = None
    frames: list = []
    capture_every: int = 3

    def __init__(self, model, data, contact_audit=None):
        super().__init__(model, data, contact_audit=contact_audit)
        self._tick = 0

    def _on_step(self) -> None:
        """Read-only observer: render a frame. Never mutates simulation state."""
        cls = type(self)
        if cls.renderer is None:
            return
        if self._tick % cls.capture_every == 0:
            cls.renderer.update_scene(self.data, camera=cls.camera)
            cls.frames.append(Image.fromarray(cls.renderer.render()))
        self._tick += 1


def _make_camera():
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.10, 0.0, 0.72]
    cam.distance, cam.azimuth, cam.elevation = 1.35, 150.0, -25.0
    return cam


def _banner(img: Image.Image, seed: int, outcome: str) -> Image.Image:
    """Burn seed + PASS/FAIL + the description into the frame."""
    out = Image.new("RGB", (img.width, img.height + 46), (17, 17, 17))
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out)
    d.text((8, img.height + 5), f"seed {seed}  -  {outcome}", fill=(255, 255, 255))
    d.text((8, img.height + 19), LABEL[:78], fill=(170, 170, 170))
    d.text((8, img.height + 31), LABEL[78:], fill=(170, 170, 170))
    return out


def _save_video(frames: list, stem: Path) -> str:
    """Write MP4 if an encoder is available, else animated GIF. Returns the file written."""
    if not frames:
        return "no frames"
    try:
        import imageio.v2 as imageio

        path = stem.with_suffix(".mp4")
        with imageio.get_writer(path, fps=FPS, codec="libx264", quality=8) as w:
            for f in frames:
                w.append_data(np.asarray(f))
        return path.name
    except Exception:
        path = stem.with_suffix(".gif")
        frames[0].save(path, save_all=True, append_images=frames[1:],
                       duration=int(1000 / FPS), loop=0, optimize=True)
        return path.name


def record_seed(seed: int, out_dir: Path) -> dict:
    """Run one evaluated seed with recording on; return its measured outcome."""
    from stage4_bimanual import reset_scene
    from stage4_bimanual.sim import MuJoCoSim

    probe = reset_scene(seed)
    if not isinstance(probe, MuJoCoSim):
        sys.exit("MuJoCo simulation unavailable; cannot record.")
    renderer = mujoco.Renderer(probe.model, height=HEIGHT, width=WIDTH)

    cls = RecordingTrajectoryExecutor
    cls.renderer, cls.camera, cls.frames = renderer, _make_camera(), []

    entry = {"seed": seed, "success": False, "attempts": None,
             "failure_reason": None, "crashed": False}
    try:
        run = run_once(EXAMPLE_COMMAND, seed=seed, executor_factory=cls)
        entry["success"] = bool(run.success)
        entry["attempts"] = run.attempts
        verif = [l for l in run.log if l.strip().startswith("-> ok=")]
        entry["final_verification"] = verif[-1].strip() if verif else "verification unavailable"
        if not run.success:
            bad = [l for l in run.log if "refused" in l or "error:" in l
                   or "not counted as success" in l]
            entry["failure_reason"] = bad[-1].strip() if bad else entry["final_verification"]
        entry["final_object_positions"] = {
            k: [round(c, 6) for c in v] for k, v in probe.get_object_positions().items()
        }
        (out_dir / "logs" / f"seed_{seed:02d}.log").write_text("\n".join(run.log), encoding="utf-8")
    except Exception as exc:
        entry["crashed"] = True
        entry["failure_reason"] = f"{type(exc).__name__}: {exc}"
        (out_dir / "logs" / f"seed_{seed:02d}.log").write_text(traceback.format_exc(), encoding="utf-8")

    outcome = "PASS" if entry["success"] else "FAIL"
    labelled = [_banner(f, seed, outcome) for f in cls.frames]
    entry["frames"] = len(labelled)
    entry["video"] = _save_video(labelled, out_dir / f"seed_{seed:02d}_{outcome}")
    cls.renderer, cls.frames = None, []
    renderer.close()
    return entry


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", type=Path, default=Path("outputs/eval10_video"))
    ap.add_argument(
        "--capture-every", type=int, default=RecordingTrajectoryExecutor.capture_every,
        help="render one frame per N physics steps. Affects only how densely frames are "
             "sampled for the video; _on_step never touches simulation state, so physics "
             "is unchanged at any value.",
    )
    ap.add_argument("--montage", action="store_true", help="also write a combined montage")
    args = ap.parse_args()

    out = (_ROOT / args.out) if not args.out.is_absolute() else args.out
    (out / "logs").mkdir(parents=True, exist_ok=True)
    RecordingTrajectoryExecutor.capture_every = max(1, args.capture_every)

    results, montage_frames = [], []
    for seed in range(args.seeds):
        e = record_seed(seed, out)
        print(f"seed {seed}: {'PASS' if e['success'] else 'FAIL'}  "
              f"frames={e['frames']}  -> {e['video']}")
        results.append(e)
        if args.montage:
            vid = out / e["video"]
            if vid.suffix in (".mp4", ".gif"):
                montage_frames.append((seed, "PASS" if e["success"] else "FAIL"))

    succ = sum(r["success"] for r in results)
    summary = {
        "label": LABEL,
        "not_act_performance": True,
        "not_an_openvino_benchmark": True,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "successes": succ, "total": len(results),
        "percent": 100.0 * succ / len(results) if results else 0.0,
        "command": EXAMPLE_COMMAND,
        "capture_every_physics_steps": RecordingTrajectoryExecutor.capture_every,
        "results": results,
        "environment": {
            "python": sys.version, "platform": platform.platform(),
            "commit": subprocess.run(["git", "-C", str(_ROOT), "rev-parse", "HEAD"],
                                     capture_output=True, text=True).stdout.strip(),
            "branch": subprocess.run(["git", "-C", str(_ROOT), "rev-parse", "--abbrev-ref", "HEAD"],
                                     capture_output=True, text=True).stdout.strip(),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n{succ}/{len(results)} = {summary['percent']:.0f}%   -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
