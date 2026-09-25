#!/usr/bin/env python3
"""Build a 5x2 grid montage from the per-seed videos written by record_evaluation.py.

    python scripts/make_montage.py --dir outputs/eval10_video

Streams frame-by-frame so memory stays flat. Shorter clips (failed runs end early)
hold their final frame so every tile stays in sync.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw
except ImportError as err:  # pragma: no cover
    sys.exit(f"Missing dependency: {err}")

COLS, ROWS, FPS = 5, 2, 30
TILE_W = 320


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", type=Path, default=Path("outputs/eval10_video"))
    ap.add_argument("--out", type=str, default="montage_all_seeds.mp4")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    d = args.dir if args.dir.is_absolute() else root / args.dir
    clips = sorted(d.glob("seed_*_*.mp4"), key=lambda p: int(re.search(r"seed_(\d+)", p.name).group(1)))
    if not clips:
        sys.exit(f"no seed_*.mp4 found in {d}")

    readers = [imageio.get_reader(str(c)) for c in clips]
    lengths = [r.count_frames() for r in readers]
    total = max(lengths)
    probe = readers[0].get_data(0)
    scale = TILE_W / probe.shape[1]
    tw, th = TILE_W, int(probe.shape[0] * scale)

    out_path = d / args.out
    writer = imageio.get_writer(str(out_path), fps=FPS, codec="libx264", quality=8)
    iters = [iter(r) for r in readers]
    last = [None] * len(readers)

    for _ in range(total):
        canvas = Image.new("RGB", (tw * COLS, th * ROWS), (17, 17, 17))
        for i in range(len(readers)):
            try:
                frame = next(iters[i])
                last[i] = frame
            except StopIteration:
                frame = last[i]
            if frame is None:
                continue
            tile = Image.fromarray(frame).resize((tw, th), Image.BILINEAR)
            canvas.paste(tile, ((i % COLS) * tw, (i // COLS) * th))
        writer.append_data(np.asarray(canvas))

    writer.close()
    for r in readers:
        r.close()
    print(f"montage: {out_path}  ({total} frames, {len(clips)} clips)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
