"""Planar footprint checks on the table surface (MuJoCo world frame, metres).

Objects are approximated by footprint circles and fixtures by axis-aligned
rectangles. These are static placement checks only. They are NOT
collision-free motion guarantees: arm links, trajectories, object orientation,
stacking and dynamics are not modelled.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Rect:
    """Axis-aligned rectangle in the table plane."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def contains(self, x: float, y: float) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max


def all_finite(values: Iterable[float]) -> bool:
    return all(math.isfinite(v) for v in values)


def circle_within_rect(cx: float, cy: float, radius: float, rect: Rect, margin: float = 0.0) -> bool:
    """True if the whole circle, grown by `margin`, lies inside `rect`."""
    reach = radius + margin
    return (
        cx - reach >= rect.x_min
        and cx + reach <= rect.x_max
        and cy - reach >= rect.y_min
        and cy + reach <= rect.y_max
    )


def circle_rect_gap(cx: float, cy: float, radius: float, rect: Rect) -> float:
    """Distance from the circle's edge to the rectangle; <= 0 means they overlap."""
    dx = max(rect.x_min - cx, 0.0, cx - rect.x_max)
    dy = max(rect.y_min - cy, 0.0, cy - rect.y_max)
    return math.hypot(dx, dy) - radius


def circle_circle_gap(ax: float, ay: float, ar: float, bx: float, by: float, br: float) -> float:
    """Distance between two circles' edges; <= 0 means they overlap."""
    return math.hypot(ax - bx, ay - by) - ar - br
