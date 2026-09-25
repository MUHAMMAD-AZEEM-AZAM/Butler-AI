from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional


@dataclass
class ObjectDetection:
    """Detected object in tabletop coordinates."""
    name: str
    x: float
    y: float
    z: float = 0.0
    confidence: float = 1.0
    pixel_center: Optional[tuple] = None
    class_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DrawerState:
    """Drawer state in table/world coordinates."""
    detected: bool = False
    center_x: float = 0.0
    center_y: float = 0.0
    open_fraction: float = 0.0
    handle_xy: Optional[tuple] = None
    pose: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SceneState:
    """Stable, planner-facing view of the scene."""
    objects: List[ObjectDetection] = field(default_factory=list)
    drawer: DrawerState = field(default_factory=DrawerState)
    stable: bool = True
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objects": [obj.to_dict() for obj in self.objects],
            "drawer": self.drawer.to_dict(),
            "stable": self.stable,
            "warnings": self.warnings,
        }
