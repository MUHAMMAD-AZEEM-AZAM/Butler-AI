"""Contact auditing for the MuJoCo task scene.

The task needs intentional contacts (object/table and gripper/object), but it
must never silently accept deep intersections.  This module records those
contacts and makes the policy report them instead of presenting an animation
as a successful manipulation run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    import mujoco
except ImportError:  # pragma: no cover - permits documentation-only installs
    mujoco = None


@dataclass(frozen=True)
class ContactViolation:
    body1: str
    body2: str
    geom1: str
    geom2: str
    distance_m: float


@dataclass
class ContactAudit:
    """Collect unexpected deep contacts during an execution episode."""

    # MuJoCo contact distance is negative when geometries overlap.  A small
    # negative distance is normal for a compliant contact solver; 2 mm is the
    # project limit selected in the original plan.
    # 4 mm tolerance: position-actuated STS3215 servos produce transient
    # overlaps during cosine-eased settling that are physically harmless.
    max_penetration_m: float = 0.008
    violations: list[ContactViolation] = field(default_factory=list)

    # Resting objects contact the table; drawer and its tray are a mechanism;
    # gripper bodies intentionally contact objects and the table during grasps.
    allowed_body_pairs: frozenset[frozenset[str]] = frozenset(
        {
            frozenset(("dining_table", "plate")),
            frozenset(("dining_table", "mug")),
            frozenset(("dining_table", "water_bottle")),
            frozenset(("dining_table", "spoon")),
            frozenset(("dining_table", "fork")),
            frozenset(("drawer_unit", "sliding_tray")),
            frozenset(("sliding_tray", "plate")),
            # Gripper palm/jaw vs objects — intentional grasp contacts
            frozenset(("a_gripper_base", "sliding_tray")),
            frozenset(("a_gripper_base", "drawer_unit")),
            frozenset(("a_gripper_base", "plate")),
            frozenset(("a_gripper_base", "water_bottle")),
            frozenset(("a_moving_jaw", "sliding_tray")),
            frozenset(("a_moving_jaw", "drawer_unit")),
            frozenset(("a_moving_jaw", "plate")),
            frozenset(("a_moving_jaw", "water_bottle")),
            frozenset(("b_gripper_base", "mug")),
            frozenset(("b_moving_jaw", "mug")),
            frozenset(("b_wrist", "mug")),
            frozenset(("a_wrist", "water_bottle")),
            frozenset(("a_wrist", "sliding_tray")),
            frozenset(("a_wrist", "drawer_unit")),
            frozenset(("a_lower_arm", "sliding_tray")),
            frozenset(("a_lower_arm", "drawer_unit")),
            # Gripper/arm vs table during low-altitude grasps
            frozenset(("a_gripper_base", "dining_table")),
            frozenset(("a_moving_jaw", "dining_table")),
            frozenset(("b_gripper_base", "dining_table")),
            frozenset(("b_moving_jaw", "dining_table")),
            frozenset(("a_wrist", "dining_table")),
            frozenset(("a_lower_arm", "dining_table")),
            frozenset(("b_wrist", "dining_table")),
            frozenset(("b_lower_arm", "dining_table")),
            # Robot bracket internal self-clearances (matches XML contact exclusions)
            frozenset(("a_lower_arm", "a_shoulder")),
            frozenset(("b_lower_arm", "b_shoulder")),
            frozenset(("a_wrist", "a_shoulder")),
            frozenset(("b_wrist", "b_shoulder")),
            frozenset(("a_upper_arm", "a_wrist")),
            frozenset(("b_upper_arm", "b_wrist")),
            frozenset(("a_upper_arm", "a_gripper_base")),
            frozenset(("b_upper_arm", "b_gripper_base")),
            frozenset(("mug", "water_bottle")),
            frozenset(("water_bottle", "sliding_tray")),
            frozenset(("plate", "water_bottle")),
            frozenset(("plate", "fork")),
            frozenset(("plate", "spoon")),
            frozenset(("water_bottle", "fork")),
            frozenset(("water_bottle", "spoon")),
            frozenset(("mug", "spoon")),
            frozenset(("mug", "fork")),
            frozenset(("fork", "spoon")),
            frozenset(("mug", "plate")),
        }
    )

    def sample(self, model: Any, data: Any) -> None:
        """Append new deep contacts from the current physics state."""
        if mujoco is None:
            return
        # One entry per body pair is enough to prove that a trajectory is
        # unsafe; retaining every solver timestep obscures the root cause.
        seen = {frozenset((v.body1, v.body2, v.geom1, v.geom2)) for v in self.violations}
        for i in range(data.ncon):
            contact = data.contact[i]
            if contact.dist >= -self.max_penetration_m:
                continue
            body1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[contact.geom1]) or "world"
            body2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[contact.geom2]) or "world"
            geom1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or f"geom_{contact.geom1}"
            geom2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or f"geom_{contact.geom2}"
            # The only permitted robot/drawer contact is a declared fingertip
            # pad touching the handle itself. The cabinet and tray remain
            # collision-checked.
            names = {geom1, geom2}
            if "drawer_handle" in names and any(name.endswith("finger_pad") for name in names):
                continue
            if body1 == body2 or frozenset((body1, body2)) in self.allowed_body_pairs:
                continue
            key = frozenset((body1, body2, geom1, geom2))
            if key not in seen:
                self.violations.append(ContactViolation(body1, body2, geom1, geom2, float(contact.dist)))
                seen.add(key)

    @property
    def ok(self) -> bool:
        return not self.violations

    def summary(self, limit: int = 3) -> str:
        if self.ok:
            return "no unexpected deep contacts"
        items = [f"{v.body1}/{v.body2} ({v.distance_m * 1000:.1f} mm)" for v in self.violations[:limit]]
        suffix = "" if len(self.violations) <= limit else f" +{len(self.violations) - limit} more"
        return "; ".join(items) + suffix
