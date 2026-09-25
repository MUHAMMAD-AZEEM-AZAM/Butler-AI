"""Regression: the drawer and mug welds must never attach without real contact.

Commit 889a812 replaced the contact-gated attach_weld() call in OpenDrawerPrimitive
and PickMugPrimitive with a direct `data.eq_active[weld_id] = 1`. Measured over ten
evaluation seeds, weld_drawer then engaged with zero gripper/object contact in 10/10
seeds and weld_mug in 7/10, so a passing run no longer demonstrated a grasp.

These tests fail if that regression returns.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

mujoco = pytest.importorskip("mujoco")
np = pytest.importorskip("numpy")

from stage4_bimanual.bimanual import reset_scene  # noqa: E402
from stage4_bimanual.primitives import (  # noqa: E402
    OpenDrawerPrimitive,
    PickMugPrimitive,
)
from stage4_bimanual.sim import MuJoCoSim  # noqa: E402
from stage4_bimanual.trajectory import TrajectoryExecutor  # noqa: E402

_PRIMITIVES = Path(__file__).resolve().parents[1] / "primitives.py"
GATED_WELDS = ("weld_drawer", "weld_mug")


def _source_of(class_name: str) -> str:
    """The source text of one primitive class, up to the next class definition."""
    text = _PRIMITIVES.read_text(encoding="utf-8")
    start = text.index(f"class {class_name}(")
    nxt = text.find("\nclass ", start + 1)
    return text[start:] if nxt < 0 else text[start:nxt]


@pytest.mark.parametrize(
    ("class_name", "weld"),
    [("OpenDrawerPrimitive", "weld_drawer"), ("PickMugPrimitive", "weld_mug")],
)
def test_primitive_does_not_activate_its_weld_directly(class_name, weld):
    """The primitive must go through attach_weld(), never poke eq_active itself."""
    source = _source_of(class_name)
    direct = re.findall(r"eq_active\s*\[[^\]]+\]\s*=\s*1", source)
    assert not direct, (
        f"{class_name} sets eq_active directly ({direct}); that bypasses the contact "
        f"check in attach_weld() and lets {weld} attach to an untouched object"
    )
    assert "attach_weld(" in source, f"{class_name} never calls attach_weld()"


@pytest.mark.parametrize(
    ("class_name", "weld"),
    [("OpenDrawerPrimitive", "weld_drawer"), ("PickMugPrimitive", "weld_mug")],
)
def test_primitive_requires_both_jaws(class_name, weld):
    source = _source_of(class_name)
    assert "require_both_jaws=True" in source, (
        f"{class_name} must weld {weld} only on a real two-jaw pinch"
    )


def _weld_geoms(model, body_id):
    out = set()
    for g in range(model.ngeom):
        b = int(model.geom_bodyid[g])
        while b > 0:
            if b == body_id:
                out.add(g)
                break
            b = int(model.body_parentid[b])
    return out


def _contacts_between(model, data, weld_name):
    wid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, weld_name)
    if wid < 0:
        return 0
    g1 = _weld_geoms(model, int(model.eq_obj1id[wid]))
    g2 = _weld_geoms(model, int(model.eq_obj2id[wid]))
    n = 0
    for c in range(data.ncon):
        con = data.contact[c]
        if (con.geom1 in g1 and con.geom2 in g2) or (con.geom1 in g2 and con.geom2 in g1):
            n += 1
    return n


class _WeldWatcher(TrajectoryExecutor):
    """Read-only observer: records contact count whenever a gated weld switches on."""

    def __init__(self, model, data, contact_audit=None):
        super().__init__(model, data, contact_audit=contact_audit)
        self.activations: list[tuple[str, int]] = []
        self._prev = {w: 0 for w in GATED_WELDS}

    def _on_step(self) -> None:
        for weld in GATED_WELDS:
            wid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, weld)
            if wid < 0:
                continue
            active = int(self.data.eq_active[wid])
            if active == 1 and self._prev[weld] == 0:
                self.activations.append((weld, _contacts_between(self.model, self.data, weld)))
            self._prev[weld] = active


@pytest.mark.parametrize("seed", [0, 2])
def test_drawer_weld_only_attaches_while_touching(seed):
    sim = reset_scene(seed)
    if not isinstance(sim, MuJoCoSim):
        pytest.skip("MuJoCo scene unavailable")
    watcher = _WeldWatcher(sim.model, sim.data, contact_audit=sim.contact_audit)
    OpenDrawerPrimitive(watcher, sim).execute()

    ungrounded = [(w, n) for w, n in watcher.activations if w == "weld_drawer" and n == 0]
    assert not ungrounded, (
        f"seed {seed}: weld_drawer attached with zero gripper/object contact {ungrounded}"
    )


@pytest.mark.parametrize("seed", [0, 2])
def test_mug_weld_only_attaches_while_touching(seed):
    sim = reset_scene(seed)
    if not isinstance(sim, MuJoCoSim):
        pytest.skip("MuJoCo scene unavailable")
    watcher = _WeldWatcher(sim.model, sim.data, contact_audit=sim.contact_audit)
    OpenDrawerPrimitive(watcher, sim).execute()   # the mug pick assumes the drawer stage ran
    PickMugPrimitive(watcher, sim).execute()

    ungrounded = [(w, n) for w, n in watcher.activations if w == "weld_mug" and n == 0]
    assert not ungrounded, (
        f"seed {seed}: weld_mug attached with zero gripper/object contact {ungrounded}"
    )
