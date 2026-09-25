"""Stage 3 rule-based planner: validate a Task against a SceneState and emit Actions.

plan_detailed() runs, in order:
  1. task validation  - unique IDs, known dependencies, no cycles, per-action fields, names
  2. support check    - unsupported actions, constraints and semantics are rejected, never ignored
  3. scene validation - finite coordinates inside the configured MuJoCo-world workspace
  4. ordering         - dependencies first; otherwise the order in which steps were listed
  5. walk             - resolve each step to one Action while tracking a PREDICTED state

The predicted state (what each arm holds, which drawers are open, where placed
objects end up) assumes every earlier action succeeds exactly as planned. It is
bookkeeping, not measurement: nothing here simulates dynamics or checks arm
motion for collisions.

When a step's pose depends on something this observation cannot show (the
contents of a drawer opened earlier in the plan, or an object held by an arm
after a pick planned in the same call), Actions stop at that step, but the
remaining steps are still checked symbolically so an impossible task fails
before anything moves. The caller executes the returned prefix, re-observes,
and calls again with completed_step_ids.
"""

from __future__ import annotations

import heapq
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from common.types import Action, ActionType, SceneState, Task, TaskStep

from stage3_policy.config import PlannerConfig, load_planner_config
from stage3_policy.errors import PreconditionError, SceneError, TaskValidationError, UnsupportedError
from stage3_policy.geometry import all_finite, circle_circle_gap, circle_rect_gap, circle_within_rect

MODE = "rule_based"

Point3 = tuple[float, float, float]

_STEP_FIELDS = ("target", "object", "destination", "source", "into")
_REQUIRED_FIELDS: dict[ActionType, tuple[str, ...]] = {
    ActionType.OPEN_DRAWER: ("target",),
    ActionType.CLOSE_DRAWER: ("target",),
    ActionType.PICK: ("object",),
    ActionType.PLACE: ("object", "destination"),
    ActionType.POUR: ("source", "into"),
    ActionType.HANDOFF: ("object",),
}
_UNKNOWN_OBJECT_PREFIX = "unknown_object:"
_CONTRACT_NOTE = "see stage3_policy/CONTRACT_PROPOSAL.md"


@dataclass(frozen=True)
class PlanResult:
    """Outcome of plan_detailed().

    actions:            executable now, in execution order, one per planned TaskStep
    complete:           True when `actions` covers every step not in completed_step_ids
    pending_step_ids:   steps to plan again after executing `actions` and re-observing
    blocked_step_id:    the first pending step, if any
    blocked_reason:     why that step needs a fresh observation
    notes:              planner decisions worth reviewing (e.g. which slot was chosen)
    warnings:           heuristic concerns that did not block planning
    predicted_holdings: PREDICTED arm -> held object after every step, assuming success
    mode:               which planner produced this result
    """

    actions: tuple[Action, ...]
    complete: bool
    pending_step_ids: tuple[int, ...]
    blocked_step_id: int | None
    blocked_reason: str | None
    notes: tuple[str, ...]
    warnings: tuple[str, ...]
    predicted_holdings: Mapping[str, str | None]
    mode: str = MODE


def plan_detailed(
    task: Task,
    scene: SceneState,
    *,
    completed_step_ids: Iterable[int] = (),
    config: PlannerConfig | None = None,
) -> PlanResult:
    """Plan `task` against `scene`; raise a PlanningError subclass if it cannot be done.

    completed_step_ids lists steps the caller has already executed (from its
    execution log). SceneState alone cannot say which steps finished or what an
    arm is holding, so without it every step is planned from scratch.
    """
    cfg = config if config is not None else load_planner_config()
    completed = frozenset(completed_step_ids)

    _validate_task(task, cfg, completed)
    _check_supported(task, cfg)
    warnings: list[str] = []
    _validate_scene(task, scene, cfg, warnings)

    ordered = _stable_order(task.steps)
    if [s.id for s in ordered] != [s.id for s in task.steps]:
        warnings.append(
            "steps were listed out of dependency order; execution order is "
            + " -> ".join(str(s.id) for s in ordered)
        )

    walker = _Walker(task, scene, cfg, ordered, completed, warnings)
    for step in ordered:
        if step.id in completed:
            walker.replay_completed(step)
    for step in ordered:
        if step.id not in completed:
            walker.plan_step(step)
    walker.note_constraint_coverage()

    blocked_id, blocked_reason = walker.blocked if walker.blocked else (None, None)
    return PlanResult(
        actions=tuple(walker.actions),
        complete=walker.blocked is None,
        pending_step_ids=tuple(walker.pending),
        blocked_step_id=blocked_id,
        blocked_reason=blocked_reason,
        notes=tuple(walker.notes),
        warnings=tuple(dict.fromkeys(warnings)),
        predicted_holdings=MappingProxyType(dict(walker.holding)),
    )


# --------------------------------------------------------------------------- validation


def _validate_task(task: Task, cfg: PlannerConfig, completed: frozenset[int]) -> None:
    issues: list[str] = []
    counts = Counter(step.id for step in task.steps)
    duplicates = sorted(step_id for step_id, n in counts.items() if n > 1)
    if duplicates:
        issues.append(f"duplicate step ids {duplicates}")

    for step in task.steps:
        label = _label(step)
        for dep in step.depends_on:
            if dep == step.id:
                issues.append(f"{label}: depends on itself")
            elif dep not in counts:
                issues.append(f"{label}: depends on unknown step {dep}")

        required = _REQUIRED_FIELDS[step.action]
        for field in _STEP_FIELDS:
            value = getattr(step, field)
            if field in required and value is None:
                issues.append(f"{label}: missing required field {field!r}")
            elif field not in required and value is not None:
                issues.append(f"{label}: field {field}={value!r} is not used by this action; refusing to ignore it")

        if step.target is not None and step.target not in cfg.drawers:
            issues.append(f"{label}: unknown drawer {step.target!r} (configured: {sorted(cfg.drawers)})")
        for field in ("object", "source", "into"):
            value = getattr(step, field)
            if value is not None and value not in cfg.objects:
                issues.append(f"{label}: {field} {value!r} has no grasp configuration (configured: {sorted(cfg.objects)})")
        if step.destination is not None and step.destination not in cfg.slots:
            issues.append(f"{label}: unknown destination {step.destination!r} (configured: {sorted(cfg.slots)})")
        if step.source is not None and step.source == step.into:
            issues.append(f"{label}: source and into are both {step.source!r}")

        hold = step.requires_hold
        if hold is not None:
            if hold.arm == step.arm:
                issues.append(f"{label}: requires_hold names the acting arm {hold.arm}; the holding arm must be the other arm")
            if hold.object not in cfg.objects:
                issues.append(f"{label}: requires_hold object {hold.object!r} has no configuration")

    for constraint in task.constraints:
        if constraint.startswith(_UNKNOWN_OBJECT_PREFIX):
            word = constraint[len(_UNKNOWN_OBJECT_PREFIX) :]
            issues.append(
                f"constraint {constraint!r}: Stage 1 could not map {word!r} to a known object; refusing to plan a partial task"
            )

    unknown_completed = sorted(completed - counts.keys())
    if unknown_completed:
        issues.append(f"completed_step_ids {unknown_completed} are not steps of this task")

    if not issues:
        cycle = _find_cycle(task.steps)
        if cycle:
            issues.append("dependency cycle (step -> depends on): " + " -> ".join(str(i) for i in cycle))
    if not issues:
        for step in task.steps:
            if step.id in completed:
                missing = sorted(set(step.depends_on) - completed)
                if missing:
                    issues.append(f"step {step.id} is marked completed but its dependencies {missing} are not")

    if issues:
        raise TaskValidationError(f"invalid task {task.command!r}", issues=issues)


def _check_supported(task: Task, cfg: PlannerConfig) -> None:
    issues: list[str] = []
    for step in task.steps:
        label = _label(step)
        if step.action is ActionType.HANDOFF:
            issues.append(
                f"{label}: handoff is not supported. The contract does not say whether `arm` gives or receives, "
                f"has no exchange pose, and Stage 4 has no handoff primitive ({_CONTRACT_NOTE})"
            )
        elif step.action is ActionType.CLOSE_DRAWER:
            issues.append(
                f"{label}: close_drawer is not supported. SceneState has no handle pose or opening distance for an "
                f"open drawer, and Stage 4 has no close primitive ({_CONTRACT_NOTE})"
            )
        elif step.action in (ActionType.PICK, ActionType.PLACE):
            spec = cfg.objects[step.object]
            if spec.requires_orientation:
                issues.append(
                    f"{label}: {step.object!r} needs its yaw for grasping or placement, which SceneState and Action "
                    f"cannot carry ({_CONTRACT_NOTE})"
                )
            elif step.action is ActionType.PLACE and step.object not in cfg.slots[step.destination]:
                issues.append(f"{label}: no layout slot is configured for {step.object!r} on {step.destination!r}")
        elif step.action is ActionType.POUR:
            if step.source not in cfg.pour.sources:
                issues.append(f"{label}: {step.source!r} is not a configured pour source {sorted(cfg.pour.sources)}")
            if step.into not in cfg.pour.receivers:
                issues.append(f"{label}: {step.into!r} is not a configured pour receiver {sorted(cfg.pour.receivers)}")

    for constraint in task.constraints:
        if constraint not in cfg.constraints:
            issues.append(
                f"constraint {constraint!r} is not supported (supported: {sorted(cfg.constraints)}); "
                "it will not be silently ignored"
            )
    if issues:
        raise UnsupportedError(f"task {task.command!r} needs unsupported behaviour", issues=issues)


def _validate_scene(task: Task, scene: SceneState, cfg: PlannerConfig, warnings: list[str]) -> None:
    non_finite = [f"{name!r}: {xyz}" for name, xyz in sorted(scene.objects.items()) if not all_finite(xyz)]
    if non_finite:
        raise SceneError("SceneState contains non-finite object coordinates", issues=non_finite)

    referenced = _referenced_objects(task)
    issues: list[str] = []
    ws, (z_low, z_high) = cfg.workspace_xy, cfg.workspace_z
    for name in sorted(scene.objects):
        x, y, z = scene.objects[name]
        if ws.contains(x, y) and z_low <= z <= z_high:
            if name not in cfg.objects:
                warnings.append(f"scene object {name!r} has no configured footprint and is ignored for placement clearance")
            continue
        message = (
            f"{name!r} at ({x:.3f}, {y:.3f}, {z:.3f}) is outside the {cfg.frame_name} workspace "
            f"x[{ws.x_min}, {ws.x_max}] y[{ws.y_min}, {ws.y_max}] z[{z_low}, {z_high}]"
        )
        if name in referenced:
            issues.append(message)
        else:
            warnings.append(message + "; ignored because the task does not use it")
    if issues:
        raise SceneError(
            f"object coordinates are not in the {cfg.frame_name} frame the planner expects "
            f"(metres, table top at z={cfg.surface_z:.2f}); perception must report MuJoCo world coordinates",
            issues=issues,
        )
    for name in sorted(scene.drawers):
        if name not in cfg.drawers:
            warnings.append(f"scene drawer {name!r} is not configured and is ignored")


def _referenced_objects(task: Task) -> set[str]:
    names: set[str] = set()
    for step in task.steps:
        names.update(v for v in (step.object, step.source, step.into) if v is not None)
        if step.requires_hold is not None:
            names.add(step.requires_hold.object)
    return names


def _find_cycle(steps: Sequence[TaskStep]) -> list[int] | None:
    deps = {s.id: list(dict.fromkeys(s.depends_on)) for s in steps}
    state: dict[int, int] = {}  # 1 = on the current path, 2 = finished
    path: list[int] = []

    def visit(node: int) -> list[int] | None:
        state[node] = 1
        path.append(node)
        for dep in deps[node]:
            if state.get(dep) == 1:
                return path[path.index(dep) :] + [dep]
            if dep not in state:
                found = visit(dep)
                if found:
                    return found
        path.pop()
        state[node] = 2
        return None

    for step in steps:
        if step.id not in state:
            found = visit(step.id)
            if found:
                return found
    return None


def _stable_order(steps: Sequence[TaskStep]) -> list[TaskStep]:
    """Topological order; among steps that are ready at the same time, the earliest-listed goes first."""
    remaining = {s.id: len(set(s.depends_on)) for s in steps}
    dependents: dict[int, list[int]] = {s.id: [] for s in steps}
    for index, step in enumerate(steps):
        for dep in set(step.depends_on):
            dependents[dep].append(index)
    ready = [i for i, s in enumerate(steps) if remaining[s.id] == 0]
    heapq.heapify(ready)
    order: list[TaskStep] = []
    while ready:
        step = steps[heapq.heappop(ready)]
        order.append(step)
        for index in dependents[step.id]:
            remaining[steps[index].id] -= 1
            if remaining[steps[index].id] == 0:
                heapq.heappush(ready, index)
    return order


def _label(step: TaskStep) -> str:
    return f"step {step.id} ({step.action.value})"


# --------------------------------------------------------------------------- walking


class _Walker:
    """Resolves ordered steps to Actions while tracking the predicted symbolic state."""

    def __init__(
        self,
        task: Task,
        scene: SceneState,
        cfg: PlannerConfig,
        ordered: Sequence[TaskStep],
        completed: frozenset[int],
        warnings: list[str],
    ) -> None:
        self.task = task
        self.scene = scene
        self.cfg = cfg
        self.completed = completed
        self.warnings = warnings
        self.notes: list[str] = []
        self.ancestors = _ancestors(ordered)
        self.holding: dict[str, str | None] = {"A": None, "B": None}
        self.hold_origin: dict[str, tuple[str, int]] = {}  # arm -> ("completed" | "planned", step id)
        # drawer -> (state, "observed" | "planned", step id that opened it)
        self.drawers: dict[str, tuple[str, str, int | None]] = {
            name: (state, "observed", None) for name, state in scene.drawers.items()
        }
        # objects not held by an arm -> (base position, "observed" | "planned")
        self.locations: dict[str, tuple[Point3, str]] = {
            name: (tuple(xyz), "observed") for name, xyz in scene.objects.items()
        }
        self.actions: list[Action] = []
        self.pending: list[int] = []
        self.blocked: tuple[int, str] | None = None

    # ---- entry points

    def replay_completed(self, step: TaskStep) -> None:
        """Apply a step the caller reports as executed; observations win over predictions."""
        if step.action is ActionType.OPEN_DRAWER:
            observed = self.scene.drawers.get(step.target)
            if observed != "open":
                raise SceneError(
                    f"step {step.id} (open_drawer) is reported complete, but {step.target!r} is observed "
                    f"{observed or 'nowhere in SceneState.drawers'}; plan it again instead of marking it complete",
                    step_id=step.id,
                )
            self.drawers[step.target] = ("open", "observed", step.id)
        elif step.action is ActionType.PICK:
            self._require_free(step)
            self._require_not_held(step, step.object)
            self._grab(step.arm, step.object, "completed", step.id)
        elif step.action is ActionType.PLACE:
            self._require_holding(step, step.object)
            self._release(step.arm)
            observed = self.scene.objects.get(step.object)
            if observed is not None:
                self.locations[step.object] = (tuple(observed), "observed")
        elif step.action is ActionType.POUR:
            self._require_holding(step, step.source)

    def plan_step(self, step: TaskStep) -> None:
        self._check_hold_requirement(step)
        handler = {
            ActionType.OPEN_DRAWER: self._open_drawer,
            ActionType.PICK: self._pick,
            ActionType.PLACE: self._place,
            ActionType.POUR: self._pour,
        }[step.action]
        action = handler(step)
        if self.blocked is None and action is not None:
            self.actions.append(action)
        else:
            self.pending.append(step.id)

    def note_constraint_coverage(self) -> None:
        for constraint in self.task.constraints:
            spec = self.cfg.constraints[constraint]
            if not any(s.action is ActionType.PLACE and s.object in spec.applies_to for s in self.task.steps):
                self.notes.append(
                    f"constraint {constraint!r} applies to placements of {sorted(spec.applies_to)} and this task places "
                    "none of them; positions chosen inside Stage 4 primitives (e.g. where a bottle is set down after "
                    "pouring) are not controlled by Stage 3"
                )

    # ---- per-action resolution (return None when the pose needs a fresh observation)

    def _open_drawer(self, step: TaskStep) -> Action | None:
        name = step.target
        spec = self.cfg.drawers[name]
        self._require_free(step)
        current = self.drawers.get(name)
        if current is None:
            raise SceneError(
                f"step {step.id} open_drawer: {name!r} has no observed state in SceneState.drawers; "
                "the planner will not assume it is closed",
                step_id=step.id,
            )
        state, source, by_step = current
        if state == "open":
            if source == "planned":
                raise PreconditionError(f"step {step.id} open_drawer: {name!r} is already opened by step {by_step}", step_id=step.id)
            raise PreconditionError(
                f"step {step.id} open_drawer: {name!r} is observed open. SceneState has no handle pose for an open "
                "drawer, so there is nothing valid to grasp. If this step already ran, pass it in completed_step_ids.",
                step_id=step.id,
            )
        self.drawers[name] = ("open", "planned", step.id)
        target = spec.handle_grasp_point_closed
        self._check_reach(step, target)
        return self._action(step, name, target, spec.grip_force, spec.approach_height_m)

    def _pick(self, step: TaskStep) -> Action | None:
        name = step.object
        spec = self.cfg.objects[name]
        self._require_free(step)
        self._require_not_held(step, name)
        pose: Point3 | None = None
        located = self.locations.get(name)
        if located is None:
            opened_here = sorted(d for d, (state, source, _) in self.drawers.items() if state == "open" and source == "planned")
            if not opened_here:
                hidden = sorted(d for d in self.cfg.drawers if self.drawers.get(d, ("unobserved",))[0] != "open")
                hint = f" Closed or unobserved drawers {hidden} could hide it: open one before step {step.id}." if hidden else ""
                raise SceneError(
                    f"step {step.id} pick: {name!r} is not in the observed scene.{hint} The planner does not invent positions.",
                    step_id=step.id,
                )
            self._block(step, f"{name!r} is not observed yet; it may be inside {opened_here}, opened earlier in this plan.")
        else:
            xyz, source = located
            drawer = self._drawer_containing(xyz)
            if drawer is None or self._drawer_allows_pose(step, name, drawer):
                if source == "planned":
                    self.notes.append(f"step {step.id}: {name!r} position is the predicted placement from an earlier step, not an observation")
                pose = _offset(xyz, spec.grasp_offset_m)
        self._grab(step.arm, name, "planned", step.id)
        if pose is None:
            return None
        self._check_reach(step, pose)
        return self._action(step, name, pose, spec.grip_force, spec.approach_height_m)

    def _place(self, step: TaskStep) -> Action | None:
        name, destination = step.object, step.destination
        spec = self.cfg.objects[name]
        self._require_holding(step, name)
        margin, margin_reason = self._edge_margin(name)
        rejected: list[str] = []
        chosen: tuple[int, float, float] | None = None
        for index, (x, y) in enumerate(self.cfg.slots[destination][name]):
            blocker = self._slot_blocker(name, x, y, spec.footprint_radius_m, margin, margin_reason)
            if blocker is None:
                chosen = (index, x, y)
                break
            rejected.append(f"slot {index} ({x:.3f}, {y:.3f}): {blocker}")
        if chosen is None:
            raise PreconditionError(
                f"step {step.id} place: no free {destination!r} slot for {name!r}", step_id=step.id, issues=rejected
            )
        index, x, y = chosen
        self._release(step.arm)
        self.locations[name] = ((x, y, self.cfg.surface_z), "planned")
        skipped = f" (skipped {'; '.join(rejected)})" if rejected else ""
        provisional = " [provisional: after a step that needs re-observation]" if self.blocked is not None else ""
        self.notes.append(f"step {step.id}: {name} -> {destination} slot {index} at ({x:.3f}, {y:.3f}){skipped}{provisional}")
        target = _offset((x, y, self.cfg.surface_z + self.cfg.release_height_m), spec.grasp_offset_m)
        self._check_reach(step, target)
        return self._action(step, name, target, spec.grip_force, spec.approach_height_m)

    def _pour(self, step: TaskStep) -> Action | None:
        source, into = step.source, step.into
        if self.holding[step.arm] != source:
            raise PreconditionError(
                f"step {step.id} pour: arm {step.arm} must already hold {source!r} (predicted: {self._describe(step.arm)}). "
                f"Add a pick step for {source!r} with arm {step.arm} before step {step.id}. The planner does not insert "
                "steps: a new step ID would break the one-Action-per-TaskStep contract.",
                step_id=step.id,
            )
        self._warn_if_undeclared(step, self.hold_origin[step.arm][1], f"arm {step.arm} holding {source!r}")

        holder = self._holder_of(into)
        if holder == step.arm:
            raise PreconditionError(f"step {step.id} pour: arm {step.arm} cannot pour into {into!r} while holding it", step_id=step.id)
        if holder is not None:
            hold = step.requires_hold
            if hold is None or hold.object != into:
                self.warnings.append(
                    f"step {step.id} pours into {into!r} held by arm {holder} without declaring requires_hold for it"
                )
            origin, by_step = self.hold_origin[holder]
            if origin == "planned":
                self._block(
                    step,
                    f"{into!r} will be held by arm {holder} after step {by_step}; where that arm holds it cannot be "
                    "predicted from SceneState.",
                )
                return None
            xyz = self.scene.objects.get(into)
            if xyz is None:
                raise SceneError(
                    f"step {step.id} pour: {into!r} is held by arm {holder} but is not in the observed scene, "
                    "so the pour point is unknown",
                    step_id=step.id,
                )
        else:
            located = self.locations.get(into)
            if located is None:
                raise SceneError(f"step {step.id} pour: receiver {into!r} is not in the observed scene", step_id=step.id)
            xyz, where = located
            if self._drawer_containing(xyz) is not None:
                raise PreconditionError(f"step {step.id} pour: receiver {into!r} is inside a drawer", step_id=step.id)
            if where == "planned":
                self.notes.append(f"step {step.id}: {into!r} position is the predicted placement from an earlier step, not an observation")

        source_spec = self.cfg.objects[source]
        rim = xyz[2] + self.cfg.objects[into].height_m
        target = (xyz[0], xyz[1], rim + self.cfg.pour.clearance_above_rim_m)
        self._check_reach(step, target)
        return self._action(step, source, target, source_spec.grip_force, source_spec.approach_height_m)

    # ---- state checks

    def _check_hold_requirement(self, step: TaskStep) -> None:
        hold = step.requires_hold
        if hold is None:
            return
        holder = self._holder_of(hold.object)
        if holder != hold.arm:
            current = f"arm {holder}" if holder else "nobody"
            raise PreconditionError(
                f"step {step.id} requires arm {hold.arm} to hold {hold.object!r} (predicted holder: {current}); "
                f"add a pick step for {hold.object!r} with arm {hold.arm} before step {step.id}",
                step_id=step.id,
            )
        self._warn_if_undeclared(step, self.hold_origin[holder][1], f"arm {holder} holding {hold.object!r}")

    def _drawer_allows_pose(self, step: TaskStep, name: str, drawer: str) -> bool:
        state, source, by_step = self.drawers.get(drawer, ("unobserved", "observed", None))
        if state != "open":
            raise PreconditionError(
                f"step {step.id} pick: {name!r} is inside {drawer!r}, which is {state}; "
                f"add an open_drawer step for {drawer!r} before step {step.id}",
                step_id=step.id,
            )
        self._warn_if_undeclared(step, by_step, f"{drawer!r} being opened")
        if source == "planned":
            self._block(
                step,
                f"{name!r} is inside {drawer!r}, which step {by_step} opens; its position will change when the drawer moves.",
            )
            return False
        return True

    def _require_free(self, step: TaskStep) -> None:
        held = self.holding[step.arm]
        if held is not None:
            raise PreconditionError(
                f"{_label(step)}: arm {step.arm} is still holding {held!r} (predicted); place it before step {step.id}",
                step_id=step.id,
            )

    def _require_not_held(self, step: TaskStep, name: str) -> None:
        holder = self._holder_of(name)
        if holder is not None:
            raise PreconditionError(f"{_label(step)}: {name!r} is already held by arm {holder} (predicted)", step_id=step.id)

    def _require_holding(self, step: TaskStep, name: str) -> None:
        if self.holding[step.arm] != name:
            raise PreconditionError(
                f"{_label(step)}: arm {step.arm} is not holding {name!r} (predicted: {self._describe(step.arm)}); "
                f"add a pick step for {name!r} with arm {step.arm} before step {step.id}",
                step_id=step.id,
            )
        self._warn_if_undeclared(step, self.hold_origin[step.arm][1], f"arm {step.arm} picking {name!r}")

    def _warn_if_undeclared(self, step: TaskStep, by_step: int | None, what: str) -> None:
        if by_step is None or by_step in self.completed or by_step in self.ancestors[step.id]:
            return
        self.warnings.append(
            f"step {step.id} relies on step {by_step} ({what}) but does not list it in depends_on; "
            "this is only safe for a strictly sequential executor"
        )

    def _slot_blocker(
        self, name: str, x: float, y: float, radius: float, margin: float, margin_reason: str
    ) -> str | None:
        cfg = self.cfg
        if not circle_within_rect(x, y, radius, cfg.table, margin):
            return f"footprint (radius {radius:.3f} m) is closer than {margin:.3f} m to the table edge ({margin_reason})"
        for zone_name in sorted(cfg.keepout_zones):
            if circle_rect_gap(x, y, radius, cfg.keepout_zones[zone_name]) < cfg.place_clearance_m:
                return f"within {cfg.place_clearance_m:.3f} m of keep-out zone {zone_name!r}"
        for other in sorted(self.locations):
            if other == name or other not in cfg.objects:
                continue
            (ox, oy, oz), source = self.locations[other]
            if not self._on_table((ox, oy, oz)):
                continue
            gap = circle_circle_gap(x, y, radius, ox, oy, cfg.objects[other].footprint_radius_m)
            if gap < cfg.place_clearance_m:
                return f"within {cfg.place_clearance_m:.3f} m of {other} ({source}) at ({ox:.3f}, {oy:.3f})"
        return None

    def _edge_margin(self, name: str) -> tuple[float, str]:
        margin, reason = self.cfg.edge_margin_m, "table.edge_margin_m"
        for constraint in self.task.constraints:
            spec = self.cfg.constraints[constraint]
            if name in spec.applies_to and spec.edge_margin_m > margin:
                margin, reason = spec.edge_margin_m, f"constraint {constraint}"
        return margin, reason

    def _check_reach(self, step: TaskStep, target: Point3) -> None:
        arm = self.cfg.arms[step.arm]
        distance = math.hypot(target[0] - arm.base_xy[0], target[1] - arm.base_xy[1])
        if distance > arm.max_reach_m:
            self.warnings.append(
                f"step {step.id}: target ({target[0]:.3f}, {target[1]:.3f}) is {distance:.3f} m from arm {step.arm}'s "
                f"base, beyond the {arm.max_reach_m:.2f} m reach heuristic; Stage 4 inverse kinematics may fail"
            )

    # ---- small helpers

    def _block(self, step: TaskStep, reason: str) -> None:
        if self.blocked is None:
            self.blocked = (step.id, reason + " Execute the actions before it, re-observe, and re-plan.")

    def _grab(self, arm: str, name: str, origin: str, step_id: int) -> None:
        self.holding[arm] = name
        self.hold_origin[arm] = (origin, step_id)
        self.locations.pop(name, None)

    def _release(self, arm: str) -> None:
        self.holding[arm] = None
        self.hold_origin.pop(arm, None)

    def _holder_of(self, name: str) -> str | None:
        return next((arm for arm, held in sorted(self.holding.items()) if held == name), None)

    def _describe(self, arm: str) -> str:
        held = self.holding[arm]
        return f"holding {held!r}" if held else "empty"

    def _drawer_containing(self, xyz: Point3) -> str | None:
        for name in sorted(self.cfg.drawers):
            spec = self.cfg.drawers[name]
            if spec.footprint.contains(xyz[0], xyz[1]) and xyz[2] < spec.interior_top_z:
                return name
        return None

    def _on_table(self, xyz: Point3) -> bool:
        cfg = self.cfg
        return (
            cfg.table.contains(xyz[0], xyz[1])
            and abs(xyz[2] - cfg.surface_z) <= cfg.on_table_tolerance_m
            and self._drawer_containing(xyz) is None
        )

    @staticmethod
    def _action(step: TaskStep, obj: str, target: Point3, grip_force: float, approach_height: float) -> Action:
        return Action(
            step_id=step.id,
            action=step.action,
            arm=step.arm,
            object=obj,
            target_pose=_rounded(target),
            grip_force=grip_force,
            approach_height=approach_height,
        )


def _ancestors(ordered: Sequence[TaskStep]) -> dict[int, frozenset[int]]:
    result: dict[int, frozenset[int]] = {}
    for step in ordered:
        found: set[int] = set()
        for dep in step.depends_on:
            found.add(dep)
            found |= result[dep]
        result[step.id] = frozenset(found)
    return result


def _offset(point: Sequence[float], offset: Sequence[float]) -> Point3:
    return (point[0] + offset[0], point[1] + offset[1], point[2] + offset[2])


def _rounded(point: Sequence[float]) -> Point3:
    # 0.1 mm resolution removes float noise such as 0.06 - 0.05 = 0.009999999; "+ 0.0" turns -0.0 into 0.0
    return (round(point[0], 4) + 0.0, round(point[1], 4) + 0.0, round(point[2], 4) + 0.0)
