# Stage 3 contract proposal — NOT APPROVED

**Status:** Design draft for discussion. None of this is in
`common/types.py` or `CONTRACTS.md`. **No stage should rely on these fields until
both files are formally updated.** Until then Stage 3 supports
only the subset listed under "What Stage 3 does today", and rejects the rest
with an explicit error.

## Why the current contracts are not enough

| Gap | Evidence in the repo (commit 3796322 + `origin/stage4-bimanual`) | Effect today |
|---|---|---|
| **Coordinate frame is unstated** | `SceneState` says only "tabletop coordinates (meters)". Stage 2 maps images to a table-corner frame (0..0.8 m × 0..0.6 m, z = 0), and `perceive(None)` returns z ≈ 0.02. Stage 4 reports MuJoCo world coordinates (origin under the table, top at z = 0.70). | The planner rejects any used object outside the MuJoCo-world workspace (`SceneError`). |
| **No orientation** | No yaw in `SceneState` or `Action`. Spoon/fork handles are 1.2 cm wide. | Spoon/fork PICK/PLACE rejected (`UnsupportedError`). |
| **Partial observability is lost** | Stage 2 computes confidence, warnings and drawer handle pose internally; the public adapter drops them. A marker-free image yields `"closed"`. Drawers only allow `"open"`/`"closed"`. | Planner cannot tell "not seen" from "closed". It cannot target the handle of an open drawer, so CLOSE_DRAWER is rejected. |
| **Held objects / completed steps** | Nothing in `SceneState` says what an arm holds or which steps finished. | Retries re-plan every step from scratch. Staged planning needs `completed_step_ids` from the caller. |
| **One pose per Action** | Pour needs the source (held) and the receiver; handoff needs giver, receiver and exchange pose. Stage 3 currently stores the drawer name in `Action.object` (the stub's convention). | Receiver name is not carried; HANDOFF rejected. |
| **Executor ignores the plan's numbers** | `origin/stage4-bimanual` dispatches on `(action, object)` and reads object poses directly from MuJoCo. `target_pose`, `grip_force` and `approach_height` are unused. Arms are hard-coded per primitive. Unknown actions do `sim.step(50)` and report success. | Planner output is not what moves the robot. Success flags are not evidence. |
| **Held receiver pose is unpredictable** | After `PICK mug` Stage 4 moves the mug to an internal hold pose. | POUR into a held mug needs a fresh observation (staged planning). |
| **No motor-policy hook** | Stage 4 primitives are scripted waypoints. | A trained ACT policy has nowhere to plug in. |

## Proposed changes (smallest first)

### P1. `SceneState`: frame + optional detail (backward compatible)

```python
class SceneState(BaseModel):
    objects: dict[str, tuple[float, float, float]]      # unchanged: base centre of each object
    drawers: dict[str, Literal["open", "closed"]]        # unchanged
    frame: Literal["mujoco_world"] = "mujoco_world"      # NEW: metres, origin under table centre, +z up
    yaw: dict[str, float] = {}                           # NEW: radians about +z, only for objects where known
    confidence: dict[str, float] = {}                    # NEW: 0..1 per object, if the detector has one
    drawer_handles: dict[str, tuple[float, float, float]] = {}  # NEW: observed handle point, any drawer state
    unobserved_drawers: list[str] = []                   # NEW: drawers perception could not see (instead of "closed")
    warnings: list[str] = []                             # NEW: perception warnings passed through
```

- **Produce:** Stage 2 `perceive`, Stage 4 `ExecutionResult.final_scene`.
- **Consume:** Stage 3 `plan`, Stage 6 `verify`, `common/pipeline.py` logs.
- **Stage 3 would then:** accept `yaw` for utensils, support CLOSE_DRAWER when `drawer_handles` is present, and treat `unobserved_drawers` as unknown.

### P2. `Action`: carry the semantics the executor needs (backward compatible)

```python
class Action(BaseModel):
    ...                                   # existing fields unchanged
    target: str | None = None             # NEW: drawer name for open/close (stop overloading `object`)
    into: str | None = None               # NEW: pour receiver
    support_arm: Arm | None = None        # NEW: arm that must keep holding (TaskStep.requires_hold.arm)
    receiver_arm: Arm | None = None       # NEW: handoff receiver; `arm` is the giver
    target_yaw: float | None = None       # NEW: radians about +z for the gripper at target_pose
```

Also define in `CONTRACTS.md`:
- `target_pose` is the gripper pinch point in the `frame` above, at the moment of grasp/release. For POUR it is the pour point above the receiver's rim.
- `grip_force` mapping, which Stage 4 must choose. Example: fraction of the gripper actuator `forcerange` (3.35 N·m in `so101_arm_*.xml`).
- The executor must use the Action's fields or fail. Unknown `(action, object)` pairs must return `False`, never `sim.step(50)` + success.

- **Produce:** Stage 3. **Consume:** Stage 4 `execute`, Stage 6 for diagnostics.

### P3. Staged observe–act loop (pipeline wiring; no model change)

Stage 3 already provides `stage3_policy.planner.plan_detailed(task, scene, completed_step_ids=...) -> PlanResult`
(`actions`, `complete`, `pending_step_ids`, `blocked_reason`, `warnings`). Proposed loop for `run_once`:

```python
completed: set[int] = set()
while True:
    result = plan_detailed(task, scene, completed_step_ids=completed)
    execution = execute(list(result.actions), sim)
    for action in result.actions:                 # only the contiguous successful prefix counts
        if not execution.action_results.get(action.step_id):
            break
        completed.add(action.step_id)
    scene = perceive(get_camera_frame(sim))       # fresh observation every stage
    if not execution.success or result.complete:
        break
verdict = verify(scene, task)                     # success requires execution.success and verdict.ok
```

- **Needs from Stage 4:** `action_results` must mean the primitive really succeeded. Today they are `True` unconditionally.
- **Location:** `common/pipeline.py`.
- Alternative to one re-observation: add `target_ref: Literal["world", "support_arm_gripper"]` to P2, so Stage 4 computes the pour point relative to the gripper holding the mug.

### P4. Stage 1 task semantics

- **POUR** requires an earlier explicit `pick` of `source` by the pouring arm. Please add `pick water_bottle, arm A` to the stub/few-shot example in `stage1_voice/voice.py`.
  - Today the example pours with an empty arm, and Stage 3 refuses it (`PreconditionError`, step 5).
  - Stage 3 will not insert the step itself: that would need a new step ID and break one-Action-per-TaskStep.
- **HANDOFF:** agree that `arm` is the giver and the receiver is the other arm, and add the exchange location (e.g. a named destination). This is required before Stage 3 supports it.

### P5. Executor motor-policy hook (needed before any learned policy runs)

```python
class MotorPolicy(Protocol):
    def reset(self) -> None: ...
    def act(self, joint_positions: Sequence[float],   # 12 values, a_* then b_*, rad (MuJoCo qpos)
            overhead_rgb: np.ndarray,                  # uint8 (480, 640, 3) from overhead_cam
            instruction: str) -> list[float]: ...      # 12 absolute position targets for MjData.ctrl, rad
```

- **Called by:** Stage 4, every 20 physics steps (25 Hz) during one skill, until that skill's success check passes or it times out.
- **Selection:** `stage3_policy.learned.inference.select_motor_policy()` returns `learned_act` or `scripted_primitives` plus a reason. Stage 4 logs it.
- **Same loop for recording:** it writes the demonstration frames described in `stage3_policy/README.md`.

## What Stage 3 does today (without any of the above)

- Coordinates must already be MuJoCo world; anything else is rejected.
- Supported actions:
  - OPEN_DRAWER: closed drawer, static handle pose from `planner_config.yaml`.
  - PICK and PLACE: plate, mug, water_bottle.
  - POUR: water_bottle → mug, source explicitly picked first. A held receiver needs a re-observation.
- Rejected with an explicit error: CLOSE_DRAWER, HANDOFF, spoon/fork PICK/PLACE, unknown constraints.
- `Action.object` holds the drawer name for OPEN_DRAWER and the source for POUR, preserving the previous stub convention.
