# stage3_policy — Policy / task reasoning

Turns a parsed `Task` plus an observed `SceneState` into an ordered `list[Action]`
for the two SO-101 arms. It has two layers:

| Layer | What it is | Status |
|---|---|---|
| **Rule-based planner** (`plan`) | Deterministic checks and pose resolution. No model, GPU, simulator or API key. | Implemented and unit-tested on synthetic inputs. **Not validated on the robot:** Stage 4 does not use its poses yet. |
| **Learned motor policy** (`learned/`) | LeRobot ACT: camera + joint state → joint targets, one policy per skill. | Schema, dataset validator, training wrapper and inference adapter written. **No dataset, no training, no checkpoint.** |

Nothing in this folder proves the robot can set the table. Passing tests show the
planner's logic on synthetic inputs, and nothing more.

---

## 1. How the planner works

```
Task (Stage 1) ─┐
                ├─> validate task ─> check support ─> validate scene ─> order steps ─> walk with predicted state ─> list[Action]
SceneState (2) ─┘         │                │                 │                                  │
                  TaskValidationError  UnsupportedError  SceneError          PreconditionError / ObservationRequired
```

1. **Validate the task.** Checks that:
   - step IDs are unique and dependencies exist, with no self-dependency and no cycle;
   - each action has its required fields and no unused ones (a `pick` with a `destination` is refused, not ignored);
   - names are configured, and `requires_hold` uses the *other* arm.
2. **Check support.** Rejects `handoff`, `close_drawer`, spoon/fork picks and placements, and any constraint not in the config. It never silently ignores them.
3. **Validate the scene.** Every coordinate must be finite. Every object the task uses must lie inside the MuJoCo-world workspace, otherwise it is a frame mismatch.
4. **Order the steps.** Dependencies come first. Steps that are ready at the same time keep the order they were listed in. Any reordering is reported as a warning.
5. **Walk the steps** while tracking a **predicted** state: what each arm holds, which drawers are open, and where placed objects end up.
   - This is bookkeeping that assumes each earlier action succeeds. It is not a measurement and not a physics simulation.
   - When a pose depends on something the camera cannot see yet, Actions stop at that step, but the remaining steps are still checked. An impossible task therefore fails *before* anything moves.

Missing information is never replaced by `(0, 0, 0)`. Missing prerequisites are **errors**: the planner does not insert steps, because a new step ID would break the one-Action-per-TaskStep convention.

### Public API

```python
from stage3_policy import plan, plan_detailed, PlanResult, PlanningError

result = plan_detailed(task, scene, completed_step_ids={1})   # PlanResult; the pipeline's contract call
actions = plan(task, scene)        # one-shot convenience: complete list[Action] or raises
```

| Exception | Meaning | Example |
|---|---|---|
| `TaskValidationError` | malformed task | duplicate IDs, cycle, `place` without `destination` |
| `UnsupportedError` | the contracts cannot express it | `handoff`, `close_drawer`, `pick spoon`, constraint `"pour_slowly"` |
| `SceneError` | the observation is missing or invalid | object not detected, drawer state absent, NaN, wrong frame |
| `PreconditionError` | impossible sequence | arm busy, placing an unheld object, pouring without picking the bottle, object inside a closed drawer, no free slot |
| `ObservationRequired` | only a prefix can be planned now | pick from a drawer opened earlier in the same plan; the prefix is attached as `.executable_actions` |

**Staged planning** is the shared contract since the `staged-planning` integration:
`common/pipeline.py::run_once` drives `plan_detailed(task, scene, completed_step_ids=...)`
in an observe–act loop (CONTRACT_PROPOSAL.md P3, now in `CONTRACTS.md`). It returns a
`PlanResult` with `actions`, `complete`, `pending_step_ids`, `blocked_reason`, `notes`,
`warnings` and `predicted_holdings`.

SceneState cannot say which steps already ran or what an arm holds, so the caller passes `completed_step_ids` from its execution log. Arm holdings are then *derived* from those steps, and positions come from the fresh observation. Without that list, a retry plans every step from scratch. For example, re-opening a drawer that is now observed open is refused, because an open drawer's handle pose is not in SceneState.

### Coordinate frame and units

- **Frame `mujoco_world`:** metres, origin on the floor under the table centre, +z up, table top at z = 0.70.
- **Arm bases:** A at (−0.20, −0.22), B at (−0.20, +0.22); both reach toward +x.
- **Object positions:** `SceneState.objects[name]` is assumed to be the object's base centre (its MuJoCo body origin).
- **`Action.target_pose`** is the gripper pinch point at grasp or release:
  - PICK: observed base + `grasp_offset_m`.
  - PLACE: slot centre at `surface_z + release_height_m`, + `grasp_offset_m`.
  - OPEN_DRAWER: the configured closed-drawer handle point.
  - POUR: receiver base + receiver height + `clearance_above_rim_m`.
- **`grip_force`:** normalized 0..1. **`approach_height`:** metres. Both are heuristics (§2).
- **`Action.object`:** holds the drawer name for `open_drawer` and the *source* for `pour`. This keeps the old stub's convention. See `CONTRACT_PROPOSAL.md` for explicit fields.

### Supported behaviour

| Action | Needs | Planner output |
|---|---|---|
| `open_drawer` | drawer observed `closed`, arm empty | handle point from config (static fixture, valid only while closed) |
| `pick` plate / mug / water_bottle | object observed, arm empty, not held by the other arm; if inside a drawer, that drawer observed open | observed base + grasp offset |
| `place` plate / mug / water_bottle on `table` | arm holds the object (predicted) | first configured slot that passes the edge-margin, keep-out and clearance checks — never the object's current position |
| `pour` water_bottle → mug | pouring arm already holds the bottle via an explicit earlier `pick`; the receiver is on the table, or held by the other arm (then `requires_hold` is checked) | pour point above the receiver; if it is held after a pick planned in the same call, `ObservationRequired` |
| constraint `keep_glasses_away_from_edge` | — | raises the edge margin to 0.15 m for mug/water_bottle placements, applied to the footprint radius, not the centre |

Placement checks are static footprint checks (circles vs rectangles). They are **not** collision-free motion planning. Reach is only a warning heuristic; Stage 4's inverse kinematics decides what is reachable.

---

## 2. Configuration — `planner_config.yaml`

Everything geometric or tunable lives in one file:
- frame and workspace, table bounds and margins;
- keep-out zones (the drawer's swept area, arm bases);
- per-object footprint, height, grasp offset, `grip_force`, `approach_height_m`;
- drawer footprint and handle point;
- destination slots, pour pairs, and supported constraints.

- **Geometry** comes from `assets/bimanual_scene.xml` and the arm XMLs. `tests/test_config.py` re-derives it from the XML, so if the drawer or table is moved the test fails instead of the planner silently using old numbers.
- **Grip forces, approach heights, margins, slots and reach** are **initial tunable heuristics**. They are not measured or safety-validated, and object masses are deliberately not used.
- **Spoon/fork** have grasp settings but `requires_orientation: true`. SceneState has no yaw, so their picks and placements are refused until the contract adds it.

---

## 3. Setup and commands (Windows PowerShell, from the repo root)

The minimal environment is Python 3.14.5 with pydantic 2.13.5, PyYAML 6.0.3 and pytest 9.1.1:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r stage3_policy\requirements.txt
```

Calling `.\.venv\Scripts\python.exe` directly avoids needing to activate the venv (activation scripts are often blocked by PowerShell's execution policy). On macOS/Linux use `python3 -m venv .venv` and `.venv/bin/python`.

| Command | What it shows |
|---|---|
| `.\.venv\Scripts\python.exe -m pytest -q stage3_policy common\tests` | Stage 3 unit tests plus the pipeline unit-integration tests |
| `.\.venv\Scripts\python.exe -m pytest -q` | the whole repo, including Stage 1's tests (3 live-API tests skip without `ANTHROPIC_API_KEY`) |
| `.\.venv\Scripts\python.exe -m stage3_policy` | readable planner decisions for 4 **synthetic** examples; exit 0 = each behaved as documented |
| `.\.venv\Scripts\python.exe -m stage3_policy --example missing` | only the missing-object refusal |
| `.\.venv\Scripts\python.exe -m stage3_policy.learned.inference` | which motor-policy mode would run, and why |
| `.\.venv\Scripts\python.exe -m stage3_policy.learned.train --dataset-root data\butler_demos\pick` | dry run: the `lerobot-train` command plus preflight problems |

Example output (Example 1 of the demo, **synthetic inputs**, nothing executed):

```
  step  action       arm  object         target_pose x, y, z (m)       grip  approach (m)
  1     pick         A    plate          (+0.030, -0.220, +0.745)    0.60  0.08
  2     place        A    plate          (-0.090, +0.020, +0.730)    0.60  0.08
  3     pick         B    mug            (+0.060, +0.180, +0.748)    0.50  0.10
  4     place        B    mug            (+0.140, +0.180, +0.753)    0.50  0.10
  5     pick         B    water_bottle   (+0.120, -0.040, +0.780)    0.55  0.12
  6     pour         B    water_bottle   (+0.140, +0.180, +0.856)    0.55  0.12
  note: step 2: plate -> table slot 1 at (-0.040, 0.020) (skipped slot 0 (0.060, 0.050): within 0.020 m of mug (observed) at (0.060, 0.180))
```

### What the tests prove, and what they do not

| File | Proves (on synthetic inputs) |
|---|---|
| `tests/test_planner_actions.py` | poses follow observations; PLACE uses slots; per-object grasp settings; drawer prefix and staged re-planning; pour with held receiver |
| `tests/test_planner_refusals.py` | IDs, dependencies, cycles, fields, unsupported actions and constraints, missing detections, NaN, wrong frame, closed drawers, arm occupancy, pour prerequisites, edge margins; Stage 1's real stub Task is refused |
| `tests/test_planner_properties.py` | determinism, no input mutation, undeclared-dependency warning, the demo runs |
| `tests/test_config.py` | config matches the MuJoCo XML and the shared vocabulary; bad config values are rejected |
| `tests/test_learned.py` | demo schema matches the MuJoCo actuators and timestep; dataset validator; fallback selection; training refuses to start without prerequisites |
| `common/tests/test_pipeline_recovery.py` | **unit-integration with doubles:** retries use the new observation; failed execution never becomes success; planning refusals report FAIL |

They do **not** show that Stage 4 moves the arms to these poses, that grasps hold, that anything is collision-free, or that the table ends up set.

---

## 4. Integration status (what `scripts/run_pipeline.py` does now)

- **Before this branch:** `RESULT: SUCCESS`. Every stage was a placeholder: plan copied steps with `(0,0,0)` fallbacks, execute marked everything successful, and verify always passed.
- **Minimal venv:** `RESULT: FAIL`, exit 1. Stage 2's `perceive(None)` returns table-corner coordinates (z ≈ 0.02), and Stage 3 refuses them as a frame mismatch. This is the honest state; merging it means `main` no longer "passes" the stub pipeline.
- **With MuJoCo but no OpenCV:** the pipeline crashes inside Stage 2 (`cv2.findHomography` on `None`) before Stage 3 runs.
- **With OpenCV:** Stage 2's README says the MuJoCo scene has no ArUco markers, so objects would be missing and Stage 3 would raise `SceneError`. Not run here.
- Even with the Stage 1 stub fixed, Stage 3 refuses it: step 5 pours with arm A without ever picking the bottle.

**Pipeline correction** (`common/pipeline.py`; a separate, backward-compatible change):
- retries re-plan from `scene_after`, not the original scene;
- a run succeeds only if `execution.success` **and** `verify.ok`;
- a `PlanningError` is logged as FAIL instead of crashing.

---

## 5. Learned policy (LeRobot ACT)

**Choice: ACT, not SmolVLA.**
- **Hardware:** Intel Core Ultra 7 270K Plus (24 cores), 31.7 GB RAM, AMD Radeon RX 9060 XT (no CUDA), ~634 GB free disk.
- **Size:** ACT is ~80M parameters (LeRobot docs); SmolVLA is ~450M and fine-tunes on a VLM backbone, which is impractical without a supported GPU.
- **Export:** ACT (ResNet-18 + transformer) is a more realistic OpenVINO export target.
- **Language:** ACT is **not language-conditioned**. Language stays in Stage 1 (Claude parse) and the Stage 3 symbolic plan, and one ACT policy per skill does low-level control. That is a hierarchical system, **not a VLA**, and an LLM producing symbolic steps is not a trained motor policy.
- **APIs:** checked against **lerobot 0.6.1** source and docs.

**Status:** not started. No demonstrations exist, lerobot is not installed, and there is no checkpoint. `configs/default.yaml` names `assets/models/policy.safetensors`, which does not exist. The inference CLI reports `scripted_primitives` with that reason.

### Proposed demonstration schema (`learned/schema.py`)

| Item | Proposal |
|---|---|
| Episode | one skill execution (e.g. "pick up the mug with arm B") from its start state to success or timeout; one LeRobot episode |
| Frame rate | **25 fps** = one frame every 20 physics steps (timestep 0.002 s); `timestamp = frame_index / 25` from 0 each episode |
| `observation.state` | float32[12] joint positions (qpos, rad), order `a_shoulder_pan, a_shoulder_lift, a_elbow_flex, a_wrist_flex, a_wrist_roll, a_gripper, b_…` (MuJoCo actuator order) |
| `action` | float32[12] absolute position targets written to `MjData.ctrl` at that frame, same order and units |
| `observation.images.overhead` | uint8 480×640×3 from `overhead_cam`, rendered at the same frame |
| `task` | the skill instruction string, e.g. `"pick up the mug with arm B"` |
| `next.done` / `next.success` | bool, true on the final frame; success from a simulator ground-truth check |
| `meta/butler_episodes.json` | per episode: `episode_index, seed, skill, arm, object, instruction, split (train/val), success, physics_only` |
| Seeds | never 0–9 (the evaluation seeds); train and val seeds disjoint |
| Physics | `physics_only: true`: objects must move only because the arm moved them. The recording/visualization scripts on `origin/stage4-bimanual` write object `qpos` directly (teleporting), which cannot teach a policy. Weld-based grasps must be triggered by a rule a policy could also trigger (e.g. gripper closed near the object), not by the script's phase. |

Data location: `data/butler_demos/<skill>/` (a LeRobotDataset v3.0 directory per skill). Training outputs go to `outputs/train/<job>/`. Consider adding both to `.gitignore`.

### Learned-policy commands (separate environment — **not run yet; large download**)

```powershell
# Needs Python 3.12 (lerobot 0.6.1 requires >= 3.12; docs use 3.12). Downloads PyTorch.
py -3.12 -m venv .venv-learned
.\.venv-learned\Scripts\python.exe -m pip install -r stage3_policy\requirements-learned.txt
.\.venv-learned\Scripts\python.exe -m stage3_policy.learned.validate_dataset --root data\butler_demos\pick
.\.venv-learned\Scripts\python.exe -m stage3_policy.learned.train --dataset-root data\butler_demos\pick --job-name act_pick_smoke          # dry run
.\.venv-learned\Scripts\python.exe -m stage3_policy.learned.train --dataset-root data\butler_demos\pick --job-name act_pick_smoke --run    # 200 CPU steps
```

- A 200-step smoke run proves the plumbing, not policy quality.
- Quality evidence means evaluating on held-out `val` seeds inside the simulator, which needs the executor hook (Contract proposal P5).
- `--run` refuses unless lerobot is installed and the dataset passes every check, including frame checks.
- An untrained or randomly initialised model is never a result.

---

## 6. Simulation viewer

MuJoCo 3.13 is installed in the *global* Python 3.14 (not in `.venv`). The scene loads: 23 bodies, 12 actuators, 1 camera. To look at the **static** scene:

```powershell
py -3.14 -m mujoco.viewer --mjcf=assets\bimanual_scene.xml
```

This shows the table, drawer, tableware and both arms settling under physics. It does **not** run the planner or any task. No viewer on `main` executes the task. `origin/stage4-bimanual` has `scripts/visualize_run.py`, but it replays scripted waypoints and teleports the plate and bottle, so it is not evidence of grasping.

---

## 7. Known limitations

- Stage 4 (`origin/stage4-bimanual`) ignores `target_pose`, `grip_force`, `approach_height` and `arm`, and marks unknown actions successful. The planner's numbers do not reach the robot yet.
- Frame mismatch with Stage 2, no yaw, no confidence, no handle pose for open drawers, drawers never "unknown".
- No handoff, no close_drawer, no utensils, no destinations other than `table`.
- One observation per `plan()` call. Pouring into a held mug and picking from a just-opened drawer need the staged loop, which `common/pipeline.py::run_once` now drives via `plan_detailed`.
- Slots are fixed candidates. Conservative utensil footprints can block the Stage 4 plate target (slot 0), in which case slot 1 is used.
- Reach is a heuristic warning; there is no inverse kinematics or collision checking.

---

## 8. Handoff

**Muhammad (Stage 4 execution + demonstrations)**
1. Make `execute` use `Action.target_pose`/`arm`/`grip_force`/`approach_height` (or agree P2). Return `False` for unsupported actions instead of `sim.step(50)` + success.
2. Report `action_results` from real success checks (grasp held, drawer opened > threshold, object on slot).
3. Record demonstrations in the schema above with `physics_only: true`, seeds ≥ 100, and train/val splits. Validate them with `validate_dataset`.
4. Agree the motor-policy hook (P5) so an ACT checkpoint can be evaluated in the simulator.

**Perception Subsystem**
1. Report MuJoCo world coordinates (base centre, z ≈ 0.70 on the table), or agree a `frame` field (P1).
2. Cover spoon/fork; pass through confidence, warnings, drawer handle pose and yaw. Report "unobserved" instead of `closed` when no drawer marker is seen.
3. Make `perceive(image)` work in the MuJoCo pipeline (OpenCV dependency, markers or a markerless detector).

**Stage 1 & Pipeline Integration**
1. ~~Add `pick water_bottle, arm A` before the pour in the stub/few-shot example (P4).~~ Done (`integration-day4`).
2. Review the `common/pipeline.py` correction and decide how to treat the now-honest `FAIL` on `main`.
3. ~~Wire the staged observe–act loop with `plan_detailed(..., completed_step_ids=...)` (P3). Catch `stage3_policy.PlanningError`.~~ Done (`staged-planning`).
4. Take `CONTRACT_PROPOSAL.md` through a `CONTRACTS.md` update (P3/P4 rows updated on `staged-planning`; P1/P2/P5 still open).

**OpenVINO Model Optimization**
There is no model to export yet. Once a checkpoint exists, it is a LeRobot ACT directory (`config.json`, `model.safetensors`, pre/post-processor configs):
- **Inputs:** `observation.state` [1, 12] float32 and `observation.images.overhead` [1, 3, 480, 640] float32 in 0..1, normalised by the saved preprocessor.
- **Output:** [1, 12] joint targets (rad).
- **Before benchmarking:** confirm `chunk_size`/`n_action_steps`, and decide whether normalisation lives inside the exported graph.

**Verification & Evaluation**
Real verification needs P1 detail (held objects, unknown drawers). Evaluate learned policies only on `val` or evaluation seeds never used in demonstrations.

Proposed contract changes: [`CONTRACT_PROPOSAL.md`](CONTRACT_PROPOSAL.md).
