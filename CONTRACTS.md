# Integration Contracts

Every stage lives in its own folder and exposes **exactly one public entrypoint**
(stage 1 also has an audio variant). All types come from [`common/types.py`](common/types.py) —
they are pydantic models, so violations fail loudly at the boundary.

**Do not change a signature or a model in this table without agreeing it here first**
(PR that updates both the code and this file).

| Stage | Subsystem | Function signature | Input | Output | Folder |
|-------|-----------|--------------------|-------|--------|--------|
| 1. Voice → Task | Voice Understanding | `parse_text(text: str) -> Task` | natural-language command | `Task` | `stage1_voice/` |
| 1b. Audio → Task | Speech Processing | `parse_command(audio_path: str) -> Task` | path to audio file | `Task` | `stage1_voice/` |
| 2. Perception | Scene State Observation | `perceive(image=None) -> SceneState` | camera frame (or None in sim) | `SceneState` | `stage2_perception/` |
| 3. Policy / planning | Task Planning & Policy | `plan_detailed(task: Task, scene: SceneState, *, completed_step_ids=()) -> PlanResult` — the pipeline's contract call (staged observe–act); `plan(task, scene) -> list[Action]` stays as the one-shot convenience | `Task`, `SceneState`, ids of already-executed steps | `PlanResult` (`actions`, `complete`, `pending_step_ids`, `blocked_reason`, …); both raise `PlanningError` if the task cannot be planned | `stage3_policy/` |
| 4. Bimanual execution | Dual-Arm Execution | `execute(actions: list[Action], sim=None) -> ExecutionResult` | `list[Action]`, MuJoCo sim handle | `ExecutionResult` | `stage4_bimanual/` |
| 4a. Scene reset | Scene Reset | `reset_scene(seed: int) -> sim` | seed; randomization ranges read from `configs/default.yaml` | MuJoCo sim handle | `stage4_bimanual/` |
| 4b. Camera | Camera Rendering | `get_camera_frame(sim) -> image` | sim handle | camera frame (`np.ndarray` in the real implementation) | `stage4_bimanual/` |
| 5. OpenVINO benchmark | Model Optimization | `run_benchmark() -> BenchmarkResult` | — (reads model from `configs/default.yaml`) | `BenchmarkResult`: model_name / device / precision / latency_ms / throughput | `stage5_openvino/` |
| 6. Verify / recover | Verification & Replanning | `verify(scene_after: SceneState, task: Task) -> VerifyResult` | post-execution `SceneState`, original `Task` | `VerifyResult` | `stage6_verify/` |
| 7. Evaluation | Robustness Evaluation | `evaluate(seeds: list[int]) -> EvalReport` | randomization seeds | `EvalReport` | `stage7_eval/` |
| 8. Integration | Pipeline Orchestration | `common/pipeline.py::run_once(command, seed=0, max_retries=None) -> RunResult` (single-run entrypoint) + `scripts/*.py` CLIs | command, seed | `RunResult` | `common/`, `scripts/`, repo root |

## Ground rules

- Import types with `from common.types import Task, SceneState, ...` — never redefine them locally.
- `SceneState` coordinates are the MuJoCo world frame of `assets/bimanual_scene.xml`,
  metres, table top at z=0.70. Every stage that produces or consumes positions
  (perception, planning, execution, verify) uses this frame — no local frames.
- Each stage's public function is re-exported from its package `__init__.py`,
  so callers write `from stage3_policy import plan`.
- Keep heavy imports (mujoco, lerobot, openvino, opencv, speechmatics, anthropic)
  **out of module top level** until your real implementation lands, or guard them —
  the stub pipeline must run with only `pydantic` + `pyyaml` + `numpy` installed.
- Data flow: `reset_scene(seed)` → `parse_text` → `perceive(get_camera_frame(sim))` →
  staged observe–act loop: `plan_detailed(task, scene, completed_step_ids=...)` →
  `execute` → add the contiguous successful prefix (by `action_results`) to the
  completed set → `perceive(get_camera_frame(sim))`, repeated until the plan is
  `complete` or execution fails → `verify` (stage 5 is standalone).
- Recovery: if `verify` returns `replan=True`, the pipeline re-runs the staged
  loop → `verify` — keeping the completed step ids and planning from the fresh
  observation — up to `max_retries` (from `configs/default.yaml`), then reports FAIL.
- The single-run loop is implemented once in `common/pipeline.py::run_once` —
  stage 7 must call `run_once` rather than re-implementing it.
