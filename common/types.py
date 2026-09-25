"""Shared data contracts for the whole pipeline.

Every stage imports from this module. These models ARE the integration
contract: do not change them without agreeing the change in CONTRACTS.md.
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Arm = Literal["A", "B"]


class ActionType(str, Enum):
    """Primitive actions the arms can perform."""

    OPEN_DRAWER = "open_drawer"
    CLOSE_DRAWER = "close_drawer"
    PICK = "pick"
    PLACE = "place"
    POUR = "pour"
    HANDOFF = "handoff"


class HoldRequirement(BaseModel):
    """Complementary dual-arm action: one arm holds while the other acts."""

    arm: Arm
    object: str


class TaskStep(BaseModel):
    """One step of a parsed task. Optional fields depend on the action type."""

    id: int
    action: ActionType
    arm: Arm
    target: str | None = None  # open_drawer/close_drawer: e.g. "top_drawer"
    object: str | None = None  # pick/place/handoff: e.g. "plate"
    destination: str | None = None  # place: e.g. "table"
    source: str | None = None  # pour: e.g. "water_bottle"
    into: str | None = None  # pour: e.g. "mug"
    depends_on: list[int] = Field(default_factory=list)  # step ids this waits on
    requires_hold: HoldRequirement | None = None


class Task(BaseModel):
    """Stage 1 output: structured task parsed from a natural-language command."""

    command: str
    steps: list[TaskStep]
    constraints: list[str] = Field(default_factory=list)


class SceneState(BaseModel):
    """MuJoCo world frame of assets/bimanual_scene.xml, metres, table top at z=0.70"""

    objects: dict[str, tuple[float, float, float]]  # name -> (x, y, z)
    drawers: dict[str, Literal["open", "closed"]]


class Action(BaseModel):
    """Stage 3 output: one executable primitive with grasp parameters."""

    step_id: int  # TaskStep.id this action realizes
    action: ActionType
    arm: Arm
    object: str | None = None
    target_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)  # placeholder pose
    grip_force: float = 1.0  # normalized 0..1
    approach_height: float = 0.10  # meters above target before descending


class ExecutionResult(BaseModel):
    """Stage 4 output: per-action outcome plus the resulting scene."""

    action_results: dict[int, bool]  # step_id -> success
    success: bool  # all actions succeeded
    final_scene: SceneState
    error: str | None = None


class VerifyResult(BaseModel):
    """Stage 6 output: does the scene after execution match the task intent."""

    ok: bool
    replan: bool = False
    details: str = ""


class SeedResult(BaseModel):
    """Outcome of one seeded pipeline run."""

    seed: int
    success: bool
    details: str = ""


class EvalReport(BaseModel):
    """Stage 7 output: aggregate results over randomization seeds."""

    seeds: list[int]
    successes: int
    success_rate: float
    results: list[SeedResult]


class BenchmarkResult(BaseModel):
    """Stage 5 output: OpenVINO benchmark numbers on Intel hardware."""

    # allow the field name "model_name" (pydantic reserves the "model_" prefix)
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    device: str
    precision: str
    latency_ms: float
    throughput: float


class RunResult(BaseModel):
    """Output of common.pipeline.run_once: one full pipeline run with recovery."""

    success: bool
    attempts: int
    task: Task
    log: list[str] = Field(default_factory=list)
