"""Stage 3 policy package.

plan_detailed(task, scene, completed_step_ids=...) -> PlanResult is the
pipeline's contract call for the staged observe-act loop; plan(task, scene)
stays as the one-shot convenience and raises when only a prefix is plannable.
PlanningError is exported so callers can report planning failures without
catching unrelated exceptions.
"""

from stage3_policy.errors import PlanningError
from stage3_policy.planner import PlanResult, plan_detailed
from stage3_policy.policy import plan

__all__ = ["plan", "plan_detailed", "PlanResult", "PlanningError"]
