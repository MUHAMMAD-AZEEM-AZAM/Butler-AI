"""Stage 3 public entrypoint: plan(task, scene) -> list[Action].

Always the deterministic rule-based planner in stage3_policy/planner.py. It
needs no model, GPU, simulator or API key. The learned ACT motor policy
(stage3_policy/learned/) is a separate, optional layer below the symbolic
plan; it never changes what plan() returns.
"""

import logging

from common.types import Action, SceneState, Task

from stage3_policy.errors import ObservationRequired
from stage3_policy.planner import plan_detailed

logger = logging.getLogger(__name__)


def plan(task: Task, scene: SceneState) -> list[Action]:
    """Return one Action per TaskStep in execution order, or raise a PlanningError.

    Raises (all subclasses of stage3_policy.PlanningError):
      TaskValidationError  malformed task (IDs, dependencies, cycles, fields, names)
      UnsupportedError     action / constraint / semantics the contracts cannot express
      SceneError           missing detections, unobserved drawers, non-finite or wrong-frame coordinates
      PreconditionError    impossible sequence (busy arm, unheld object, closed drawer, no free slot)
      ObservationRequired  only a prefix is plannable from this observation; the prefix is attached.
                           plan() returns a complete list or nothing - use
                           stage3_policy.plan_detailed() for staged observe-act execution.
    """
    result = plan_detailed(task, scene)
    logger.info("stage3_policy: %s planner produced %d action(s)", result.mode, len(result.actions))
    for warning in result.warnings:
        logger.warning("stage3_policy: %s", warning)
    if not result.complete:
        total = len(result.actions) + len(result.pending_step_ids)
        raise ObservationRequired(
            f"only {len(result.actions)} of {total} steps can be planned from this observation; "
            f"blocked at step {result.blocked_step_id}: {result.blocked_reason} "
            "plan() needs a staged observe-act loop here (stage3_policy.planner.plan_detailed).",
            step_id=result.blocked_step_id,
            executable_actions=result.actions,
            pending_step_ids=result.pending_step_ids,
        )
    return list(result.actions)
