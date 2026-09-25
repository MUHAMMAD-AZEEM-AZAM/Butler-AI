"""Smooth trajectory interpolation and actuation executor for MuJoCo simulation."""

from typing import Any
import numpy as np

try:
    import mujoco
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False

from stage4_bimanual.constants import DEFAULT_SUBSTEPS_PER_TRAJECTORY


class TrajectoryExecutor:
    """Executes smooth joint-space trajectories in MuJoCo with physics stepping."""

    def __init__(self, model: Any, data: Any, contact_audit: Any | None = None):
        self.model = model
        self.data = data
        self.contact_audit = contact_audit

    def _on_step(self) -> None:
        """Observer hook called after every mj_step; no-op by default.

        Subclasses may read state here (e.g. to render a frame) but must never
        modify model, data or ctrl, so the simulation stays bit-identical.
        """

    def interpolate(
        self,
        target_ctrl: np.ndarray | list[float],
        steps: int = DEFAULT_SUBSTEPS_PER_TRAJECTORY,
    ) -> None:
        """Smoothly interpolate actuator controls using cosine easing and step physics.

        Args:
            target_ctrl: Desired 12-element actuator target array.
            steps: Number of simulation substeps for smooth transition.
        """
        if not HAS_MUJOCO or self.model is None or self.data is None:
            return

        start_ctrl = np.copy(self.data.ctrl)
        target_ctrl_arr = np.asarray(target_ctrl, dtype=np.float64)

        # Step simulation along smooth cosine S-curve
        for s in range(steps):
            alpha = 0.5 * (1.0 - np.cos(np.pi * (s + 1) / steps))
            self.data.ctrl[:] = start_ctrl + alpha * (target_ctrl_arr - start_ctrl)
            mujoco.mj_step(self.model, self.data)
            if self.contact_audit is not None:
                self.contact_audit.sample(self.model, self.data)
            self._on_step()

        # Hold final target for settling
        self.data.ctrl[:] = target_ctrl_arr
        settle_steps = min(25, max(10, steps // 3))
        for _ in range(settle_steps):
            mujoco.mj_step(self.model, self.data)
            if self.contact_audit is not None:
                self.contact_audit.sample(self.model, self.data)
            self._on_step()
