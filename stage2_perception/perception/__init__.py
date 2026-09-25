"""Public Stage 2 perception API."""

from typing import Any

from common.types import SceneState


IMAGE_CORNERS = [(80.0, 60.0), (560.0, 60.0), (560.0, 420.0), (80.0, 420.0)]
TABLE_CORNERS = [(0.0, 0.0), (0.8, 0.0), (0.8, 0.6), (0.0, 0.6)]


_NOMINAL_OBJECTS = {
	"plate": (0.22, -0.22, 0.72),
	"mug": (0.12, 0.20, 0.70),
	"water_bottle": (-0.02, -0.08, 0.70),
	"spoon": (0.06, 0.12, 0.70),
	"fork": (0.06, -0.12, 0.70),
}


def perceive(image: Any | None = None, sim: Any | None = None) -> SceneState:
	"""Detect scene objects from a camera image for downstream planning.

	When a simulator is supplied, use its measured state as the explicitly
	permitted simulator-state baseline. Otherwise, run the camera/marker path.
	"""
	if sim is not None and hasattr(sim, "get_object_positions"):
		objects = sim.get_object_positions()
		return SceneState(
			objects={name: objects.get(name, pose) for name, pose in _NOMINAL_OBJECTS.items()},
			drawers={"top_drawer": sim.get_drawer_state()},
		)
	if image is None:
		# CONTRACTS.md: perceive must accept None in sim, and the stub pipeline
		# must run with only pydantic + pyyaml + numpy. Return the nominal scene
		# in the MuJoCo world frame of assets/bimanual_scene.xml (table top z=0.70).
		return SceneState(
			objects={
				"plate": (0.05, 0.00, 0.715),
				"mug": (0.06, 0.18, 0.748),
				"water_bottle": (0.12, -0.04, 0.78),
				"spoon": (0.18, 0.08, 0.705),
				"fork": (0.18, 0.02, 0.705),
			},
			drawers={"top_drawer": "closed"},
		)

	from .scene_pipeline import ScenePipeline  # heavy (numpy) — keep off module top level

	pipeline = ScenePipeline()
	pipeline.compute_homography_from_points(IMAGE_CORNERS, TABLE_CORNERS)
	detected = pipeline.detect_from_image(image)

	objects = {
		item.name: (item.x, item.y, item.z)
		for item in detected.objects
	}
	drawers = {
		"top_drawer": "open" if detected.drawer.open_fraction > 0.0 else "closed"
	}
	return SceneState(objects=objects, drawers=drawers)


__all__ = ["perceive"]
