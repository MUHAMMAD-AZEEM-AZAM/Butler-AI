# Part 2 - Scene Perception

This project builds a tabletop perception pipeline for a robot task: detect objects in a camera image, map them into the physical table frame, and produce a structured scene state that downstream planning logic can use.

## Project summary

The current work focuses on a core robotics problem: a camera sees the tabletop in image coordinates, but the robot needs to reason in tabletop coordinates. The solution is a planar homography between the camera image and the table surface.

In practical terms:

- image frame = pixel coordinates from the camera view
- tabletop frame = physical coordinates on the table surface in meters
- transform = camera frame -> tabletop frame using calibration points

This is exactly the right approach for a flat tabletop because the scene is planar. Instead of trying to estimate a full 3D world model from scratch, we calibrate the camera to the table plane and then project detections into the table coordinate system.

## What is implemented

### Active scene-perception workflow

The working implementation is in [perception/lighting_experiment.py](perception/lighting_experiment.py), which includes:

- synthetic tabletop scene generation
- calibration and homography logic
- image-to-tabletop projection
- contour-based detection and candidate filtering
- structured scene labels for drawer, handle, table objects, and staging regions
- multiple scene modes and perception-oriented output

### Calibration and tabletop mapping

The camera-to-tabletop mapping is handled by:

- [perception/calibration.py](perception/calibration.py)
- [perception/tabletop_transform.py](perception/tabletop_transform.py)
- [perception/scene_pipeline.py](perception/scene_pipeline.py)
- [perception/scene_state.py](perception/scene_state.py)

This gives the project its key capability: estimate where a detected object sits on the real table, not just where it appears in pixels.

### Legacy robustness baseline

The earlier lighting work is kept as a separate reference in [perception/lighting_experiment_legacy_robustness.py](perception/lighting_experiment_legacy_robustness.py). It includes:

- gamma correction
- CLAHE enhancement
- synthetic fog generation
- dehazing
- low-light and haze comparison runs

This baseline is intentionally preserved as historical robustness analysis and is not the main runtime pathway for the tabletop perception task.

## Camera frame -> tabletop frame

This is the key concept behind the project.

A camera sees a point on the table as a pixel coordinate like $(u, v)$ in the image. To make that useful to a robot, we transform that point into a planar tabletop coordinate like $(x, y)$ in meters.

This is done by:

1. selecting known calibration points in the image
2. defining the matching points on the table plane
3. computing a homography
4. projecting detections into the tabletop coordinate frame

The result: object positions are expressed in a coordinate system that matches the robot’s physical task space.

## Repo structure

- [perception/](perception/) - tabletop perception and calibration code
  - [perception/calibration.py](perception/calibration.py) - homography calibration logic
  - [perception/calibration_demo.py](perception/calibration_demo.py) - example camera-to-tabletop calibration output
  - [perception/demo.py](perception/demo.py) - polished demo summary for scene state
  - [perception/scene_pipeline.py](perception/scene_pipeline.py) - planner-facing scene pipeline
  - [perception/scene_state.py](perception/scene_state.py) - structured scene representation
  - [perception/tabletop_transform.py](perception/tabletop_transform.py) - transform helpers
  - [perception/lighting_experiment.py](perception/lighting_experiment.py) - active scene perception implementation
  - [perception/lighting_experiment_legacy_robustness.py](perception/lighting_experiment_legacy_robustness.py) - legacy robustness analysis
- [optimization/](optimization/) - benchmarking and optimization support
- [evaluation/](evaluation/) - evaluation artifacts and benchmark outputs
- [requirements.txt](requirements.txt) - Python dependencies

## Visual outputs in the repo

The repo already contains example visual artifacts that help communicate the project story:

- [scene_demo.png](scene_demo.png)
- [bright_version.png](bright_version.png)
- [dark_version.png](dark_version.png)
- [dark_gamma_version.png](dark_gamma_version.png)
- [fog_25_processed.png](fog_25_processed.png)
- [fog_40_processed.png](fog_40_processed.png)

These outputs demonstrate the perception workflow and the lighting robustness comparison work.

## Verified commands

The following commands were successfully run from the project root.

```bash
cd stage2_perception
source .venv/bin/activate
python -m perception.demo
python -m perception.calibration_demo
./.venv/bin/python perception/lighting_experiment.py
```

## Current validation workflow

Install the Stage 2 dependencies from the repository root:

```powershell
python -m pip install -r stage2_perception/requirements.txt
```

Run the deterministic four-marker validation:

```powershell
python stage2_perception/evaluation/validate_perception.py --drawer-state closed
python stage2_perception/evaluation/validate_perception.py --drawer-state open
python stage2_perception/evaluation/randomized_eval.py --num-seeds 10 --source synthetic
```

The validation image uses ArUco IDs 0-3 for plate, mug, drawer handle, and
water bottle. It maps the 640x480 image corners
`(80,60), (560,60), (560,420), (80,420)` to tabletop coordinates
`(0,0), (0.8,0), (0.8,0.6), (0,0.6)` in meters. A localization error of at
most 0.01 m is required for each marker.

The public API is now image-based:

```python
from stage2_perception import perceive

scene = perceive(camera_frame)
```

It returns the shared planner `SceneState` and does not invent object
positions when the image contains no markers. The MuJoCo camera frame is real,
but the current MuJoCo scene has no ArUco marker textures, so
`--source mujoco` correctly reports no detections until Stage 4 adds markers
or the detector is replaced with a markerless method.

For MuJoCo camera testing:

```powershell
python stage2_perception/evaluation/randomized_eval.py --num-seeds 10 --source mujoco
```

This is expected to fail currently with `No markers detected`; that is an
integration limitation, not a passing perception result.

## Why this project matters

This work matters because a robot cannot act reliably on objects it cannot localize in a meaningful coordinate system. The project turns scene understanding into a usable spatial representation for tabletop work, which is the practical bridge between perception and action.

## Project direction

This repo is intentionally split into two useful tracks:

- the current scene-perception pipeline for the task itself
- a legacy robustness baseline for lighting and haze evaluation

This keeps the project honest, clear, and easier to explain in a hackathon setting or in a technical interview.
