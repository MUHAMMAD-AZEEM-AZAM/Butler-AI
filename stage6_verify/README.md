# stage6_verify — Postcondition verification & replan signal

Stage 6 checks the post-execution `SceneState` against requirements in the original `Task` and returns a `VerifyResult`.

It currently verifies:

- all objects referenced by pick, place, handoff, and pour steps are present;
- requested drawer states are correct; and
- optional expected object positions are within a 1 cm tolerance.

Missing objects, incorrect drawer states, and position errors return `replan=True` so the integration pipeline can retry. The current `Task` contract does not contain destination coordinates, so callers may pass `expected_positions` when those targets are available from the controller.

Standalone check:

```powershell
python -c "from common.types import SceneState, Task; from stage6_verify import verify; print(verify(SceneState(objects={}, drawers={}), Task(command='x', steps=[])))"
```

Run the Stage 6 unit tests with:

```powershell
python -m pytest stage6_verify/tests -q
```

The full pipeline is expected to fail until the MuJoCo camera produces valid
Stage 2 detections and Stage 4 reports real manipulation results. That failure
is intentional: verification must not accept a scene with missing objects or
an unopened drawer.
