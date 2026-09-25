# Implementation approach and current status

## Why this approach

The official challenge is simulation-first. Its largest scored areas are
end-to-end bimanual completion (30 points), multi-modal reasoning (20), Intel
OpenVINO deployment (20), and ten-seed robustness (15). A reliable
collision-aware state-machine baseline therefore comes before a VLA
fine-tune. It makes the required demo reproducible, generates useful expert
trajectories, and remains a safety fallback if a learned policy fails.

LeRobot is the data/training/deployment toolkit, not one VLA model. The
recommended policy progression is:

1. Validate the MuJoCo scene and scripted expert with contact checks.
2. Collect aligned, varied demonstrations using teleoperation (or a clearly
   labelled scripted expert bootstrap).
3. Train an ACT or SmolVLA baseline for atomic skills first.
4. Fine-tune a VLA only after it beats the baseline on held-out seeds; retain
   the verified state machine as recovery/fallback.

OpenVLA/OFT is a viable later option, but it is not the fastest route to a
credible Core Ultra demo or to task-specific data collection.

## Implemented in this pass

- Removed unsafe MuJoCo exclusions that allowed grippers through the tabletop
  and drawer tray, and the plate through the cabinet.
- Added a continuous contact audit. Contacts deeper than 2 mm are reported as
  failures, except physical resting contacts (object/table) and the drawer
  mechanism. A run with an unexpected deep intersection cannot be successful.
- Added the missing Stage 2 shared perception adapter. When a simulator is
  supplied, it uses measured simulator state, which the challenge explicitly
  permits. This is an honest simulation-state baseline; it must not be called
  camera-only perception.
- Replaced the always-success verifier with drawer, placement, and pour-proxy
  postcondition checks. The scene has no liquid simulation, so pour is
  explicitly a pose-only proxy, not a false water-transfer claim.
- Removed the fake success path used when MuJoCo is missing and made the
  pipeline require both successful execution and successful verification.

## What remains before a submission claim

1. Rebuild the gripper collision approximation and add orientation-aware IK.
   The current five-DoF position-only IK can put the large gripper-base mesh
   through a target while the fingers never form a valid pinch. The visual
   runner now stops at this failure, rather than animating a fake grasp.
2. Retune the pick/place trajectories after contact exclusions are removed.
   Existing scripted welds are an oracle grasp approximation, not a physical
   grasp proof. Replace or gate weld attachment on verified finger/object
   contact.
3. Connect the existing ArUco/camera pipeline to the shared Stage 2 contract
   and report whether each evaluation run used camera perception or simulator
   state.
4. Record full synchronized image sequences, joint state, gripper state,
   actions, language prompt, seed, and success/recovery labels. The current
   collector saves only first/last image previews, so it is **not** VLA-ready.
5. Add train/validation/test scenario splits by episode/seed, not random
   frames. Hold out layouts, lighting, friction, mass, and object poses.
6. Run the full ten-seed evaluation only after the above physics checks pass;
   report actual failures rather than synthetic 100% success.
7. Benchmark an exported supported perception/policy model with OpenVINO on
   the required Intel Core Ultra Series 2/3 hardware. Do not invent latency,
   throughput, NPU usage, or quantization results.

## Run and verify

From `ai-infra-summit-hack` on Windows:

```powershell
..\venv\Scripts\python.exe scripts\run_pipeline.py
..\venv\Scripts\python.exe scripts\evaluate.py --seeds 10
```

The evaluation is meaningful only when MuJoCo is installed and the result log
contains neither `MuJoCo is unavailable` nor `collision audit failed`.
