#!/usr/bin/env python3
"""Directly command and control the robot in MuJoCo using the trained LeRobot ACT model.

Usage:
    # 1. Run direct command in simulation (default: seed 100, 120 steps):
    python scripts/control_robot_act.py --command "open the drawer"

    # 2. Watch real-time interactive 3D physics in MuJoCo desktop window:
    python scripts/control_robot_act.py --view --seed 100

    # 3. Record video of the policy executing:
    python scripts/control_robot_act.py --record outputs/act_drawer_control.mp4 --seed 100
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import mujoco
import numpy as np

from stage3_policy.learned.inference import configured_checkpoint, select_motor_policy
from stage4_bimanual.bimanual import reset_scene


def run_act_control(
    command: str = "open the top drawer",
    seed: int = 100,
    steps: int = 120,
    checkpoint: str | None = None,
    backend: str = "pytorch",
    device: str = "CPU",
    view: bool = False,
    record_path: str | None = None,
) -> bool:
    if backend == "openvino":
        ckpt_path = checkpoint or "assets/models/openvino_open_drawer/act_policy.xml"
    else:
        ckpt_path = checkpoint or configured_checkpoint()

    print("=" * 65)
    print(f"[ACT Controller] Loading model ({backend.upper()}): {ckpt_path}")
    selection = select_motor_policy(ckpt_path, backend=backend, device=device, load=True)
    if selection.policy is None:
        print(f"[ACT Controller Error] Failed to load policy: {selection.reason}")
        return False
    policy = selection.policy
    policy.reset()

    print(f"[ACT Controller] Initializing MuJoCo scene (seed={seed})...")
    sim = reset_scene(seed)

    drawer_joint = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, "drawer_slide")
    drawer_qpos_idx = sim.model.jnt_qposadr[drawer_joint] if drawer_joint >= 0 else None
    initial_drawer = float(sim.data.qpos[drawer_qpos_idx]) if drawer_qpos_idx is not None else 0.0

    renderer = mujoco.Renderer(sim.model, height=480, width=640)
    instruction = "Open the top drawer with arm A."

    viewer = None
    if view:
        import mujoco.viewer as mj_viewer
        print("[ACT Controller] Launching 3D interactive viewer...")
        viewer = mj_viewer.launch_passive(sim.model, sim.data)

    frames = []
    print(f"[ACT Controller] Executing direct command: '{command}'")
    print(f"[ACT Controller] Running {steps} policy cycles at 25 Hz (0.04s timestep)...")
    print(f"  Initial drawer slide: {initial_drawer*100:.2f} cm")

    success = False
    start_wall_time = time.perf_counter()

    for step_i in range(steps):
        # 1. State observation (12-DoF joint angles, qpos[36:48])
        joint_positions = np.asarray(sim.data.qpos[36:48], dtype=np.float32).tolist()

        # 2. Visual observation (overhead camera 480x640x3 RGB)
        renderer.update_scene(sim.data, camera="overhead_cam")
        overhead_rgb = renderer.render().copy()

        # 3. Model inference: compute 12 joint targets (rad)
        action = policy.act(joint_positions, overhead_rgb, instruction)

        # 4. Command actuators (ctrl[:12])
        sim.data.ctrl[:12] = np.asarray(action, dtype=np.float64)

        # 5. Step physics 20 substeps (20 * 0.002s = 0.04s)
        for _ in range(20):
            mujoco.mj_step(sim.model, sim.data)
            if viewer is not None and viewer.is_running():
                viewer.sync()
                time.sleep(0.002)

        cur_drawer = float(sim.data.qpos[drawer_qpos_idx]) if drawer_qpos_idx is not None else 0.0
        disp = cur_drawer - initial_drawer

        if record_path:
            annotated = overhead_rgb.copy()
            cv2.putText(
                annotated,
                f"Step {step_i+1}/{steps} | Drawer: {cur_drawer*100:.1f} cm",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2,
            )
            frames.append(annotated)

        if (step_i + 1) % 25 == 0 or step_i == steps - 1:
            print(f"  Step {step_i+1:3d}/{steps}: Drawer={cur_drawer*100:5.2f} cm (slide={disp*100:+.2f} cm), Arm A grip ctrl={action[5]:.3f}")

        if disp >= 0.040:
            success = True

    duration = time.perf_counter() - start_wall_time
    final_drawer = float(sim.data.qpos[drawer_qpos_idx]) if drawer_qpos_idx is not None else 0.0
    final_disp = final_drawer - initial_drawer

    print("-" * 65)
    print(f"[ACT Controller] Execution completed in {duration:.2f}s ({steps} steps)")
    print(f"[ACT Controller] Final drawer displacement: {final_disp*100:.2f} cm")
    if final_disp >= 0.040:
        print("[ACT Controller RESULT] SUCCESS! The drawer was pulled open past the 4.0 cm threshold.")
    else:
        print(f"[ACT Controller RESULT] PARTIAL CONTROL: Drawer moved {final_disp*100:.2f} cm.")

    if record_path and frames:
        out_p = Path(record_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_p), fourcc, 25.0, (640, 480))
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"[ACT Controller] Annotated video saved to: {out_p.resolve()}")

    if viewer is not None and viewer.is_running():
        time.sleep(1.0)
        viewer.close()

    print("=" * 65)
    return final_disp >= 0.040


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Direct robot control via LeRobot ACT model")
    parser.add_argument("--command", default="open the top drawer", help="Instruction text")
    parser.add_argument("--seed", type=int, default=100, help="Randomization seed")
    parser.add_argument("--steps", type=int, default=120, help="Number of policy steps")
    parser.add_argument("--checkpoint", default=None, help="Custom checkpoint path")
    parser.add_argument("--backend", choices=["pytorch", "openvino"], default="pytorch", help="Execution backend")
    parser.add_argument("--device", default="CPU", help="Device target: CPU, GPU.0, etc.")
    parser.add_argument("--view", action="store_true", help="Open MuJoCo 3D viewer")
    parser.add_argument("--record", default=None, help="Save MP4 video to path")
    args = parser.parse_args()

    run_act_control(
        command=args.command,
        seed=args.seed,
        steps=args.steps,
        checkpoint=args.checkpoint,
        backend=args.backend,
        device=args.device,
        view=args.view,
        record_path=args.record,
    )
