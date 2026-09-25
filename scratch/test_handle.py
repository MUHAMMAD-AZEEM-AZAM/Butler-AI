import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np
from stage4_bimanual.trajectory import TrajectoryExecutor
from stage4_bimanual.constants import ALTITUDE_SAFE_TRANSIT, GRIPPER_OPEN, GRIPPER_CLOSED
from stage4_bimanual.kinematics import DLSInverseKinematics
from stage4_bimanual.safety import ContactAudit

xml_path = Path("assets/bimanual_scene.xml")
xml_text = xml_path.read_text(encoding="utf-8")

old_mug = '''    <!-- 3D Upright Ceramic Mug (Standing flat on table in Arm B's reach) -->
    <body name="mug" pos="0.06 0.18 0.70">
      <freejoint/>
      <!-- Visual 3D Mesh -->
      <geom type="mesh" mesh="mug_mesh" material="ceramic_blue_mat" contype="0" conaffinity="0" group="2"/>
      <!-- Physical Collision Cylinder -->
      <geom name="mug_geom" type="cylinder" size="0.044 0.048" pos="0 0 0.048" mass="0.14" friction="1.8 0.01 0.001" rgba="0 0 0 0" group="3"/>
      <site name="mug_site" pos="0 0 0.048" rgba="0 0 0 0" group="3"/>
    </body>'''

new_mug = '''    <!-- 3D Upright Ceramic Mug (Standing flat on table in Arm B's reach) -->
    <body name="mug" pos="0.06 0.18 0.70">
      <freejoint/>
      <!-- Visual 3D Mesh -->
      <geom type="mesh" mesh="mug_mesh" material="ceramic_blue_mat" contype="0" conaffinity="0" group="2"/>
      <!-- Physical Collision Cylinder -->
      <geom name="mug_geom" type="cylinder" size="0.044 0.048" pos="0 -0.018 0.048" mass="0.12" friction="1.8 0.01 0.001" rgba="0 0 0 0" group="3"/>
      <!-- Physical Collision Handle Capsule -->
      <geom name="mug_handle" type="capsule" fromto="0 0.050 0.030 0 0.050 0.075" size="0.010" mass="0.02" friction="2.0 0.01 0.001" rgba="0 0 0 0" group="3"/>
      <site name="mug_site" pos="0 -0.018 0.048" rgba="0 0 0 0" group="3"/>
      <site name="mug_handle_site" pos="0 0.050 0.065" size="0.005" rgba="0 0 0 0" group="3"/>
    </body>'''

xml_mod = xml_text.replace(old_mug, new_mug)
temp_xml = Path("assets/temp_scene.xml")
temp_xml.write_text(xml_mod, encoding="utf-8")

try:
    m = mujoco.MjModel.from_xml_path(str(temp_xml))
    d = mujoco.MjData(m)

    from stage4_bimanual.constants import ARM_A_STANDBY, ARM_B_STANDBY
    d.qpos[36:41] = ARM_A_STANDBY
    d.qpos[41] = GRIPPER_OPEN
    d.qpos[42:47] = ARM_B_STANDBY
    d.qpos[47] = GRIPPER_OPEN
    d.ctrl[0:5] = ARM_A_STANDBY
    d.ctrl[5] = GRIPPER_OPEN
    d.ctrl[6:11] = ARM_B_STANDBY
    d.ctrl[11] = GRIPPER_OPEN

    audit = ContactAudit()
    executor = TrajectoryExecutor(m, d, contact_audit=audit)
    ik = DLSInverseKinematics(m, d)

    mug_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "mug")
    handle_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "mug_handle_site")
    mujoco.mj_forward(m, d)

    handle_pos = np.copy(d.site_xpos[handle_site_id])
    print("Initial handle pos:", np.round(handle_pos, 4))

    # 1. Approach high above handle
    ctrl = np.copy(d.ctrl)
    ctrl[6:11] = ik.solve("B", [handle_pos[0], handle_pos[1], ALTITUDE_SAFE_TRANSIT], wrist_roll=0.0)[1]
    ctrl[11] = GRIPPER_OPEN
    executor.interpolate(ctrl, steps=25)

    # 2. Descend vertically to handle grasp altitude (0.768)
    grasp_z = 0.768
    for step_idx, z in enumerate(np.linspace(ALTITUDE_SAFE_TRANSIT, grasp_z, 6)[1:]):
        ctrl[6:11] = ik.solve("B", [handle_pos[0], handle_pos[1], z], wrist_roll=0.0)[1]
        executor.interpolate(ctrl, steps=10)
        print(f"2.{step_idx} After descend z={z:.3f}: mug={np.round(d.xpos[mug_id], 4)}")

    # 3. Close gripper firmly on handle
    ctrl[11] = GRIPPER_CLOSED
    executor.interpolate(ctrl, steps=20)
    print("3. After grip: mug=", np.round(d.xpos[mug_id], 4))

    # Check contact with mug
    def in_subtree(model, body_id, root_id):
        while body_id > 0:
            if body_id == root_id:
                return True
            body_id = int(model.body_parentid[body_id])
        return body_id == root_id

    b1 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "b_gripper_base")
    b2 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "mug")

    has_contact = any(
        (in_subtree(m, int(m.geom_bodyid[d.contact[i].geom1]), b1)
         and in_subtree(m, int(m.geom_bodyid[d.contact[i].geom2]), b2))
        or
        (in_subtree(m, int(m.geom_bodyid[d.contact[i].geom1]), b2)
         and in_subtree(m, int(m.geom_bodyid[d.contact[i].geom2]), b1))
        for i in range(d.ncon)
    )
    print("Contact established with mug?", has_contact)
    print("Contact audit ok?", audit.ok, audit.summary())

    for i in range(d.ncon):
        g1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, d.contact[i].geom1)
        g2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, d.contact[i].geom2)
        b1_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[d.contact[i].geom1])
        b2_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[d.contact[i].geom2])
        if "mug" in (b1_name, b2_name):
            print(f"  {b1_name}:{g1} <-> {b2_name}:{g2}, dist={d.contact[i].dist*1000:.1f} mm")

finally:
    if temp_xml.exists():
        temp_xml.unlink()
