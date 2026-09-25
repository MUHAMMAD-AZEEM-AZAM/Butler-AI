import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stage4_bimanual.bimanual import reset_scene
from stage4_bimanual.trajectory import TrajectoryExecutor
from stage4_bimanual.primitives import (
    OpenDrawerPrimitive,
    PickPlatePrimitive,
    PlacePlatePrimitive,
    PickMugPrimitive,
)

print("Evaluating Mug Grasp Contact across Seeds 0 to 9...")
for s in range(10):
    sim = reset_scene(seed=s)
    executor = TrajectoryExecutor(sim.model, sim.data, contact_audit=sim.contact_audit)
    OpenDrawerPrimitive(executor, sim).execute()
    PickPlatePrimitive(executor, sim).execute()
    PlacePlatePrimitive(executor, sim).execute()
    p = PickMugPrimitive(executor, sim)
    ok = p.execute()
    touching = len(p.gripper_contact_bodies("weld_mug"))
    print(f"Seed {s}: PickMug = {ok} | Touching Jaws = {touching}")
