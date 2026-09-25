import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from stage4_bimanual.bimanual import reset_scene
from stage4_bimanual.trajectory import TrajectoryExecutor
from stage4_bimanual.primitives import (
    OpenDrawerPrimitive,
    PickPlatePrimitive,
    PlacePlatePrimitive,
    PickMugPrimitive,
    PickBottlePrimitive,
    PourWaterPrimitive,
)

print("=" * 65)
print("Evaluating All 10 Seeds (Full 6-Step Task Pipeline)")
print("=" * 65)

results = []
for s in range(10):
    sim = reset_scene(seed=s)
    executor = TrajectoryExecutor(sim.model, sim.data, contact_audit=sim.contact_audit)

    p1 = OpenDrawerPrimitive(executor, sim).execute()
    p2 = PickPlatePrimitive(executor, sim).execute()
    p3 = PlacePlatePrimitive(executor, sim).execute()
    p4 = PickMugPrimitive(executor, sim).execute()
    p5 = PickBottlePrimitive(executor, sim).execute()
    p6 = PourWaterPrimitive(executor, sim).execute()

    plate_id = sim.model.body("plate").id
    plate_pos = sim.data.xpos[plate_id]
    p_err = np.linalg.norm(plate_pos[:2] - np.array([0.06, 0.00]))

    bottle_id = sim.model.body("water_bottle").id
    bottle_z = sim.data.xpos[bottle_id][2]

    all_ok = p1 and p2 and p3 and p4 and p5 and p6 and sim.contact_audit.ok and (p_err < 0.02)
    results.append((s, all_ok, p_err, sim.contact_audit.ok))
    status = "PASS" if all_ok else "FAIL"
    print(f"Seed {s:02d}: {status} | Plate Error: {p_err*1000:4.1f}mm | Bottle Z: {bottle_z:.3f}m | Contact Clean: {sim.contact_audit.ok}")

passed = sum(1 for r in results if r[1])
print("=" * 65)
print(f"Final Result: {passed} / 10 PASSED ({passed*10}%)")
print("=" * 65)
