<!-- <p align="center">
  <img src="docs/visualizations/bimanual_simulation_full.gif" alt="Butler AI — full bimanual sequence" width="720">
</p> -->

<h1 align="center">Butler AI</h1>
<p align="center"><b>Two simulated SO-101 arms set a dinner table from a spoken command.</b></p>
<p align="center">
Intel Physical AI Online Challenge — Bimanual VLA Manipulation with Multi-Modal Reasoning<br>
<a href="https://lablab.ai/ai-hackathons/ai-infra-summit-hackathon/">lablab.ai AI Infra Summit Hackathon</a> · Challenge option: <i>Setting Up a Dinner Table</i>
</p>

<p align="center">
<a href="https://drive.google.com/file/d/1tM_mr5yj0RUUEhJGEbOZgIqykB2NKViq/view?usp=sharing">Trained checkpoint</a> ·
<a href="https://colab.research.google.com/drive/1hUY1LIP8rjpQOGafFnSko-S1TPc5XygI?usp=sharing">Training notebook</a> ·
<a href="docs/TECHNICAL_DEEP_DIVE.md">Technical deep dive</a> ·
<a href="#installation--running-it">Quick start</a>
</p>

> **"Open the top drawer, pick up the plate with arm A, place it on the table, pick up the
> mug with arm B, pour water into the mug with arm A."**

<p align="center">
  <img src="docs/visualizations/divider.svg" width="100%" height="4">
</p>

## Overview

- A voice or text command drives **two MuJoCo-simulated SO-101 arms** through a full
  drawer → plate → mug → bottle → **pour** sequence, with a genuine simultaneous
  **complementary dual-arm action**: Arm B tilts the held mug while Arm A tilts the bottle.
- Language understanding is **Speechmatics + Claude**; scene state is read live from the
  simulator; task planning is a **rule-based planner** with an opt-in, trained **LeRobot
  ACT** policy (exported to OpenVINO and benchmarked on Intel hardware) integrated into
  execution, with automatic fallback to scripted control if the policy isn't available.
- Physics is real, not scripted: grasps are **contact-gated MuJoCo welds** — a grasp only
  attaches once both gripper jaws register contact — caught and regression-tested after an
  earlier version silently "grasped" with zero contact 10/10 times. See the
  [gripper engineering story](docs/TECHNICAL_DEEP_DIVE.md#gripper--physics-engineering-story).
- **Measured, not claimed**: 60/60 skill-seed physics validation, 146/147 tests passing, and
  real OpenVINO benchmarks on two Intel machines (including a Core Ultra 7 + NPU run).
- Every claim below is tied to a file, commit, or number. Where something is a placeholder
  (demo video, a fresh 10-seed pipeline run) it's marked `[ADD]`, not filled in with a guess.

<p align="center">
  <img src="docs/visualizations/divider.svg" width="100%" height="4">
</p>

## Scenes of the MuJoCo-simulated SO-101 arms

<table>
<tr>
<td width="33%"><img src="https://github.com/user-attachments/assets/a02f2e06-9f53-4b2e-b372-53df9c2483fd" width="100%"/><p align="center"><sub>Full scene — dinner table, drawer, place setting</sub></p></td>
<td width="33%"><img src="https://github.com/user-attachments/assets/3d16a44f-b834-4bea-8180-bdb8ea4ac92b" width="100%"/><p align="center"><sub>Drawer open, plate visible inside</sub></p></td>
<td width="33%"><img src="https://github.com/user-attachments/assets/719e0b17-d8ab-4a04-af2c-0b3c67b427fa" width="100%"/><p align="center"><sub>Top-down view of both arms</sub></p></td>
</tr>
<tr>
<td width="33%"><img src="https://github.com/user-attachments/assets/c980867f-093d-4525-89bc-2a38d4f73f09" width="100%"/><p align="center"><sub>Drawer-tray detail, close-up</sub></p></td>
<td width="33%"><img src="https://github.com/user-attachments/assets/3fd2a7a2-2901-4e02-b9aa-92be2fce8384" width="100%"/><p align="center"><sub>Angled overhead, full place setting</sub></p></td>
<td width="33%"><img src="https://github.com/user-attachments/assets/15c55911-6a3f-43b7-97ee-890a6b84ad93" width="100%"/><p align="center"><sub>Isometric view, both arms parked</sub></p></td>
</tr>
<tr>
<td width="33%"><img src="https://github.com/user-attachments/assets/d8c987ec-0b24-4476-adbe-c7e3f45f4a0f" width="100%"/><p align="center"><sub>Wide isometric, arms lowered toward drawer/bottle</sub></p></td>
<td width="33%"><img src="https://github.com/user-attachments/assets/38093214-a897-4c92-a342-a4483906035f" width="100%"/><p align="center"><sub>Drawer-tray detail, alternate crop</sub></p></td>
<td width="33%"><img src="https://github.com/user-attachments/assets/e5585fee-52b4-465d-81e6-2b94b4b5d74b" width="100%"/><p align="center"><sub>Close overhead detail — drawer, mug, bottle</sub></p></td>
</tr>
</table>

## Architecture

```mermaid
flowchart TD
    A[Command: text, audio, or mic] --> B[stage1_voice: Speechmatics + Claude]
    B -->|Task| C[stage2_perception: MuJoCo sim-state]
    C -->|SceneState| D[stage3_policy: planner plus optional ACT]
    D -->|Action| E[stage4_bimanual: dual-arm execution]
    E --> F[stage2_perception: re-observe]
    F --> G[stage6_verify]
    G -->|ok| H[Done]
    G -->|replan| D
    E -.-> I[stage5_openvino: benchmark]
    D -.-> J[stage7_eval: 10-seed robustness]
```

This is the real call graph of [`common/pipeline.py::run_once()`](common/pipeline.py) — not
an aspirational diagram. All inter-stage data is typed pydantic (`common/types.py`),
signatures pinned in [`CONTRACTS.md`](CONTRACTS.md). Full stage-by-stage walkthrough,
including exactly how language + vision + task state combine into the next action:
→ [deep dive](docs/TECHNICAL_DEEP_DIVE.md#multi-modal-reasoning).

<p align="center">
  <img src="docs/visualizations/divider.svg" width="100%" height="4">
</p>

## The task

| Step | Primitive | Arm | Result |
|---|---|---|---|
| 1 | Open drawer | A | contact-gated pinch grasp → weld → pull → clean release |
| 2 | Pick plate | A | live-pose rim grasp, lift clear of the drawer tray |
| 3 | Place plate | A | levelling-pitch search so the plate lands flat |
| 4 | Pick mug | B | handle-relative grasp, lift to holding station |
| 5 | Pick bottle | A | side/body grasp, pre-computed pour-side roll |
| 6 | **Pour** | A + B | Arm B tilts the mug **while** Arm A tilts the bottle — closed-loop, simultaneous |

**Not implemented**, stated plainly: spoon/fork retrieval, closing the drawer, and a literal
object hand-off between grippers (the pour is complementary, not a hand-off — the two arms
never exchange an object). Each arm owns disjoint objects throughout. This still satisfies the
challenge brief's own example of complementary action ("one arm holding a mug while the other
pours"). Full coordination mechanics (standby parking, contact-audit collision checks, why
this is harder than single-arm pick-and-place) → [deep dive](docs/TECHNICAL_DEEP_DIVE.md).

<p align="center">
  <img src="docs/visualizations/divider.svg" width="100%" height="4">
</p>

## Results at a glance

| | |
|---|---|
| **Skill-level physics validation** | **60/60** — all 6 primitives, 10 randomized seeds each, contact audit clean |
| **Test suite** | **146 / 147** passing (`pytest -q -m "not live"`), 150 tests total, 3 gated behind a real API key |
| **Domain randomization** | placement, yaw, friction, mass, lighting — seeded, reproducible per-seed |
| **OpenVINO — CPU** (i7-10510U) | 630 ms/chunk, matches PyTorch to `2.4e-7` |
| **OpenVINO — Intel iGPU** | 501 ms/chunk |
| **OpenVINO — Core Ultra 7 NPU** | 572 ms/chunk, matches PyTorch to `~1e-3 rad` (correct; **not yet faster** than CPU — no quantization attempted) |
| **Learned ACT policy** | trained (10k steps, external GPU run), exported, benchmarked — **not yet validated for closed-loop task success** |
| **Full pipeline, 10-seed eval** | harness implemented and runnable; committed report predates recent physics fixes — see [below](#robustness--evaluation) |

Full benchmark tables (both machines, all devices, precision/latency/throughput) →
[deep dive](docs/TECHNICAL_DEEP_DIVE.md#intel--openvino-benchmark-results-in-full).

<p align="center">
  <img src="docs/visualizations/divider.svg" width="100%" height="4">
</p>

## Robustness & evaluation

**Skill-level (measured, real)**: `VOICE_STUB=1 python scripts/evaluate.py --seeds 10` and
`python scripts/verify_all_10_seeds.py` — 10/10 on all six primitives, seeds 100–109, with
per-skill contact-audit checks. This isolates physics/grasping robustness from the full
voice→perception→policy pipeline.

**Full-pipeline (the official deliverable)**: `stage7_eval/evaluate.py` runs the complete
pipeline per seed and is fully implemented — but the `evaluation_report.json` currently
committed predates the [gripper/weld fix](docs/TECHNICAL_DEEP_DIVE.md#gripper--physics-engineering-story)
and shows only 2 seeds, 0 successes. Rather than replace that with an invented number, the
table below is a placeholder to fill in with a fresh run before the submission video:

| Seed | Scene variation | Result | Notes |
|---|---|---|---|
| 0–9 | `[ADD]` | `[ADD PASS/FAIL]` | `[ADD]` |

**Success rate = successful seeds ÷ 10 × 100 = `[ADD]`%** — regenerate with:
```bash
python scripts/evaluate.py --seeds 10 --output evaluation_report.json
```

<p align="center">
  <img src="docs/visualizations/divider.svg" width="100%" height="4">
</p>

## Media

| | |
|---|---|
| ![Open drawer](docs/visualizations/open_drawer.gif) | **Open drawer** — Arm A |
| ![Pick plate](docs/visualizations/pick_plate.gif) | **Pick plate** — Arm A |
| ![Place plate](docs/visualizations/place_plate.gif) | **Place plate** — Arm A |
| ![Pick mug](docs/visualizations/pick_mug.gif) | **Pick mug** — Arm B |
| ![Pour water](docs/visualizations/pour_water.gif) | **Complementary pour** — both arms |
| ![Final state](docs/visualizations/final_table_set.png) | **Final table state** |

### Still needed

#### Submission demo video
`[ADD LINK]` — command → randomized initial scene → perception/policy inference →
coordinated dual-arm execution including the pour → final state, per the official
recommended demonstration sequence.

#### 10-seed evaluation video
`[ADD LINK]` — one clip per seed from a fresh `python scripts/evaluate.py --seeds 10` run,
showing command + scene variation + outcome for each seed (feeds the seed table in
[Robustness & evaluation](#robustness--evaluation)).

#### OpenVINO benchmark video
`[ADD LINK]` — the benchmark script running live on confirmed Intel Core Ultra
Series 2/3 hardware, terminal output visible.

#### Failure / recovery clip
`[ADD LINK]` — one deliberately-induced failure and the `stage6_verify` replan loop
recovering from it.

Full placeholder descriptions → [deep dive media notes](docs/TECHNICAL_DEEP_DIVE.md).

<p align="center">
  <img src="docs/visualizations/divider.svg" width="100%" height="4">
</p>

## Challenge rubric coverage

No self-scoring — only where the evidence for each official criterion lives.

| Criterion (pts) | Evidence |
|---|---|
| End-to-End Task Completion & Bimanual Manipulation (30) | [The task](#the-task), [primitives.py](stage4_bimanual/primitives.py), `stage4_bimanual/README.md` |
| VLA / Multi-Modal Reasoning (20) | [Architecture](#architecture), `stage1_voice/`, `stage3_policy/` |
| Robustness & Generalization (15) | [Results at a glance](#results-at-a-glance), [Robustness & evaluation](#robustness--evaluation) |
| OpenVINO & Intel Core Ultra Optimization (20) | [Results at a glance](#results-at-a-glance), [full benchmarks](docs/TECHNICAL_DEEP_DIVE.md#intel--openvino-benchmark-results-in-full) |
| Technical Quality & Reproducibility (10) | [Testing](docs/TECHNICAL_DEEP_DIVE.md#testing), [Installation](#installation--running-it), [`CONTRACTS.md`](CONTRACTS.md) |
| Innovation & Technical Demonstration (5) | Hybrid learned/scripted policy with fallback, closed-loop mouth-tracking pour, source-inspecting regression test |

<p align="center">
  <img src="docs/visualizations/divider.svg" width="100%" height="4">
</p>

## Repository structure

```
.
├── common/            # shared pydantic types + run_once() pipeline entrypoint
├── stage1_voice/       # ASR + Claude instruction parsing -> Task
├── stage2_perception/   # sim-state adapter (live) + ArUco/homography vision (dormant)
├── stage3_policy/        # rule-based planner + learned/ (ACT schema, inference, training)
├── stage4_bimanual/       # MuJoCo dual-arm primitives, physics, contact audit
├── stage5_openvino/        # OpenVINO export + benchmark
├── stage6_verify/           # postcondition verification + replan signal
├── stage7_eval/               # multi-seed robustness harness
├── scripts/                    # CLI entrypoints (see below)
├── assets/                      # MuJoCo scene/meshes, SO-ARM100 MJCF, checkpoints*
├── configs/default.yaml          # seeds, randomization, model paths, hardware target
├── docs/                          # this deep dive + visualizations/
└── data/                            # collected datasets*

  * assets/models/, data/*, outputs/ are gitignored (generated/large) — see below.
```

<p align="center">
  <img src="docs/visualizations/divider.svg" width="100%" height="4">
</p>

## Installation & running it

**Requirements**: Python 3.12.10 confirmed working (torch 2.11.0+cpu, openvino
2026.3.1–2026.4.0, lerobot 0.6.1, mujoco 3.13.0). The base pipeline only needs
`pydantic`/`pyyaml`/`numpy` — heavier deps stay optional by design.

```bash
git clone https://github.com/MUHAMMAD-AZEEM-AZAM/Butler-AI.git && cd Butler-AI
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# per-stage extras, as needed:
pip install mujoco opencv-python speechmatics-python anthropic python-dotenv
pip install -r stage3_policy/requirements-learned.txt   # lerobot, torch
pip install openvino

cp .env.example .env   # set ANTHROPIC_API_KEY, SPEECHMATICS_API_KEY
```

**Model checkpoint** (gitignored, 329MB): download from the [checkpoint link](https://drive.google.com/file/d/1tM_mr5yj0RUUEhJGEbOZgIqykB2NKViq/view?usp=sharing)
into `assets/models/act_open_drawer/` and `assets/models/openvino_open_drawer/` — fully
optional, the pipeline falls back to scripted control without it.

```bash
# Stub-mode sanity check (no API keys, no MuJoCo required)
VOICE_STUB=1 python scripts/run_pipeline.py

# Full pipeline from text
python scripts/run_pipeline.py --command "open the top drawer, pick up the plate with arm A, place it on the table, pick up the mug with arm B, pour water into the mug with arm A"

# From audio / live mic, with viewer or recording
python scripts/run_voice_pipeline.py --audio stage1_voice/samples/official_command.wav --view
python scripts/run_voice_pipeline.py --mic --seconds 6 --record outputs/demo.mp4

# Skill-level physics validation (10 seeds)
python scripts/verify_all_10_seeds.py

# Full pipeline randomized evaluation (the official deliverable)
python scripts/evaluate.py --seeds 10 --output evaluation_report.json

# Learned-policy robot control (PyTorch or OpenVINO backend)
python scripts/control_robot_act.py --command "open the top drawer" --backend openvino --device GPU.0 --view

# OpenVINO export + accuracy/latency comparison
python -m stage5_openvino.export_act --checkpoint assets/models/act_open_drawer --dataset data/butler_demos/open_drawer

# Tests
pytest -q -m "not live"   # 146/147 passing, no API key needed
```

More commands (data collection, replay, montage, interactive viewer) →
[deep dive](docs/TECHNICAL_DEEP_DIVE.md).

<p align="center">
  <img src="docs/visualizations/divider.svg" width="100%" height="4">
</p>

## Limitations

- No true object hand-off — the pour is simultaneous complementary action, not a transfer.
- No fluid simulation — pour success is a geometric pose proxy.
- Live perception uses ground-truth sim state; the separately-validated vision pipeline
  isn't yet wired to the camera feed.
- The full-pipeline 10-seed evaluation report needs a fresh run before submission.
- NPU inference is ~19× slower than CPU for this model — no quantization attempted yet.

Full limitations list (9 more items, each with the same file-level specificity) →
[deep dive](docs/TECHNICAL_DEEP_DIVE.md#full-limitations-list).

## Future work

Scale demonstrations for `pick`/`place`/`pour`, wire the vision pipeline into the live camera
feed, evaluate a language-conditioned policy (e.g. SmolVLA), measure closed-loop ACT task
success, INT8/NNCF quantization for NPU throughput, consume the planner's pose/force output
in the executor, real SO-101 hardware deployment. Details → [deep dive](docs/TECHNICAL_DEEP_DIVE.md#full-limitations-list).

<p align="center">
  <img src="docs/visualizations/divider.svg" width="100%" height="4">
</p>

## Author

**Muhammad Azeem Azam**
- GitHub: [@MUHAMMAD-AZEEM-AZAM](https://github.com/MUHAMMAD-AZEEM-AZAM)

<p align="center">
  <img src="docs/visualizations/divider.svg" width="100%" height="4">
</p>

## Acknowledgements

**Intel** and **lablab.ai** for the [Physical AI Online Challenge](https://lablab.ai/ai-hackathons/ai-infra-summit-hackathon/)
and OpenVINO/Core Ultra tooling · **Hugging Face LeRobot** for ACT and `LeRobotDataset` ·
**TheRobotStudio / SO-ARM100** for the SO-101 models used directly in `assets/SO-ARM100/` ·
**MuJoCo** · **Anthropic Claude** and **Speechmatics**.

## License

MIT — see [LICENSE](LICENSE).
