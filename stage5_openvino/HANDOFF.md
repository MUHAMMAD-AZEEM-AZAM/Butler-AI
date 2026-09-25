# Stage 5 — OpenVINO Export and Benchmarking Baseline

`stage5_openvino/export_act.py` exports our ACT checkpoint to OpenVINO IR and compares it against
PyTorch on every device this machine actually exposes. It is a **compatibility test and measured
baseline**, not a benchmark report.

> **The checkpoint used here has NOT been validated for closed-loop robot control.**
> It is a 200-step smoke run over 6 `open_drawer` episodes. It has never driven the simulated
> robot, no success rate has ever been measured with it, and every recorded evaluation to date
> used the scripted primitives, not this policy. Do not describe it as a working table-setting
> policy or quote it as evidence of task capability.

`stage5_openvino/benchmark.py` remains the original stub returning hardcoded values
(`latency_ms=42.7`, `throughput=23.4`). **Nothing in this document comes from it, and those
numbers must not reach any report.** Replacing that stub is the natural next task.

## Reproduction

```powershell
cd <repo root>
# real TRAIN frame
.\.venv-learned\Scripts\python.exe -m stage5_openvino.export_act `
  --checkpoint "outputs\train\act_open_drawer_smoke\checkpoints\000200\pretrained_model" `
  --dataset "data\butler_demos\open_drawer" --frame-index 0 `
  --out "outputs\openvino\act_open_drawer_real_train" --repeats 20 --warmup 3

# held-out VAL frame (first frame of the first val episode)
.\.venv-learned\Scripts\python.exe -m stage5_openvino.export_act `
  --checkpoint "outputs\train\act_open_drawer_smoke\checkpoints\000200\pretrained_model" `
  --dataset "data\butler_demos\open_drawer" --frame-index 1058 `
  --out "outputs\openvino\act_open_drawer_real_val" --repeats 20 --warmup 3
```

Each run writes the IR (`act_policy.xml` / `.bin`) plus a machine-readable `export_report.json`
into `--out`.

**Always pass `--dataset`.** Without it the tool falls back to a synthetic all-zeros observation
and labels the run `SYNTHETIC zeros`. That input is not representative and produces NaN on fp16
hardware — see *Limitations*.

## Requirements

| | |
|---|---|
| Python | 3.12 (3.12.10 used) |
| torch | 2.11.0+cpu |
| openvino | 2026.3.1 (already listed in `requirements.txt`) |
| lerobot | 0.6.1 |
| numpy | 2.2.6 |

**Checkpoint:** a LeRobot ACT `pretrained_model` directory containing `config.json`,
`model.safetensors`, and the `policy_preprocessor*` / `policy_postprocessor*` files. The
processors are required — they are used, not reimplemented.

**Dataset:** a LeRobotDataset v3.0 directory (`meta/info.json` + `data/*.parquet` + videos),
used only to draw one real observation. Produced by
`python -m stage3_policy.learned.convert_demos` and checkable with
`python -m stage3_policy.learned.validate_dataset`.

Neither the checkpoint nor the dataset is in git. See *Artifacts shared separately*.

## What is exported, and what the timing covers

Exported: `ACTPolicy.predict_action_chunk` — the network mapping one observation to a
`(1, 100, 12)` action chunk.

- Timing is **per action chunk**, not per action. With `n_action_steps = 100` the network runs
  once per 100 emitted actions (`select_action()` pops the rest from a queue). Per-action cost is
  roughly chunk ÷ 100 (~0.3 ms on CPU), but the chunk figure is the real inference cost.
- **Preprocessing and postprocessing are EXCLUDED from the timed region.** They run once, outside
  it, identically for both paths, using the checkpoint's own MEAN_STD processors. Accuracy is
  compared after the same unnormalizer, so the comparison is end to end.
- **Compilation is EXCLUDED** (`convert_model` / `compile_model` happen before timing; export
  takes ~1.5 s).
- **Warm-up is EXCLUDED**: 3 warm-up iterations, then 20 measured. Batch size 1.

Tracing leaves shapes dynamic (`state [?,?]`, `image [?,3,?,?]`); the tool reshapes them to
`[1,12]` and `[1,3,480,640]`, which the NPU plugin requires.

## Devices — queried, not assumed

```
available_devices: ['CPU', 'NPU']
  CPU -> Intel(R) Core(TM) Ultra 7 270K Plus
  NPU -> Intel(R) AI Boost
```

**There is no GPU target.** The discrete AMD Radeon RX 9060 XT is not an OpenVINO device and no
Intel iGPU is exposed. Any claim of GPU acceleration would be false on this machine.

## Measured results

Model: 51,609,484 parameters. IR: FP32, 136.94 MB. 20 repeats, 3 warm-up, batch 1.

| Input | Path | Device | Mean ms | Median | Stdev | Finite | Final-action max abs diff vs PyTorch |
|---|---|---|---|---|---|---|---|
| train frame | PyTorch | CPU fp32 | 32.442 | 33.482 | 2.266 | yes | — |
| train frame | OpenVINO | CPU fp32 | 30.044 | 30.115 | 1.170 | yes | 5.7e-07 |
| train frame | OpenVINO | NPU fp16 | 571.768 | 573.539 | 4.761 | yes | 5.5e-04 rad |
| val frame | PyTorch | CPU fp32 | 29.475 | 29.680 | 1.208 | yes | — |
| val frame | OpenVINO | CPU fp32 | 30.500 | 30.425 | 0.641 | yes | 4.2e-07 |
| val frame | OpenVINO | NPU fp16 | 571.681 | 573.618 | 4.609 | yes | 1.0e-03 rad |

**CPU** matches PyTorch to ~5e-07 rad — equivalent within fp32 rounding. The export path is sound.

**NPU** produces correct output; its fp16 error on the final action is 5e-04 to 1e-03 rad
(~0.03–0.06°), acceptable for joint targets. It is ~19x slower than CPU for this model.

## Limitations

1. **No speedup is claimed.** CPU: 30.0 vs 32.4 ms (train) but 30.5 vs 29.5 ms (val) — opposite
   directions. At 20 repeats PyTorch and OpenVINO CPU are indistinguishable. Do not quote a
   speedup from this data.
2. **NPU is ~19x slower** than CPU here (~572 ms vs ~30 ms). A 52M-parameter model with a
   640x480 ResNet-18 encoder is not what this NPU is fastest at.
3. **Synthetic inputs break fp16.** Arm B is parked throughout `open_drawer`, so its joints have
   near-zero training std (smallest 1.83e-10). MEAN_STD divides by that std. A real frame sits at
   its mean, so the numerator is ~0 and the normalized value stays at O(10) (measured \|max\| 16.47,
   0 of 12 dims exceeding fp16). A synthetic zero is 0.1 rad from the mean, giving ~1e7 — past
   fp16's 65504 limit, overflowing to inf on the NPU and NaN through the output. **This is why
   `--dataset` matters.** Nothing is clamped and no normalization was altered; doing so would
   change model behaviour.
4. **Residual risk:** with std ~1e-10, a genuinely out-of-distribution arm-B reading at inference
   would still overflow fp16. The robust fix is training data where both arms move, not an input
   clamp.
5. **Not attempted:** INT8 / NNCF quantization, `benchmark_app`, throughput mode, multi-request
   async, any long benchmark.
6. **Scope of the accuracy check:** equivalence is established between PyTorch and OpenVINO on the
   same machine and inputs. It says nothing about behaviour on the robot.

## Suggested next steps

1. Retrain on episodes where both arms move (multi-skill, or `pick_mug` for arm B), then re-run
   this tool. That removes the near-zero-std fragility at its source.
2. Replace the `benchmark.py` stub so no hardcoded number can reach a report.
3. Only then: INT8 via NNCF, and a longer benchmark with throughput mode.
4. Evaluating a learned policy in the simulator needs the motor-policy hook (P5 in
   `stage3_policy/CONTRACT_PROPOSAL.md`), which does not exist yet.

## Artifacts shared separately (not in git)

- `outputs/openvino/` — IR files and `export_report.json` per run, plus the corrected
  `STAGE5_NOTES.md`
- `outputs/train/act_open_drawer_smoke/` — the ACT smoke checkpoint (~197 MB)
- `data/butler_demos/open_drawer/` — the converted LeRobotDataset v3.0
- `data/smoke_demos/` — the raw demonstration bundles it was built from
