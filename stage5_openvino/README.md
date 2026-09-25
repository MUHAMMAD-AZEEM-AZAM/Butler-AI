# stage5_openvino — Intel OpenVINO Model Optimization & Benchmarking
Converts the model to OpenVINO and benchmarks it on Intel hardware. Standalone — not part of the run_pipeline chain.

**`benchmark.py` is still a stub returning hardcoded numbers (`latency_ms=42.7`, `throughput=23.4`). Do not quote them.**

**`export_act.py` is real**: it exports an ACT checkpoint to OpenVINO IR and compares it against PyTorch on every device actually present (CPU and NPU here — there is no GPU target on this machine). See [`HANDOFF.md`](HANDOFF.md) for reproduction commands, dependency versions, measured results, timing scope and limitations.
Input: none (real version reads model from `configs/default.yaml`).
Output: `BenchmarkResult` (`model_name`, `device`, `precision`, `latency_ms`, `throughput`).
Test: `python3 -c "from stage5_openvino import run_benchmark; print(run_benchmark())"`
