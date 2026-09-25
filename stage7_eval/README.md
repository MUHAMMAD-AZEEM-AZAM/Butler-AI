# stage7_eval — Multi-seed robustness evaluation harness

Stage 7 runs the integrated pipeline once per seed and reports the aggregate
success rate plus per-seed outcomes. Each run calls `reset_scene(seed)` through
`common.pipeline.run_once`, so the simulation owns the actual randomization.

Run ten configured seeds:

```powershell
python scripts/evaluate.py --seeds 10
```

Save the complete report:

```powershell
python scripts/evaluate.py --seeds 10 --output evaluation_report.json
```

The evaluator counts a seed as successful only when the pipeline returns
success after Stage 6 verification. A pipeline exception or failed verification
is recorded as a failed seed with details; a completed pipeline is not treated
as success automatically.

Standalone API:

```powershell
python -c "from stage7_eval import evaluate; print(evaluate([0, 1, 2]).model_dump_json(indent=2))"
```

Final evaluation still depends on real Stage 2 camera detections, Stage 4
manipulation, and Stage 6 verification. Until those are integrated, failures
are expected and should be reported rather than hidden.
