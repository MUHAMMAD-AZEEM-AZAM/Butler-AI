from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any


def benchmark_simulation(iterations: int = 50, warmup: int = 10) -> Dict[str, Any]:
    """Simulated benchmark used when actual Intel hardware or model export is unavailable.

    This gives a reproducible benchmark script and reporting format for the project.
    """
    start = time.perf_counter()
    for _ in range(warmup):
        _ = 0.0
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = 1.0 / 3.0
    elapsed = time.perf_counter() - t0

    return {
        "device": "simulation",
        "iterations": iterations,
        "warmup": warmup,
        "avg_latency_ms": (elapsed / iterations) * 1000.0,
        "throughput_fps": iterations / elapsed,
        "precision": "float32",
        "notes": "This is a placeholder benchmark; run on Intel hardware for official results.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a model or a simulated workload.")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", type=str, default="benchmark_result.json")
    args = parser.parse_args()

    result = benchmark_simulation(iterations=args.iterations, warmup=args.warmup)
    output_path = Path(args.output)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
