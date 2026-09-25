#!/usr/bin/env python3
"""Run the pipeline across randomization seeds and report the success rate.

Works today with stubs: python scripts/evaluate.py --seeds 10
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stage7_eval import evaluate


def seeds_from_config() -> list[int]:
    """Fall back to configs/default.yaml when --seeds is not given."""
    try:
        import yaml

        cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
        return [int(s) for s in cfg["seeds"]]
    except Exception:
        return list(range(10))


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the pipeline over N seeds.")
    parser.add_argument(
        "--seeds",
        type=int,
        default=None,
        help="Number of seeds to run (uses seeds 0..N-1). Default: configs/default.yaml.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON path for the complete evaluation report.",
    )
    args = parser.parse_args()

    seeds = list(range(args.seeds)) if args.seeds is not None else seeds_from_config()
    print(f"Evaluating over {len(seeds)} seeds: {seeds}\n")

    report = evaluate(seeds)
    for r in report.results:
        print(f"  seed {r.seed}: {'PASS' if r.success else 'FAIL'}  {r.details}")

    print(f"\nSuccess rate: {report.successes}/{len(report.seeds)} = {report.success_rate:.0%}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            report.model_dump_json(indent=2),
            encoding="utf-8",
        )
        print(f"Report written to {args.output}")
    return 0 if report.successes == len(report.seeds) else 1


if __name__ == "__main__":
    sys.exit(main())
