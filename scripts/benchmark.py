#!/usr/bin/env python3
"""Print the OpenVINO benchmark results as a table.

Works today with stubs: python scripts/benchmark.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage5_openvino import run_benchmark


def main() -> int:
    result = run_benchmark()
    rows = result.model_dump()

    width = max(len(str(k)) for k in rows)
    print("OpenVINO benchmark")
    print("-" * (width + 28))
    for key, value in rows.items():
        print(f"{key:<{width}}  {value}")
    print("-" * (width + 28))
    return 0


if __name__ == "__main__":
    sys.exit(main())
