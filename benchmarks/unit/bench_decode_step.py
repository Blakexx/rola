"""UNIT BENCH: one carried single-token decode step.

Invoked BARE (`bench.discipline.disciplined` takes `gpu_lock()` itself):

    python benchmarks/unit/bench_decode_step.py --tier landing
"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "benchmarks"))
    sys.path.insert(0, str(root))
    from bench.driver import main as drive

    return drive({"decode_step"}, description=__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
