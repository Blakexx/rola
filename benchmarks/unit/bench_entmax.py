"""UNIT BENCH: the entmax solves — the producer's batched multi-level solve.

Invoked BARE (`bench.discipline.disciplined` takes `gpu_lock()` itself):

    python benchmarks/unit/bench_entmax.py --tier landing
"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "benchmarks"))
    sys.path.insert(0, str(root))
    from bench.driver import main as drive

    return drive({"entmax_solve"}, description=__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
