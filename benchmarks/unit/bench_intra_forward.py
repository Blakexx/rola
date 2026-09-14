"""UNIT BENCH: the within-window term at the shared window.

Invoked BARE (`bench.discipline.disciplined` takes `gpu_lock()` itself):

    python benchmarks/unit/bench_intra_forward.py --tier landing
"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "benchmarks"))
    sys.path.insert(0, str(root))
    from bench.driver import main as drive

    return drive({"intra_forward"}, description=__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
