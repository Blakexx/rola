#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""a DAG tuple (`rola.engine.dags.*`) as Graphviz DOT, and an SVG.

`rola/engine/runner.py`'s `Node.needs`/`.gives` already declare the edges
`docs/internals/engine/*.md`'s prose diagrams describe by hand; this renders
the SAME edges MECHANICALLY from the committed tuple, so a doc page and the
DAG it describes can be checked against each other rather than trusted by
eye. Not wired into any gate -- a mismatch between the prose and this render
is a review-time finding (`rola-review`'s pointer), not a CI failure.

Usage:
    python tools/build/dag_to_dot.py rola.engine.dags.chunk_dag:CHUNK_DAG \
        --svg docs/internals/engine/chunk_dag.svg
    python tools/build/dag_to_dot.py rola.engine.dags.decode_dag:DECODE_STEP \
        --svg docs/internals/engine/decode_dag.svg
"""
from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
import sys


def load_dag(spec: str):
    """`spec` is `module.path:ATTR_NAME` -- the DAG tuple itself."""
    if ":" not in spec:
        raise SystemExit(f"expected module.path:ATTR_NAME, got {spec!r}")
    mod_name, attr = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, attr)


def to_dot(dag, name: str) -> str:
    lines = [f"digraph {name} {{", "  rankdir=TB;", '  node [shape=box, fontname="monospace"];']
    #: A fact's PRODUCER is the first node whose `gives` names it (a DAG is
    #: import-time-validated to be walkable in tuple order, `validate_dag`, so
    #: this lookup never has to search past the current node).
    producer_of: dict[str, str] = {}
    for step in dag:
        lines.append(f'  "{step.name}";')
        for fact in step.gives:
            producer_of[fact] = step.name
    for step in dag:
        for need in step.needs:
            src = producer_of.get(need, f"seed:{need}")
            if src.startswith("seed:"):
                lines.append(f'  "{src}" [shape=ellipse, style=dashed];')
            lines.append(f'  "{src}" -> "{step.name}" [label="{need}"];')
    lines.append("}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dag_spec", help="module.path:ATTR_NAME of the DAG tuple")
    ap.add_argument("--dot", default=None, help="write the raw DOT source here too")
    ap.add_argument("--svg", default=None, help="render to SVG here (needs `dot` on PATH)")
    args = ap.parse_args()

    dag = load_dag(args.dag_spec)
    name = args.dag_spec.split(":", 1)[1]
    dot_src = to_dot(dag, name)

    if args.dot:
        with open(args.dot, "w") as f:
            f.write(dot_src)
        print(f"wrote {args.dot}")

    if args.svg:
        if shutil.which("dot") is None:
            print("no `dot` (graphviz) on PATH -- cannot render --svg; use --dot for "
                  "the raw source instead", file=sys.stderr)
            return 1
        proc = subprocess.run(["dot", "-Tsvg", "-o", args.svg], input=dot_src, text=True)
        if proc.returncode != 0:
            return proc.returncode
        print(f"wrote {args.svg}")

    if not args.dot and not args.svg:
        print(dot_src)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
