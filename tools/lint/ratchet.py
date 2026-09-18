#!/usr/bin/env python3
"""THE RATCHET: a lint the tree does not yet pass gates on what a commit ADDS, and its old findings can only go away.

    python3 tools/lint/ratchet.py work_codes            # the gate: a new finding fails, and so does a fixed one
    python3 tools/lint/ratchet.py work_codes --shrink   # drop the fixed entries from the baseline (it never adds one)
    python3 tools/lint/ratchet.py work_codes --init     # record today's findings; refused when a baseline exists

A lint's findings are its `path:line: message` lines. A finding is known by its path, its message's first sentence and
a digest of the line it names, never by the line number, which any edit above it moves. `tools/lint/baselines/<lint>.json`
holds the findings the tree carried when the lint began gating, with their multiplicity: the backlog. The gate fails on

- a finding the baseline does not hold: fix it, or exempt it the lint's own way (vulture's whitelist, a `reserved:`
  marker), with the reason;
- a baseline entry the lint no longer finds: `--shrink` removes it in the commit that fixed it, so the backlog on disk
  is the backlog in the tree;
- a baseline that holds more of an entry than HEAD's: a backlog only shrinks.

A lint that cannot run (a missing tool, a crash) fails the gate; it never reads as zero findings.
Docs: docs/internals/tools/lint.md#ratchets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINES = ROOT / "tools" / "lint" / "baselines"
LINTS = {
    "work_codes": [sys.executable, "tools/lint/work_codes.py"],
    "vulture": ["bash", "tools/lint/run_vulture.sh"],
    "constant_parameters": [sys.executable, "tools/lint/constant_parameters.py"],
    "drift_guards": [sys.executable, "tools/lint/drift_guards.py"],
    "r9_enforcement": [sys.executable, "tools/lint/r9_enforcement.py"],
    "burst_tier": [sys.executable, "tools/lint/burst_tier.py"],
}
FINDING = re.compile(r"^\s*([^\s:]+):(\d+): (.+)$")


def findings(lint: str) -> Counter:
    done = subprocess.run(LINTS[lint], cwd=ROOT, capture_output=True, text=True)
    if done.returncode:
        raise SystemExit(f"ratchet {lint}: the lint did not run (exit {done.returncode}):\n{(done.stdout + done.stderr)[-1500:]}")
    found: Counter = Counter()
    for line in done.stdout.splitlines():
        match = FINDING.match(line)
        if match:
            path, lineno, message = match.group(1), int(match.group(2)), match.group(3).strip().split(". ")[0]
            source = ROOT / path
            text = source.read_text(errors="replace").splitlines() if source.is_file() else []
            named = text[lineno - 1].strip() if 0 < lineno <= len(text) else ""
            found[(path, message, hashlib.sha256(named.encode()).hexdigest()[:16])] += 1
    return found


def _decode(blob: str) -> Counter:
    return Counter({(e["path"], e["message"], e["line_sha256"]): e["count"] for e in json.loads(blob)["findings"]})


def _encode(lint: str, entries: Counter) -> str:
    rows = [{"path": p, "message": m, "line_sha256": d, "count": n} for (p, m, d), n in sorted(entries.items()) if n > 0]
    return json.dumps({"lint": lint, "findings": rows}, indent=1) + "\n"


def _head(path: Path) -> Counter | None:
    shown = subprocess.run(["git", "show", f"HEAD:{path.relative_to(ROOT)}"], cwd=ROOT, capture_output=True, text=True)
    return _decode(shown.stdout) if shown.returncode == 0 else None


def verdict(found: Counter, baseline: Counter, head: Counter | None) -> tuple[Counter, Counter, Counter]:
    """`(new, fixed, grown)`: findings the baseline lacks, entries no longer found, entries HEAD's baseline lacks.

    A MOVE IS NOT GROWTH: a finding known to HEAD by its message and line digest under a path the baseline no longer
    carries is the same finding after a rename, so a renamed package moves its backlog and does not add to it."""
    if head is None:
        return found - baseline, baseline - found, Counter()
    left_behind = Counter()
    for (path, message, digest), n in (head - baseline).items():
        left_behind[(message, digest)] += n
    grown = Counter()
    for (path, message, digest), n in (baseline - head).items():
        moved = min(n, left_behind[(message, digest)])
        left_behind[(message, digest)] -= moved
        if n - moved:
            grown[(path, message, digest)] = n - moved
    return found - baseline, baseline - found, grown


def _show(title: str, entries: Counter) -> None:
    print(title)
    for (path, message, _digest), n in sorted(entries.items()):
        print(f"  {path}: {message}" + (f"  (x{n})" if n > 1 else ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lint", choices=sorted(LINTS))
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--shrink", action="store_true", help="remove the entries the lint no longer finds")
    mode.add_argument("--init", action="store_true", help="record the current findings as a new lint's baseline")
    a = ap.parse_args()
    path = BASELINES / f"{a.lint}.json"
    found = findings(a.lint)

    if a.init:
        if path.exists():
            raise SystemExit(f"ratchet {a.lint}: {path.relative_to(ROOT)} exists; a baseline is recorded once")
        BASELINES.mkdir(parents=True, exist_ok=True)
        path.write_text(_encode(a.lint, found))
        print(f"ratchet {a.lint}: baseline recorded, {sum(found.values())} finding(s)")
        return 0
    if not path.exists():
        raise SystemExit(f"ratchet {a.lint}: no baseline at {path.relative_to(ROOT)} (python3 tools/lint/ratchet.py "
                         f"{a.lint} --init records one)")
    baseline = _decode(path.read_text())

    if a.shrink:
        kept = baseline & found
        path.write_text(_encode(a.lint, kept))
        print(f"ratchet {a.lint}: baseline {sum(baseline.values())} -> {sum(kept.values())}")
        return 0

    new, fixed, grown = verdict(found, baseline, _head(path))
    if new:
        _show(f"ratchet {a.lint}: {sum(new.values())} NEW finding(s) -- fix each, or exempt it the lint's own way:", new)
    if fixed:
        _show(f"ratchet {a.lint}: {sum(fixed.values())} baseline entr(ies) no longer found -- run "
              f"`python3 tools/lint/ratchet.py {a.lint} --shrink` and commit the smaller baseline:", fixed)
    if grown:
        _show(f"ratchet {a.lint}: the baseline gained {sum(grown.values())} entr(ies) over HEAD's -- a backlog only "
              f"shrinks:", grown)
    total = sum(baseline.values())
    print(f"ratchet {a.lint}: {'FAIL' if new or fixed or grown else 'OK'} ({total} in the backlog)")
    return 1 if new or fixed or grown else 0


if __name__ == "__main__":
    sys.exit(main())
