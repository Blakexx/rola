#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""ONE wrapper over `ast-grep scan`, two scopes.

**THE BUG THIS REPLACES.** `sgconfig.yml`'s rules each carry `files:
[csrc/**, tools/lint/fixtures/**]` -- the fixtures glob is there so the rule's
own must-fire fixture is reachable at all, per rule, without a second config
file. But `.pre-commit-config.yaml` invoked `ast-grep scan --config
sgconfig.yml` with NO path argument, which defaults to the whole repo -- so
every commit re-triggered every fixture's deliberate violation as a gating
error. `docs/build.md`'s OWN documented invocation already said `... csrc/`;
only the wired hook had drifted from it. A run of stages bypassed the
hook rather than fix the four-word drift.

**Commit gate (default, no flag):** `ast-grep scan --config sgconfig.yml
csrc` -- csrc/ only. A rule's fixtures/** glob is irrelevant to this
invocation because the scan root excludes it outright, not because the rule
was told to ignore it.

**Fixture self-test (`--test-fixtures`):** `ast-grep scan --config
sgconfig.yml --json tools/lint/fixtures` and asserts every rule id under
`tools/lint/rules/*.yml` appears at least once in the findings -- a rule
whose fixture stops firing (a rewrite that accidentally narrows the pattern)
is a silently disabled rule, which is exactly what "must fire on its own
fixture" is supposed to catch.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
RULES_DIR = ROOT / "tools" / "lint" / "rules"
FIXTURES_DIR = ROOT / "tools" / "lint" / "fixtures"


def rule_ids() -> list[str]:
    return sorted(yaml.safe_load(p.read_text())["id"] for p in RULES_DIR.glob("*.yml"))


def commit_gate() -> int:
    proc = subprocess.run(["ast-grep", "scan", "--config", "sgconfig.yml", "csrc"], cwd=ROOT)
    return proc.returncode


def test_fixtures() -> int:
    ids = rule_ids()
    proc = subprocess.run(
        ["ast-grep", "scan", "--config", "sgconfig.yml", "--json", str(FIXTURES_DIR)],
        cwd=ROOT, capture_output=True, text=True)
    try:
        findings = json.loads(proc.stdout)
    except json.JSONDecodeError:
        print(f"ast-grep --json produced no parseable output (rc={proc.returncode}):\n"
              f"{proc.stdout}{proc.stderr}")
        return 1
    fired = {f["ruleId"] for f in findings}
    missing = [r for r in ids if r not in fired]
    if missing:
        print(f"FIXTURE SELF-TEST FAILED: {len(missing)}/{len(ids)} rule(s) did not fire "
              f"on their own fixture under {FIXTURES_DIR.relative_to(ROOT)}: {missing}")
        return 1
    print(f"FIXTURE SELF-TEST: all {len(ids)} rule(s) fired on their own fixture "
          f"({len(findings)} finding(s) total)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-fixtures", action="store_true",
                     help="assert every rule fires on tools/lint/fixtures/, instead of "
                          "running the csrc/ commit-gate scan")
    args = ap.parse_args()
    return test_fixtures() if args.test_fixtures else commit_gate()


if __name__ == "__main__":
    sys.exit(main())
