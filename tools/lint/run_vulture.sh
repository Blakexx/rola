#!/usr/bin/env bash
# PYTHON dead-code check, gated by `tools/lint/ratchet.py vulture`:
# vulture over the Python surface, with the whitelist in
# `tools/lint/vulture_whitelist.py` (genuine dynamic uses only, each with a
# one-line reason -- see that file's own docstring). `ruff check --select F401,F841`
# is the exact-match belt (unused imports / unused locals ruff already gates
# elsewhere via its full "F" selection in pyproject.toml); vulture additionally
# catches unused FUNCTIONS, CLASSES and DEAD ATTRIBUTES ruff's F-codes do not.
#
# `vulture` is not a project dependency (a lint tool, never imported by
# shipped code): this script REFUSES per KERNEL_STANDARDS §18 rather than
# silently skipping if it is not on PATH -- install it via the pinned
# `.pre-commit-config.yaml` entry's `additional_dependencies`, or by hand
# (`pip install vulture==2.16`) for a manual run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ "${1:-}" == "--test-fixtures" ]]; then
  # Self-test (excluded from the commit gate, K46 convention).
  if ! command -v vulture >/dev/null 2>&1; then
    echo "run_vulture.sh --test-fixtures: SKIPPED -- vulture not on PATH" >&2
    exit 0
  fi
  out=$(vulture tools/lint/fixtures/vulture_dead_code.py 2>&1 || true)
  if ! echo "$out" | grep -q "dead_function"; then
    echo "SELF-TEST FAILED: expected finding on 'dead_function' did not fire" >&2
    exit 1
  fi
  if echo "$out" | grep -q "used_function"; then
    echo "SELF-TEST FAILED: 'used_function' fired unexpectedly" >&2
    exit 1
  fi
  echo "run_vulture.sh --test-fixtures: PASS"
  exit 0
fi

if ! command -v vulture >/dev/null 2>&1; then
  echo "run_vulture.sh: refusing -- 'vulture' not on PATH." >&2
  echo "  pip install vulture==2.16   (pinned in .pre-commit-config.yaml)" >&2
  exit 1
fi

set +e
# `--exclude` keeps tools/lint/fixtures/**'s deliberate "must fire" violations
# out of the real report (K46's ast-grep bug: a fixture re-triggering the
# gate it exists to test).
# `setup.py` is IN THE SCAN because it is a CALLER: it runs the pre-build and post-build
# gates, and a scan that could not see it reported `tools/build_flags.py`'s DEPFILE_FLAGS
# and `tools/gen_shards.py`'s declaration_matches_table as dead when setup.py calls both.
out=$(vulture --exclude '*/tools/lint/fixtures/*' \
       rola/ tools/ tests/ measure/ setup.py tools/lint/vulture_whitelist.py 2>&1)
set -e

n=$(printf '%s\n' "$out" | grep -c . || true)
echo "$out"
echo "run_vulture.sh: $n finding(s) (whitelist: tools/lint/vulture_whitelist.py)"
exit 0
