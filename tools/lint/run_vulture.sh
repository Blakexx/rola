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
# silently skipping if it is not on PATH -- the graph's environment carries
# it (requirements.lock pins vulture 2.16), on PATH for the graph's lints.
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
  echo "  run it through the graph: bazel test //:ratchet_vulture   (requirements.lock pins vulture 2.16)" >&2
  exit 1
fi

set +e
# `--exclude` keeps tools/lint/fixtures/**'s deliberate "must fire" violations
# out of the real report (K46's ast-grep bug: a fixture re-triggering the
# gate it exists to test). `bazel/` is IN THE SCAN because its action scripts are CALLERS: the record, the arm
# gate and the flags call the tools' gates and lists.
out=$(vulture --exclude '*/tools/lint/fixtures/*' \
       rola/ tools/ tests/ measure/ bazel/ tools/lint/vulture_whitelist.py 2>&1)
set -e

n=$(printf '%s\n' "$out" | grep -c . || true)
echo "$out"
echo "run_vulture.sh: $n finding(s) (whitelist: tools/lint/vulture_whitelist.py)"
exit 0
