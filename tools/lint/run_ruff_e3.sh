#!/usr/bin/env bash
# RUFF BLANK-LINE SPACING, GATING (a finding fails the commit):
# pycodestyle's blank-line-spacing family, preview-gated in ruff
# 0.16.5 (see pyproject.toml's `[tool.rola_lint.preview]` comment for the
# measurement). Reads the code list from THAT table rather than hardcoding
# "E3" a second time, so the two stay in one place.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# A tiny, deliberately non-tomllib extraction (this hook runs under
# `language: system`, whatever `python3` is on PATH -- tomllib needs >=3.11,
# and pre-commit's own invoking shell is not guaranteed to have that; a real
# TOML parser would be overkill for one single-line array this file owns).
CODES=$(python3 -c "
import re
text = open('pyproject.toml').read()
m = re.search(r'ruff_preview_select\s*=\s*\[(.*?)\]', text, re.S)
codes = re.findall(r'\"([^\"]+)\"', m.group(1))
print(','.join(codes))
")

set +e
out=$(ruff check --preview --select "$CODES" --exit-zero --output-format=concise .)
set -e

n=$(printf '%s\n' "$out" | grep -cE ':[0-9]+:[0-9]+: E30[0-9] ' || true)
echo "$out"
echo "run_ruff_e3.sh: $n finding(s) for codes [$CODES]"

# GATING. The codes stay in their own table and this script keeps its own invocation
# because ruff preview-gates this family: `ruff check`'s select cannot carry it without
# turning preview on for every other family too, which would gate on rules nobody read.
if [[ "$n" -gt 0 ]]; then exit 1; fi
exit 0
