#!/usr/bin/env bash
# clang-tidy over this codebase's HOST-ONLY translation units, using the
# compile database `tools/lint/compdb.sh` writes from an already-built tree.
# This is the GATE script (pre-commit/CI); `tools/lint/try_clang_tidy_device_tu.sh`
# is a separate, non-gating exploration of whether a device TU can be added.
#
# SCOPE. A "host-only TU" here means a `.cpp` file compiled by the C++
# compiler directly (as opposed to nvcc): the compdb's `file` entries whose
# suffix is `.cpp`. Right now that is exactly one file, `csrc/rola/rola_api.cpp`.
# Every other translation unit is a `.cu` compiled by nvcc; clang-tidy 14
# cannot reliably parse this codebase's device TUs even in `--cuda-host-only`
# mode (see try_clang_tidy_device_tu.sh's header for the measured reason), so
# they are not part of this gate.
#
# The compdb is built from WHATEVER checkout produced `build/temp` (docs/build.md
# names it as the fixed, repo-relative build dir) -- which may not be this
# worktree if you're iterating on a branch. This script rewrites the compdb's
# absolute source-tree prefix to THIS repo's root before invoking clang-tidy,
# so it lints the file actually on disk here, not the one that was built.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CC_JSON="${1:-$ROOT/compile_commands.json}"

if [[ ! -f "$CC_JSON" ]]; then
  # Blake ruling, 2026-08-28: "pre-commit can never be green by
  # absence." Try to generate one from an already-built tree
  # (`docs/build.md`'s documented, fixed `build/temp` dir); if there is
  # nothing to generate it FROM, FAIL with the exact command to run, never
  # exit 0 on a check this script did not actually perform.
  if ! "$ROOT/tools/lint/compdb.sh" "$CC_JSON" >/dev/null 2>&1; then
    echo "run_clang_tidy.sh: FAILED -- no build/temp to read a compile database from." >&2
    echo "  Run this first, then re-run this script:" >&2
    echo "    pip install -e . --no-build-isolation" >&2
    exit 1
  fi
fi

overall_status=0

host_files=$(python3 - "$CC_JSON" << 'PYEOF'
import json, sys
db = json.load(open(sys.argv[1]))
for e in db:
    if e["file"].endswith(".cpp"):
        print(e["file"])
PYEOF
)

if [[ -z "$host_files" ]]; then
  echo "run_clang_tidy.sh: no .cpp (host-only) TU found in the compile database" >&2
  exit 1
fi

for src_from_db in $host_files; do
  # The compdb's own source-tree prefix (everything up to csrc/) -- rewrite it
  # to THIS repo's root so we lint the on-disk file, using that build's flags.
  rel="${src_from_db#*/csrc/}"
  target="$ROOT/csrc/$rel"
  if [[ ! -f "$target" ]]; then
    echo "run_clang_tidy.sh: $target not found under this worktree, skipping" >&2
    overall_status=1
    continue
  fi
  flags=$(python3 - "$CC_JSON" "$src_from_db" "$ROOT" << 'PYEOF'
import json, shlex, sys
cc_json, src, root = sys.argv[1], sys.argv[2], sys.argv[3]
db = json.load(open(cc_json))
entry = next(e for e in db if e["file"] == src)
old_root = entry["file"].split("/csrc/")[0]
cmd = entry["command"].replace(old_root, root)
toks = shlex.split(cmd)
skip_flags_with_arg = {"-MF", "-o"}
skip_flags_bare = {"-MMD", "-c"}
out = []
skip_next = False
for t in toks[1:]:
    if skip_next:
        skip_next = False
        continue
    if t in skip_flags_with_arg:
        skip_next = True
        continue
    if t in skip_flags_bare:
        continue
    if t.endswith(".cpp"):
        continue
    out.append(t)
print(" ".join(shlex.quote(x) for x in out))
PYEOF
)
  echo "== clang-tidy: ${target#"$ROOT"/} =="
  # shellcheck disable=SC2086
  clang-tidy "$target" -- $flags
  status=$?
  if [[ $status -ne 0 ]]; then
    overall_status=1
  fi
done

exit $overall_status
