#!/usr/bin/env bash
# INFORMATIONAL, NON-GATING: can clang-tidy 14 parse one of this codebase's
# device (.cu) TUs at all, translated to `--cuda-host-only`?
#
# MEASURED ANSWER (2026-08-28, clang-tidy 14.0.0, decode.cu, compdb from an
# ordinary `pip install -e .` build): PARTIALLY. clang's CUDA front end DOES
# reach and diagnose this codebase's OWN code -- real
# `bugprone-narrowing-conversions` hits on `const int tid = threadIdx.x;` and
# `bugprone-implicit-widening-of-multiplication-result` hits on pointer
# arithmetic like `fold + kDecodeWarps * (cols + 3)` -- but it does not
# complete a clean parse of the whole TU:
#   1. clang 14's bundled CUDA wrapper headers are INCOMPATIBLE with this
#      host's installed CUDA toolkit's libcu++ (`<cuda/std/...>`, reached
#      transitively through torch/c10's CUDA headers): "CUDA device code
#      does not support variadic functions", "no template named 'texture'"
#      (from `__clang_cuda_texture_intrinsics.h`). ~99 such errors on
#      decode.cu, all inside toolkit headers, none in this codebase's files.
#      This is a clang14-vs-newer-CUDA-toolkit version mismatch, not
#      something a flag or a `.clang-tidy` check-set change fixes.
#   2. A small number of genuine device-only-intrinsic gaps: `__ldcg` (a
#      cache-global load) has no host-mode overload under `--cuda-host-only`,
#      so a real call site to it is a hard error there even though it is
#      correct device code.
# The default `-ferror-limit=20` aborts the whole run after the toolkit-header
# errors; `-ferror-limit=0` gets further but still cannot finish cleanly.
#
# CONCLUSION: no device TU is part of the clang-tidy gate (run_clang_tidy.sh).
# This script exists so the attempt is reproducible without re-deriving the
# flag translation, and so a future clang/CUDA toolkit pairing that fixes (1)
# can be checked by rerunning it -- it is deliberately NOT wired into
# pre-commit or CI.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CUDA_PATH="$(python3 "$ROOT/tools/dev.py" get cuda-home)"
CC_JSON="${1:-$ROOT/compile_commands.json}"

if [[ ! -f "$CC_JSON" ]]; then
  echo "try_clang_tidy_device_tu.sh: no compile database at $CC_JSON" >&2
  echo "  Generate one first: tools/lint/compdb.sh" >&2
  exit 1
fi

src_from_db=$(python3 -c "
import json
db = json.load(open('$CC_JSON'))
for e in db:
    if e['file'].endswith('decode.cu'):
        print(e['file']); break
")
if [[ -z "$src_from_db" ]]; then
  echo "try_clang_tidy_device_tu.sh: decode.cu not in $CC_JSON" >&2
  exit 1
fi

rel="${src_from_db#*/csrc/}"
target="$ROOT/csrc/$rel"

flags=$(python3 - "$CC_JSON" "$src_from_db" "$ROOT" "$CUDA_PATH" << 'PYEOF'
import json, shlex, sys
cc_json, src, root, cuda_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
db = json.load(open(cc_json))
entry = next(e for e in db if e["file"] == src)
old_root = entry["file"].split("/csrc/")[0]
cmd = entry["command"].replace(old_root, root)
toks = shlex.split(cmd)[1:]  # drop the nvcc wrapper itself
skip_with_arg = {"-MF", "-o", "--dependency-output"}
drop_prefixes = ("--threads", "--ptxas-options", "-gencode", "-Xfatbin")
drop_bare = {"-MMD", "-c", "--generate-dependencies-with-compile", "-MP",
             "--expt-relaxed-constexpr", "-lineinfo", "-compress-all"}
out = []
skip_next = False
for t in toks:
    if skip_next:
        skip_next = False
        continue
    if t in skip_with_arg:
        skip_next = True
        continue
    if t in drop_bare:
        continue
    if t.startswith(drop_prefixes):
        continue
    if t == "--compiler-options":
        skip_next = True
        continue
    if t.endswith(".cu"):
        continue
    out.append(t)
out += ["-x", "cuda", "--cuda-host-only",
        f"--cuda-path={cuda_path}", "--cuda-gpu-arch=sm_80"]
print(" ".join(shlex.quote(x) for x in out))
PYEOF
)

echo "== clang-tidy (informational, --cuda-host-only): ${target#"$ROOT"/} =="
# shellcheck disable=SC2086
clang-tidy "$target" --header-filter='csrc/rola/.*' -- $flags -ferror-limit=0
echo "(exit $? -- expected to be non-clean; see this script's header)"
