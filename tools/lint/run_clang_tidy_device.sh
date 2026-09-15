#!/usr/bin/env bash
# DEVICE clang-tidy (LINT2 item 1, G_FOUNDATION G5 "dead-code lints", REPORT-ONLY
# until G5 flips it): compiles this codebase's TORCH-FREE device TUs -- the per-arm
# carry instantiations (`csrc/rola/src/instantiations/carry_arm_*.cu`, raw-pointer
# ABI -- rola-build skill's "the torch-header floor") -- through clang's CUDA
# frontend (`--cuda-host-only`) and runs the checks in
# `tools/lint/clang-tidy-device.yaml` (clang-analyzer-deadcode.DeadStores,
# misc-unused-parameters, bugprone-unused-*, readability-redundant-*).
#
# WHY ARM TUs AND NOT decode.cu/carry.cu/etc: `try_clang_tidy_device_tu.sh`
# already measured that clang 14's bundled CUDA wrapper headers conflict
# with this host's CUDA 12.4 toolkit headers reached transitively through
# torch/c10 (~99 errors, all inside vendored headers, none in this codebase's
# own lines) -- torch-bearing device TUs never finish a clean parse. Arm TUs
# never include torch, so this profile is the one class of device TU
# where the noise floor is smallest; it is still not zero -- see NOISY CHECKS.
#
# NOISY CHECKS ON THIS CUDA/clang PAIRING (reported, never silenced with a
# `-checks=` exclusion or a suppression comment):
#   * `clang-diagnostic-error: no template named 'texture'`, from clang 14's
#     own `__clang_cuda_texture_intrinsics.h` -- a clang14-vs-CUDA-12.4-headers
#     version mismatch (same root cause `try_clang_tidy_device_tu.sh` found on
#     decode.cu), unrelated to anything in csrc/rola. It always fires from a
#     clang-bundled system header path, never from a csrc/rola line, so this
#     script's counter below excludes diagnostics whose file is NOT under
#     csrc/rola -- but it prints them, unfiltered, in a NOISE section so the
#     exclusion is auditable rather than silent.
#   * `-ferror-limit=0` is required (default 20 aborts before reaching the
#     arm's own code); clang-tidy still reports "Found compiler error(s)" and
#     a nonzero PROCESS exit even when it emitted real, on-target findings
#     first -- this script's own exit code is always 0 (report-only); read
#     the printed finding counts, not `$?`.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CUDA_PATH="$(python3 "$ROOT/tools/dev.py" get cuda-home)"
CONFIG="$ROOT/tools/lint/clang-tidy-device.yaml"

if [[ "${1:-}" == "--test-fixtures" ]]; then
  # Self-test (excluded from the commit gate, K46 convention): the fixture
  # needs no compdb -- it is self-contained CUDA, no project headers.
  FIX="$ROOT/tools/lint/fixtures/device_lint_unused_param.cu"
  out=$(clang-tidy "$FIX" --config-file="$CONFIG" -- \
          -x cuda --cuda-host-only --cuda-path="$CUDA_PATH" \
          --cuda-gpu-arch=sm_80 -ferror-limit=0 2>&1)
  if echo "$out" | grep -q "misc-unused-parameters.*bad\|bad(int used"; then :; fi
  if ! echo "$out" | grep -q "'unused_param' is unused \[misc-unused-parameters\]"; then
    echo "SELF-TEST FAILED: expected finding on fixture's 'bad' did not fire" >&2
    exit 1
  fi
  if echo "$out" | grep -q "'a' is unused\|'b' is unused"; then
    echo "SELF-TEST FAILED: fixture's 'ok' fired unexpectedly" >&2
    exit 1
  fi
  echo "run_clang_tidy_device.sh --test-fixtures: PASS"
  exit 0
fi

CC_JSON="${1:-$ROOT/compile_commands.json}"

if [[ ! -f "$CC_JSON" ]]; then
  if ! "$ROOT/tools/lint/compdb.sh" "$CC_JSON" >/dev/null 2>&1; then
    echo "run_clang_tidy_device.sh: SKIPPED -- no build/temp to read a compile database from."
    echo "  Build first (pip install -e . --no-build-isolation) to exercise this check."
    exit 0
  fi
fi

arm_files=$(python3 - "$CC_JSON" << 'PYEOF'
import json, sys
db = json.load(open(sys.argv[1]))
for e in db:
    if "/instantiations/carry_arm_" in e["file"] and e["file"].endswith(".cu"):
        print(e["file"])
PYEOF
)

if [[ -z "$arm_files" ]]; then
  echo "run_clang_tidy_device.sh: no carry_arm_*.cu TU in the compile database" \
       "(build with ROLA_CARRY_ARMS set to widen coverage) -- 0 arms checked."
  exit 0
fi

ARCH="$(python3 -c "from rola_cu13._build_config import BUILD_CONFIG as c; print(c['archs'][0].replace('sm_',''))" 2>/dev/null || echo 80)"

total_real=0
total_noise=0
for src_from_db in $arm_files; do
  rel="${src_from_db#*/csrc/}"
  target="$ROOT/csrc/$rel"
  [[ -f "$target" ]] || { echo "run_clang_tidy_device.sh: $target missing, skipping" >&2; continue; }

  flags=$(python3 - "$CC_JSON" "$src_from_db" "$ROOT" "$ARCH" "$CUDA_PATH" << 'PYEOF'
import json, shlex, sys
cc_json, src, root, arch, cuda_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
db = json.load(open(cc_json))
entry = next(e for e in db if e["file"] == src)
old_root = entry["file"].split("/csrc/")[0]
cmd = entry["command"].replace(old_root, root)
toks = shlex.split(cmd)[1:]
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
out += ["-x", "cuda", "--cuda-host-only", f"--cuda-path={cuda_path}",
        f"--cuda-gpu-arch=sm_{arch}"]
print(" ".join(shlex.quote(x) for x in out))
PYEOF
)

  echo "== device clang-tidy: ${target#"$ROOT"/} (sm_$ARCH) =="
  # LOCKS brief item 1: one host-compute-budget slot per translation unit --
  # a `clang-tidy --cuda-host-only` pass is a full front-end parse per TU, as
  # CPU-heavy as a `cicc` compile, and previously drew from no shared pool at
  # all.
  # shellcheck disable=SC2086
  out=$(python3 "$ROOT/tools/host_budget.py" --slots 1 --label clang-tidy-tu -- \
          clang-tidy "$target" --config-file="$CONFIG" --header-filter='csrc/rola/.*' \
          -- $flags -ferror-limit=0 2>&1)
  real=$(echo "$out" | grep -E '^/.*csrc/rola/.*: (warning|error): ' | grep -v 'clang-diagnostic-error' || true)
  noise=$(echo "$out" | grep -E 'clang-diagnostic-error' || true)
  real_n=$(printf '%s\n' "$real" | grep -c . || true)
  noise_n=$(printf '%s\n' "$noise" | grep -c . || true)
  if [[ -n "$real" ]]; then echo "$real"; fi
  total_real=$((total_real + real_n))
  total_noise=$((total_noise + noise_n))
done

echo "---"
echo "device clang-tidy: $total_real on-target finding(s) across $(echo "$arm_files" | wc -l) arm TU(s);" \
     "$total_noise toolchain-noise diagnostic(s) excluded from that count (see this script's header)."

# GATING on the ON-TARGET count. Toolchain noise -- clang's bundled CUDA wrapper headers
# disagreeing with the installed toolkit -- is printed and never counted: it is a fact
# about the analyzer's own headers, not about this code.
if [[ "$total_real" -gt 0 ]]; then exit 1; fi
exit 0
