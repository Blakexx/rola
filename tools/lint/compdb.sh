#!/usr/bin/env bash
# Generate compile_commands.json for clangd/clang-query/clang-tidy.
#
# The build directory is repo-relative and FIXED (docs/build.md #sccache):
# `pip install -e . --no-build-isolation` drives setuptools' build_ext into
# `build/temp`, and that is where ninja's build.ninja (and therefore its
# compdb) lives. This script does not build anything itself -- it only asks
# an ALREADY-BUILT tree for its compile database, because building is
# expensive (the AOT closed-world build: ~7 min clean, both archs -- see
# docs/build.md's cost table) and this is a lint convenience, not a gate.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="$ROOT/build/temp"
OUT="${1:-$ROOT/compile_commands.json}"

if [[ ! -f "$BUILD_DIR/build.ninja" ]]; then
  echo "compdb.sh: no build.ninja under $BUILD_DIR" >&2
  echo "  Run 'pip install -e . --no-build-isolation' first (docs/build.md)." >&2
  exit 1
fi

ninja -C "$BUILD_DIR" -t compdb > "$OUT"
echo "compdb.sh: wrote $OUT ($(grep -c '"file"' "$OUT" || true) entries)"
