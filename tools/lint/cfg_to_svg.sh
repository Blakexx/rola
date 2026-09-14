#!/usr/bin/env bash
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
#
# One kernel's control-flow graph, from SASS to a viewable SVG.
#
# `nvdisasm -cfg` emits Graphviz DOT directly; this script is the one-line
# wrapper (`nvdisasm -cfg <cubin> | dot -Tsvg -o <out>.svg`) plus the argument
# checking a raw pipeline leaves to a confusing dot/nvdisasm error instead.
#
# Usage: tools/lint/cfg_to_svg.sh <cubin-or-fatbin> <kernel-name-regex> <out.svg>
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 <cubin-or-fatbin> <kernel-name-regex> <out.svg>" >&2
  echo "  <cubin-or-fatbin>: extract with 'cuobjdump --extract-elf <arch> <so>'" >&2
  echo "                     or pass a .so/.o directly -- nvdisasm reads either." >&2
  exit 2
fi

CUBIN="$1"
KERNEL_RE="$2"
OUT="$3"

command -v nvdisasm >/dev/null || { echo "nvdisasm not on PATH (CUDA toolkit bin/)" >&2; exit 1; }
command -v dot >/dev/null || {
  echo "dot (graphviz) not on PATH. This host's own tooling has no apt-installable" >&2
  echo "graphviz package (no root, not in the base image) -- see docs/build.md for" >&2
  echo "the user-local vendoring recipe (apt-get download + dpkg -x, no install)." >&2
  exit 1
}
[ -f "$CUBIN" ] || { echo "no such file: $CUBIN" >&2; exit 1; }

nvdisasm -cfg "$CUBIN" > /tmp/cfg_to_svg.$$.dot 2>/tmp/cfg_to_svg.$$.err || {
  echo "nvdisasm -cfg failed:" >&2
  cat /tmp/cfg_to_svg.$$.err >&2
  rm -f /tmp/cfg_to_svg.$$.dot /tmp/cfg_to_svg.$$.err
  exit 1
}
if [ -s /tmp/cfg_to_svg.$$.dot ] && ! grep -q "$KERNEL_RE" /tmp/cfg_to_svg.$$.dot; then
  echo "warning: '$KERNEL_RE' matched nothing in nvdisasm's output -- check the" >&2
  echo "kernel name against 'cuobjdump --dump-sass $CUBIN | grep Function'" >&2
fi

dot -Tsvg -o "$OUT" /tmp/cfg_to_svg.$$.dot
rm -f /tmp/cfg_to_svg.$$.dot /tmp/cfg_to_svg.$$.err
echo "wrote $OUT"
