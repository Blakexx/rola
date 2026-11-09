# `tools/kernels.py` — the kernel list

ONE LIST, DERIVED FROM THE TREE, of what the device build compiles and what kernels it defines. The graph's unit
list (`csrc/rola/units.bzl`, which `tools/gen_shards.py` writes) takes its translation units from it, `tools/ratify.py` compiles the same units and names each manifest entry's shard from it,
and the gates take their targets from it -- so a kernel added under `csrc/rola/src` is built, ratified and
gated without anyone adding it to a list, and no tool can hold a second enumeration that drifts. The one it replaced
did: `ratify.py`'s hand-written tuple of units had lost three kernels (`liveness_pass`, `carry_stamp_kernel`,
`sm_clock_kernel`), which were built and shipped and never ratified.

## <a id="units"></a>Units

`hand_units()` is every `.cu` under `csrc/rola/src` outside `instantiations/`; `arm_units(arms)` is the per-arm carry
units `gen_shards` generates there, for the arms named (every declared arm by default -- the build passes its arm
selection, so an arm-subset build compiles a file subset). `units(arms)` is both. A unit with no kernel (the host
seams `geom.cu`, `arch_runtime.cu`, `vmm_owner.cu`) is still a unit: the build compiles it, and ratify's compile of
it records no entry. `runtime_units()` is the hand units under `common/` that define no kernel (the card's facts, the
geometry, the selection): every kernel's host side calls them, so an instrument's library links them beside its own
kernels (`//csrc/rola:runtime`, [`build-graph.md`](../../build-graph.md#compile)).

`closure(unit)` is the files under `src` a unit compiles -- itself and every quoted `#include` it reaches, resolved
against the including file's directory and then `src`, the build's include root -- and `units_of(kernel)` the units
whose closure holds the kernel's definition: where its SASS is.

## <a id="kernels"></a>Kernels

`kernels()` is every `__global__` function definition in a `.cu`/`.cuh` under `csrc/rola/src`: its name, its FAMILY
(the directory under `src/`), its file and line, and whether it is a template. `parse()` reads one file with its
comments and string literals blanked (`blank()`, which keeps every offset and line), matches `__global__`, an
optional `__launch_bounds__(...)` holding one level of nested parentheses, `void` and the name; a definition is a
template when the declaration run before it (back to the last `;`, `{` or `}`) ends in `template <...>`.

## <a id="check-manifest"></a>The manifest check

`check_manifest(arch)` compares the source's kernel names with the names every toolchain's ratified `sm_<arch>.json`
records (each mangled key demangled by `c++filt`; `function_name()` takes the identifier before the depth-0 argument
list, template arguments cut). A source kernel no manifest records is UNRATIFIED; a recorded name no source defines is
STALE. `--check-manifest` prints both and exits nonzero on either.
