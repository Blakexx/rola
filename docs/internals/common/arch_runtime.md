# `common/arch_runtime.cu` — the closed-world arch rule's runtime half

Mirrors `csrc/rola/src/common/arch_runtime.{cu,cuh}`.

`arch_caps.cuh`'s `static_assert(tabulated(kArch))` fixes the compiling target: a
binary refuses to BUILD for a compute capability the table has no row for. That
static assertion cannot see the device it eventually runs on — the compiling
machine and the running machine are not always the same one, and even on one box a
binary is portable to a different card. `check_arch_table()` is the other half: it
runs on the first launch, reads the device the process is actually holding, and
refuses if that architecture was never tabulated or if the driver's own numbers
disagree with the row a compiled derivation was closed over.

<a id="check-arch-table"></a>
## `check_arch_table()`

Keyed on `denseref::arch::tabulated(cc)` and `denseref::arch::caps_of(cc)`, the same
table `arch_caps.cuh` declares — there is no second table to drift from the first.
The check runs once per device per process: a `std::mutex`-guarded `std::set<int>`
of admitted device ordinals turns every call after the first into a lock and a
set lookup, so the entries that call it on every invocation do not pay a device
query per call.

Three things are asserted, in order:

1. `tabulated(cc)` — the table carries a row for this compute capability at all.
2. `supported(caps_of(cc))` — the row's REQUIRED capability clauses are all true
   (an untabulated arch and an unsupported one are different refusals: the first
   means "measure it and add the row", the second means "this architecture cannot
   run the baseline no matter what row is added").
3. The row's `smem_per_sm`, `smem_per_cta_max`, `regs_per_sm` and
   `max_threads_per_sm` equal what `cudaGetDeviceProperties`/`cudaDeviceGetAttribute`
   read off the actual device — every launch shape compiled into the binary was
   derived from the table's numbers, so a disagreement means those shapes are wrong
   for the card that is about to run them, independent of whether the compute
   capability itself is tabulated.

## Where it is called

Every entry point that reaches a kernel launch — the carry forward pass, the SM clock
read, the liveness pass, the standalone intra forward pass, the entmax and softmax
routing forward/backward families, and the T=1 decode step and its residency query —
calls `check_arch_table()` as the first statement of its body.
`tests/unit/test_closed_world_arch_gate.py` scans the registration translation unit's own
operator list and asserts this for every entry that is not named as launch-less (an arm
census, a build stamp, a host-only derivation over the descriptor, a ledger binding, or a
VMM driver operation).

## Why its own translation unit

`arch_caps.cuh`'s facts are `constexpr` and paid for at compile time; the refusal
that reads the device is unavoidably a runtime cost, however small, and giving it
one translation unit means every calling family links one definition rather than
each carrying its own copy that could drift from the others — the same reason the
gate insists there is exactly one `void check_arch_table() {` in the tree.
