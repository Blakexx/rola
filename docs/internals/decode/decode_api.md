# `csrc/rola/src/decode/decode_api.cuh` — the decode path's host-facing declarations

<a id="note-l3"></a>The ONLY decode header `csrc/rola/rola_api.cpp` includes. It declares six
symbols and carries no definitions: the two capacity accessors, the build stamp, the arm list, the phase-ledger bind
and `rola_decode_forward`. The decode
path itself is documented in
[`decode.md`](decode.md); this file exists for a compilation reason, and the
reason is the whole content of this doc.

---

## <a id="why-split"></a>1. Why it is split from `decode.cuh`

The split looks gratuitous until you hit it.

`rola_api.cpp` is HOST-ONLY and is compiled by the HOST compiler, not by `nvcc`.
`decode.cuh` carries `__device__` helpers — the bitmap funnel shift, the set-bit
boundary cuts ([`decode.md` §14](decode.md#extract-bits)) — that the host compiler
cannot parse. A binding TU that included `decode.cuh` would therefore fail to
compile.

Guarding those helpers with `#ifdef __CUDACC__` would work, and would be worse:
the header would then mean two different things to two compilers, and which
declarations a caller got would depend on which compiler read it. **One header per
audience instead.**

## <a id="bound-surface"></a>2. The bound surface

<a id="rola-decode-capacity"></a>`rola_decode_capacity()` exists so a Python test can assert the C++ level-width
capacity against `rola.routing.topology.MAX_BRANCH_WIDTH`. A `static_assert`
cannot read a Python constant, so the pair of them is what makes the
producer/decode capacity relation CHECKABLE rather than merely written down.

`rola_decode_producer_width_mirror()` is the second half of that pair; why it is a
separate claim from the capacity is [`decode.md`
§33](decode.md#producer-width-mirror), and the capacity gate itself is
[`decode.md` §6](decode.md#level-width-capacity).

<a id="arm-subset"></a><a id="roladecodearms"></a>**THE ARM LIST IS DATA, AND A BUILD MAY NAME A SUBSET OF IT.**
the decode family's instantiation matrix is the product `d_v(2) x D(4) x decay(2)`, declared
one row per arm in `tools/manifests/decode_set.json` (an arm family of `tools/gen_shards.py`:
[`../tools/gen_shards.md#families`](../tools/gen_shards.md#families)) so a BUILD can name a subset
by index: `--//:decode_arms=5` compiles arm 5 and nothing else, through the generated
`decode_selection.inc` that `decode.cu` alone includes
([`build.md#arm-subset`](../../build.md#arm-subset)). Instantiation is the dominant build
resource, so this is what makes iterating on one arm cost one arm.

**THE LAUNCH IS STILL WRITTEN ONCE.** [`dispatch_switch.md`](../dispatch_switch.md) refuses
the MACRO LADDER — a `#define` per axis whose BODY restates the `<<<...>>>`, so an
`N`-axis dispatch spells the launch `2^N` times inside an expansion the compiler cannot
type-check. This is not one: the macro carries an arm's three CONSTANTS and its
comparison, the launch itself is a single generic lambda taking three `integral_constant`s,
and no `<<<...>>>` is spelled twice. What replaces the nested `int_switch`/`bool_switch` is
only the SOURCE of the candidate set — a list the build can cut, instead of three sets
written into the call — and `matched` is `int_switch`'s own refusal mechanism, kept.

**THE BUILT SET IS READ BACK OFF THE SAME LIST.** `rola_decode_arms()` expands
`ROLA_DECODE_ARM_SET_X` a second time to push `[d_v, D, decay]` rows into a vector, so
`rola.ops.decode.arms()` cannot disagree with what was compiled — there is one list and
both readings expand it. An arm the binary does not carry is a `STD_TORCH_CHECK` refusal
naming the shape, never a silent miss.

**A SUBSET BINARY IS NOT SHIPPABLE** and its record says so: ratify proves EVERY ratified
instantiation is present, which a subset contradicts by construction, so an iteration
build is never ratified and never a wheel ([`build.md#iteration`](../../build.md#iteration)).

<a id="build-stamp"></a>
<a id="roladecodebuildstamp"></a>`rola_decode_build_stamp()` is a DEVICE-SIDE FACT a stale `.so` at the right
path cannot fake: it launches a one-thread kernel, compiled with this translation unit,
that folds four facts into one `int64_t` in a positional scheme and reads it back:

    stamp = ((kKinds * kMaxTerms * DecodePlan<B>::kWarps + kUnitBlockWarps + kFieldWidthMax) * 4096
             + sizeof(DecodeParams)) * 1000003
          + sizeof(Unit<4>) * 101
          + kDecodeMaxSpan

A path check or a hash of the sources checks what is ON DISK; this checks what the LOADED
BINARY actually built from, which is the fact a stale extension can silently disagree
about (the standing "extension trap" lesson). It is bound because a Python-side
recomputation of the same quantities would just be a second mirror with its own drift.

THE ENUMERATION-SHAPE FIELD IS THE MOST SIGNIFICANT ONE: a change to the enumeration
([`decode_lattice.md`](decode_lattice.md#the-terms)) can redefine what every unit index in
this path MEANS while moving neither struct's size nor any ceiling, and a stamp without it
would certify the new binary as the old one. A stamp that cannot separate two binaries is
not a stamp. `tests/unit/test_decode_capacity.py` inverts the stamp term by term.

<a id="the-step"></a><a id="note-l43"></a>`rola_decode_forward` is the single data-plane entry, and it launches
ONE kernel: the fold, the `(k, m)` lattice's factor tables, the walk and the step's OWN
per-batch-head residency verdict are phases of it. There is no probe launch in front of it:
the write set the walk enumerates is the set the residency test reads, so the test is a pass
over shared memory. The verdict (`growth`, `done`, and the `growth_any` reduction the host
reads) sits beside the write-atom set it was reached from (`atom_bits`, what a host
admission plans from), and under a slack pool a growth step admits itself.
`write_dense` is the bundle's flag that the innermost level's write side is dense
(`docs/api.md` §2.1), which lets the write term walk whole atoms (`decode.md#whole-atoms`). Its operands are the
caller's RAW
`[B, 1, H, *]` views — `read`, `write` and their scales, `v` — because the fold is inside the kernel and there is no operand plane
to hand it ([`decode_fold.md`](decode_fold.md)). `heads` is not an argument: `v`'s own
`[B, 1, H, d_v]` shape carries it. `lattice_k`/`lattice_m` name the `(k, m)` box
(`decode_lattice.md#the-refusals`) the caller's state plane and routing levels are
already written in — the leaf order travels with the call rather than being assumed. Its
argument checks and workspace contracts are documented with the definition, in
[`decode.md`](decode.md#instantiation-matrix).

## <a id="paged-state"></a>3. `page_table` — which backing the state is

The trailing `std::optional<Tensor> page_table` is the whole ABI difference
between the two backings, and it is what the entry validates differently:

* **absent** — `state` is the dense `[BH, N, d_v + 1]` plane and `N` is its own middle
  extent. This is the pre-P67 signature exactly, defaulted so a caller that never
  pages never mentions paging.
* **present** — `page_table` is `[BH, N / 16]` int32 and NAMES the leaf space, so `N`
  comes from it; `state` is then the arena's `[slots, 16, d_v + 1]` plane, which holds
  only the atoms a plan committed.

Slot VALUES are not bounds-checked. That would be a device-side read of the table, and
the plan owns the bound. The address the kernel builds from a slot, and what a `-1` means, are
[`decode.md` §4](decode.md#paged-address).

## <a id="the-verdict"></a>4. The verdict buffers — required whichever backing this is

`growth`, `growth_any`, `growth_ctr`, `done` and `atom_bits` are the step's OWN
per-batch-head residency answer, written by each head's owner and finisher
([`decode.md#one-writer`](decode.md#one-writer)) and read by nobody's prologue but the
step's own next call. They are validated whichever
backing the call is — a dense step has nothing to admit and simply never writes a `1` into
`growth` — because holding one ABI for both backings is what keeps a dense-vs-paged
disagreement from being expressible in the first place: [`decode.md`
§4c](decode.md#absorbed-verdict).

`atom_bits` is `[BH, N / 16]` bool, this step's own write-atom set condensed on the
device, unconditionally, by every batch-head's owner. It is what a host
admission (`admit_growth`) plans from, so the host derives no support of its own
([`decode.md` §39](decode.md#second-derivation)).

## <a id="the-pool"></a>5. `pool_slots`, `pool_cursor` — the slack pool, both or neither

Present exactly on a paged step whose state reserved slack (`rola.state(pool_slack=...)`
above `0.0`): the physical slots the host pre-committed and pre-zeroed, and how many of each
batch-head's are spent. The head's owner writes its claim into the page table itself
([`decode.md#queue-claim-pool`](decode.md#queue-claim-pool)), so `page_table` is mutable.
`rola_decode_forward` checks that both are present together or neither is — the pool is not a mode a caller can half-configure — and
that a pool requires `page_table` (a dense state has no table to admit into). The design and
the device-side claim are [`paging/paging.md`
§the-slack-pool](../paging/paging.md#the-slack-pool).

## <a id="decode-ledger-bind"></a>`rola_decode_ledger_bind`

Binds the phase ledger the next decode steps book into (`decode.md#phase-ledger`), or unbinds it with `None`: a
contiguous CUDA int64 tensor `[ctas][warps][kDecodePhases]` (`warps` the step's plan's at its device's build), a row per resident CTA, which the host entry
checks before each launch. Measurement only; nothing on the shipped path binds one.

<a id="decode-ledger-warps"></a>`rola_decode_ledger_warps()` is that tensor's warps, the step's plan's at the current
card's build (`DecodePlan<B>::kWarps`), so a caller sizes a ledger by asking rather than holding the number.
