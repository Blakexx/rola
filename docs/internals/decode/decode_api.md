# `csrc/rola/src/decode/decode_api.cuh` — the decode path's host-facing declarations

The ONLY decode header `csrc/rola/rola_api.cpp` includes. It declares four
symbols and carries no definitions: the two capacity accessors, the build
stamp and `rola_decode_forward`. The decode path itself is documented in
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

`rola_decode_capacity()` exists so a Python test can assert the C++ level-width
capacity against `rola.routing.topology.MAX_BRANCH_WIDTH`. A `static_assert`
cannot read a Python constant, so the pair of them is what makes the
producer/decode capacity relation CHECKABLE rather than merely written down.

`rola_decode_producer_width_mirror()` is the second half of that pair; why it is a
separate claim from the capacity is [`decode.md`
§33](decode.md#producer-width-mirror), and the capacity gate itself is
[`decode.md` §6](decode.md#level-width-capacity).

<a id="residency"></a>`rola_decode_residency(decay, levels)` returns the step's DECLARED
CTAs per SM for that arm, read off the same `constexpr` the kernel's `__launch_bounds__`
are built from. The host's split policy sizes ONE WAVE from it
([`decode.md` §7](decode.md#declared-residency)), and it is bound rather than mirrored in
Python for the reason the whole bound surface exists: a launch bound is exactly the
constant that drifts when it is written down twice. The two arms differ because the decay
arm and the deep arms each carry more live state through the same loop.

<a id="arm-subset"></a>**THE ARM LIST IS DATA, AND A BUILD MAY NAME A SUBSET OF IT.**
the decode family's instantiation matrix is the product `d_v(2) x D(4) x decay(2)`, and
`decode.cu` writes it out one row per arm (`DECODE_ARM_<i>`) so a BUILD can name a subset
by index: `ROLA_DECODE_ARMS=5` compiles arm 5 and nothing else
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
`ROLA_DECODE_ARMS` a second time to push `[d_v, D, decay]` rows into a vector, so
`rola.ops.decode.arms()` cannot disagree with what was compiled — there is one list and
both readings expand it. An arm the binary does not carry is a `TORCH_CHECK` refusal
naming the shape, never a silent miss.

**A SUBSET BINARY IS NOT SHIPPABLE** and the build announces it: the fatbin and manifest
gates prove EVERY ratified instantiation is present, which a subset contradicts by
construction, so `setup.py` skips exactly those two gates behind a banner when
`ROLA_DECODE_ARMS` is set — and nowhere else.

<a id="build-stamp"></a>`rola_decode_build_stamp()` is a DEVICE-SIDE FACT a stale `.so` at the right
path cannot fake: it launches a one-thread kernel, compiled with this translation unit,
that folds four facts into one `int64_t` in a positional scheme and reads it back:

    stamp = ((kKinds * kMaxTerms * kDecodeWarps + kUnitBlockWarps + kFieldWidthMax) * 4096
             + sizeof(DecodeParams)) * 1000003
          + sizeof(Unit<4>) * 101
          + kDecodeMaxSpan

A path check or a hash of the sources checks what is ON DISK; this checks what the LOADED
BINARY actually built from, which is the fact a stale extension can silently disagree
about (the standing "extension trap" lesson). It is bound because a Python-side
recomputation of the same quantities would just be a second mirror with its own drift.

THE ENUMERATION-SHAPE FIELD IS THE MOST SIGNIFICANT ONE, AND IT WAS ADDED FOR A MEASURED
REASON: the union-of-products enumeration ([`decode_lattice.md`](decode_lattice.md#the-terms))
redefined what every unit index in this path MEANS while moving neither struct's size and
neither ceiling, so the three older fields alone would have certified the new binary as
the old one. A stamp that cannot separate two binaries is not a stamp. The three older
fields still divide out unchanged, which is why the inversion in
`tests/unit/test_decode_capacity.py` only gained a term.

<a id="the-step"></a>`rola_decode_forward` is the single data-plane entry, and it launches
ONE kernel: the fold, the `(k, m)` lattice's factor tables, the walk and the step's OWN
per-batch-head residency verdict are phases of it. Its operands are the caller's RAW
`[B, 1, H, *]` views — `read`, `write`, `g_write`, `v`, plus the per-read-level
`normalize` flags — because the fold is inside the kernel and there is no operand plane
to hand it ([`decode_fold.md`](decode_fold.md)). `heads` is not an argument: `v`'s own
`[B, 1, H, d_v]` shape carries it. `lattice_k`/`lattice_m` name the `(k, m)` box
(`decode_lattice.md#the-refusals`) the caller's state plane and routing levels are
already written in — the leaf order travels with the call rather than being assumed. Its
argument checks and workspace contracts are documented with the definition, in
[`decode.md`](decode.md#instantiation-matrix).

## <a id="paged-state"></a>3. `page_table` — which backing the state is

The trailing `c10::optional<at::Tensor> page_table` is the whole ABI difference
between the two backings, and it is what the entry validates differently:

* **absent** — `state` is the dense `[BH, N, d_v + 1]` plane and `N` is its own middle
  extent. This is the pre-P67 signature exactly, defaulted so a caller that never
  pages never mentions paging.
* **present** — `page_table` is `[BH, N / 16]` int32 and NAMES the leaf space, so `N`
  comes from it; `state` is then the arena's `[slots, 16, d_v + 1]` plane, which holds
  only the atoms a plan committed.

Slot VALUES are not bounds-checked. That would be a device-side read of the table, and
the plan owns the bound — the same division of labour `chunk.cu`'s `page_table()`
states. The address the kernel builds from a slot, and what a `-1` means, are
[`decode.md` §4](decode.md#paged-address).

## <a id="the-verdict"></a>4. The verdict buffers — required whichever backing this is

`growth`, `growth_any`, `growth_ctr`, `done` and `atom_bits` are the step's OWN
per-batch-head residency answer, written by the last-arriving CTA of each batch-head and
read by nobody's prologue but the step's own next call. They are validated whichever
backing the call is — a dense step has nothing to admit and simply never writes a `1` into
`growth` — because holding one ABI for both backings is what keeps a dense-vs-paged
disagreement from being expressible in the first place: [`decode.md`
§4c](decode.md#absorbed-verdict).

`atom_bits` is `[BH, N / 16]` bool, this step's own write-atom set condensed on the
device, unconditionally, by the `seg == 0` CTA of every batch-head. It is what a host
admission (`admit_growth`) plans from, so the host derives no support of its own
([`decode.md` §39](decode.md#second-derivation)).

## <a id="the-pool"></a>5. `pool_slots`, `pool_cursor`, `pool_map` — the slack pool, all three or none

Present exactly on a paged step whose state reserved slack (`rola.state(pool_slack=...)`
above `0.0`): the physical slots the host pre-committed and pre-zeroed, how many of each
batch-head's are spent, and the claim the walk addresses through until the batch-head's
last-arriving CTA folds it into the real table. `rola_decode_forward` checks that all three
are present together or none are — the pool is not a mode a caller can half-configure — and
that a pool requires `page_table` (a dense state has no table to admit into). The design and
the device-side claim are [`paging/paging.md`
§the-slack-pool](../paging/paging.md#the-slack-pool).

## Comments moved from source

Verbatim `//:` prose blocks from `csrc/rola/src/decode/decode_api.cuh`, in source order, one heading per declaration. Source keeps only a short decl block pointing here.

<a id="rola-decode-capacity"></a>
### `rola_decode_capacity`

THE BOUND SURFACE. The two accessors exist so a Python test can assert the C++
level-width capacity against `rola.routing.topology.MAX_BRANCH_WIDTH` -- a
`static_assert` cannot read a Python constant:
docs/internals/decode/decode_api.md#bound-surface

<a id="note-l43"></a>
### near line 43

THE WHOLE STEP IN ONE LAUNCH -- fold, factor, walk, AND ITS OWN ADMISSION QUESTION --
over the caller's RAW `[B, 1, H, *]` views. There is no operand plane between the
producer and the walk, and no probe launch in front of it: the write set the walk
enumerates is the set the residency test reads, so the test is a pass over shared
memory. `lattice_k`/`lattice_m` name the `(k, m)` box the state plane is written in --
the leaf order travels with the plane rather than being assumed.
The verdict is PER BATCH-HEAD (`growth`, `done`, and the `growth_any` reduction the
host reads) beside the write-atom set it was reached from (`atom_bits`, which is what
a host admission then plans from), and under a slack pool a growth step admits
ITSELF: docs/internals/decode/decode_api.md#the-step

## More comments moved from source

Verbatim trailing/banner `//` comments from `csrc/rola/src/decode/decode_api.cuh` condensed at their call site into a short pointer.

<a id="note-l3"></a>
### near line 3

THE DECODE PATH'S HOST-FACING DECLARATIONS, and the ONLY decode header
`csrc/rola/rola_api.cpp` includes.

The split from `decode.cuh` is mechanical and looks gratuitous until you hit it:
`rola_api.cpp` is HOST-ONLY, and `decode.cuh` carries `__device__` helpers the host
compiler cannot parse. ONE HEADER PER AUDIENCE, rather than an `#ifdef __CUDACC__`
that would make one header mean two things to two compilers.

The full argument, and what each bound symbol claims: docs/internals/decode/decode_api.md

<a id="roladecoderesidency"></a>
### `rola_decode_residency`

THE STEP'S DECLARED RESIDENCY in CTAs per SM, so the split policy sizes one wave from
the launch bound itself rather than from a python mirror of it:
docs/internals/decode/decode_api.md#residency

<a id="roladecodebuildstamp"></a>
### `rola_decode_build_stamp`

THE DEVICE-SIDE BUILD STAMP, a fact only the loaded binary can produce. A path check
or a hash of the sources cannot catch a stale `.so` at the right path:
docs/internals/decode/decode_api.md#build-stamp

<a id="roladecodearms"></a>
### `rola_decode_arms`

THE ARMS THIS BINARY CARRIES, `[d_v, D, decay]` per row. A full build lists the whole
declared matrix; an ITERATION build (`ROLA_DECODE_ARMS`) lists its subset and is not
shippable: docs/internals/decode/decode_api.md#arm-subset
