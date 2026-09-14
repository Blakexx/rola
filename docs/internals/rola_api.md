<!-- ABSENT: flash_api carry_owner_rows carry_backward_stamp rola_consumer_forward rola_consumer_occupancy rola_consumer_check_arch_table rola_consumer_smem_bytes rola_sm_shared_bytes page_plan_abi_version page_plan_status_codes page_plan_workspace_bytes rola_page_plan_prepare rola_page_plan_finalize rola_page_plan_publish rola_page_admit_mask rola_page_cache_reset -->
# `csrc/rola/rola_api.cpp` — the extension's Python binding

The only translation unit that knows pybind exists. One `PYBIND11_MODULE`, four
groups of `m.def`, and one `py::class_`.

`m.def` bodies live here and nowhere else; the declarations they name come from
`src/chunk/chunk_api.cuh`, `src/decode/decode_api.cuh`, `src/entmax/entmax.cuh` and
`src/paging/vmm_owner.cuh`.

Symbols: a TU is a translation unit; `-gencode` is the nvcc flag that emits one device
code path per target architecture; `TORCH_EXTENSION_NAME` is the macro the build
substitutes for the module name.

---

## <a id="why-cpp"></a>1. Why it is a `.cpp` and not part of a kernel `.cu`

Binding code is host-only. Left inside a `.cu` it is recompiled once per `-gencode`
target for no benefit, and **every future architecture added to the ratified set
multiplies that cost.** `flash_api.cpp` is the convention; this is the same split.

Every declaration the binder needs is a `*_api.cuh`-shaped host header, so nothing
here needs a device body in scope.

## <a id="one-module"></a>2. One module, not two

The consumer and the §7.1 paging arena register into the SAME extension: the consumer
reads the page table the plan publishes, so splitting them would put a **cross-module
ABI between two halves of one contract.**

Registration happens HERE and nowhere else: every `m.def` body is in this file. No
fragment header registers anything, so there is one place to read to know what the
module exports.

The T=1 decode path registers here for the same reason: it is a SIBLING of the
prefill consumer, not a specialization of it — shared produce contract, own kernel,
frozen config — and one contract deserves one ABI boundary.

## <a id="module-name"></a>3. The module name comes from the build

`TORCH_EXTENSION_NAME` is set by `setup.py` to `rola._C` — **namespaced inside the
package**, so the binary lands in `rola/` for both an in-place build and a wheel.
`rola/ops/_ext.py` imports it package-relative and never builds it.

**Nothing in this file or under `src/` spells the name**, so the packaging layout can
move without touching a kernel symbol. Hard-coding `_C` anywhere in the C++ would undo
that.

## <a id="registered-entries"></a>4. What is registered

| group | entries |
|---|---|
| producer union entmax solve | `entmax_union_forward`, `entmax_union_backward` |
| producer per-side solves | `entmax_factor_forward`, `entmax_factor_backward`, `routing_softmax_forward`, `routing_softmax_backward` |
| the addressing block (G2-EXTRACT), a foundation feature | `carry_geometry`, `carry_sub_boxes` |
| the within-window term | `intra_forward`, `intra_arms`, `intra_build_stamp` |
| the T=1 decode path (thunder §6b) | `rola_decode_forward`, `rola_decode_build_stamp`, `rola_decode_capacity`, `rola_decode_producer_width_mirror` |
| the VMM owner | `rola_vmm_probe`, `rola_vmm_create`, and the `RoLAVmmOwner` class |

Three of the four decode entries are gates rather than operations, and their docstrings
say so: `rola_decode_capacity` reports the decode path's maximum level width, which must
be `>=` the producer's `MAX_BRANCH_WIDTH` (`tests/unit/test_decode_capacity.py` pins the
pair); `rola_decode_producer_width_mirror` is the C++ mirror of that producer
constant, pinned EQUAL to the Python value so a stale mirror cannot silently satisfy the
capacity `static_assert`; and `rola_decode_build_stamp` is the device-side fact a stale
extension cannot fake ([`decode/decode_api.md`
§build-stamp](decode/decode_api.md#build-stamp)).

`rola_decode_forward` and the intra entries are bound with explicit `py::arg` names.

**NOT REGISTERED ON THIS LINE (C0, card C).** The carry family's six entries
(`carry_forward_inter`, `carry_build_stamp`, `carry_owner_rows`, `carry_arms`,
`carry_backward_inter`, `carry_backward_stamp`) and the stats passes' three
(`chunk_facts`, `chunk_block_bits`, `chunk_arms`) were bound here until the clean
slate deleted their kernels. `rola/ops/carry.py` and `rola/engine/facts/` still CALL
them -- that is the contract, kept deliberately -- and fail at `rola._C` with an
honest missing-attribute error until C2 and C3 build them back
([`DELETIONS.md`](DELETIONS.md)).

The tiled consumer's five entries — `rola_consumer_forward`,
`rola_consumer_occupancy`, `rola_consumer_check_arch_table`,
`rola_consumer_smem_bytes` and `rola_sm_shared_bytes` — were bound here until P67
D2 retired that arm; none of them is in the module today, and the device-side
entry list is the check that says so ([`DELETIONS.md`](DELETIONS.md)).

## <a id="deleted-page-plan"></a>5. There is no CUDA page-plan producer

The page table the consumer reads is published by the torch PLANNED-EXACT-ASYNC arena
in `rola/ops/paging.py`, and nothing under `rola/`, `tests/` or `benchmarks/` reaches
a CUDA page planner. Recorded here because the absence is the fact worth keeping:
these eight names are NOT bound, and a reader looking for them is looking for the
arena.

- `page_plan_abi_version`
- `page_plan_status_codes`
- `page_plan_workspace_bytes`
- `rola_page_plan_prepare`, `rola_page_plan_finalize`, `rola_page_plan_publish`
- `rola_page_admit_mask`
- `rola_page_cache_reset`

Their removal was safe on the no-dead-code standard because no Python caller ever
reached them; the register of that removal is
[`DELETIONS.md`](DELETIONS.md).

## <a id="vmm-binding"></a>7. The VMM owner's binding shape

`rola_vmm_probe` is bound as a lambda that flattens `VmmProbe` into a `py::dict`
(`device`, `supported`, `allocation_granularity`, `reason`) rather than exposing the
struct — a probe result is diagnostic data, and a dict keeps it printable and stable
across additions.

`RoLAVmmOwner` is held by `std::shared_ptr`, which is required rather than stylistic:
the class is `enable_shared_from_this` and every exported tensor view captures a
`shared_from_this` to keep the driver mapping alive. See
[`paging/vmm_owner.md`](paging/vmm_owner.md#view-lifetime).

The mutating operations (`grow`, `rollback_to`, `reset`, `close`) are methods; every
observable (`closed`, `mapped_capacity_pages`, `dense_limit_pages`, `committed_bytes`,
`virtual_bytes`, `allocation_granularity`, `chunk_pages`, `backing_kind`) is a
`def_property_readonly`, so nothing on the Python side can write owner state except
through the four verbs.

`rola_vmm_create` is the factory. Its page geometry arrives as two extents,
`page_rows` and `page_cols`, and the owner never learns what either counts — the arena
passes `MMA_K_QUANTUM` and `d_v + 1`, which is the fused-mass page it addresses, and
keeping the names dimensionless is what stops the owner acquiring an opinion about page
meaning it has no business holding ([`paging/vmm_owner.md`](paging/vmm_owner.md)).
`target_chunk_bytes` defaults to 64 MiB at this boundary; the arena overrides it with the
device's own allocation granularity, which is the finest commitment the driver admits.

## <a id="the-step"></a>7a. `rola_decode_forward` — the whole step, one launch

The decode step is ONE kernel: the operand fold, the `(k, m)` lattice's factor tables, the
token-stationary walk and the step's OWN per-batch-head residency verdict are
phases of it, so what the boundary passes is the caller's RAW `[B, 1, H, *]` views and not
a widened plane. There is no separate fold entry and no `heads` argument — the fold is
inside the kernel and `v`'s own shape carries `H`. There is no separate admission-probe
entry either: the residency test reads the write map the walk was going to build anyway,
so what used to be a launch ahead of the step is now a pass over shared memory inside it.
The phases, and why neither the fold nor the admission question is its own launch, are
[`decode/decode_fold.md`](decode/decode_fold.md) and [`decode/decode.md`
§4c](decode/decode.md#absorbed-verdict); the step is [`decode/decode.md`](decode/decode.md).

`growth`, `growth_any`, `growth_ctr`, `done` and `atom_bits` are the verdict's own
buffers — bound as required arguments whichever backing the call is, so one ABI serves
both. `pool_slots`, `pool_cursor` and `pool_map` are the slack pool, present exactly on a
paged step whose state reserved one. Both groups are [`decode/decode_api.md`
§4-5](decode/decode_api.md#the-verdict).

<a id="decode-build-stamp"></a>`rola_decode_build_stamp` is bound beside it for the same
reason `rola_decode_capacity` and its mirror are: a fact worth reading back from the
loaded binary rather than trusting from the source tree — the only kind of check an
extension trap cannot survive ([`decode/decode_api.md`
§build-stamp](decode/decode_api.md#build-stamp)).

## The facts entries (operator adoption)

The facts group was a PRIVATE seam, `rola/engine/facts/` its only intended caller.
Its kernels and its declarations left with C0 (`docs/internals/DELETIONS.md`); C2
rebuilds the pass as the liveness pass, a foundation feature, and C1 adjusts the
Python seam where its outputs replace the atom and block bitmaps.

## Comments moved from source

Verbatim `//:` prose blocks from `csrc/rola/rola_api.cpp`, in source order, one heading per declaration. Source keeps only a short decl block pointing here.

<a id="pybind11-module-note-l41"></a>
### in `PYBIND11_MODULE`, near line 41

ONE PASS, BOTH ROW SETS: the membership union table always, and the atom-grain
write bitmap when this call has residency to plan.  `mask` is the CALL CLASS
(`rola/engine/types.py`'s M1/M2), which is compiled in; `atoms` is the
runtime-null output skip within the stateful class.

<a id="pybind11-module-note-l126"></a>
### in `PYBIND11_MODULE`, near line 126

A SIBLING of the chunk consumer, not a specialization of it: shared produce
contract, own consumer, frozen config. Two of its three entries are capacity GATES
rather than operations -- docs/internals/rola_api.md#registered-entries
THE WHOLE STEP: the operand fold, the lattice factor tables, the gather-GEMV and the
step's OWN admission question in ONE launch, over the caller's raw `[B, 1, H, *]`
views -- docs/internals/rola_api.md#the-step

<a id="pybind11-module-note-l203"></a>
### in `PYBIND11_MODULE`, near line 203

The page geometry is the ARENA's -- `page_rows x page_cols` fp32, which is
`MMA_K_QUANTUM x (d_v + 1)` with the mass column fused. The owner takes it as two
extents rather than naming them, because it owns no meaning for either:
docs/internals/rola_api.md#vmm-binding

<a id="near-line-31"></a>
### near line 31

K31 R1 -- the box-native CARRY family, built ALONGSIDE the shipped facts
producers and reachable from no shipped dispatch (development/plans/
K31_BOX_NATIVE_KERNEL_STUDY_2026-08-18.md section 6).

<a id="near-line-58"></a>
### near line 58

the sparse grid's predicate INPUT: three liveness facts per (bh, block),
reduced from the union table this call already built. The predicate OVER
them is the dispatch's, because it differs by call class and by backing.

<a id="near-line-113"></a>
### near line 113

One entry per direction per family, each covering as many LEVELS as the caller
groups into one width class -- docs/internals/entmax/factor.md#batched-levels

<a id="near-line-133"></a>
### near line 133

The verdict's buffers: the per-batch-head answer, its reduction, the reducing
CTA's arrival counter, the done flags a replay reads, and the write-atom set a
host admission plans from.

<a id="near-line-140"></a>
### near line 140

`page_table` resolves each touched ATOM to its slot; `None` is the dense
backing, where the slot IS the atom and the bytes are the same.

<a id="near-line-143"></a>
### near line 143

THE SLACK POOL, all three or none: the slots the host pre-committed and
pre-zeroed, how many of each batch-head's are spent, and the claim the walk
addresses through until the table is folded.

<a id="near-line-149"></a>
### near line 149

THE DEVICE-SIDE BUILD STAMP: a fact only the loaded binary can produce, which is the
only kind of check an extension trap cannot survive --
docs/internals/rola_api.md#decode-build-stamp
