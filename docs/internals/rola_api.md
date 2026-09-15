<!-- ABSENT: flash_api carry_owner_rows carry_backward_stamp rola_consumer_forward rola_consumer_occupancy rola_consumer_check_arch_table rola_consumer_smem_bytes rola_sm_shared_bytes page_plan_abi_version page_plan_status_codes page_plan_workspace_bytes rola_page_plan_prepare rola_page_plan_finalize rola_page_plan_publish rola_page_admit_mask rola_page_cache_reset -->
# `csrc/rola/rola_api.cpp` — the extension's operator registration

The only translation unit that registers with torch, and it does so through torch's **stable ABI**: every entry is an
operator in the `rola` namespace, `torch.ops.rola.<entry>`, declared by a schema (`STABLE_TORCH_LIBRARY`) and
implemented by a boxed function (`STABLE_TORCH_LIBRARY_IMPL`, `TORCH_BOX`). The binary depends on torch's stable C shim
and header-only types, never on torch's C++ ABI, so one build loads under every torch from the targeted minimum on
(`tools/build_flags.py`'s `TORCH_MIN`; [`docs/build.md`](../build.md#stable-abi)). The kernels meet torch only through
[`common/torch_seam.cuh`](common/torch_seam.md).

Symbols: a TU is a translation unit; `-gencode` is the nvcc flag that emits one device code path per target
architecture; a schema is torch's operator signature string (`int[] widths`, `Tensor(a!) num` for a tensor the call
writes).

---

## <a id="why-cpp"></a>1. Why it is a `.cpp` and not part of a kernel `.cu`

Registration is host-only. Left inside a `.cu` it is recompiled once per `-gencode` target for no benefit, and every
architecture added to the ratified set multiplies that cost. Every declaration it names is a host header
(`src/carry/carry_api.cuh`, `src/decode/decode_api.cuh`, `src/entmax/entmax.cuh`, `src/entmax/factor.cuh`,
`src/common/geom_api.cuh`, `src/facts/liveness_api.cuh`, `src/intra/intra.cuh`, `src/paging/vmm_owner.cuh`), so no
device body is in scope.

## <a id="one-library"></a>2. One library, registered here only

Every family registers into the one `rola` namespace, because they are halves of one contract: the carry reads the page
table the arena publishes, and the decode step is a sibling of the prefill, not a specialization of it. Every `m.def`
and `m.impl` is in this file, so there is one place to read to know what the extension offers;
`tests/unit/test_closed_world_arch_gate.py` scans exactly these registrations.

## <a id="module-name"></a>3. The module object is only a handle

`rola_cu13._C` is what an import names, in the binary plugin the build ships as ([build.md#wheels](../build.md#wheels)):
`PyInit__C` builds an empty module against Python's limited API (3.10 and on), and loading the library is what runs the
static registrations. `rola/ops/_ext.py` imports it after checking the plugin's build record is this `rola`'s version, and returns
`torch.ops.rola` under the names the package calls, so one binary serves every CPython from 3.10 and nothing in Python
reaches into the module itself.

## <a id="registered-entries"></a>4. What is registered

| group | operators |
|---|---|
| the carry launch surface | `carry_forward`, `carry_backward` (the reverse surface's refusal only), `carry_ledger_bind`, `carry_build_stamp`, `carry_census`, `carry_arms` |
| build and clock facts | `csrc_build_stamp`, `sm_clock_ghz` |
| the addressing block and the one liveness pass | `carry_geometry`, `carry_sub_boxes`, `liveness_words` |
| the within-window term | `intra_forward`, `intra_arms`, `intra_build_stamp` |
| the producer's solves | `entmax_union_forward`, `entmax_union_backward`, `entmax_factor_forward`, `entmax_factor_backward`, `routing_softmax_forward`, `routing_softmax_backward` |
| the T=1 decode path | `rola_decode_forward`, `rola_decode_residency`, `rola_decode_build_stamp`, `rola_decode_arms`, `rola_decode_capacity`, `rola_decode_producer_width_mirror` |
| the VMM owner, by handle | `vmm_probe`, `vmm_create`, `vmm_release`, `vmm_base`, `vmm_grow`, `vmm_rollback_to`, `vmm_reset`, `vmm_close`, `vmm_facts` |

Every operator that launches a kernel opens with `check_arch_table()`, the closed-world refusal of an architecture the
binary was never measured on ([`common/arch_runtime.md`](common/arch_runtime.md)). A schema names every argument, so
`torch.ops.rola` takes the same keywords the pybind binding did; the defaults (`sm_clock_ghz`'s spin, the decode step's
optional page table and slack pool, `vmm_create`'s chunk size) are the schema's.

Three of the decode entries are gates rather than operations: `rola_decode_capacity` reports the decode path's maximum
level width, which must be `>=` the producer's `MAX_BRANCH_WIDTH` (`tests/unit/test_decode_capacity.py` pins the pair);
`rola_decode_producer_width_mirror` is the C++ mirror of that producer constant, pinned equal to the Python value so a
stale mirror cannot satisfy the capacity `static_assert`; and <a id="decode-build-stamp"></a>`rola_decode_build_stamp`
is the device-side fact a stale extension cannot fake ([`decode/decode_api.md`
§build-stamp](decode/decode_api.md#build-stamp)).

The tiled consumer's entries and the CUDA page-plan producer's eight are not registered and have not been since their
kernels were deleted ([`DELETIONS.md`](DELETIONS.md)); the page table the carry reads is published by the torch arena in
`rola/ops/paging.py`.

## <a id="vmm-binding"></a>5. The VMM owner, by handle

An operator can return tensors, integers, floats, strings and lists of them, and not a native object. So the owner
lives in this file's registry, `std::unordered_map<int64_t, std::shared_ptr<VmmOwner>>`, and a caller holds the integer
`vmm_create` returns. `vmm_release` drops the registry's reference; every other operator takes the handle.

The `shared_ptr` is required rather than stylistic: `VmmOwner` is `enable_shared_from_this`, and every tensor view
`vmm_base` returns captures a `shared_from_this` in its deleter, so a view keeps the driver mapping alive past the
handle's release ([`paging/vmm_owner.md`](paging/vmm_owner.md#view-lifetime)).

`rola/ops/_ext.py`'s `RoLAVmmOwner` is the handle's Python face and keeps the surface the arena reads: the verbs
(`grow`, `rollback_to`, `reset`, `close`) are methods, the observables (`closed`, `mapped_capacity_pages`,
`dense_limit_pages`, `committed_bytes`, `virtual_bytes`, `allocation_granularity`, `chunk_pages`, `backing_kind`) are
read-only properties over one `vmm_facts` call, and dropping the object releases the handle. `rola_vmm_probe` returns
`vmm_probe`'s four facts as a dict (`device`, `supported`, `allocation_granularity`, `reason`), and `rola_vmm_create`
is the factory.

The page geometry arrives as two extents, `page_rows` and `page_cols`, and the owner never learns what either counts --
the arena passes `MMA_K_QUANTUM` and `d_v + 1`, the fused-mass page it addresses. `target_chunk_bytes` defaults to
64 MiB at this boundary; the arena overrides it with the device's own allocation granularity.

## <a id="the-step"></a>6. `rola_decode_forward` — the whole step, one launch

The decode step is ONE kernel: the operand fold, the `(k, m)` lattice's factor tables, the token-stationary walk and the
step's own per-batch-head residency verdict are phases of it, so what the boundary passes is the caller's raw
`[B, 1, H, *]` views and not a widened plane. There is no separate fold entry and no `heads` argument -- `v`'s own shape
carries `H` -- and no separate admission-probe entry: the residency test reads the write map the walk was going to build
anyway ([`decode/decode_fold.md`](decode/decode_fold.md), [`decode/decode.md` §4c](decode/decode.md#absorbed-verdict)).

`growth`, `growth_any`, `growth_ctr`, `done` and `atom_bits` are the verdict's own buffers, required whichever backing
the call is, so one schema serves both. `pool_slots`, `pool_cursor` and `pool_map` are the slack pool, present exactly
on a paged step whose state reserved one ([`decode/decode_api.md` §4-5](decode/decode_api.md#the-verdict)).
