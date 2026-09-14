# `csrc/rola/src/paging/vmm_owner.{cuh,cu}` — the CUDA driver VMM backing

`VmmOwner` reserves one virtual address range at the dense limit up front and commits
physical memory into it in chunks, so a long-lived page pool can grow without ever moving
a base pointer.

Both source files are already at the mirror's target comment density; this doc is their
reference rather than a relocation of their prose.

**Scope discipline — the one thing to preserve.** `VmmOwner` owns *only* the CUDA-driver
backing. Logical page identity, allocation plans, and publication remain in
[`PageArena`](paging.md), so the dense backing and the VMM backing share ONE ABI. Anything
that knows what a page *means* does not belong in this class — which is why the
constructor takes the page's two extents as numbers and never learns what either counts.

Symbols: a PAGE is one slot of the state pool, `page_rows x page_cols` fp32 (the arena
passes `MMA_K_QUANTUM x (d_v + 1)`, the mass column fused); a CHUNK is one driver
allocation handle mapped at the end of the mapped prefix; *granularity* is
`cuMemGetAllocationGranularity`'s minimum; *committed* means physical memory exists,
*reserved* means only address space does.

---

## <a id="one-plane"></a>1. One plane, reserved whole, mapped in a prefix

`reserve_plane()` reserves `dense_limit_pages * page_bytes`, rounded up to the allocation
granularity. The reservation is address space, not memory, and it is why the arena has no
ceiling parameter: reserving for the worst case costs nothing to reserve.

`base()` then hands out an `at::from_blob` tensor spanning the **entire** reserved extent,
shaped `[dense_limit_pages, page_rows, page_cols]` fp32.

**Only the prefix reported by `mapped_capacity_pages()` is mapped and may be touched.**
The tensor's own shape does not say so; the contract does, and `PageArena.state` is what
enforces it — the arena bounds the view to the mapped prefix before any caller sees it, so
reaching past the commitment is an index error rather than an unmapped-address fault.

Keeping the reservation-wide view (rather than re-reserving on every growth) is the point
of the design: the base pointer is stable for the owner's life, so nothing downstream
re-publishes a pointer when the pool grows.

## <a id="probe"></a>2. `probe()` — capability is proven, not queried

The attribute query (`CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED`) and a
non-zero granularity are necessary but not sufficient. **A probe is not complete until the
driver can reserve, map, grant access, unmap, and release one granularity-sized allocation
in this context.** The probe performs that whole round trip and reports `supported` only
if every step returned `CUDA_SUCCESS`.

Its cleanup is written to follow the ACTUAL MAP STATE, not the final status: mapping may
have succeeded even when `cuMemSetAccess` failed, so `cuMemUnmap` is gated on the `mapped`
flag and `cuMemAddressFree` / `cuMemRelease` on the handles being non-zero. Gating cleanup
on `status` instead would leak a mapping on exactly the failure path that most needs
unwinding.

The round trip is far too expensive to repeat per arena, which is why `PageArena` caches
the verdict per device index rather than asking again. `create()` calls `probe()` itself
and turns a negative into a refusal — the arena never falls back, because a pool that
quietly became a dense allocation would still be reported as paged.

## <a id="chunking"></a>3. Chunk sizing and the one-chunk lookahead

The constructor derives

```
chunk_pages_ = min(dense_limit_pages, max(1, target_chunk_bytes / page_bytes))
```

The arena passes the device's own allocation granularity as `target_chunk_bytes`, which is
the finest commitment the hardware admits: a smaller target is rounded straight back up by
`round_up`, and a larger one over-commits for nothing. A pool smaller than one granule
therefore commits its whole dense limit in a single mapping and saves no memory at all —
that is a hardware fact about the granularity, not a defect, and it is why the savings
receipt in `tests/integration/test_page_arena_vmm.py` is taken on a pool many chunks
wide.

`grow(required_pages)` returns immediately when the requirement is already met, and
otherwise commits to a chunk boundary with a **one-chunk lookahead**: the target is
`required_chunks + 1` chunks, *except* on the first growth (`current_pages == 0`) and
except when `required_chunks` already reaches the maximum — in both of those cases it is
exactly `required_chunks`. The result is clamped to `dense_limit_pages_` and then rounded
up to the granularity.

The lookahead is why a steadily growing pool does not call into the driver on every
admission: it is always one chunk ahead of the request that would need it. The no-op path
costs 0.38 µs (RTX 3090), which is what makes an unconditional `grow` in front of every
admission cheaper than any test that would decide to skip it.

`map_initial_chunk()` is just `grow(min(dense_limit_pages_, chunk_pages_))`, so the first
chunk exists before any caller sees the object.

## <a id="growth-atomicity"></a>4. Growth is all-or-nothing

`grow` maps ONE chunk covering the whole distance to its target, and installs it in the
order that makes every failure invisible: `cuMemCreate`, then `cuMemMap`, then
`cuMemSetAccess`, and only then `chunks_.push_back` and the `mapped_bytes_` update. Every
earlier failure therefore observes an owner that never saw this chunk.

The `catch` closes the one window where a chunk is half-installed — a handle exists and the
mapping may or may not have been made: it unmaps only if the map actually succeeded,
releases the handle, and rethrows. `committed_bytes_` is assigned from `mapped_bytes_`
after the push, so it cannot describe a chunk the owner does not hold.

## <a id="rollback"></a>5. `rollback_to` — quiesce, validate, then unmap

Shrinking is far more dangerous than growing, so the method refuses everything it cannot
prove:

1. **Quiesce first.** `cudaDeviceSynchronize()` must succeed before anything is unmapped.
   In-flight work may still hold addresses in the range being removed.
2. **The target must SURVIVE the rounding.** The requested page count is converted to bytes
   and rounded up to the granularity like every other extent; if that rounded extent holds
   more pages than were asked for, the target was inside a chunk and the method says so by
   name. Without this check the request would silently keep more than the caller asked for
   and then trip a consistency assertion further down, which names the symptom instead of
   the mistake.
3. **The target must be a committed CHUNK BOUNDARY.** The chunk list is walked accumulating
   byte offsets and the target must equal one of them (or be zero). Chunks are the
   allocation unit and there is no way to release half of one.
4. **The initial headroom chunk is never removed** (`target >= chunks_.front().bytes`), so
   the owner always retains the chunk `map_initial_chunk` installed and `base()` always has
   something behind it.

`remove_latest_chunk` then re-checks, per chunk, that the chunk being removed actually ENDS
at the mapped extent — a defence against a chunk list that has drifted out of address order
— and the method closes by asserting the plane landed exactly on the target and that
`mapped_capacity_pages()` equals the requested count.

Unlike growth, rollback's driver calls go through `check()` (which throws) rather than being
swallowed: a failed unmap during a deliberate shrink is a real error, not a teardown nicety.

**`reset()` is deliberately not this.** It is logical only — mappings and their stable
addresses survive — because what resets is the arena's page table, and re-mapping for the
next sequence would pay the driver twice for memory that never went away.

## <a id="view-lifetime"></a>6. View lifetime — why `close()` can refuse

Every tensor `base()` returns carries a deleter that captures two things: the shared
`ViewTracker` (an atomic `live_views` counter) and `shared_from_this()`.

The `shared_from_this` capture keeps the driver allocation alive for every tensor view.
**This prevents a caller holding a slice from outliving the native owner and observing an
unmapped raw pointer.** It is the reason `VmmOwner` derives from `enable_shared_from_this`
and the reason the pybind class is held by `std::shared_ptr` (see
[`../rola_api.md`](../rola_api.md#vmm-binding)). It is also the whole answer to "can torch
view driver memory without a copy?": `at::from_blob` over the reservation, with a deleter
that owns the lifetime, is the mechanism — no pluggable allocator, no copy, and nothing the
caching allocator can undo.

The counter is the other half: `close()` `TORCH_CHECK`s that `live_views == 0` and refuses
to proceed while any exported view exists. Lifetime extension alone would let `close()`
unmap under a live view; the counter turns that into a loud error at the call site instead.

The increment happens BEFORE `at::from_blob`, and the `catch` decrements it if construction
throws — so a failed view never leaks a count that would permanently block `close()`.

## <a id="destruction"></a>7. Destruction leaks rather than races

`~VmmOwner` runs `release_plane()` only if `quiesce()` (a guarded `cudaDeviceSynchronize`)
succeeds.

**Never unmap while work may still reference the reserved range.** If the context is already
unavailable — process teardown, a device reset, a CUDA error latched earlier — leaking the
mapping is safer than tearing it down under in-flight CUDA work; process teardown reclaims
it either way.

The whole body is exception-swallowing by necessity: destructors cannot report driver
failures. `release_plane()` is `noexcept` and unmaps chunks in REVERSE order before freeing
the address reservation, mirroring the order they were mapped in.

`close()` is the path that *can* report: it quiesces, releases, and marks `closed_`, and
every mutating method `TORCH_CHECK`s `!closed_` afterwards.

## <a id="overflow-discipline"></a>8. Arithmetic that cannot silently wrap

Every size computed from caller-supplied dimensions goes through a checked helper, and each
takes a `what` string so the failure names the quantity:

| helper | guards |
|---|---|
| `checked_mul` | `uint64` multiply overflow |
| `checked_size` | `uint64` value exceeding host `size_t` |
| `round_up` | `value + alignment - 1` overflowing `size_t`, and a zero alignment |

The page-byte derivation in the constructor (`page_rows * page_cols * sizeof(float)`), the
virtual extent in `reserve_plane`, and the growth and rollback targets all pass through
them. Page geometry arrives from Python; a wrapped size here would reserve or map the wrong
extent with no diagnostic, which is why the checks are unconditional rather than
debug-only.

`driver_error()` renders any `CUresult` as `NAME: description` via `cuGetErrorName` /
`cuGetErrorString`, falling back to `CUDA_ERROR_UNKNOWN`, so every `check()` failure carries
the driver's own words.

`initialize_driver()` calls `cuInit(0)` under a `std::call_once` and re-checks the STORED
result on every subsequent call — so a failed init reports on each entry rather than being
silently skipped after the first.
