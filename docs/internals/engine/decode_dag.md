# The decode family's DAG — M3 and M4

`rola/engine/dags/decode_dag.py`. The kernel-side page is
[`internals/decode/decode.md`](../decode/decode.md); this one is the HOST side — what
the step's facts are, in what order they are found, and which of the framework's
declarations carry the two properties a decode step lives or dies by: it must be
capturable, and its growth must self-correct.

Symbols used below: `B` batch, `H` heads, `BH = B*H`, `D` level count, `N` leaf count,
`d_v` value width, **atom** = 16 consecutive leaves = the page granule.

## <a id="the-two-masks"></a>1. M3 and M4 are the two BACKINGS, and a growth step is neither

`Mask.M3` is decode-dense and `Mask.M4` is decode-paged. After the step absorbed its own
residency question, the node SET is one tuple for both — a dense call simply never finds
an atom the table does not map, because it has no table. What the mask still separates is
the host-visible verdict READ: a dense step has no residency fact to read back, so
`read_verdict` carries `Mask.M4` alone and `DECODE_STEP` itself carries none of the
band-specific work either mask would drop.

**A GROWTH STEP IS NOT A THIRD MASK.** It is `DECODE_STEP` walked twice around an
admission the host performs between the two walks. A mask selects a subgraph; growth
selects the same one, which is exactly why a CUDA graph captured over the step stays
valid across an admission and why a replay is self-correcting: the verdict is INSIDE the
captured region, so on the second walk a batch-head that could not advance the first time
carries no done flag yet and the step it had skipped performs.

## <a id="the-order"></a>2. The step, read top to bottom

| node | gives | stream | mask |
|---|---|---|---|
| `decode_call` | `widths`, `d_v`, `has_decay`, `BH`, `device`, `paged` | host | both |
| `decode_mask` (rule) | `mask` | host | both |
| `geometry` | `geometry` | host, `pure_of` | both |
| `scratch` | `scratch` | host, `per_state` | both |
| `backing` | `carried_plane`, `page_table`, `pool` | main | both |
| `state_plane` | `state_plane` | main | both |
| `build_plan` | `plan` | main | both |
| `step` | `y` | main | both |

**THERE IS NO OPERAND NODE AND NO ADMISSION NODE.** The step's per-token operands are
folded inside the step kernel, from the producer's own views
([`decode/decode_fold.md`](../decode/decode_fold.md)), and the residency question the
walk needs answered is asked off the write map that same kernel builds for itself, in the
same launch ([`decode.md#absorbed-verdict`](../decode/decode.md#absorbed-verdict)). So a
decode step is ALWAYS one launch, dense or paged, growing or steady, and neither backing
has a stage whose only output is a widened copy of an input or a residency answer for a
launch still to come. The detached clock is the write allocation itself
([`decode.md#detached-clock`](../decode/decode.md#detached-clock)), so it is not a stream
of its own either.

`backing` is `RoLAState._decode_entry`: the plane, table and slack pool the state already
has, over the residency it already has — it admits nothing itself. `build_plan` carries a
stream because the decay arm's `dials` are derived there: a per-HEAD constant, and the one
operand the step kernel does not fold. Everything else on the plan is a view the producer
already holds.

`state_plane` carries the per-step SHAPE agreements — `d_v` against the frozen config,
the plane's extent against the backing, the page table's against `[BH, N/16]`. They are
checked per step and not at plan build because that is where the plane the step actually
indexes is known.

## <a id="the-verdict"></a>3. The verdict is ONE node, appended, and it is the step's own output

`DECODE_VERDICT` is `DECODE_STEP + (read_verdict,)` — the captured recipe with the
host-visible read spliced on the end, never a different walk. `read_verdict` is a single
`copy_(non_blocking)` of the step's own `growth_any` output into a pinned mirror, and it
CANNOT be enqueued ahead of the step any more: the verdict the step publishes is its own
last-arriving CTA's write, so the read is behind it by data dependence, not by host
policy. It is the family's ONE `host_sync` node, and it is the last node in the tuple —
`DECODE_VERDICT[:len(DECODE_STEP)] == DECODE_STEP` is asserted directly
(`tests/unit/test_facts_memo.py`).

`decode_step_gated` runs `DECODE_STEP` alone — the caller owns the verdict, at whatever
synchronization it already has, through `growth_pending` and `admit_growth`
([`decode.md#capture`](../decode/decode.md#capture)).

## <a id="capturable"></a>4. `capturable` is a declaration, not a hope

Every node of `DECODE_STEP` declares `capturable=True`, and
`rola.engine.runner.validate_capturable` refuses the tuple at import if one does not. A
node cannot declare both `capturable` and `host_sync`: reading a device value on the host
is precisely what a capture forbids, and the two attributes are one claim written from
two ends — which is exactly why `read_verdict` lives only in `DECODE_VERDICT`, appended
after the captured tuple, and never inside `DECODE_STEP` itself.

The declaration is what makes the property survive editing. A node added to the step that
reads a device value, or allocates per step, fails at import rather than at some caller's
first `graph.replay()` — and `tests/integration/test_decode_graph_step.py` is what proves
the declaration is TRUE of a real capture.

## <a id="host-carrier"></a>5. The carrier and the scratch, and why each is memoized the way it is

`DecodeGeometry` (`rola/engine/plan.py`) answers "what shape is the launch?" — widths,
per-level dial row offsets, the `(k, m)` lattice the state plane is written in
(`lattice_k`, `lattice_m`, [`decode_lattice.md`](../decode/decode_lattice.md)), `d_v`, the
decay arm, `eps`, `n_split`. None of that is a derived axis. `BC` is deliberately NOT a
field: owner-blocked AMORTIZATION is a property of how a tiled consumer shares a
`[BC, cols]` block's cost across many tokens, and with one token there is nothing to
amortize, so the decode kernels are BC-blind by construction in that sense — the
lattice's OWN `BC` (the leaf count one owner holds) is a layout fact the `(k, m)` fields
carry instead, never a launch-shape one
([`decode.md#bc-absent`](../decode/decode.md#bc-absent)).

It is `DecodePlan`'s CONFIG HALF rather than a field of it, because the memo key is over
this alone: `geometry` is declared `pure_of` `(widths, d_v, has_decay, BH, device)` and
the terminal facts beside it — the folded amplitudes, the plane, the verdict buffers —
are per-call and never cached.

`scratch` is declared `per_state`, which makes its table a `WeakKeyDictionary` keyed on
the sequence's state object. The weakness is REQUIRED and not incidental: a dropped
sequence must drop its buffers, and two live sequences must never share one. Being
memoized is also what hoists the allocation, and the once-only zeroing of `ctr`,
`growth_ctr` and `done`, above any captured region — the last-arriving CTA of every step
resets each of them, so no `memset` ever stands between two decode steps
([`decode.md#capture`](../decode/decode.md#capture)).

Because the launch config is FROZEN for the sequence, a state has exactly one live
scratch; `growth_pending` and `admit_growth` read it through that table and refuse a
state holding more than one, which is the observable form of the decay arm changing
mid-sequence.

## <a id="reference-only"></a>6. `reference_only` — the enumerated gate reference, refused mechanically

`write_atom_bitmap` is declared as a node and belongs to no DAG. It is the gate's
INDEPENDENT second derivation of the write-atom set, written from the definition rather
than from the kernel's structure
([`decode.md#second-derivation`](../decode/decode.md#second-derivation)); the shipped path
derives that set on the device, inside the step, as `atom_bits`. Splicing it into a
shipped DAG would make the gate compare the step against itself, and `validate_dag`
refuses any DAG containing a `reference_only` node — which is the mechanical form of the
warning the function's docstring used to carry alone.

## <a id="the-pool"></a>7. The slack pool is storage mechanics, and neither node nor mask

A step admits into its state's slack pool INSIDE the kernel, from the pool buffers
`backing` hands the plan — there is no node for it, because it is not a fact the host
finds: the claim and the fold-back are the kernel's own, elected from the split-K
arrival counter it already keeps ([`decode.md#the-claim`](../decode/decode.md#the-claim)).
`pool_headroom` and `refill_pool` (`rola/engine/dags/decode_dag.py`) are HOST reads and
writes off the step's path entirely — a serving loop polls the first at leisure and calls
the second on its own schedule, never as a precondition of a step. Pool-covered growth
raises no verdict, because nothing is urgent for the host: `growth_pending` stays `False`
and the batch-head simply advanced. Exhaustion is not a third path — an unpooled or
pool-exhausted batch-head is `needy` exactly as it always was, and the existing
`growth_pending` / `admit_growth` pair is what a caller reaches for it. The pool's own
design is [`paging/paging.md` §the-slack-pool](../paging/paging.md#the-slack-pool).
