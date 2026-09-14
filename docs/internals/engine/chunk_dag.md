# The chunk family's DAG

Mirror of `rola/engine/dags/chunk_dag.py` and the chunk half of `rola/engine/facts/nodes.py`.
Framework vocabulary — masks, seeds, the fact bag, the Join — is
[the engine's page](engine.md); the band-2 folds this tuple walks are
[the rule layer's](rules.md).

Symbols: `C` chunk tokens, `BC` owner block, `N` leaf count, `BH = B*H`, atom = 16
consecutive leaves in canonical order (the page granule).

## <a id="the-two-masks"></a>1. M1 and M2 are one tuple

```
paging_preference   -> paging                        @host  RULE
topology            -> widths, d_v                   @host
classify_call       -> call_class                    @host
mask                -> mask                          @host  RULE
manifest            -> manifest                      @host
arm                 -> arm                           @host  RULE
pack_planes         -> read/write plane, g_write, read_mass   @main
pack_v              -> v_packed                      @main
facts_tables        -> union_table, atom_bits        @main
residency           -> state_in, arena               @side   [M2] host_sync
join                                                 @main   [M2] waits
backing             -> page_table, state_out         @host
block_bits          -> block_bits                    @main
run_table           -> run_table                     @host  RULE
build_plan          -> plan                          @host
```

`run_table` sits last before the terminal node rather than beside `backing`, which
produces two of the three presences it folds: a host rule placed between `backing` and
`block_bits` would step into the middle of the enqueue stream for no reason, and this
one has no consumer until the plan is built.

`backing` READS `state_in`, which is why it sits below `residency` rather than beside
it: a CONTINUATION's two state planes are one plane, so the exit plane is the entry
plane and only a fresh dense bind allocates
([`state.md`](../state.md#the-dense-continuation)).

M1 (prefill-stateless) is M2 (prefill-stateful) with the residency band masked out:
`residency` and `join` do not run and their facts are null, and `facts_tables` runs
under the call class that compiles the atom-grain emission out, so `atom_bits` is
null. The state plane is not allocated and the kernel variant with no state I/O
compiled into it is the one that runs, so a stateless call pays nothing for the
feature.

The critical path is `pack_planes -> facts_tables -> block_bits -> execute`. The three
rules, the topology, the manifest and the call class are host-only and sit on the
parallel front — which is why band 2 costs nothing: no rule is ever on the critical
path.

The mask is a FACT the MaskRule produces, and the runner reads it rather than the call
class: the band a node belongs to is a decision, and the walk consults the decision.
Every node before that rule runs under every mask by construction.

## <a id="the-absent-arm-edge"></a>2. One pass, both row sets, and a BC-free bitmap

The union table and the atom bitmap are ONE node because they are one pass over the
amplitude planes: the plane is `BH · T · sumW · 2` bytes, it is the pass' whole cost,
and two nodes would read it twice
([`liveness.md#what-makes-it-one-pass`](../facts/liveness.md#what-makes-it-one-pass)). The
node's `gives` is asserted at import to carry both facts, because a DAG that split
them again would pay those bytes back silently.

What must not follow from the shared pass is a bitmap that moves with the arm. The
published bitmap is keyed to the 16-leaf quantum and to nothing else, so the same
routing is the same residency at every built `BC` — that is what makes residency an
OUTPUT of the plan rather than a consequence of it. The atom rows are voted under
`LevelPlan<D, B, 16>` inside a `BC`-templated pass and their reduction is instantiated
`(D, B)` alone, and
`tests/oracle/test_atom_bitmap.py::test_the_bitmap_is_bc_free` holds the property
where it now lives: on the FACT, at every built arm.

**The ordering this buys and the ordering it costs.** `facts_tables` feeds
`residency`, so the pass now sits BEFORE the admission rather than straddling it. The
one host synchronization of the sequence is `plan_exact`'s size read, and the pass is
enqueued ahead of it — so the device is working through the facts pass while the host
blocks, which is the larger of the two windows. What is given up is the smaller one:
the admission's own side-stream launches no longer overlap a main-stream table pass.
The trade is stated as a measurement, not as an argument: the perf ledger (archived with the store's
pre-library record) carries the pass at two token counts, and the saving is the one that grows with `T`.

## <a id="the-join"></a>3. The Join sits before the block bitmap

`residency` runs the admission — the page-table write and the slot zero-init — on the
arena's side stream, and records the one event. `join` waits on it.

**Where the wait sits is a correctness fact.** `block_bits` READS the page table: the
`S` bit is a residency fact, so the block bitmap is downstream of the admission's
TABLE WRITE and not merely of the launch that follows. A wait placed immediately
before the consumer launch would leave a race between the side-stream table write and
the block bitmap's residency read. The Join is therefore upstream of `block_bits`,
never between it and `execute`.

## <a id="host-order"></a>4. The host order of a stateful call

The DAG's order IS the host order, and the two things that used to make it delicate
are now declarations rather than conventions:

* `residency` is the only node permitted a device-to-host read. `plan_exact`'s single
  size read is the only host synchronization in the sequence, and it is the read that
  also SIZES the admission. A fresh state plans `fresh=True` (call-scoped, the
  unpaged "allocate zeros every call" semantics); a continuation plans `fresh=False`,
  so residency becomes the UNION, which is what continuing a sequence means.
* `BC` reaches the state's extent policy from the plan's own `arm`, so the refusal
  check, the allocator policy and the launch cannot disagree about which arm this
  call runs. It is never stored on the state: it is allocator policy, and every
  ADDRESS is `BC`-invariant by the atom re-key
  ([`state.md` §4](../state.md#backing-policy)).

## <a id="entering-without-a-state"></a>5. Driving the arm without a state object

`ChunkContext` takes `state_in` and `arena` directly as an alternative to a
`RoLAState`. The `residency` node hands those straight back when no state is engaged,
which is how a test or a bring-up driver reaches the kernel arm without the facade's
state lifecycle — through the same DAG, the same plan and the same launch.
