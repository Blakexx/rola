# `csrc/rola/src/common/cost.cuh` — the cost of a form's load

What a step form costs on the two pipes that bind it: its MMAs on its busiest scheduler and its shared-memory
traffic on its SM, the larger. A choice among exact forms -- a sparse side's digits stepping in groups of g --
takes the cheapest by this cost, per warp and window, from the window's exact counts; every form is exact, so the
choice changes speed, never results.

<a id="cost"></a>

## The cost

A form's `Load`: the MMAs of each kind on its busiest scheduler; on its SM, the shared-memory load wavefronts,
and the atomic instructions with their wavefronts. Its cost, in quanta (1/256 cycle, integer, so the same load
always prices the same):

- tensor: each kind's MMAs at its rate;
- shared memory: the load wavefronts at the wavefront rate, plus the larger of the atomics at an atomic's least and
  their wavefronts at a wavefront's.

The max assumes the two overlap, as a kernel built to the ideal regime makes them. Issue is not a term: a form
whose step cannot hide its instructions under its MMAs is not admitted on that architecture (the issue-hiding
gate, which reads the issue rate and the taxes from [`hardware.json`](hardware.md)), so for every admitted form
issue never binds. Global traffic -- staging a window's tokens -- is the same for every form and drops out.

<a id="quanta"></a>

## Quanta

`Quanta` holds a product's rates for the cost: an MMA of each kind (in `Mma` order), a load wavefront, a shared
atomic's least and each of its wavefronts. They are resolved on the host -- `measure/hardware.py`'s `quanta` over
`resolve(product=...)` -- and reach a kernel at launch, since a binary is built per compute capability and the
product is known only at run time.

The header's two `static_assert`s price two loads with the RTX 3080 Ti's quanta, one bound by MMAs and one by
shared memory; `tests/unit/test_hardware.py` resolves the same quanta from the configuration and prices the same
loads in Python, so the constants and the arithmetic agree on both sides.
