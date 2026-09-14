# `common/static_for.cuh` — the family's compile-time loop

Mirrors `csrc/rola/src/common/static_for.cuh`.

<a id="static-for"></a>
## `static_for<N>(f)`

`f` is called once per index `0 .. N-1`, in order, with the index as a
`std::integral_constant<int, I>` — so the body may use it in a template argument, a
`constexpr` expression or an array subscript the compiler resolves. There is no runtime
induction variable and no branch: the recursion is fully expanded at compile time.

It exists as its own header, rather than inside the first body that wanted it, because
two of the family's foundations need it and neither may depend on the other:
`common/structure_switch.cuh`'s `uniform_switch` walks the generated member set with it,
and the carry body walks its accumulator's fragment modes with it. One loop, not two.

## What it is FOR, and what it is not for

Constitution §1: register-resident data exists only as MMA fragments, and a loop over
register-resident state iterates FRAGMENT MODES — m-tiles, n-tiles, k-steps — with every
finer index folded into constexpr address math. `static_for` is how that loop is written,
and its body count is therefore stated at the call site as a function of the arm.

It is not a general unroller. A loop whose trip count is a runtime quantity carries
`#pragma unroll 1` and stays a loop (§6); expanding it here would be the per-leaf
unrolling that once put 4,096 bodies in one nest.
