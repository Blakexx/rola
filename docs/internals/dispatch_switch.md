# `csrc/rola/src/dispatch_switch.cuh` — host-side compile-time dispatch

A kernel whose structural axes are template parameters has to be selected at runtime
from runtime values. This header is the mechanism for that selection in the
hand-written launchers: `decode.cu`'s `launch_decode_step` and `entmax.cu`'s forward
and backward launchers.

The carry family does not use it, and does not use a switch at all. Its dispatch
walks a GENERATED ARM LIST: `tools/gen_shards.py` defines `CARRY_ARMS` (the closed-world
`(D, DV, warps_per_cta)` table) and emits `carry_selection.inc`, one X-macro over every
SHIPPED arm, and `carry/carry.cu` expands that same macro four times via
`ROLA_CARRY_ARM_SET_X` — the arm declarations, the launch dispatch, the census and the
per-arm row listing — so the enumeration exists once and is single-sourced with the
per-arm translation units it partitions into (`carry_arm_0.cu`,
[`carry/carry_kernel.md`](carry/carry_kernel.md), `instantiations/README.md`).

THE RULE BETWEEN THE TWO IS WHERE THE ENUMERATION LIVES. A launcher with a handful of
axes and one kernel writes its selection here, in the one place its `<<<...>>>`
configuration exists. A matrix large enough that spelling it out would be a SECOND
COPY of the instantiation set is generated instead — and once it is generated, a table
is the cheaper form than a nested switch, because the axis order then exists exactly
once instead of once per nesting level.

## <a id="one-launch"></a>1. The launch is written ONCE

The axis walkers are function templates. The launch is a generic lambda. Each axis
arrives as an `integral_constant` whose `::value` is a compile-time constant at the
point of use, so a launcher's template arguments and its `<<<...>>>` configuration
exist in exactly one place.

```cpp
int_switch<32, 64>(d_v, "decode d_v", [&](auto DV) {
  bool_switch(global_norm, [&](auto GLOBAL) {
    bool_switch(decay, [&](auto DECAY) {
      int_switch<1, 2, 3, 4>(p.D, "D (spec §1)", [&](auto LEVELS) {
        decode_gemv_kernel<decltype(DV)::value, decltype(LEVELS)::value,
                           decltype(DECAY)::value, decltype(GLOBAL)::value>
            <<<dim3(p.n_split, p.BH), kDecodeThreads, smem, stream>>>(p);
      });
    });
  });
});
```

**Uniqueness of the launch text is the property this file exists to hold.**
`entmax.cu`'s forward launch takes FIFTEEN arguments across thirty-two selectable
arms. In a form where each arm carries its own copy of that argument list, an argument
that disagrees between two copies is a wrong-result bug on whichever arm the suite
reaches second, and nothing checks the copies against each other. With one copy the
question cannot arise.

Two further properties follow from the body being real code rather than a
preprocessor expansion: a swapped axis is a TYPE error at the call site, and every
line has a source location a debugger and a profiler can name.

## <a id="axes"></a>2. The two primitives, and why only two

| | axis | unmatched value |
|---|---|---|
| `bool_switch(value, body)` | one boolean | impossible — both arms exist |
| `int_switch<VALUES...>(value, what, body)` | one integer over a CLOSED set | `TORCH_CHECK` naming the axis and the value |

**`int_switch`'s candidate set is part of the call.** The set is where the instantiated
arms are declared, so a runtime value outside it is a NAMED REFUSAL and never a
silently missing arm; `what` is the axis's name in that message.

The sets in this engine are two to six values wide, so the expansion is a comparison
chain rather than a jump table, which is the right shape at that width.

**A derived arm parameter stays derived.** `entmax.cu`'s width axis carries a
`(lane width, items per thread)` pair, computed from the padded width inside the
lambda:

```cpp
constexpr int kLaneWidth      = kWPad < 32 ? kWPad : 32;
constexpr int kItemsPerThread = kWPad / kLaneWidth;
```

A table of pairs, one row per width, is a second statement of an arithmetic relation,
and a second statement can disagree with the width it was chosen for. The relation is
stated once.

## <a id="what-is-gated"></a>3. What the gates establish about a dispatch, and what they do not

A dispatch has two halves, covered by different things. Which is which is worth
stating: a reader who assumes the binary gates cover both will trust an argument they
do not have.

| half | what it is | what covers it |
|---|---|---|
| the instantiation SET | which `(template arguments)` tuples exist in the binary | `tools/ratify.py` (347 entries per arch: 17 decode + 320 entmax + 10 intra), reading the built `.so`. A tuple that stops being instantiated fails it. |
| the SELECTION | which tuple a given set of runtime values reaches | the CUDA suites, which exercise every arm |

`tools/sass_bodies.py` hashes DEVICE entry points. Host `.text` is not in that hash, so
a SASS comparison is evidence about kernels and never about the host code that chooses
between them: it cannot tell a correct selection from one that reaches the wrong
instantiated arm. Only the suites can, which is why a dispatch is not verified without
running them.

## Comments moved from source

Verbatim `//:` prose blocks from `csrc/rola/src/dispatch_switch.cuh`, in source order, one heading per declaration. Source keeps only a short decl block pointing here.

## More comments moved from source

Verbatim trailing/banner `//` comments from `csrc/rola/src/dispatch_switch.cuh` condensed at their call site into a short pointer.

<a id="note-l3"></a>
### near line 3

HOST-SIDE COMPILE-TIME DISPATCH, AS FUNCTION TEMPLATES TAKING A GENERIC LAMBDA.

A kernel with `N` structural template axes has to be selected at runtime from `N`
runtime values, and the shape that selection takes is a real engineering choice.
The form this project refuses is the MACRO LADDER: a `#define` per axis whose body
restates the launch, so an `N`-axis dispatch spells the launch `2^N` times inside a
preprocessor expansion the compiler cannot type-check, the debugger cannot step and
a reader cannot see. It was measured to be exactly the place a `<<<...>>>` argument
drifts between arms.

The form here is flash-attention's `BOOL_SWITCH` idiom with the macro removed: the
axis walker is a function template, the launch is written ONCE as a generic lambda,
and each axis arrives as an `integral_constant` whose `::value` is a compile-time
constant at the point of use.

WHAT THIS DOES AND DOES NOT CHANGE. The set of DEVICE instantiations is identical --
the same template arguments are named, so the fatbin carries the same entry points
and the ratification manifest is unmoved, which is a gate and not a hope. The HOST
selection code is genuinely different (a comparison chain over the candidate set
rather than a `switch` statement), it is not device code, and no SASS comparison can
speak for it; what covers it is the CUDA suites, which exercise every arm.

The idiom, what it replaced, and the axis-order rule: docs/internals/dispatch_switch.md

<a id="near-line-14"></a>
### near line 14

ONE BOOLEAN AXIS. `body` is called with `std::true_type` or `std::false_type`, so
`decltype(FLAG)::value` is a template argument at the call site.

<a id="near-line-25"></a>
### near line 25

ONE INTEGER AXIS OVER A CLOSED SET. The candidate values are template arguments,
so the set is part of the call and an unmatched runtime value is a REFUSAL rather
than a silently missing arm -- `what` names the axis in the message.

<a id="near-line-31"></a>
### near line 31

An expansion over the candidate set, evaluated left to right: exactly one arm
runs. The sets in this engine are two to six values wide, so the chain is not a
jump table and does not need to be.
