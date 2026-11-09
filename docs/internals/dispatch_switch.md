# `csrc/rola/src/dispatch_switch.cuh` — host-side compile-time dispatch

A kernel whose structural axes are template parameters has to be selected at runtime
from runtime values. This header is the mechanism for that selection in a hand-written
launcher: `entmax.cu`'s forward and backward launchers (`factor.cu`'s four launchers use
it the same way).

The carry and decode families select from a GENERATED ARM LIST instead. The carry's is
the declared shipped set (`tools/manifests/shipped_set.json`, keyed on the carve's plan
pair): `tools/gen_shards.py` emits `carry_selection.inc`, one X-macro over every built
arm, and `carry/carry.cu` expands `ROLA_CARRY_ARM_SET_X` four times — the arm
declarations, the launch dispatch through `arm_switch` ([`common/arm_switch.md`](common/arm_switch.md)),
the census and the per-arm row listing — so the enumeration exists once and is
single-sourced with the per-arm translation units it partitions into
(`instantiations/README.md`). Decode's is `ROLA_DECODE_ARM_SET_X`, generated the same way
(`decode_selection.inc` from `tools/manifests/decode_set.json`), an X-macro over the declared
`(DV, D, decay)` rows, tried in order and refused by name if none matches.

THE RULE BETWEEN THE TWO IS WHERE THE ENUMERATION LIVES. A launcher with a handful of
axes and one kernel writes its selection here, in the one place its `<<<...>>>`
configuration exists. A matrix large enough that spelling it out would be a SECOND
COPY of the instantiation set is generated instead — and once it is generated, a table
is the cheaper form than a nested switch, because the axis order then exists exactly
once instead of once per nesting level.

## <a id="one-launch"></a>1. The launch is written ONCE

<a id="note-l3"></a>The form this file refuses is the MACRO LADDER: a `#define` per axis whose body restates the
launch, so an `N`-axis dispatch spells the launch `2^N` times inside a preprocessor expansion the compiler cannot
type-check, the debugger cannot step and a reader cannot see -- exactly the place a `<<<...>>>` argument drifts
between arms. The form here is flash-attention's `BOOL_SWITCH` idiom with the macro removed.

The axis walkers are function templates. The launch is a generic lambda. Each axis
arrives as an `integral_constant` whose `::value` is a compile-time constant at the
point of use, so a launcher's template arguments and its `<<<...>>>` configuration
exist in exactly one place.

```cpp
int_switch<2, 4, 8, 16, 32, 64, 128, 256>(
    w_pad, "union entmax padded width (MAX_BRANCH_WIDTH = 256)", [&](auto WPAD) {
      constexpr int kWPad = decltype(WPAD)::value;
      constexpr int kLaneWidth = kWPad < 32 ? kWPad : 32;
      constexpr int kItemsPerThread = kWPad / kLaneWidth;
      bool_switch(alpha_is_1p5, [&](auto ALPHA) {
        union_forward_kernel<Bt, kLaneWidth, kItemsPerThread, decltype(ALPHA)::value, LogitT, OutT>
            <<<grid, BLOCK_THREADS, 0, stream>>>(/* ... */);
      });
    });
```

(`entmax.cu`'s `launch_forward_at<Bt, ...>`, the kernel at the build tag `Bt`; `launch_backward_at` and `factor.cu`'s
four launchers follow the same shape. Each `launch_*` above them is the one branch to the card's build,
`build_switch`, outside every launch: [`common/select.md#build-switch`](common/select.md#build-switch).)

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

<a id="near-line-14"></a><a id="near-line-25"></a>

| | axis | unmatched value |
|---|---|---|
| `bool_switch(value, body)` | one boolean | impossible — both arms exist |
| `int_switch<VALUES...>(value, what, body)` | one integer over a CLOSED set | `STD_TORCH_CHECK` naming the axis and the value |

**`int_switch`'s candidate set is part of the call.** The set is where the instantiated
arms are declared, so a runtime value outside it is a NAMED REFUSAL and never a
silently missing arm; `what` is the axis's name in that message.

<a id="near-line-31"></a>The candidates are tried left to right and exactly one arm runs. The one
`int_switch` set live in this engine is eight values wide (entmax/factor's
padded width, `{2, 4, 8, 16, 32, 64, 128, 256}`) and its `bool_switch` axes are all
two-way, so the expansion is a comparison chain rather than a jump table, which is the
right shape at that width.

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
| the instantiation SET | which `(template arguments)` tuples exist in the binary | `tools/ratify.py`, reading the built `.so`. A tuple that stops being instantiated fails it. |
| the SELECTION | which tuple a given set of runtime values reaches | the CUDA suites, which exercise every arm |

`tools/sass_bodies.py` hashes DEVICE entry points. Host `.text` is not in that hash, so
a SASS comparison is evidence about kernels and never about the host code that chooses
between them: it cannot tell a correct selection from one that reaches the wrong
instantiated arm. Only the suites can, which is why a dispatch is not verified without
running them.
