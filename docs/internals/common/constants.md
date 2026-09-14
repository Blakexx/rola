# `constants.cuh` — the file constants, the launch table, and the S-from-SMEM rule

The mirror of `csrc/rola/src/common/constants.cuh`, and of its Python twin
`rola/engine/constants.py`. The axis law (`development/queue/K50_mo-runtime.md`,
`docs/ARCHITECTURE.md` §5) sorts every quantity the carry family reads into four
classes: **compiled** (`D`, `DV`), **derived** (`BC`, the spans, `kWarps`, `S`),
**runtime addressing** (`MO`, `B_l`, the orders, the row maps) and **constant** (`W`,
`C`). This file is where the CONSTANT class is declared and where the DERIVED
quantities that are pure arithmetic on constants and the register shape (`BC`, the
per-arch launch table, the S-from-SMEM rule) live, so no second declaration of any of
them can quietly disagree with this one.

**Symbols.** `W` the kernel's window, in tokens. `C` the batch tile. `DV` the value
width. `leaves_per_warp(dv)` the warp sub-box's leaf count (`common/geom.cuh`'s
derivation, consumed here, never forked). `warps_per_sm = 8` the register law
(`common/geom.cuh`). `warps_per_cta` the launch shape's one free field. `BC` the CTA's
box, `leaves_per_warp * warps_per_cta`. `S` a side's stream count.

<a id="window"></a>
## `kWindow` = 512

The axis law: `W` is a kernel-internal CONSTANT, never an argument. The inter/intra
decomposition (`docs/ARCHITECTURE.md` §6) is defined on this grid, so the fp64
reference has to mirror the same number or the two terms are not one recurrence. A
sequence shorter than a window runs a partial window inside the kernel; a caller never
selects a different `W`. `rola/ops/carry.py`'s own `WINDOW` constant is this same
value; a follow-on consolidation stage (`C_CLEAN_SLATE.md`'s C2e) is where that site
is repointed to import this module rather than restate the literal.

<a id="warps-per-cta"></a>
## `warps_per_cta` ∈ {8, 4} — K50 "the 2×4 issue"

`warps_per_sm` splits into `ctas_per_sm × warps_per_cta`. The CTA's box is the
ownership unit, carved per side among its own warps, so the member set the carve
generates is `f(D, leaves_per_warp, warps_per_cta)` — `warps_per_cta` is therefore
part of the compile-time arm key `(D, DV, warps_per_cta)`, never a runtime argument
(`rola/ops/carry.py`'s `LaunchShape`, C1c). 8 is the SHIPPED shape (one CTA per SM,
the register file filled exactly, KERNEL_STANDARDS §R15); 4 is the two-CTA latency
diagnostic named in `docs/ARCHITECTURE.md` §7 — it never ships, it exists only to
measure how far the one-CTA body is from hiding its own latency.

<a id="bc"></a>
## `BC = leaves_per_warp(DV) × warps_per_cta` — derived, never an axis

`leaves_per_warp` is `common/geom.cuh`'s derivation (`leaves_per_sm(dv) / warps_per_sm`
= `16384 / dv / 8`): 32 leaves at `DV ≤ 64`, 16 at `DV = 128`, uniform across
sm_80/86/90 today because the register budget the `16384` numerator encodes does not
yet vary by architecture (a part that changes it gets its own numerator, never a
branch on a compute capability). `box_leaves`/`BC` multiplies that by the launch's
`warps_per_cta` — a derived quantity from two other numbers, never itself a dial a
caller sets or a template parameter enumerated over.

<a id="launch-table"></a>
## The per-arch launch table

Data the seam reads, not a branch the kernel takes: for each architecture
`arch_caps.cuh` tabulates, the `(ctas_per_sm, warps_per_cta)` pair the shipped binary
launches at.

| `cc` | `ctas_per_sm` | `warps_per_cta` | why |
|---|---|---|---|
| 800 | 1 | 8 | the one-CTA shape (KERNEL_STANDARDS §R15) |
| 860 | 1 | 8 | |
| 870 | 1 | 8 | |
| 890 | 1 | 8 | |
| 900 | 1 | 8 | |

The table is ONE row repeated, and `launch_table_is_one_cta_per_sm` asserts that it is: one
CTA per SM holding every one of the SM's warps, on every architecture. The local dev part
carried a two-CTA row for as long as the retired body's shared-memory coupling made one CTA
of eight warps impossible there; the design's peak now fits that part's budget, so the
exception's reason is gone. `warps_per_cta = 4` stays in the arm key's DOMAIN as the
two-CTA latency diagnostic — a measurement of how far the one-CTA body is from hiding its
own latency — and no shipped row names it.

`sm_100` is not a row: `arch_caps.cuh` does not tabulate it yet
(`future-proof-for-hopper-blackwell`), so there is nothing this table could assert
about it without guessing; when a Blackwell row lands in `arch_caps.cuh` its launch
row follows sm_90's `(1, 8)` by the same register law, and this table's own
`launch_table_rows_are_all_tabulated`/`launch_table_covers` cross-checks (against
`arch_caps.cuh::tabulated`) fail loudly until that row is added here too.

`launch_row(cc)` REFUSES an untabulated `cc` rather than approximating it from a
neighbor — the same "never guess" discipline `arch_caps.cuh::caps_of` follows for its
own table.

<a id="segment"></a>
## `kSegmentTokens` = 16 — the stream's buffering grain

A SEGMENT is a 16-token slice of a stream's buffer. Sixteen because it is the MMA's `K`:
one segment is exactly one k-step of the fold, so the buffering grain and the arithmetic
grain are the same number and a segment is never partly consumed.

The value rows and the raw factor rows of one segment are the two things the fold's fill
stages; `carry/box.cuh` sizes both from this constant and `docs/internals/carry/
smem_ledger.md` names them.

It is the BUFFERING grain and not the chain length. A consumer chains every sealed segment
back to back, so a dense window's MMA chain is as deep as any; short chains occur only where
sparsity has left little work, which is the segment fixed cost's own crossover and not a
property of this constant.

<a id="cluster"></a>
## The cluster size

`kClusterCtas` lives in `common/geom.cuh` beside `warps_per_sm`, because it is a level of the
ownership hierarchy and not a launch-table field; see
`docs/internals/common/geom.md#cluster`.

<a id="s-from-smem"></a>
## The S-from-SMEM rule

K50: *"S (streams) = min(kWarps, what SMEM admits without losing residency)."* No
stream segments exist on this branch yet — the fold's private streams are C3-the —
so `derive_stream_count` is the RULE against a STATED layout, not a read of a real
one: given the bytes one stream's segment costs and the SMEM residency has left over
for streams, it returns the largest power of two at or below `warps_per_cta` whose
streams all fit, or `0` — a refusal, not a silent "run with no streams" — if even one
stream's segment does not fit.

```
S = max { s = warps_per_cta, warps_per_cta/2, ..., 1 : s * per_stream_bytes <= residency_smem_bytes }
    or 0 if no such s exists
```

K3 supplies the real `per_stream_bytes` (the own measurement: 72,960 B for the
literal per-stream V segment at one flagship cell, `K50_mo-runtime.md`, "THE CTA
POLICY") and the real `residency_smem_bytes` (the launch's SMEM budget minus every
other resident tenant — the transit region, the panels, the factor staging). Until
then, `tests/unit/test_carry_constants.py::TestDeriveStreamCount` is a MODEL test: it
states its own bytes and checks the rule's arithmetic (power-of-two search, the
floor-at-one refusal, monotonicity in the budget), never a real kernel's numbers.

## On the tip

Nothing on this branch calls any of these functions from a kernel yet — `C3-K0`
onward is what will (`development/queue/C_CLEAN_SLATE.md`). This stage's job is
naming the ONE PLACE these values live so no kernel stage re-derives or re-states
them: `csrc/rola/src/common/constants.cuh` for device code, `rola.engine.constants`
for the host, `tests/unit/test_carry_constants.py` the agreement gate between them.
`rola/ops/carry.py`'s pre-existing `WINDOW`/`WARPS_PER_CTA`/`leaves_per_warp`
and `common/geom.cuh`'s `warps_per_sm`/`leaves_per_warp` are read here, not
duplicated; repointing those call sites at this module is `C_CLEAN_SLATE.md`'s C2e
consolidation stage, not this one.
