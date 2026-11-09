# `csrc/rola/src/common/arch_caps.cuh` — the card's facts at the compiling target

Its rows are data in [`hardware.json`](hardware.md), generated into `common/hardware.cuh` (`Caps`, `tabulated`,
`caps_of`): per architecture the INSTRUCTIONS an atom reads and the NUMBERS a plan reads. This file names the compiling
target (`kArch`, `kCaps`) and the row's digest. STRUCTURE is not here: which warps do what, the pipeline, the tiling
are a family's schedule, selected by the build's family (`Family`, [`hardware.md#families`](hardware.md#families);
KERNEL_STANDARDS section 8), never derived from capability flags.

<a id="note-l125"></a>
## The row's fields

The generated rows initialize the fields by name, in this order, and an untabulated compute capability gets an
empty row (`tabulated` refuses it first).

| field | kind | what it states |
|---|---|---|
| `bf16_cvt_packed` | INSTRUCTION | the packed two-float to bf16 pair conversion (`cvt.rn.bf16x2.f32`), asserted by the atoms that pack |
| <a id="reg-transpose-b16"></a>`reg_transpose_b16` | INSTRUCTION | an 8x8 b16 tile spread across the MMA unit's own registers is transposed IN PLACE. A contraction over the accumulator's own m axis puts the accumulator on the k side, and the accumulator map and the k-side map differ by exactly this transpose. |
| <a id="bf16-simd-arith"></a>`bf16_simd_arith` | INSTRUCTION | `mul.rn.bf16x2`. ptxas rejects it below sm_90 (measured 2026-08-09): Ampere has bf16 conversion and bf16 MMA but no bf16 SIMD multiply, so the product costs 1 FMUL + 1/2 pack an element there against ~1/2 instruction on sm_90 -- the archetypal flag |
| <a id="bf16-simd-fma"></a>`bf16_simd_fma` | INSTRUCTION | `fma.rn.bf16x2`, available at sm_80 (measured 2026-08-28): a product is one instruction through a `-0.0` addend, bit-identical to `mul.rn.bf16x2` (one rounding of the exact product; the addend's sign keeps a signed zero) |
| `smem_per_sm`, `smem_per_cta_max`, `regs_per_sm`, `max_threads_per_sm` | FACT | the SM's budgets the admission checks against the driver and the plans read (an area's shared memory, the state's registers) |

AN INSTRUCTION FLAG is read by an atom, where only the instruction differs between architectures of one family
(`docs/internals/common/atoms/math.md`'s bf16 pair arithmetic picks its form by `bf16_simd_arith` and `bf16_simd_fma`).
A FACT is an input to a plan (the shared memory an area holds) or to the admission; a fact nothing reads is not tabulated. Neither is ever a branch
of a schedule. PER-SM FACTS LIVE HERE, PER-DEVICE FACTS DO NOT: SM count is not an architectural constant (GA102 ships
at 80 and 82 SMs, GA100 at 108), so it is a run-time `cudaDeviceProp` query.

## The queries over a row

<a id="caps-of-note-l147"></a>**sm_90 is TABULATED AND NOT BUILT** (`caps_of`): its row carries its facts for the hopper
family's schedules, which are not written yet; a binary built without sm_90 refuses an sm_90 card by name
([`select.md#card-of`](select.md#card-of)).

<a id="tabulated"></a>**`tabulated` is the RATIFIED SET.** A compute capability that is not a row fails the build; it
is never approximated by a neighbouring row.

<a id="capsdigest"></a>**`caps_digest`** is a digest of the active row's instruction flags (`kCapsDigest` at the
compiling target), written by the carry's stamp kernel and read on the host. A binary compiled against a different
table cannot masquerade as this one -- no path or file hash can establish that, only a device-side fact.
