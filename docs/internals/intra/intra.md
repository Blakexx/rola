# `csrc/rola/src/intra/intra.cuh` / `intra.cu` — the intra's constants, operand block and translation unit

Mirror doc for the header every intra source includes and for the one translation unit that derives a call's geometry
from its widths and forms on the device's card and builds the arms. The kernel is
[`ampere/intra_kernel.md`](ampere/intra_kernel.md); the host entries this unit defines, with their refusals, are
[`intra_api.md`](intra_api.md).

<a id="constants"></a>

## The constants

A window is the device's W ([`../common/window.md`](../common/window.md)) cut into `kTile` = 64-token tiles; a warp owns
`kWarpRows` = 16 readers; the values are `kValueWidth` = 64 f16 channels (`kValueBytes` = 128 a token). The rows serve
depth at most `kMaxLevels` = 2, each level `kDigitGranule` = 16 to `kMaxLevelWidth` = 256 digits, u8 or f16 on either
side. A copy moves `kChunk` = 16 bytes; a stage is `kStageBytes` = 128 bytes of a writer tile's rows -- 128 u8 or 64
f16 digits of one level (`form_stage_digits`), so the three buffers hold any level -- a level at most `kLevelStages` = 4
of them and a call `kMaxStages` = 8; a slab is `kSlabDigits` = 32 digits, the gate's grain. `kStageBuffers` = 3 stage
buffers. `kStagingBytes` is a warp's deposit staging tile (16 rows of 72 floats). A `LevelForm` names a level's
storage, bit 0 its read rows f16 and bit 1 its write rows f16. `IntraPhase` is the phase ledger's column order and
`kLedgerWarps` = 8 its rows.

<a id="block"></a>

## A card's walk

A card's walk is its build's PLAN, `Walk<B>` (`intra_kernel.cuh`; [`../common/hardware.md#builds`](../common/hardware.md#builds)),
read by the device code it compiles and by the host at the device's build (`Shape`, `shape_of`, through
`build_switch`: [`../common/select.md`](../common/select.md)) -- one derivation both ways, from the card's traits, no
table of compute capabilities (ruled 2026-10-31: W and the structural constants are a card's; 2026-11-07: every kernel
adapts to its card). The BLOCK -- the readers a CTA holds resident -- is the most the card's area holds at the widest
rows, (256, 256) f16 on both sides (1 KB a reader), a multiple of the tile and at most a row group of 16 a warp:
`kFits<128>` at the card's `smem_per_cta_max` gives 128 on sm_80 and sm_87 (164 KB), 64 on sm_86 and sm_89 (99 KB). A
CTA is 8 warps on every card, two a scheduler, so a warp's WRITER COLUMNS are what the block leaves: 64 where it is 128
readers (8 warps of one reader group each) and 32 where it is 64 (two warps a reader group). A block of 64 re-reads a
window's writer tiles twice as often as one of 128: the area's price, fixed per card, with no threshold inside one.

<a id="pieces"></a>

## The pieces

`kPieceTiles` = 4 caps a CTA's sweep: block `b`'s tiles `0 .. (b + 1) R - 1` (`R` the block's row tiles) cut into
pieces of at most four, so the diagonal pair lands in the block's last piece. `PiecesAt<W, Rows>` builds the window's
table at compile time, longest first and the heavier block first among equals, and the kernel reads its entry from
constant memory by `blockIdx.x` modulo the count (6 at W = 512 with 128-reader blocks, 12 with 64). Why four is
[`ampere/intra_kernel.md`](ampere/intra_kernel.md#the-grid).

<a id="stages"></a>

## The stages: a call's geometry from its widths and forms

Each level is read at its rows' own width `w`, a whole number of 16-digit granules from 16 to 256: a level has
`ceil(w / 32)` slabs, numbered across the levels in order (the gate word's bits, `slab0` a level's first), and its
stages are its writer rows' 128-byte slices, `form_stage_digits` digits each, the last holding the rest. A `Stage` is
one: its `index` (its reader part and its place in the copy order), `level`, first digit, first slab, slab `mask`, and
`reader_chunk`, its reader slot's first 16-byte chunk in a reader row -- the stage's digits in the reader's own
storage (64, 128 or 256 bytes). A level's stages are contiguous (`level_stage0`, `level_stages`), so the kernel walks a
level's under its form. A 16-digit tail is a slab's second half, which the copies zero-fill; a stage's slots past its
slabs are filled the same way and never read. `geometry()` derives all of it and the footprint -- the card's
`sizeof(SmemAt<W, Rows, Warps>)` plus its block of reader rows at `pitch`, the slots rounded up to 128 bytes, at least
three stages and every warp's staging tile -- and the same function answers `intra_footprint`.

The facts pass's union table is laid out by its own widths, the topology's padded ones, which are never narrower than
the rows: a level's union groups start at the layout's `group_base` (`group0`), and its slabs read the first
`ceil(w / 32)` of them.

<a id="intraparams"></a>

## `IntraParams` — the operand block, a grid constant

Per level a `RowView` of each side's rows (base, and the batch, token and head strides in bytes), its width, form,
first slab, first union group, first stage, stage count and slab mask; the stage table; the values' view, the facts
pass's `fm` and tile unions; the output planes; the ledger; the stage and slab counts, the union table's group count
and the reader row's pitch; `H`, the query heads a write head (`group`), `L`, the window count; level 0's lift and the
digits' `unit` ([`ampere/intra_kernel.md`](ampere/intra_kernel.md#widening)); and each side's dense-level bitmask. It is a
`__grid_constant__` kernel parameter of under a kilobyte, so a launch allocates nothing to learn its shape.

<a id="the-unit"></a>

## The translation unit

The arms are the depth and the accumulation, `intra_kernel<D, kF16>` at `D` = 1, 2: four; a call whose last window
is partial runs the four arms `kEven` false, `intra_kernel<D, kF16, false>`, compiled in `intra_partial.cu`
([`ampere/intra_kernel.md`](ampere/intra_kernel.md#partial-window)). The host entry reads the device's card, derives the geometry
from the rows' widths and dtypes, turns each tensor into a view after checking it (`row_view`: shape, 16-byte starts
and strides), fills the block and launches the arm on a `(windows × pieces, B·H)` grid of the card's threads with the
call's footprint, `windows` the call's tokens over W rounded up. A footprint past the 48 KB a kernel gets unasked raises
the arm's attribute to it, once a size: only a call wider than any before it pays the attribute call. Every footprint
fits the card: the walk is sized for the widest rows ([`#block`](#block), asserted at compile time).

The build stamp's kernel lives in a NAMED namespace, `stampdet`: nvcc hashes the build path into an anonymous
namespace's kernel names, and a manifest ratified in one worktree would refuse the identical kernel built in another.
