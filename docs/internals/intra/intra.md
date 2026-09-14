# `intra.cu` / `intra.cuh` — the intra family's dispatch and vocabulary

The mirror of `csrc/rola/src/intra/intra.cu` and `csrc/rola/src/intra/intra.cuh`. One
translation unit: the support-mode vocabulary, the arm dispatch, the arm listing and the
device build stamp.

## The support modes are a DECLARATION, not a switch

Two bits per level, level 0 in the low pair: bit 0 says the READ side's rows are
value-sparse, bit 1 the WRITE side's. Neither bit is `DENSE_BOTH`, both is
`BOTH_SPARSE`. The kernel is pattern-generic — the slab mask machinery answers
all-ones for an undeclared side and therefore cannot gate — so `DENSE_BOTH` everywhere
is always CORRECT and only ever slower. That is what makes the mode word a statement
about the routing rather than a code path, and it is why a caller who declares wrongly
loses a harvest instead of an answer.

## The window is an instantiation axis

`W` pairs with the carry family's `W` template argument, because the two kernels run ONE
global window grid: an arm is a `(D, B, W)` triple on both sides. `kWindowFlagship = 384` is
the architecture doc's two-constraint window — the intra's ideal arithmetic intensity is
`W/4` flop/byte, which must clear sm_86's compute/bandwidth balance — and
`kWindowWide = 512` is the end of the admitted band that also DIVIDES a power-of-two
length, which is the third constraint and the reason both ends are built.
`kWindowSmall = 64` is the conformance cell's.

The LEVEL WIDTH is the second instantiation axis, mirroring the carry family's
topology arms: `kWidthFlat = 256`, `kWidthDeep3 = 16` and `kWidthDeep4 = 16`. The other
shape constants are the tile grid the causal staircase is expressed on: `kTile = 64`
output tokens, `kWarpRows = 16` rows to a warp, `kSlabDigits = 16` digits to an MMA
k-step, and `kValueWidth = 64`.

## It reduces into `o` and `den`

The kernel's write-back is a RED, not a store, so the caller owns those buffers and their
initial contents. The carry allocates and fills them; the intra then folds its
within-window term into the same buffers on the same stream. Nothing is summed on the
host, and there is no second output pair to reconcile — which is the whole reason the
intra was built as a reducer rather than as a returner.

## The stamp is a footprint and an ABI width

`intra_build_stamp` returns the heaviest built arm's shared-memory footprint together
with its thread count, folded against `sizeof(IntraParams)`, so the value moves whenever
any arm's SHAPE moves or the kernel's OPERAND SET does. A shape-only stamp cannot see a
signature change — the certificate rewire added two support-word pointers and moved no
footprint at all — and an extension trap is exactly the case where an unchanged stamp
would be read as an unchanged binary. A path check or a hash check cannot catch that
trap; a fact only the loaded binary can produce can.

## The kernel names are not anonymous

`nvcc` mangles an anonymous-namespace kernel with a hash that depends on the BUILD PATH,
so a manifest ratified in one worktree certifies a name no other checkout's build
produces and the fatbin gate then refuses a byte-identical kernel. A ratifiable kernel
needs a named symbol, which is why the stamp lives in a named `stampdet` namespace.

## Comments moved from source

Verbatim `//:` prose blocks from `csrc/rola/src/intra/intra.cu`, in source order, one heading per declaration. Source keeps only a short decl block pointing here.

<a id="note-l12"></a>
### near line 12

NOT an anonymous namespace: nvcc mangles anonymous-namespace kernels with a
hash that depends on the BUILD PATH, so a manifest ratified in one worktree
certifies a name no other checkout's build produces -- the fatbin gate then
refuses a byte-identical kernel. A ratifiable kernel needs a named symbol.

## Comments moved from source

Verbatim `//:` prose blocks from `csrc/rola/src/intra/intra.cuh`, in source order, one heading per declaration. Source keeps only a short decl block pointing here.

<a id="kmodereadsparse"></a>
### `kModeReadSparse`

Per-level sparse-side declaration, two bits per level, level 0 in the low
pair.  Bit 0: the READ side's rows are value-sparse.  Bit 1: the WRITE
side's.  Neither bit set is DENSE_BOTH; both set is BOTH_SPARSE, which says both
sides are sparse there and nothing more -- each votes its own support.  The
kernel is pattern-generic: the slab mask machinery reads all-ones for an
undeclared side and therefore never gates.

<a id="kwindowflagship"></a>
### `kWindowFlagship`

The window is the intra term's unit and the CTA's whole subject; the tile is
the [C x C] output block the causal staircase is expressed on; a warp owns
kWarpRows of a tile's rows and reduces nothing with its siblings.

THE WINDOW IS AN INSTANTIATION AXIS, mirroring the carry's `W` template
argument (integration rung I1(b)): the two kernels share ONE global window
grid, so an arm is a (D, W) pair on both sides.  `kWindowFlagship` is the
architecture doc's two-constraint window -- the intra's ideal arithmetic
intensity is `W/4` flop/byte, which must clear sm_86's compute/bandwidth
balance (~380) -- and `kWindowSmall` is the conformance cell's.

<a id="kslabdigits"></a>
### `kSlabDigits`

the MMA's contraction step, and therefore the QUANTUM a level's operand slab is
staged in.  A level narrower than one step is staged PADDED, its tail zeroed
once: there is no `k < 16` tensor op, so the digits below the step are the only
way to contract eight of them.

<a id="intra-forward"></a>
### `intra_forward`

`window` selects the built arm; `L % window` must be zero, refused per arm.
``sread``/``swrite`` are the producer's frozen support words for the plane of the
same name -- int32 ``[BH, ceil(L/32), sum-of-widths]``, one bit per (token, digit),
bit ``token & 31``.  The zero certificate READS them; it does not re-derive the
support from the amplitudes.

<a id="gstampprobe"></a>
### `g_stamp_probe`

Recompiled with the translation unit, so a test that reads it back is
reading a fact only the loaded binary can produce.

<a id="stampkernel"></a>
### `stamp_kernel`

THE HEAVIEST BUILT ARM's footprint and the launch ABI's own width: the pair
moves whenever any arm's shape or the kernel's operand set does, which is what
makes it a fact only THIS binary can produce.

<a id="checksupport"></a>
### `check_support`

The frozen support word, in the producer's own storage: int32 bit patterns, one
bit per (token, digit), 32 tokens to a word, DIGIT-MAJOR INNERMOST -- which is
what makes a warp's 32 lanes read 32 consecutive digits as one line.

<a id="intraarmsx"></a>
### `INTRA_ARMS_X`

THE BUILT SET, as one macro so the dispatch and the arm list cannot drift --
the carry family's `CARRY_ARMS_X` discipline, mirrored.  `W` pairs with the
carry's window template argument: the two kernels run ONE grid.

<a id="near-line-64"></a>
### near line 64

The block's whole reader side is resident, which puts the footprint past the
48 KB a kernel gets without asking; the opt-in is per device, so it is renewed
at every launch rather than cached behind a once-flag.

<a id="checksupport-2"></a>
### `check_support`

THE WORD AXIS IS 32 TOKENS, the producer's block (`entmax/factor.cu:236-239`),
and a window is a whole number of 64-token tiles, so `L` is a whole number of
words wherever an arm exists.

<a id="kwindowwide"></a>
### `kWindowWide`

the doc's band runs to 512, and a THIRD constraint picks it out: a window must
DIVIDE `L`, and 384 does not divide the flagship length 65536 while 512 divides
every power-of-two length.  Both ends are built and measured.

<a id="kwidthflat"></a>
### `kWidthFlat`

THE LEVEL WIDTH IS AN INSTANTIATION AXIS, not a constant: an arm is a
`(D, B, W)` triple, matching the carry family's topology arms.

<a id="intrabackward"></a>
### The reverse pass

Gone (`DELETIONS.md`, 2026-09-04): the reverse pass is rebuilt from its fp64 reference
(`tests/oracle/backward_reference.py`) on the position-carved form once the forward is
final.

<a id="intrabuildstamp"></a>
### `intra_build_stamp`

The device-side build stamp: a fact only THIS binary can produce, read back
by the test battery so that a green run against a stale extension is
impossible to mistake for a green run against the tree.
