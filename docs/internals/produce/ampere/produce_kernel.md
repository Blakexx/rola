# `csrc/rola/src/produce/ampere/produce_kernel.cuh` — the producer's kernel on ampere

RULED 2026-09-29: the producer is a declared configuration compiled into ONE kernel launch -- from a stretch of
tokens to the bundle the ops take: every routing level's projection, solve and unit-max rows, the write gain, and the
f16 values. The layer declares its routing (`rola.Routing`), owns the weights, and compiles a producer plan once
(`rola/ops/produce.py`); `job_table` turns the plan into the rows this kernel reads, so every width, kind and offset
is a runtime field and the family is ONE instantiation.

`rola.ops.produce.produce` runs this kernel on CUDA when no gradient flows (`run_kernel` allocates the outputs;
`_kernel_bundle` views its packed planes as the bundle's per-level rows and scales); the torch composition behind the
same call is the fp64 and host reference, and until the native backward the path a gradient takes.

The components land one at a time into the skeleton, each gated (KERNEL_STANDARDS §19); a component not yet landed
has its real signature, tenant and edges and does the cheapest thing that keeps the composition's shape (zero digits
and scales). THE REGISTER-RESIDENT REBUILD: LANDED the projection into
the accumulators, and on them every job's solve, bundle and values: one-row jobs, narrow unions in one warp, wide
unions across a warp pair. `tests/integration/test_produce_kernel.py` holds the skeleton's claim (every element of every output is
written, whatever the routing) and each landed component's gate.

## <a id="jobs"></a>The job table

One int32 row per job (`JobField`), its ADDRESSING: the head (the first of a value job's heads, and their count), the
level's width, the `route_W` columns of its one or two logit rows, the packed-plane columns its rows land at on each
side (`-1` for a side it does not write) and the level. Its CONTROL -- the kind, the solve (`SolveKind`), the solve
row's width class, the projection's column-tile class and whether it is a wide union's pair form, the sides it writes
and which of them is dense -- is one SHAPE WORD (int16) a job in the launch's
parameters (`ProduceParams::shapes`, `rola.ops.produce._shape`), read by block index at each `load_job` (made opaque,
so no k-loop holds it: on sm_80 the held word was the kernel's one spill): a value ptxas knows is warp-uniform, so a branch on it compiles its collectives inline alone, where a branch on a loaded job field gave each
a guarded fallback subroutine as well (692 of them: cold code, KERNEL_STANDARDS §20). The parameters are
`__grid_constant__`, so the body's lambdas may hold their address without copying them to local memory (a copy made
every shape word a local load, and every branch on one a fallback). At most `kMaxJobs` jobs a launch.
A ROUTE job is one head's one solve: a union level's two logit rows (read and write), a tied level's one row filling
both planes, or one side of an independent level. A VALUE job is the heads whose channels fit one job (`256 / d_v`
of them, at most `kMaxColumns` channels). A row is at most `kMaxWidth` leaves: the widest a warp solves.

<a id="grid"></a>THE GRID is token tiles of `kTokenTile` (64) tokens by jobs, the tiles fastest: the blocks resident together
are one job's tiles, so at prefill its weights are read from DRAM once and served to the rest from L2. A block is 4
warps; two blocks an SM (`__launch_bounds__(128, 2)`, so up to 255 registers a thread: a warp's 128 accumulators and
its working set without spilling; the carveout set to shared), so one block's solve runs beside the other's MMAs.

## <a id="accumulate"></a>The projection

The job's columns for the block's tokens, fp32, into REGISTERS: the MMA's M is the job's columns (16-column tiles) and
its N a warp's 16 tokens in two n-tiles (`m16n8k16` bf16 with fp32 accumulate), so each weight fragment feeds two
MMAs; a lane holds, a column tile and an n-tile, two columns (`g`, `g + 8`) for two tokens (`2t`, `2t + 1`), and a
token's row lies across the 8 lanes sharing `lane & 3` -- 32 values a lane at 256 columns, four tokens a lane, a row
reduction three shuffles. Nothing passes through shared memory but the operands.

<a id="form"></a>A JOB'S FORM is its column count against one warp's 16 tiles (`kWarpColumns`, 256): a one-row job (a
split, tied or independent side, a union of width at most 128 -- its write row at its width class's tile offset,
`32 << rows` columns (`job_column`), so a column's two logits share a lane -- a value job's heads) is every warp's whole
row for its 16
tokens, the tiles in the job's class (4, 8 or 16: the shape word's -- a narrower job runs at 4 tiles until narrow
rows are packed into one job, its stage otherwise its x rows' copies under sm_80's MMA rate -- chosen once a block by `common/structure_switch.cuh`'s `uniform_switch`), 64 tokens a block;
a WIDE union (two rows past 256 columns) is a warp PAIR a token group -- warp `2j + h` columns `128 h ..` of both rows,
eight tiles a row, so each warp holds both rows of the same columns and forms their midpoint and difference in its own
registers -- in two 32-token passes, the threshold's row sums crossing the pair.

<a id="stage"></a>THE STAGE is `S` k-steps (16 hidden each) of every column the job holds and of the tile's x rows, staged
once a block and read by every warp (the weights' one read a block). Its depth is the job's class: `32 / M` k-steps
at `M` column tiles, at most four (two at 256 columns, four at 128 and at 64), so a one-row job's weights are 16 KB a
stage (8 KB at 64 columns) and its x rows follow them (to 8 KB), and one for a wide union's 512 columns (16 KB) -- 64
MMAs a warp between barriers at 128 and 256 columns, with each tile's A fragment loaded a tile ahead of its MMAs
(`mma_stage`). Two k-steps a stage, not one, because a warp issues in order: at one the 256-column class barriered
every 32 MMAs and issued its 17 `ldmatrix` in one burst ahead of them (measured 2026-11-01: two k-steps a stage, 2.88 ms). A row is 32 S bytes in 16-byte chunks, chunk `c`
at `c` XOR the row's swizzle (`chunk_at<S>`: `(row >> 2) & 1`, `(row >> 1) & 3`, `row & 7` at one, two and four
k-steps), so an `ldmatrix`'s eight rows at one chunk lie in eight bank groups at every depth -- A by `ldmatrix.x4` a
column tile, B by `ldmatrix.x4` the warp's 16 tokens, one wavefront a matrix, the copies four wavefronts a warp
(`tools/bank_model.py`). Each thread's copy sources are formed once a job as 32-bit element offsets (`Sources`, the
host bounding x and the weights under 2^32 elements); a stage is one `cp.async` group of every thread (`stage_step`),
a chunk past the hidden size zero-filled and a column past the job's never issued (only dead accumulators read it).
Two stages (`kProduceSmem`, 48 KB a block, two blocks ~98 KB of sm_86's 100): at a stage's barrier its copy has
landed and the other is released, and its refill issues there, a stage ahead of its MMAs (`k_loop`,
KERNEL_STANDARDS 4's sm_80 form). The routing bias SEEDS the accumulators before the loop (`seed_bias`, a predicated
load a column, while no accumulator is live); the logit probe is written from them after it (`probe_logits`).
A warp whose tokens are past the call's multiplies zeros: at prefill only a call's last tile has such warps.

## <a id="route-row"></a>A token's row

One warp a token row: the solve on the row, then the bundle's form of it. Every warp makes the same passes over the
tile's rows (a count of its live tokens, uniform), a row past them solving and reducing zeros with its stores off, so
no collective sits under a branch ptxas cannot prove warp-uniform.

<a id="bundle"></a>THE BUNDLE'S FORM (`bundle_rows`, `bundle_side`), one n-tile at a time: after the projection a warp
PARKS its second n-tile's accumulators in the released stages (`kParkedWarpBytes`, 16 bytes a lane, lane-major) and
solves and bundles each n-tile on 64 live accumulators, so the solve never sits at the register cap. The two sides are
one `#pragma unroll 1` loop, the side's operands selected at run time, so a class carries two `bundle_side` copies
(the f16 plane and the u8 plane), not one a side and plane. A side's rows are
staged one token slot at a time -- a lane quad's 4 rows, `kStagedRowBytes` a row, the widest slot plus 16 bytes so the
four rows' bytes lie in distinct banks (`tools/bank_model.py`: a byte store one wavefront, the read-back four) -- and
stored 16 bytes a lane; a coefficient is the level times `unit`, its row's reciprocal max at the plane's full scale
(one quotient a row, within an fp32 unit of the IEEE quotient, never its slow-path call). A u8 digit is ONE FFMA,
`level unit + 1.5 2^23`: the sum's low mantissa bits hold the nearest-even integer of the exact product, and the
byte store keeps that byte (`F2I` is a conversion-pipe instruction, and the old form two multiplies before it).
Every value of the class is staged, a dead column's too (its level 0, a 0 digit); a staged column past the slot is
never stored. A dense row's floor lands on its dead columns as well, so an f16 side whose width is not a multiple
of 16 zeroes its slot's tail (under 16 a row) after the values. For each side the job writes (the shape word's bits: whether it
writes it and whether its level is dense): the row against
its max as u8 digits (round to nearest even of `level x unit`) -- or, for a dense level, f16 at 1 with the floor
taken in a live row (`max(level / max, 2^-13)`, an IEEE divide rounded to nearest, the torch quantizer's own
rounding) -- the max clamped at `1e-30` as the forms' `quantize` (`rola.routing.quant.Form`) clamps it; and the max as the side's scale: the
level as its solve produced it, whatever its sum ([`docs/CONSTITUTION.md` R23](../../../CONSTITUTION.md#r23)). A side
reduces only what the solve did not give: a one-activation row's max is its top element's own value (`top`:
`(-tau/2)^2`, `-tau` or `1 / sum`, the same expressions the elements are), so only a union's sides reduce their max.
Gate: every row's digits (within one), f16 units (within one unit in the last place at 1), scales (within twice the
solve's bound; write level 0's, the gain folded in ([#gain](#gain)), within the gain times that plus 2^-12 of it)
against the fp64 composition on the same bf16 operands, with and without a gain column or a routing bias; a dense level's f16 rows within one digit's worth
(1/255) of it, every live coefficient at or above the floor.

<a id="dense"></a>A SIDE'S TWO PLANES: its sparse levels in a u8 plane (`amp_read`, `amp_write`) and its dense levels
in an f16 plane (`dense_read`, `dense_write`, `[B, H, T, rows]`), each level's slot its bytes rounded up to the row
alignment law's 32 (`rola.ops.produce.row_layout`); a route job writes each side to the plane its level's bit in the
side's dense mask names, at its offset there.

EVERY OUTPUT IS STORED HEAD-MAJOR, `[B, H, T, ...]` (`slot_of`: the flat token split into its batch and step), and the
bundle's `[B, T, H, ...]` tensors are views of it. At one token the two orders are the same bytes; over a prefill a
head's token rows are contiguous, and every prefill reader walks one head's tokens -- the facts pass measured 212
us a head at `(256, 256)`, `L = 64K`, `H = 16` on token-major rows (an 8 KB token stride, DRAM at 360 GB/s) against
113 us head-major (676 GB/s), the rate of a single head (`docs/internals/facts/facts.md#the-grid`).

<a id="solve"></a>THE SOLVE runs on the accumulators (`solve_tile`, a row class's body `solve_class`): a lane holds four token rows (`row_value`: its
two n-tiles' two tokens), 32 values each at 256 columns, and a row's reductions are a pairwise tree over the lane's
values (`row_tree`: log2 of the count dependent steps, where a running fold is the count) and three shuffles over the
8 lanes sharing `lane & 3` (`row_all`). A route job's DEAD columns (past its width) seed at -inf (`seed_bias`; their
weight rows, never copied, are zeroed in both stages once a block by `k_loop`, so they stay there), so no pass tests membership: the max ignores them, the softmax's `2^-inf` is 0,
every threshold lies above them, and every level of one is 0. The two n-tiles are one rolled loop (the second parked,
[#bundle](#bundle)), so the solve and the bundle are one copy each in the kernel. The softmax is `2^(z log2 e - m log2 e)` (one FFMA and `ops::exp2_approx`).
The row's class (2, 4, 8 or 16 column tiles, its width's) and its activation are the shape word's, so every
branch over them, and every shuffle and vote under it, is warp-uniform to ptxas; no value past the class's tiles is
touched. A tied or independent row is its one logit row: softmax (`softmax_rows`) or entmax 1.5 or 2
(`entmax_rows`), the closed registry's kerneled activations, with the reference `rola/routing/entmax/production.py`'s
`_torch_activation`. A UNION's split -- entmax on the midpoint of its read and write logits, the read side the value
times the sigmoid of their overflow-free difference, the write side the entmax family's `union_write`
(`docs/internals/entmax/entmax.md#union-write`), reference `_torch_union_split` -- runs on the same registers for a
NARROW union ([#union](#union)) and, across a warp pair, for a WIDE one ([#wide-union](#wide-union)). Gate: every route job's sides (`probe_levels`) against the fp64 reference solve on the
kernel's own logits, per slot within `(width + 16) 2^-23`.

<a id="union"></a>A NARROW UNION (at most 128 columns a row, both rows in a warp's 16 tiles: the read row at values
`v`, the write row at `v + 2 M`, the class's tile offset) is two halves around the row's one entmax, so the class
carries one entmax copy: `union_midpoints` leaves each column's midpoint `(r + w) / 2` at `v` and the logits'
overflow-free difference (`row_solve::overflow_free_delta`; a dead column's, both logits -inf, is 0) at `v + 2 M`;
the entmax solves the midpoints; `union_sides` forms the read side `level / (1 + e^-d)` in place and the write side in
the log domain over the support (a supported level is positive -- the reference's `z > tau` support differs only where
a level underflows to 0, whose write weight is 0 either way): `log level + log sigmoid(-d)`, its max subtracted,
exponentiated, normalized, so `top_write` is the argmax's own quotient `1 / den`; `top_read` is reduced. The forms are
the special-function unit's (`__logf`, `__expf`, `__fdividef`: within ~1e-6 at these ranges, under the solve's
`(width + 16) 2^-23`), so the split is ~4.5K instructions of the kernel, not the ~9K of the IEEE forms. The bundle's
side loop moves the write row down to `v` once the read side is stored, so a union's sides share the one-row
`bundle_side`.

<a id="wide-union"></a>A WIDE UNION (past 128 columns, the pair form: warp `2 j + h` holds columns `128 h ..` of both
rows for the pair's 16 tokens) is solved pass by pass on its accumulators with the narrow union's value layout (the
read half at `v`, the write half at `v + 16`), each warp its 128 columns: every row reduction -- the max, the Newton
sums, the union's write max and normalizer, the read side's max -- combines the pair's halves
through `Pair`'s exchange past the parked and staged rows (one 64-thread named barrier a round, two buffers
alternating), the two warps forming the same bits (the ops commute), and the Newton exit is the pair's vote voted
again, so ptxas sees it uniform (a value read from shared memory is not: unvoted it guarded every collective). Each
warp bundles its half row (`col0`); the half-0 warp stores the scale, and after the passes folds the write gain into
write level 0's ([#gain](#gain)).

<a id="threshold"></a>THE THRESHOLD IS NEWTON FROM THE LEFT (`entmax_rows`), not a sort. With the row shifted by
its max, `f(tau) = sum_i g(z_i - tau) - 1` (`g(d) = max(d, 0)^2 / 4` at alpha 1.5, `max(d, 0)` at 2) is convex and
decreasing, and `f(-2) >= 0` (`f(-1)` at alpha 2): Newton from there never passes the root, and each step needs only
the count, sum and sum of squares of the support `{z > lo}` -- one compare and three predicated adds a value
(`sums_past`), and the settled test's count one compare and one predicated add (`count_at`), never a select: a select
and a compare are each an ALU-pipe instruction at half the FP32 rate, and the masked form issued 6.6 of them a value a
step (812 instructions a step at 256 columns, now 513). The levels are `relu(h (z - tau))` times `fma(base, w1, w0)`
-- `base^2` at alpha 1.5's `(1, 0)`, `base` at 2's `(0, 1)`, both exactly -- so the activation is no select a value. The closed form on that support -- the formula the sorted
solve evaluates at its chosen rank -- IS the root once no member lies below it, which one vote over the warp's 16
rows decides (a vote is warp-uniform to ptxas, so the loop's exit outlines nothing; a settled row's further steps
keep its threshold, the closed form on an unchanged support). At alpha 2 the steps are Michelot's iteration. Measured
in fp32 over Gaussian, uniform, tied and Cauchy rows of widths 16 to 256 at scales 0.3 to 30: at most 8 steps, 1 to 5
typically, within 2.2e-6 of the fp64 sorted threshold (8e-8 at alpha 2); `kNewtonSteps` caps it at 16, and a capped
result is still the closed form on a support whose extra members lie within the last step of the root. WHY NOT THE
SORT: at one decoded token a CTA has one live row, so its warp's whole solve is the kernel's latency, and the sort's
network is ~36 dependent shuffle stages of long straight-line code executed once from a cold instruction cache (~30
cycles an instruction there, ncu's no-instruction stall the top one); the sorted threshold measured 13.8K cycles a
256-wide row, twelve-round multisection 10K, Newton 2.7K. The sort's bitwise tie-order independence (its sums run in
sorted order) does not carry over -- these sum in lane order, deterministic run to run; the gate is per slot against
fp64.

<a id="budget"></a>THE BUDGET (`tools/budgets/produce.json`, KERNEL_STANDARDS §19): instructions a job tile (one job's 64
tokens, the block's 4 warps) by component, at nl64k-alt-h4-g4 (16 one-row route jobs of 256 columns a token tile, 8
softmax and 8 entmax, ~3.7 Newton steps a row; one value job). The design's counts a warp: the k-loop 16 k-steps x (32
HMMAs, 17 `ldmatrix`, 5 copies, ~12 address and edge) ~1.1K; entmax a n-tile max and shift 128, 3.7 steps x (64 x 6 +
40), levels 192 ~1.85K; softmax ~400 a n-tile; a u8 or f16 bundle side ~180 / ~310 a n-tile; the gain a quad a token,
8 chunk pairs and their products; the gain pass two passes x ~45 a warp (the softplus, the scale's address, its load
and store) in the cell's 4 write level 0 jobs of 17, ~90 a job tile. It is checked by attribution (`tools/stall_census.py --kernel produce`).

<a id="phase-ledger"></a>THE PHASE LEDGER (`rola.ops.produce.ledger_bind`, `[jobs, 4 warps, 6 phases]` int64 cycles,
booked over the tiles): job, project, threshold, levels, bundle (a write level 0 job's gain pass with it), values -- a
warp's cycles between the kernel's laps (`common/phase_clock.cuh`).

## <a id="gain"></a>The write gain

The producer's write gain, `softplus(logit + bias)` (as torch's, linear past 20), is folded into write level 0's scale
by the job that writes that level ([`docs/CONSTITUTION.md` R23](../../../CONSTITUTION.md#r23)), so no job waits on
another: each level's scale is its own job's, and a reader multiplies a side's scales as it loads them. The pass
(`apply_gain`) runs after the job's tile loop, its accumulators dead by then -- the kernel holds its 255-register cap, so
the gain cannot ride the bundle step without spilling: each warp multiplies the 16 scales its bundle stored by their
tokens' gains, a lane quad a token, 8 tokens a pass (in the pair form, the half-0 warp, which stored them). The gain's
logit (`gain_logit`) is the routing's trailing gain column against the token's row, a quad's lanes over the hidden
axis in 16-byte chunks, plus the column's routing bias and `gain_bias`; a routing without a gain column has a gain of
1 and runs no pass. Gate: write level 0's scale against the fp64 composition's level max times its gain
([#bundle](#bundle)).

## <a id="value-rows"></a>The values

On the accumulators (`value_tile`, `value_class`), a value job's heads `d_v / 8` of a lane's values each (the head size
the shape word's class, `d_v = 32 << rows`: 32, 64, 128 or 256 channels, so a value's head is an immediate): per
(token, head) the channel max |v| (clamped at `1e-30`) over the 8 lanes holding the head as the scale, and each channel
times its reciprocal rounded to f16 -- the `SIGNED` form's `quantize` (`rola.routing.quant`), on the fp32
projection rather than the bf16 the composition's linear rounds it to. Gate: every value within one f16 unit in the
last place plus the projection's fp32 error (the channel's and the scale's, over the scale), every scale within that
error, against the fp64 composition.
