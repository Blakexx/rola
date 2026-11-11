# `wgmma_regs`

Mirrors `measure/lab/experiments/wgmma_regs.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

A fold's fp32 accumulators -- an `m64n16k16` wgmma's output, 8 a thread -- become the next wgmma's register operand A
once each pair is narrowed to bf16 in place. What does that narrowing cost a step, by truncation and by rounding,
beside an A read from shared memory?

THE FORM. One or two warpgroups a CTA, a CTA an SM. A step is a fold (`m64n16k16`, both operands shared memory) and a
readout (`m64n64k16`) whose A is shared memory, registers loaded once, or the fold's accumulators narrowed each step;
the two MMAs of a step are one commit group, and the wait for them opens the next step. The fold alone and the
readout alone give the sum a step is read against. Every form's accumulators are compared, exactly, with a host
recurrence of the same integers at 24 steps, through the register map CUTLASS documents (`CLayout_64x16`,
`ALayout_64x16`): a thread's accumulator `4q + 2j + i` is row `16 w + g + 8j`, column `8q + 2t + i` (warp `w`, lane
`4g + t`).

WHAT THE SASS SAYS (sm_90a, the pinned toolkit; no device): the relayout is the identity. An `m64n16` accumulator's
values `2j`, `2j + 1` are register A's word `j`, so the narrowing is one instruction a pair and nothing moves between
threads: 4 `PRMT` a thread a step truncated (each value's high 16 bits), or 4 `F2FP.BF16.F32.PACK` rounded. The step
loop is the wait (`WARPGROUP.DEPBAR.LE`), those 4, `WARPGROUP.ARRIVE`, the two `HGMMA` and the branch.

MEASURED (H100 SXM, 132 SMs, a rented box: vast.ai instance 60164819, driver 595.91.07; 2026-11-11, one session, 7
repetitions, every form's accumulators exact against the host's; cycles a step, the median -- the spread under 0.3%):

| a step | one warpgroup | two warpgroups |
|---|---|---|
| the fold alone (`m64n16k16`) | 81.1 | 81.1 |
| the readout alone (`m64n64k16`), A shared | 105.1 | 103.1 |
| the readout alone, A registers loaded once | 136.4 | 139.9 |
| fold + readout, A shared | 121.1 | 124.1 |
| fold + readout, A registers loaded once | 165.4 | 174.8 |
| fold + readout, A the state TRUNCATED | 110.1 | 118.1 |
| fold + readout, A the state ROUNDED | 110.1 | 118.1 |

WHAT IT SHOWS.
- The register map is CUTLASS's: every accumulator matched through it, so the state is the readout's operand with no
  relayout, on the device as in the SASS.
- Truncating and rounding cost the same step. Which one the kernel takes is the accuracy model's question
  ([`arith_model`](../internals/tools/arith_model.md)), not a cost's.
- The narrowed step, 110, is under the fold alone plus the readout alone (186) and under the step with A shared (121):
  the two MMAs of one commit group overlap.
- A second warpgroup in the CTA costs a step nothing to 7%, so at these steps one warpgroup does not hold the SM's
  tensor pipe: a step that waits for its own group is bound by the group's round trip. The pipe's own time a wgmma is
  [`wgmma_rate`](wgmma_rate.md)'s: 32 cycles.
- THE FIXED-REGISTER ROWS (A in registers loaded once: 136 against A shared's 105 alone, 165 against 121 with the
  fold) are not the register operand's cost: [`wgmma_rate`](wgmma_rate.md) reads the same readout at 96, under A
  shared's 105. Its kernel differs from this one's in three ways at once -- every descriptor held in a register, A's
  words held, launch bounds stated -- so which of them is the 40 cycles is NOT isolated; ptxas re-deriving B's
  descriptor inside this loop is one candidate the SASS shows, and the narrowed form here keeps three of those five
  instructions and is fast. No design rests on the rows here.

Record: `measure/lab/results/wgmma_regs/sm_90a.json`; its counters are the written exception (no profiler on that
box).
