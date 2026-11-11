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

UNMEASURED: no hopper card has run it. Its cycles a step, and so whether the narrowing leaves the tensor pipe
covered, are owed to its first run (`bazel run //tools:lab -- pack wgmma_regs --arch sm_90a`, then `carried` on the
card), which writes its sm_90a record beside the other experiments'.
