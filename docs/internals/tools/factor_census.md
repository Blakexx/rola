# `tools/factor_census.py` — the scaled operands' f16 range

Operand scaling moves a readout's factor onto the read rows before the MMA, so the rows must stay inside f16's range
once scaled. This census reads where they land, on a cell's own draw and the reference's states
(`tests/oracle/readout_reference.py`): a token `t` reading digit `d` scales its rows `a` by
`f = c 2^(e_d - kImageTop - s_t - 1)` (its coefficient, the digit's image exponent, the token's scale byte), and the
census reports the exponents of `f`, of `a f`, and of `a f 2^(s_t + 1)` (everything but the byte): the largest, the
smallest, the median and the shares under 2^-14 (subnormal) and 2^-24 (flushed).

    bazel run //tools:factor_census -- flagship-alt-k4 [--json census.json]

MEASURED (sm_86, 2026-11-09), the share of scaled operands under 2^-14 -- with the byte / without it: flagship-alt-k4
32% / 3.7%, flagship-alt-k16 71% / 11%, flagship-cohort-k16 57% / 7.6% (listed read sides), flagship-dense 100% / 38%.
The scale byte's frame sits at the scale pass's bound, whose slack over the real sums (median 2^9.2) puts operands
scaled into it near 2^-13 to 2^-17; a factor never overflows (unit-max rows bound it near 2^1). This refuted the fp16
readouts' operand scaling at the token's scale as planned (plans' FP16_READOUT, section 3.1).
