# `tensor_tax`

Mirrors `measure/lab/experiments/tensor_tax.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

Warps 0-3 (one a scheduler) run 8 MMAs a loop into 8 accumulators; warps 4-7 issue one instruction class in 8 chains,
each instruction's operands other chains' values; the class is sized to land inside the chain; the tax is the chain's
slowdown over the class's count. sm_86, CUDA 13.0, cycles an instruction beside an fp16-accumulate HMMA (16 cycles) /
an fp32-accumulate HMMA (32 cycles): FP32 `FMUL`/`FFMA`/`FADD` 1.0 / 0.51; packed `HMUL2`/`HFMA2`/`HADD2`/bf16 FMA
0.48 / 0.21; `HMNMX2` 0.44 / 0.16; an add and xor pair 1.2 / 0.71; `IMAD` 0.19 / 0.10; `LOP3`, `PRMT`, `FMNMX`, the
conversions 0.13-0.22 / 0.05-0.08; `LDS` (with its address math) 0.85 / 0.53. Beside an int8 IMMA (16 cycles) the
numbers match the fp16-accumulate column. Record: `measure/lab/results/tensor_tax/sm_86.json`.
