# `pass_fixed`

Mirrors `measure/lab/experiments/pass_fixed.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

The fold's fixed work alone, one block a SM: the fill (a pass of 128 scattered rows, waited each pass), the restage (a
pass's 256 items) and the window's entries (8 warps' 16 union words). sm_86: the fill a pass, DRAM / L2 -- as built
2.35K / 1.37K cycles (47 instructions a row: four factor planes, the u8 dense chunk), the factor products precomputed
2.22K / 1.29K (39 a row), with f16 dense digits too 2.47K / 1.29K (more bytes from DRAM), 32-bit offsets from window
bases no better; the restage 294 cycles a pass, 224 with the products precomputed; the entries a window, 1 in 4 / 1 in
16 -- a serial loop over set bits 2.10K / 1.28K, word-parallel by popc 2.65K / 2.62K (sixteen word iterations whatever
the density). Inside the load-ahead pipeline the fill and restage instructions are ~4-7% of a sparse pass; what costs
there is in `sparse_pass`'s ledger. Record: `measure/lab/results/pass_fixed/sm_86.json`.
