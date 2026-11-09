# `mma_pairing`

Mirrors `measure/lab/experiments/mma_pairing.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

Every warp of a 256-thread block runs a fold-like step of 32 fp16-accumulate HMMAs (8 lists x 4 n-tiles), plus one atom
of a kind. The median launch sits on the floor (1024 cycles a step a warp at two warps a scheduler); some launches
phase-lock up to 1.22x. One fp32-accumulate atom among them costs 32 cycles a step pair, not its 64. Record:
`measure/lab/results/mma_pairing/sm_86.json`.
