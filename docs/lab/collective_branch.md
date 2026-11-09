# `collective_branch`

Mirrors `measure/lab/experiments/collective_branch.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

Eight shuffle chains in a loop, one shuffling warp a scheduler: with no role-dependent control flow the shuffle is
inline (8.3 cycles a shuffle), and so under a branch on the block index or a kernel argument (8.3); under a branch on
the warp id (13.6) or with the role setting the trip count (13.9) it is a CALL into a convergence subroutine, ~1.65x.
Record: `measure/lab/results/collective_branch/sm_86.json`.
