# `image_ring`

Mirrors `measure/lab/experiments/image_ring.cu`, a part harness of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

L2: the state image streamed from the fold's warps to the readout's through a ring in shared memory. Each fold warp
holds 4 lists x 32 digits x 32 channels of fp32 state in registers; each readout warp needs 4 digits x 32 lists. The
hand-off is therefore an all-to-all transpose of 64 KB a window. A round is one 8-digit group x U 8-channel chunks:
- **Producers:** each warp packs its lists' values to bf16 and stores them with full-warp 4-byte stores. A fold lane's
  g is its digit, so a store covers 8 digits.
- **Consumers:** each warp pulls its digit's slice into its tokens-on-N A fragments by `ldmatrix.trans`, the same
  loads a static image costs.
- **Swizzle:** a slot's 16-byte units are swizzled by `g ^ s(list)`, so both the stores and the loads are
  conflict-free (KERNEL_STANDARDS §30; measured 0.01 conflicts an MMA).

The lab runs the readout of `read_image.cu` (its inputs and its exact check) with a warp's 128 fp32 state registers
live. Between windows a fold stand-in rewrites them: one m16n8k8 MMA into each of the 32 accumulator tiles, ~0.7x the
fold's tensor work at alt-k4, its operands zeros from memory so ptxas cannot fold the values. Every variant's outputs
are exact.

| sm_86, alt-k4, a CTA-window | cycles | over a static image |
|---|---|---|
| a static image (the readout and the fold stand-in) | 2744 | |
| the publish without handshakes (stores and packs; an isolation) | 3504 - 3520 | +0.76K |
| the ring on mbarriers, 8-channel rounds (4 KB) x 2 / 4 / 8 slots | 11.8K / 11.7K / 11.5K | +8.8K - 9.1K |
| the ring on mbarriers, 16-channel rounds (8 KB) x 2 / 4 slots | 7.96K / 7.76K | +5.0K - 5.2K |
| the ring on mbarriers, 32-channel rounds (16 KB) x 2 slots | 6.01K | +3.3K |
| a named barrier a round, 8 / 16 / 32-channel rounds x 2 slots | 5.56K / 5.18K / 4.73K | +2.8K / +2.4K / +2.0K |

Every round needs all 8 warps, so each round is a CTA-wide rendezvous whichever primitive signals it. On sm_80 and
sm_86 an mbarrier is software: each arrive is a `MEMBAR.ALL.CTA`, which waits for the warp's outstanding panel
atomics, plus a shared atomic, and the waiting warps poll shared memory. That comes to ~600 cycles a round. A named
barrier (`bar.sync`, hardware) costs instead the warps' skew between barriers, ~300 cycles a round at 32-channel
rounds. The publish's bytes alone (512 store wavefronts and 64 packs a thread a window) cost +0.76K, where the
tokens-on-N readout already runs shared memory at ~92% of its tensor floor. The line was rounds within 5% of a sparse
window, about 0.2K; the ring misses it by 10-45x.

Then EXCHANGE THEN COMPUTE: every warp does the same stores and loads between named barriers, holding its digits'
fragments in registers, and the tiles run after.

| sm_86, alt-k4, a CTA-window | cycles |
|---|---|
| a static image, a barrier a window (the rendezvous alone) | 3228 (+0.48K) |
| the exchange alone (64 KB stored and loaded, 4 barriers; an isolation) | 1770 |
| exchange then compute, 4 digits held (242 registers) / 2 (210) | 4858 / 5151 |
| exchange, then the tiles and the next fold interleaved, 4 held | 4757 |

Holding the exchanged copy frees the next fold to run beside the tiles, but that recovers little. A rendezvous aligns
the warps' phases: the exchange (shared memory only, tensor idle), the readout and the fold (tensor only) run one
after another, where free-running warps overlap one warp's fold with another's readout. The exchange's 1024 wavefronts
take 1.77K with nothing to hide them. Record: `measure/lab/results/image_ring/sm_86.json`.
