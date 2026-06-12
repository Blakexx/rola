# RoLA — Routed Linear Attention

Model library for **RoLA (Routed Linear Attention)**: a linear-attention head whose recurrent
state is expanded into multiple sub-states sharing the head's projections, with learned read
and write routing over the states. Effective attention factors as a Hadamard product of a
routing Gram and a kernel Gram, scaling realized attention rank with the state count at one
shared projection. The routing composes with any decomposable inner kernel (vanilla, gated,
delta-rule), and the normalization's placement is a learned per-token quantity (the kappa
family) whose endpoints are the per-state and global divisions.

Blake Bottum, 2026. Paper: [rola-paper](https://github.com/Blakexx/rola-paper). Fused kernels:
[flash-linear-attention-rola](https://github.com/Blakexx/flash-linear-attention-rola).
Experiments: [rola-zoology](https://github.com/Blakexx/rola-zoology).

## Layout

- `rola.py` — the library: the `RoLA` orchestrator, composition kernels (`KERNEL_REGISTRY`:
  additive RLA family with elu/hedgehog/based/rebased feature maps, scalar-gated GLA,
  virtual-head GLA, GDN), normalization family (global / per-state / kappa), named instances
  (`rola_instance`), canonical monolith baselines, and eval-time diagnostics (distributional
  realized-rank, routing peakiness).
- `rola_kernel.py` — kernel correctness and benchmark harness (chunked shared-gram torch paths,
  Triton variants).
- `rola_fla_dev/` — verification and benchmark suite for the fused kernels: correctness gates
  (`check.py`, `check_den_gla.py`), parity model, kernel benchmarks (`paper_bench.py`,
  FlopCounter-verified analytical FLOPs).
- `rola_hf/`, `train_lm.py`, `eval_lm.py` — HF-style language-model pipeline (FLA-compatible
  model class, HF Trainer training, lm-eval evaluation).
- `cla_bench.py` — deprecation shim re-exporting `rola` for legacy configs.

## Verification

Every kernel path carries a first-use correctness gate against a reference implementation, and
`rola_fla_dev/` holds the offline suites: fused paths vs dense oracles (forward and all
gradients, including the decayed denominator's decay gradient), fp64 gradchecks, deep-decay
regimes, and ragged sequence lengths.
