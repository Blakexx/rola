# RoLA — Routed Linear Attention

> **This repository is now a stub.** It was the original RoLA model library; the model, kernels,
> baselines, and benchmark suite have moved to the dedicated repositories below. This repo is kept
> as the project namespace (and for potential future use) — it no longer ships an importable module.

**RoLA (Routed Linear Attention)** is a linear-attention head whose recurrent state is expanded into
multiple sub-states sharing the head's projections, with learned read and write routing over the
states. Effective attention factors as a Hadamard product of a routing Gram and a kernel Gram,
scaling realized attention rank with the state count at one shared projection. The routing composes
with any decomposable inner kernel (vanilla, gated, delta-rule), and the normalization's placement is
a learned per-token quantity (the kappa family) whose endpoints are the per-state and global divisions.

Blake Bottum, 2026.

## Where everything lives now

| Component | Repository | What's there |
|---|---|---|
| **Model + fused kernels** (source of truth) | [flash-linear-attention-rola](https://github.com/Blakexx/flash-linear-attention-rola) | `fla_rola.layers.RoLA` (the layer) and `fla_rola.models.rola` (`RoLAConfig` / `RoLAModel` / `RoLAForCausalLM`), as a first-class FLA-style arch, plus the chunk-parallel + recurrent Triton kernels and their correctness suite. |
| **Baselines + MQAR mixer** | [rola-zoology](https://github.com/Blakexx/rola-zoology) | The zoology fork: clean-room baselines and `zoology.mixers.rola.RoLAMixer` (wraps the fla layer for the MQAR harness). |
| **Benchmarks + orchestration** | `rola-bench` | The benchmark suite — MQAR, LM, kernel-perf, similarity — as declarative specs over one config-driven runner, with a uniform `results/<bench>/<config>/<cell>.json` store and a fleet of cloud GPUs. |
| **Paper** | [rola-paper](https://github.com/Blakexx/rola-paper) | The write-up. |

The model is the single source of truth in the `fla_rola` fork; this repo's former `rola.py` (the
orchestrator, kernels, instances, and eval dens) was retired — its model code became the fla layer,
and the few remaining helpers (named instances, eager denominators) moved into `rola-bench`.
