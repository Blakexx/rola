"""rola_kernels — Routed Linear Attention kernels, FLA-derived but a SEPARATE package.

The single canonical, optimized implementation of the RoLA Triton kernels (RLA + scalar-GLA):
shared-gram forward + fused analytic backward (no virtual-head replication, no forward recompute),
autograd-wrapped. Built ON canonical `fla` utilities but kept in its own namespace so the installed
`fla` package stays PRISTINE — every baseline (`fla.ops.gla.chunk_gla`, `gated_delta_rule`,
`fused_chunk_linear_attn`) runs the unmodified canonical FLA. RoLA imports `rola_kernels`; baselines
import `fla`; the two diverge with zero cross-contamination (we never edit installed `fla`)."""
from . import ops as _ops
# Re-export everything (public + internal underscore helpers used by the test oracle) so callers
# can `from rola_fla import triton_rola_ag, triton_gla_ag, ...`.
globals().update({k: v for k, v in vars(_ops).items() if not k.startswith('__')})
