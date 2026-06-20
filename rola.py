"""RoLA support helpers used by the orchestration layer.

The RoLA model itself lives in the `fla_rola` fork (single source of truth): `fla_rola.layers.RoLA`
and `fla_rola.models.rola`. This module holds only the pieces that are NOT part of the model and
have no home in the fork:

  * `rola_instance` / `ROLA_INSTANCES` — preset name → RoLAMixer kwargs (the MQAR configs pass these
                        to `zoology.mixers.rola.RoLAMixer`, which maps them onto the fla layer).
  * `_rola_perstate_den` / `_rola_gla_perstate_den` — eager per-state denominators the
                        routing/similarity evaluator uses to reconstruct the read-gate rescale.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# FLA kernel for the single-state RLA baseline.


# ============================================================================
# Eager per-state denominators (used by the routing/similarity evaluator).
# ============================================================================
def _rola_perstate_den(qf, kf, wg, chunk=64):
    """Per-state denominator d_i^c = sum_{j<=i} w_j^c (φ(q_i)·φ(k_j)) — the mass each state
    contributes to token i's global partition function. [B,L,H,*] in → [B,L,H,nc] out.
    Used by the KAPPA / per_state read-gate rescale r̃ = r·(d+ε)^{-κ(x)}."""
    B, L, H, dqk = qf.shape
    nc = wg.shape[-1]
    fold = lambda t: t.permute(0, 2, 1, 3).reshape(B * H, L, t.shape[-1])
    q, k, w = fold(qf), fold(kf), fold(wg)
    pad = (-L) % chunk
    if pad:
        q = F.pad(q, (0, 0, 0, pad)); k = F.pad(k, (0, 0, 0, pad)); w = F.pad(w, (0, 0, 0, pad))
    Lp = L + pad; n = Lp // chunk
    qc = q.view(B * H, n, chunk, dqk); kc = k.view(B * H, n, chunk, dqk); wc = w.view(B * H, n, chunk, nc)
    G = torch.einsum('bnid,bnjd->bnij', qc, kc)
    causal = torch.tril(torch.ones(chunk, chunk, device=q.device, dtype=q.dtype))
    intra = torch.einsum('bnij,bnjc->bnic', G * causal, wc)
    KZ = torch.einsum('bnjc,bnjd->bncd', wc, kc)                  # per-chunk z increments
    Z = torch.cumsum(KZ, dim=1) - KZ                              # exclusive prefix
    inter = torch.einsum('bnid,bncd->bnic', qc, Z)
    d = (intra + inter).reshape(B * H, Lp, nc)[:, :L]
    return d.view(B, H, L, nc).permute(0, 2, 1, 3)                # [B,L,H,nc]


def _rola_gla_perstate_den(qf, kf, wg, ld, chunk=64):
    """Per-state denominator UNDER per-state log-decay ld:
    d_i^c = sum_{j<=i} (φq_i·φk_j) w_j^c e^{Λ_ic-Λ_jc}, Λ = inclusive cumsum(ld).
    [B,L,H,*] in → [B,L,H,nc] out (decay absorbed chunk-locally, decayed cross-chunk carry)."""
    B, L, H, dqk = qf.shape
    nc = wg.shape[-1]
    fold = lambda t: t.permute(0, 2, 1, 3).reshape(B * H, L, t.shape[-1])
    q, k, w, g = fold(qf), fold(kf), fold(wg), fold(ld)
    pad = (-L) % chunk
    if pad:
        q = F.pad(q, (0, 0, 0, pad)); k = F.pad(k, (0, 0, 0, pad))
        w = F.pad(w, (0, 0, 0, pad)); g = F.pad(g, (0, 0, 0, pad))
    Lp = L + pad; n = Lp // chunk
    qc = q.view(B * H, n, chunk, dqk); kc = k.view(B * H, n, chunk, dqk)
    wc = w.view(B * H, n, chunk, nc); gc = g.view(B * H, n, chunk, nc)
    a = gc.cumsum(2)                                              # chunk-local Λ [b,n,t,c]
    Lam = a[:, :, -1, :]                                          # chunk decay totals [b,n,c]
    G = torch.einsum('bnid,bnjd->bnij', qc, kc)
    causal = torch.tril(torch.ones(chunk, chunk, device=q.device, dtype=q.dtype))
    intra = torch.exp(a) * torch.einsum('bnij,bnjc->bnic', G * causal, wc * torch.exp(-a))
    w_end = wc * torch.exp(Lam.unsqueeze(2) - a)                  # writes decayed to chunk end
    KZ = torch.einsum('bnjc,bnjd->bncd', w_end, kc)               # per-chunk carry increments
    acc = torch.zeros(B * H, nc, dqk, device=q.device, dtype=q.dtype)
    Zs = []
    for i in range(n):                                            # decayed exclusive prefix
        Zs.append(acc)
        acc = torch.exp(Lam[:, i]).unsqueeze(-1) * acc + KZ[:, i]
    Z = torch.stack(Zs, 1)                                        # [b,n,c,d] state at chunk start
    inter = torch.exp(a) * torch.einsum('bnid,bncd->bnic', qc, Z)
    d = (intra + inter).reshape(B * H, Lp, nc)[:, :L]
    return d.view(B, H, L, nc).permute(0, 2, 1, 3)                # [B,L,H,nc]


# ============================================================================
# RoLA named instances — preset name -> RoLAMixer kwargs. The MQAR configs pass these to
# zoology.mixers.rola.RoLAMixer (which maps them onto fla_rola.layers.RoLA).
# ============================================================================
def rola_instance(name, d_qk, d_v, num_chunks, n_heads=4):
    """RoLAMixer kwargs for a named instance. The INNER KERNEL (rla / gla_scalar) is the only
    parameterized axis; tie_routers => symmetric (one router)."""
    common = dict(d_qk=d_qk, d_v=d_v, num_chunks=num_chunks, n_heads=n_heads, use_short_conv=False)
    # RLA family (phi = feature map; global/per_state/kappa norm).
    if name == 'rola-rla-asym':
        return dict(kernel='rla', phi='elu', tie_routers=False, **common)
    if name == 'rola-rla-sym':
        return dict(kernel='rla', phi='elu', tie_routers=True, **common)
    if name == 'rola-rla-asym-ps':
        return dict(kernel='rla', phi='elu', tie_routers=False,
                    kernel_kwargs={'state_norm': 'per_state'}, **common)
    if name == 'rola-rla-sym-ps':
        return dict(kernel='rla', phi='elu', tie_routers=True,
                    kernel_kwargs={'state_norm': 'per_state'}, **common)
    if name == 'rola-rla-kappa-asym':
        return dict(kernel='rla', phi='elu', tie_routers=False,
                    kernel_kwargs={'state_norm': 'kappa'}, **common)
    if name == 'rola-rla-kappa-sym':
        return dict(kernel='rla', phi='elu', tie_routers=True,
                    kernel_kwargs={'state_norm': 'kappa'}, **common)
    if name == 'rola-rla-asym-tieinit':
        return dict(kernel='rla', phi='elu', tie_routers=False, tie_router_init=True, **common)
    if name in ('rola-hedgehog-sym', 'rola-hedgehog-asym'):
        return dict(kernel='rla', phi='hedgehog', tie_routers=(name.endswith('-sym')), **common)
    if name in ('rola-based-sym', 'rola-based-asym'):
        return dict(kernel='rla', phi='based', tie_routers=(name.endswith('-sym')), **common)
    if name in ('rola-rebased-sym', 'rola-rebased-asym'):
        return dict(kernel='rla', phi='rebased', tie_routers=(name.endswith('-sym')), **common)
    # scalar-GLA family (optimized shared-gram).
    if name.startswith('rola-gla-scalar'):
        return dict(kernel='gla_scalar', tie_routers=name.endswith('-sym'),
                    kernel_kwargs={'normalized': '-norm-' in name}, **common)
    if name.startswith('rola-gla-kappa'):
        return dict(kernel='gla_scalar', tie_routers=name.endswith('-sym'),
                    kernel_kwargs={'state_norm': 'kappa'}, **common)
    if name.startswith('rola-gla-ps'):
        return dict(kernel='gla_scalar', tie_routers=name.endswith('-sym'),
                    kernel_kwargs={'state_norm': 'per_state'}, **common)
    raise ValueError(f"unknown RoLA instance: {name!r}")


ROLA_INSTANCES = ('rola-rla-asym', 'rola-rla-sym', 'rola-rla-asym-ps', 'rola-rla-sym-ps',
                  'rola-rla-kappa-asym', 'rola-rla-kappa-sym', 'rola-rla-asym-tieinit',
                  'rola-gla-scalar-sym', 'rola-gla-scalar-norm-sym',
                  'rola-gla-scalar-asym', 'rola-gla-scalar-norm-asym',
                  'rola-gla-kappa-sym', 'rola-gla-kappa-asym',
                  'rola-hedgehog-sym', 'rola-hedgehog-asym',
                  'rola-based-sym', 'rola-based-asym', 'rola-rebased-sym', 'rola-rebased-asym')
