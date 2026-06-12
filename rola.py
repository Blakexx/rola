"""RoLA — Routed Linear Attention. The generic model library (kernels + orchestrator
+ canonical monolith baselines), decoupled from usage (datasets / benchmarking / LM
harnesses live with the caller, e.g. zoology). Single source of truth for the model."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import warnings

# Suppress the misleading FLA deprecation warning
warnings.filterwarnings("ignore", message="Input tensor shape suggests potential format mismatch.*")


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

DEVICE = get_device()

# FLA kernels (canonical baselines + virtual-head GLA/GDN paths).
from fla.ops.linear_attn import chunk_linear_attn, fused_chunk_linear_attn
from fla.ops.gla import chunk_gla
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

# Fused Triton kernels (shared-gram, tiled over states), verified vs torch refs (fwd + all grads,
# nc≤256, bf16-safe). Empirically (measured fwd+bwd at model scale):
#   GLA: Triton fwd + FLA virtual-head fused bwd is ~4× faster than the eager torch GLA (its
#        [Cc,Cc,C]/Python-loop path) → DEFAULT ON for global-norm scalar GLA.
#   RLA: eager cuBLAS chunk-parallel is already optimal (not kernel-bound); the Triton path is
#        neutral (elu) to slower (hedgehog), and the fully-fused Triton bwd LOSES to cuBLAS
#        (0.23-0.37×). So RLA defaults to EAGER; Triton-RLA is opt-in (ROLA_TRITON_RLA=1) for the
#        low-memory / inference forward only. The Triton RLA path is φ-agnostic + global-norm only.
# Per-state norm, non-CUDA, or import failure → torch fallback automatically.
_USE_TRITON = os.environ.get('ROLA_USE_TRITON', '1') != '0'          # default ON for both RLA + GLA
_USE_TRITON_RLA = _USE_TRITON                                        # RLA uses the bespoke fused path too
try:
    # The FLA fork's routed simple_gla: additive r/w/g params on the canonical kernel
    # ([B,T,H,*] layout, un-normalized readout — we normalize via the ones-column here).
    # Gated against the oracle + canonical baselines by rola_fla_dev/check.py.
    from fla_rola.ops.simple_gla import chunk_simple_gla as _fla_routed
except Exception:
    _fla_routed = None


def _triton_compute_dtype(fallback):
    """The dtype the Triton kernels should run in. Under autocast use the AUTOCAST dtype (bf16) —
    NOT whatever the feature map emits (e.g. hedgehog's softmax is fp32 under autocast, which would
    force slow fp32 matmuls). Outside autocast use the input dtype. One policy → no dtype mismatches
    (the router softmax is also fp32 under autocast; casting all inputs to this single dtype fixes it)."""
    if torch.is_autocast_enabled():
        return torch.get_autocast_gpu_dtype()
    return fallback

# CUDA caps grid y/z dims at 65535. FLA's chunked kernels grid over B*(H*nc)
# (batch * virtual heads), so at high state count (H*nc large) the launch fails
# with "Triton Error [CUDA]: invalid argument". Heads are independent in linear
# attention, so we split the virtual-head dim into grid-safe groups and call the
# kernel once per group, concatenating the outputs (mathematically exact).
# Chunking only triggers when B*HC exceeds the cap — by then the work is tens of
# waves past the GPU's resident-block capacity, so the sub-launches are saturated
# and sequential execution costs ~nothing (no concurrency/stream gain to be had).
_GRID_CAP = 60000   # < 65535, with margin; the observed failure was at B*HC=65536
def _headchunked(fn, tensors, head_dim=2, **kw):
    """Run an FLA chunk kernel `fn` over per-head tensors (all sized H*nc on
    head_dim), splitting that dim so each launch keeps B*group <= _GRID_CAP.
    `fn` returns (out, final_state); returns the out concatenated over groups."""
    B = tensors[0].shape[0]
    HC = tensors[0].shape[head_dim]
    if B * HC <= _GRID_CAP:
        return fn(*tensors, **kw)[0]
    max_heads = max(1, _GRID_CAP // max(1, B))
    outs = []
    for s in range(0, HC, max_heads):
        n = min(max_heads, HC - s)
        chunk = [t.narrow(head_dim, s, n).contiguous() for t in tensors]
        outs.append(fn(*chunk, **kw)[0])
    return torch.cat(outs, dim=head_dim)


# ============================================================================
# Fused RoLA-RLA forward (GLOBAL normalization).
#
# The paper's Eq. rola_output uses a single global partition function per query:
#   O_i = (sum_k a_i^k sum_j b_j^k (q_i.k_j) v_j) / (sum_k a_i^k sum_j b_j^k (q_i.k_j)).
# Equivalently linear attention on Kronecker features q⊗a, k⊗b with V augmented
# by a ones-column (the denominator), then one divide. This fuses the read combine
# so we NEVER materialize the per-state [B,L,H,C,d_v] tensor — that, plus sharing
# the content gram across states, is the memory + speed win over virtual heads.
#
# `_rola_chunked_parallel` is the shipping path: chunk-parallel pure torch (grams
# batched over chunks, the inter-chunk recurrence is an exclusive cumsum of
# per-chunk Kronecker states). Validated byte-identical (fwd 1e-7, grads <=4e-5)
# vs `_rola_global_ref`; ~2x faster AND ~2x less memory than the virtual-head path
# at high nc / long L. Triton would push memory/speed further (deferred — only
# needed if full LM scale hits a wall). Inputs are POST-feature-map q,k and the
# read/write routing distributions (dense or top-k; top-k just zeros the
# non-routed gates, so the global denominator ranges over routed states for free).
# ============================================================================
def _rola_global_ref(q, k, v, wg, rg, eps=1e-5):
    """O(L^2) global-norm reference. q,k:[B,L,H,dqk] v:[B,L,H,dv] wg,rg:[B,L,H,C].
    Ground truth for the first-use correctness gate."""
    L = q.shape[1]
    G = torch.einsum('bthd,bshd->bhts', q, k)            # content gram (shared)
    R = torch.einsum('bthc,bshc->bhts', rg, wg)          # routing gram
    causal = torch.tril(torch.ones(L, L, device=q.device, dtype=q.dtype))
    W = G * R * causal
    num = torch.einsum('bhts,bshv->bthv', W, v)        # [B,T,H,dv]
    den = torch.einsum('bhts->bth', W).unsqueeze(-1)   # [B,T,H,1]
    return num / (den + eps)


def _rola_chunked_parallel(q, k, v, wg, rg, eps=1e-5, chunk=64):
    """Chunk-parallel global-norm RoLA. q,k:[B,L,H,dqk] v:[B,L,H,dv] wg,rg:[B,L,H,C]
    -> [B,L,H,dv]. Heads fold into the batch; the chunk loop is vectorized (grams
    batched over chunks, inter-chunk = exclusive cumsum of Kronecker chunk-states)."""
    B, L, H, dqk = q.shape
    dv = v.shape[-1]
    C = wg.shape[-1]
    bh = B * H

    def fold(t):
        return t.permute(0, 2, 1, 3).reshape(bh, t.shape[1], t.shape[-1])
    q, k, v, wg, rg = fold(q), fold(k), fold(v), fold(wg), fold(rg)   # [BH,L,*]

    pad = (-L) % chunk
    if pad:
        z = lambda t: F.pad(t, (0, 0, 0, pad))
        q, k, v, wg, rg = z(q), z(k), z(v), z(wg), z(rg)
    Lp = L + pad
    n = Lp // chunk
    v1 = torch.cat([v, torch.ones_like(v[..., :1])], dim=-1)
    qc = q.view(bh, n, chunk, dqk); kc = k.view(bh, n, chunk, dqk); vc = v1.view(bh, n, chunk, dv + 1)
    rgc = rg.view(bh, n, chunk, C); wgc = wg.view(bh, n, chunk, C)
    # intra-chunk: shared content gram, routing gram, causal mask
    G = torch.einsum('bnid,bnjd->bnij', qc, kc)
    R = torch.einsum('bnic,bnjc->bnij', rgc, wgc)
    causal = torch.tril(torch.ones(chunk, chunk, device=q.device, dtype=q.dtype))
    A = G * R * causal
    o_intra = torch.einsum('bnij,bnjv->bniv', A, vc)
    # inter-chunk: per-chunk Kronecker states, exclusive prefix-sum over chunks
    KV = torch.einsum('bnjc,bnjd,bnjv->bncdv', wgc, kc, vc)       # [BH,n,C,dqk,dv+1]
    S_before = torch.cumsum(KV, dim=1) - KV
    M = torch.einsum('bnic,bncdv->bnidv', rgc, S_before)
    o_inter = torch.einsum('bnid,bnidv->bniv', qc, M)
    O = (o_intra + o_inter).reshape(bh, Lp, dv + 1)[:, :L]
    num, den = O[..., :-1], O[..., -1:]
    out = num / (den + eps)                                       # [BH,L,dv]
    return out.view(B, H, L, dv).permute(0, 2, 1, 3).contiguous()  # [B,L,H,dv]



def _rola_perstate_ref(q, k, v, wg, rg, eps=1e-5):
    """O(L²) PER-STATE-norm reference: each state self-normalizes, THEN the read gates combine
    (o_i = Σ_c r_i^c · num_i^c/den_i^c). Kept ONLY as the endpoint oracle for the kappa/per_state
    first-use check — the compute path rides the global combine on mass-rescaled gates."""
    L = q.shape[1]
    G = torch.einsum('bihd,bjhd->bhij', q, k)
    causal = torch.tril(torch.ones(L, L, device=q.device, dtype=q.dtype))
    W = G * causal
    num = torch.einsum('bhij,bjhc,bjhv->bihcv', W, wg, v)
    den = torch.einsum('bhij,bjhc->bihc', W, wg)
    o_c = num / (den.unsqueeze(-1) + eps)
    return torch.einsum('bihc,bihcv->bihv', rg, o_c)


def _rola_perstate_den(qf, kf, wg, chunk=64):
    """Per-state denominator d_i^c = sum_{j<=i} w_j^c (φ(q_i)·φ(k_j)) — the mass each state
    contributes to token i's global partition function. [B,L,H,*] in → [B,L,H,nc] out.
    Mirrors _rola_chunked_parallel restricted to the ones-column, with the state axis kept.
    Used by the KAPPA normalization: r̃ = r·(d+ε)^{-κ(x)} interpolates global (κ=0, exact)
    ↔ per-state (κ=1, exact — read gates sum to 1 so the outer divide collapses)."""
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
    [B,L,H,*] in → [B,L,H,nc] out. Chunked (decay absorbed chunk-locally, decayed
    cross-chunk carry) — the torch fallback/oracle for the fork's GLA den kernel."""
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



# ----------------------------------------------------------------------------
# Optimized SCALAR-gated RoLA-GLA. With a scalar per-state forget gate the
# cumulative log-decay factors out of the content contraction, so the effective
# weight keeps the shared-gram form W_tj = (q.k)(sum_c r_t^c w_j^c e^{A_t-A_j}) =
# G_tj (r~_t . w~_j), r~=r e^A, w~=w e^{-A}. i.e. additive RoLA with decay-absorbed
# routing gates -> the SAME fused chunk path, decay absorbed CHUNK-LOCALLY for
# stability, Kronecker state carried across chunks with a per-chunk decay scan.
# This avoids the virtual-head [B,L,H,nc,d_qk] materialization (the nc-scaling
# blowup): ~4x faster, ~9x less memory at the real training batch. Validated by
# rola_fla_dev/check.py (fork routed path vs O(L^2) ref + recurrent, fwd+grads).
# Inputs are POST-feature-map q,k, the routing distributions, and ld = log of the
# per-state decay alpha_chunk = 1 - w*(1-alpha_base).  Per-channel (vector) gate
# does NOT factor (decay entangled in the d-sum) -> no shared path, use eager.
# ----------------------------------------------------------------------------
def _rola_gla_ref(q, k, v, wg, rg, ld, eps=1e-5, normalized=False):
    """O(L^2) ref for scalar-gated RoLA-GLA (first-use gate truth). q,k:[B,L,H,dqk]
    v:[B,L,H,dv] wg,rg,ld:[B,L,H,C]. normalized=False = raw gated sum (GLA convention)."""
    B, L, H, dqk = q.shape
    bh = B * H
    fold = lambda t: t.permute(0, 2, 1, 3).reshape(bh, t.shape[1], t.shape[-1])
    q, k, v, wg, rg, ld = fold(q), fold(k), fold(v), fold(wg), fold(rg), fold(ld)
    A = torch.cumsum(ld, dim=1)                                   # [bh,L,C]
    G = torch.einsum('btd,bsd->bts', q, k)
    # clamp exponent ≤0: a no-op on the causal triangle (t≥s ⟹ A_t≤A_s, exact) and a guard on
    # the upper triangle (masked below) so deep decay can't overflow exp before the mask applies.
    decay = torch.exp((A[:, :, None, :] - A[:, None, :, :]).clamp(max=0.0))   # [bh,L,L,C]
    D = torch.einsum('btc,bsc,btsc->bts', rg, wg, decay)
    causal = torch.tril(torch.ones(L, L, device=q.device, dtype=q.dtype))
    W = G * D * causal
    v1 = torch.cat([v, torch.ones_like(v[..., :1])], dim=-1) if normalized else v
    O = torch.einsum('bts,bsv->btv', W, v1)
    out = O[..., :-1] / (O[..., -1:] + eps) if normalized else O
    dv = v.shape[-1]
    return out.view(B, H, L, dv).permute(0, 2, 1, 3).contiguous()


def _rola_gla_chunked(q, k, v, wg, rg, ld, eps=1e-5, chunk=64, normalized=False):
    """Shared-gram chunked scalar-gated RoLA-GLA. q,k:[B,L,H,dqk] v:[B,L,H,dv]
    wg,rg,ld:[B,L,H,C] -> [B,L,H,dv]. Heads fold into batch; decay absorbed
    chunk-locally; Kronecker state carried across chunks (decayed scan).
    normalized=False (GLA convention, like FLA's GLA / the per-channel baseline):
    raw gated sum, no partition function. normalized=True: global V+1 divide."""
    B, L, H, dqk = q.shape
    dv = v.shape[-1]; C = wg.shape[-1]
    bh = B * H
    fold = lambda t: t.permute(0, 2, 1, 3).reshape(bh, t.shape[1], t.shape[-1])
    q, k, v, wg, rg, ld = fold(q), fold(k), fold(v), fold(wg), fold(rg), fold(ld)
    v1 = torch.cat([v, torch.ones_like(v[..., :1])], dim=-1) if normalized else v
    dvp = v1.shape[-1]
    S = q.new_zeros(bh, C, dqk, dvp)
    outs = []
    for c0 in range(0, L, chunk):
        c1 = min(c0 + chunk, L); Cc = c1 - c0
        qc, kc, vc = q[:, c0:c1], k[:, c0:c1], v1[:, c0:c1]
        rgc, wgc, ldc = rg[:, c0:c1], wg[:, c0:c1], ld[:, c0:c1]
        a = torch.cumsum(ldc, dim=1)                             # chunk-local cumulative log-decay (≤0)
        Lam = a[:, -1, :]                                        # chunk-total log-decay [bh,C]
        G = torch.einsum('bid,bjd->bij', qc, kc)                 # shared content gram
        # Intra-chunk decayed routing gram D_ij = Σ_c rg_i^c wg_j^c exp(a_i^c - a_j^c), as a
        # cheap factored MATMUL rt·wtᵀ. To keep it fp32-safe we (a) MIDPOINT-shift by s=Lam/2
        # so each exponent is ≤|Lam|/2 (not |Lam|), and (b) FLOOR the per-token decay (see
        # _GLA_LD_FLOOR) so |Lam| stays inside fp32 range. NO exponent clamp — that would break
        # the shift's exact cancellation rt_i·wt_j = exp(a_i-a_j) and silently corrupt the O(1)
        # near-diagonal entries (the bug that the [bh,Cc,Cc,C] direct form avoided but ran ~3×
        # slower). The floor is a no-op for healthy gates (rel diff 0.0); it only clips the
        # degenerate deep-decay corner — the same corner that produced the original overflow.
        # Intra-chunk decayed routing gram D_ij = Σ_c rg_i^c wg_j^c exp(a_i^c - a_j^c), formed
        # DIRECTLY (per-state relative decay) with the exponent clamped ≤0. This is the ONLY
        # fp32-safe form: a factored matmul rt·wtᵀ forms exp(a_i-a_j) for the UPPER triangle
        # (small i, large j → exp(+|Lam|)) which overflows BEFORE the causal mask — and the
        # midpoint shift cancels in the product, so it gives the matmul no safety. Here the
        # clamp(max=0) is a NO-OP on the causal triangle (i≥j ⟹ a_i≤a_j, exact) and caps the
        # masked upper triangle, so exp never overflows in any decay regime. Cost: a
        # [bh,Cc,Cc,C] tensor — cheap at the LM's nc≤16; a Triton scan (masks before exp) is
        # the real speedup, deferred.
        dec = torch.exp((a[:, :, None, :] - a[:, None, :, :]).clamp(max=0.0))   # [bh,Cc,Cc,C], ≤1
        D = torch.einsum('bic,bjc,bijc->bij', rgc, wgc, dec)     # decayed routing gram (exact)
        causal = torch.tril(torch.ones(Cc, Cc, device=q.device, dtype=q.dtype))
        A = G * D * causal
        o_intra = torch.einsum('bij,bjv->biv', A, vc)
        # cross-chunk read references the chunk-START state S, so it needs exp(a_i) (a≤0, bounded).
        rt_state = rgc * torch.exp(a)
        M = torch.einsum('bic,bcdv->bidv', rt_state, S)
        o_inter = torch.einsum('bid,bidv->biv', qc, M)
        outs.append(o_intra + o_inter)
        w_end = wgc * torch.exp(Lam[:, None, :] - a)            # writes referenced to chunk end (≤0)
        KV = torch.einsum('bjc,bjd,bjv->bcdv', w_end, kc, vc)
        S = torch.exp(Lam)[:, :, None, None] * S + KV           # decayed cross-chunk carry
    O = torch.cat(outs, dim=1)
    out = O[..., :-1] / (O[..., -1:] + eps) if normalized else O
    return out.view(B, H, L, dv).permute(0, 2, 1, 3).contiguous()


def _rank_stats(sv, l, tols, prefix=''):
    """Rank statistics from singular values sv:[N,l] (already sorted desc per row), over
    the N = b·H (sequence × head) slices. The paper's barrier claim is DISTRIBUTIONAL —
    "the level past which the head's realized rank distribution no longer covers the
    task's demand" (§4) — and a mean over slices can report a rank no slice realizes
    (verified: 4×rank-8 + 4×rank-380 slices → mean 194). So every estimator is emitted
    both as the legacy mean and as the full sorted per-slice distribution `*_dist`.
    Estimators: numerical rank #{σ>tol·σmax} at each tol; Roy & Vetterli (2007)
    effective rank exp(H(σ/Σσ)); participation ratio (Σσ)²/Σσ² (the appendix
    apd:spectra estimator, so main-figure and appendix numbers are comparable);
    stable rank Σσ²/σmax²."""
    smax = sv[..., :1].clamp_min(1e-20)
    out = {}

    def put(name, vals):                       # vals: [N] tensor of per-slice values
        out[f'{prefix}{name}'] = round(vals.mean().item(), 2)
        out[f'{prefix}{name}_dist'] = [round(v, 2) for v in vals.sort().values.tolist()]

    for t in tols:
        put(f'rank_{t:.0e}', (sv > t * smax).sum(-1).float())
    # Roy & Vetterli effective rank: exp(Shannon entropy, nats, of the L1-normalized
    # singular-value distribution). xlogy gives the 0·log0→0 convention exactly.
    p = sv / sv.sum(-1, keepdim=True).clamp_min(1e-20)
    put('eff_rank', torch.exp(-torch.special.xlogy(p, p).sum(-1)))
    put('pr_rank', sv.sum(-1) ** 2 / (sv ** 2).sum(-1).clamp_min(1e-20))
    # stable (numerical) rank ‖W‖_F²/σmax² = Σσ² / σmax² — threshold-free, in [1, rank].
    put('stable_rank', (sv ** 2).sum(-1) / smax.squeeze(-1) ** 2)
    return out


@torch.no_grad()
def _effective_attention_rank(qf, kf, rg, wg, max_b=4, max_l=4096, tols=(1e-1, 1e-2, 1e-3, 1e-4)):
    """Realized effective-attention rank on real inputs — the direct empirical test of the
    paper's rank argument. The per-head effective weight is W = G ∘ R (Hadamard), G=φ(Q)φ(K)ᵀ
    (rank ≤ d_qk), R=(read)(writeᵀ) (rank ≤ nc), so by the Schur/Oppenheim bound
    rank(W) ≤ rank(G)·rank(R) ≤ nc·d_qk. The relevant ceiling is PER-HEAD: a plain
    linear-attention head's score matrix has rank ≤ its feature dim (d_qk, or ≤ d_model for a
    width-d_model "wide monolith"); routing lifts the per-head ceiling to nc·d_qk. So these
    are PER-HEAD means — compare them to the per-head monolith ceiling (d_qk, or d_model for
    the wide monolith), NOT to n_heads·d_v. To demonstrate the claim, pair this with the same
    measurement on the matched-state monolith.

    Measured object: the UNMASKED W — the score-structure rank the nc·d_qk bound is about,
    and the paper's measured object (§7: the causal mask is a Hadamard with the FULL-RANK
    lower-triangular ones matrix, so it can only inflate rank — verified: masked rank-1 ones
    measures 1024/1024 — and the masked map carries no rank signal). Un-normalized: row
    normalization is a positive-diagonal left-scaling, rank-preserving for strictly positive
    entries. The masked object (prefix 'm') can still be emitted as an appendix honesty row
    via CLA_RANK_MASKED=1; off by default, with the SVD budget spent on more sequences
    (max_b=4) so the per-slice DISTRIBUTIONS in `*_dist` have support.
    Also emits the spectrum itself — per-slice σ/σmax at log-spaced indices, quantiles
    across slices — the "supply thins" plot of §4.
    qf,kf:[B,L,H,d_qk]  rg,wg:[B,L,H,nc].  Eval-only, subsampled."""
    B, L, H, _ = qf.shape
    b, l = min(max_b, B), min(max_l, L)
    q, k, r, w = qf[:b, :l], kf[:b, :l], rg[:b, :l], wg[:b, :l]
    measure_masked = bool(os.environ.get('CLA_RANK_MASKED'))
    causal = torch.tril(torch.ones(l, l, device=qf.device)) if measure_masked else None
    # SVD ONE l×l matrix at a time (loop over batch×head) rather than batching the full
    # [b,H,l,l] tensor + cuSOLVER workspace ×(b·H) at once — that peak OOMs/corrupts the
    # heavy cell (nc=256, l=4096: stacked 4096² SVDs) and produced an anomalous, non-monotone
    # rank. Per-slice is identical math, peak memory = one l×l matrix.
    svs, svs_m = [], []
    for bi in range(b):
        for hi in range(H):
            W = (q[bi, :, hi] @ k[bi, :, hi].T) * (r[bi, :, hi] @ w[bi, :, hi].T)   # [l,l]
            W = W.float()
            svs.append(torch.linalg.svdvals(W))                 # unmasked: score-structure rank
            if measure_masked:
                svs_m.append(torch.linalg.svdvals(W * causal))  # masked: inflated, no rank signal
    sv = torch.stack(svs)                                 # [b·H, l] = per-(sequence,head) slices
    out = _rank_stats(sv, l, tols)                        # unmasked (score structure)
    if measure_masked:
        out.update(_rank_stats(torch.stack(svs_m), l, tols, prefix='m'))
    smax = sv[..., :1].clamp_min(1e-20)
    # The spectrum itself, σ_i/σmax at log-spaced indices i, quantiles over slices: the
    # supply distribution. A thinning supply shows as p90 collapsing toward p10 past the
    # barrier; legacy scalar probes sv_ratio_128/256 kept for old-run comparability.
    idx = sorted({0, 1} | {2 ** i for i in range(1, l.bit_length())} | {l - 1})
    idx = [i for i in idx if i < l]
    rel = (sv / smax)[:, idx]                              # [N, len(idx)]
    qs = torch.quantile(rel, torch.tensor([0.1, 0.5, 0.9], device=rel.device), dim=0)
    out['spec_idx'] = idx
    out['spec_p10'] = [round(v, 5) for v in qs[0].tolist()]
    out['spec_p50'] = [round(v, 5) for v in qs[1].tolist()]
    out['spec_p90'] = [round(v, 5) for v in qs[2].tolist()]
    out['sv_ratio_128'] = round((sv[..., min(127, l - 1)] / smax.squeeze(-1)).mean().item(), 4)
    out['sv_ratio_256'] = round((sv[..., min(255, l - 1)] / smax.squeeze(-1)).mean().item(), 4)
    out['seq_len'] = l
    out['n_slices'] = sv.shape[0]
    return out


# ==========================================
# 1. Dataset Handling
# ==========================================
class RecurrentLinearAttention(nn.Module):
    """
    Enhanced Recurrent Linear Attention (RLA).
    
    Improvements over Standard RLA:
    1. NORMALIZATION: Uses the "V+1 Denominator Trick" for stable weighted averaging.
    2. POSITIVITY: Applies ELU+1 to Q/K to ensure strictly positive attention weights.
    """
    def __init__(self, d_model, d_qk, d_v, n_heads=4, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.d_qk = d_qk
        self.d_v = d_v
        self.n_heads = n_heads
        
        # Projection output size: Heads * State_Dim
        self.dim_inner_qk = n_heads * d_qk
        self.dim_inner_v = n_heads * d_v

        self.w_q = nn.Linear(d_model, self.dim_inner_qk, bias=False)
        self.w_k = nn.Linear(d_model, self.dim_inner_qk, bias=False)
        self.w_v = nn.Linear(d_model, self.dim_inner_v, bias=False)
        
        self.w_o = nn.Linear(self.dim_inner_v, d_model, bias=False)

    def forward(self, x):
        # x: [Batch, Length, Dim]
        B, L, _ = x.shape
        H = self.n_heads
        
        # 1. Project and reshape to [B, L, H, D]
        q = self.w_q(x).view(B, L, H, self.d_qk)
        k = self.w_k(x).view(B, L, H, self.d_qk)
        v = self.w_v(x).view(B, L, H, self.d_v)

        # 2. Enforce Positivity (Critical for Denominator Trick)
        # Guarantees that Q * K^T > 0, preventing zero/negative divisors.
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0

        # 3. The Denominator Trick (RWKV-Style)
        # Append 1.0 to V to track the sum of weights.
        # Shape: [B, L, H, D+1]
        v_aug = torch.cat([v, torch.ones_like(v[..., :1])], dim=-1)

        # 4. Kernel Call
        raw_out, _ = fused_chunk_linear_attn(q, k, v_aug, normalize=False, scale=1.0)

        # 5. Normalization
        # Split Signal (num) and Weight Mass (den)
        num = raw_out[..., :-1]
        den = raw_out[..., -1:]
        
        # Stable Division
        out = num / (den + 1e-5) 

        # 6. Output Projection
        out = out.reshape(B, L, self.dim_inner_v)
        return self.w_o(out)

    def get_stats(self):
        """Returns a dictionary of key hyperparameters and calculated state size."""
        # For RLA, num_chunks is effectively 1
        state_floats = self.n_heads * 1 * (self.d_qk * self.d_v + self.d_qk)
        return {
            'd_qk': self.d_qk,
            'd_v': self.d_v,
            'n_heads': self.n_heads,
            'num_chunks': 1, # RLA is not chunked
            'state_floats': state_floats
        }

    def state_size(self, sequence_length: int = None, **kwargs) -> int:
        # Zoology's Hybrid.state_size / model.state_size_total call this on
        # every mixer. Recurrent state is independent of sequence_length.
        return self.get_stats()['state_floats']


class RecurrentGLA(nn.Module):
    """Single-state Gated Linear Attention. Adds a learnable per-(head,key-dim) forget gate.

    State update: S_t = diag(g_t) · S_{t-1} + k_t^T v_t   with g_t = sigmoid(w_g x_t) in (0,1).
    V+1 normalization still applied on the output.
    """
    def __init__(self, d_model, d_qk, d_v, n_heads=4, use_short_conv=False, conv_size=4, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.d_qk = d_qk
        self.d_v = d_v
        self.n_heads = n_heads
        self.dim_inner_qk = n_heads * d_qk
        self.dim_inner_v = n_heads * d_v
        self.w_q = nn.Linear(d_model, self.dim_inner_qk, bias=False)
        self.w_k = nn.Linear(d_model, self.dim_inner_qk, bias=False)
        self.w_v = nn.Linear(d_model, self.dim_inner_v, bias=False)
        self.w_g = nn.Linear(d_model, self.dim_inner_qk, bias=False)  # log-forget-gate per (head, d_qk)
        self.w_o = nn.Linear(self.dim_inner_v, d_model, bias=False)
        self.use_short_conv = use_short_conv
        if use_short_conv:
            self.conv_size = conv_size
            self.conv_pad = conv_size - 1
            self.q_conv = nn.Conv1d(self.dim_inner_qk, self.dim_inner_qk,
                                    kernel_size=conv_size, groups=self.dim_inner_qk, bias=False)
            self.k_conv = nn.Conv1d(self.dim_inner_qk, self.dim_inner_qk,
                                    kernel_size=conv_size, groups=self.dim_inner_qk, bias=False)
            self.v_conv = nn.Conv1d(self.dim_inner_v, self.dim_inner_v,
                                    kernel_size=conv_size, groups=self.dim_inner_v, bias=False)

    def forward(self, x):
        B, L, _ = x.shape
        H = self.n_heads
        q_raw = self.w_q(x)
        k_raw = self.w_k(x)
        v_raw = self.w_v(x)
        if self.use_short_conv:
            def _cconv(xx, conv):
                xx = xx.transpose(1, 2)
                xx = F.pad(xx, (self.conv_pad, 0))
                xx = conv(xx)
                return F.silu(xx.transpose(1, 2))
            q_raw = _cconv(q_raw, self.q_conv)
            k_raw = _cconv(k_raw, self.k_conv)
            v_raw = _cconv(v_raw, self.v_conv)
        q = F.elu(q_raw.view(B, L, H, self.d_qk)) + 1.0
        k = F.elu(k_raw.view(B, L, H, self.d_qk)) + 1.0
        v = v_raw.view(B, L, H, self.d_v)
        g = F.logsigmoid(self.w_g(x).view(B, L, H, self.d_qk))
        v_aug = torch.cat([v, torch.ones_like(v[..., :1])], dim=-1)
        raw_out, _ = chunk_gla(q, k, v_aug, g, scale=1.0)
        num = raw_out[..., :-1]
        den = raw_out[..., -1:]
        out = num / (den + 1e-5)
        return self.w_o(out.reshape(B, L, self.dim_inner_v))

    def get_stats(self):
        state_floats = self.n_heads * 1 * (self.d_qk * self.d_v + self.d_qk)
        return {'d_qk': self.d_qk, 'd_v': self.d_v, 'n_heads': self.n_heads,
                'num_chunks': 1, 'state_floats': state_floats}

    def state_size(self, sequence_length: int = None, **kwargs) -> int:
        return self.get_stats()['state_floats']


class RecurrentGatedDelta(nn.Module):
    """Single-state Gated Delta Rule (the kernel MoM uses by default).

    State update: S_t = g_t · (I - k_t^T k_t) · S_{t-1} + β_t · k_t^T v_t
    No V+1 normalization (delta rule doesn't factor into simple weighted average).
    """
    def __init__(self, d_model, d_qk, d_v, n_heads=4, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.d_qk = d_qk
        self.d_v = d_v
        self.n_heads = n_heads
        self.dim_inner_qk = n_heads * d_qk
        self.dim_inner_v = n_heads * d_v
        self.w_q = nn.Linear(d_model, self.dim_inner_qk, bias=False)
        self.w_k = nn.Linear(d_model, self.dim_inner_qk, bias=False)
        self.w_v = nn.Linear(d_model, self.dim_inner_v, bias=False)
        self.w_g = nn.Linear(d_model, n_heads, bias=False)     # scalar forget-gate per head
        self.w_beta = nn.Linear(d_model, n_heads, bias=False)  # scalar delta amplitude per head
        self.w_o = nn.Linear(self.dim_inner_v, d_model, bias=False)

    def forward(self, x):
        B, L, _ = x.shape
        H = self.n_heads
        q = self.w_q(x).view(B, L, H, self.d_qk)
        k = self.w_k(x).view(B, L, H, self.d_qk)
        v = self.w_v(x).view(B, L, H, self.d_v)
        g = F.logsigmoid(self.w_g(x).view(B, L, H))
        beta = torch.sigmoid(self.w_beta(x).view(B, L, H))
        out, _ = chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
        return self.w_o(out.reshape(B, L, self.dim_inner_v))

    def get_stats(self):
        # GDN has no V+1 denominator trick — state is just d_qk * d_v per head.
        state_floats = self.n_heads * 1 * (self.d_qk * self.d_v)
        return {'d_qk': self.d_qk, 'd_v': self.d_v, 'n_heads': self.n_heads,
                'num_chunks': 1, 'state_floats': state_floats}

    def state_size(self, sequence_length: int = None, **kwargs) -> int:
        return self.get_stats()['state_floats']


# ============================================================================
# Routed linear-attention KERNELS (composition). One class per inner update rule;
# each maps (x, q, k, v [B,L,H,*], write_gates, read_gates [B,L,H,C]) -> [B,L,H,d_v].
# Projections + routing live in the orchestrator (RoLA); a kernel owns ONLY its
# kernel-specific params (feature map / forget gate / beta) and its own optimal
# compute path. Selection is the registry — the orchestrator has NO kernel branches.
#   proj_qk        : q/k projection width the orchestrator allocates (d_qk, except
#                    hedgehog = d_qk//2 — the +/- map doubles it back to d_qk).
#   feat_dim       : feature dim governing recurrent state (d_qk for elu; bigger for
#                    hedgehog/based/rebased).
#   uses_v_plus_one: state includes the +feat_dim global-partition-fn term.
# ============================================================================
class RoutedKernel(nn.Module):
    uses_v_plus_one = True

    def __init__(self, d_model, n_heads, num_chunks, d_qk, d_v):
        super().__init__()
        self.d_model, self.n_heads, self.num_chunks = d_model, n_heads, num_chunks
        self.d_qk, self.d_v = d_qk, d_v
        self.feat_dim = d_qk
        self.proj_qk = d_qk

    def forward(self, x, q, k, v, write_gates, read_gates):
        raise NotImplementedError


class AdditiveKernel(RoutedKernel):
    """RoLA-RLA family: global-norm shared-gram linear attention with feature map
    phi in {elu, hedgehog, based, rebased}. Fused via _rola_chunked_parallel (shares
    the content gram, fuses the read-combine, single global partition fn). Carries
    the eval-only realized-rank diagnostic and a first-use correctness gate."""
    uses_v_plus_one = True

    def __init__(self, d_model, n_heads, num_chunks, d_qk, d_v, phi='elu', state_norm='global'):
        super().__init__(d_model, n_heads, num_chunks, d_qk, d_v)
        self.phi = phi
        assert state_norm in ('global', 'per_state', 'kappa'), state_norm
        self.state_norm = state_norm
        # 'kappa': input-dependent interpolation global↔per-state. The read gates are rescaled
        # r̃ = r·(d+ε)^{-κ(x)} with d the per-state denominator (_rola_perstate_den) and
        # κ = sigmoid(w_κ x) per head; the GLOBAL combine then runs unchanged (Triton-compatible).
        # κ=0 reproduces global exactly; κ=1 per-state exactly (read gates sum to 1).
        # (κ, not α: the paper uses α for the read-routing distribution.)
        # ALL norms ride the global-form combine (Triton or eager): per_state and kappa rescale
        # the read gates first (per_state ≡ kappa=1, exact). _rola_perstate_ref remains as the
        # endpoint ORACLE in the first-use check only.
        self._kfn = _rola_chunked_parallel
        self._ref = _rola_global_ref
        if state_norm == 'kappa':
            self.w_kappa = nn.Linear(d_model, n_heads)
            nn.init.zeros_(self.w_kappa.weight)
            nn.init.constant_(self.w_kappa.bias, -4.0)   # start ≈ global (κ≈0.018), learn upward
        if phi == 'hedgehog':
            from fla.modules.feature_map import HedgehogFeatureMap
            assert d_qk % 2 == 0, "hedgehog needs even d_qk (feature dim 2*(d_qk//2))"
            self.hh_q = HedgehogFeatureMap(head_dim=d_qk // 2)
            self.hh_k = HedgehogFeatureMap(head_dim=d_qk // 2)
            self.proj_qk = d_qk // 2; self.feat_dim = d_qk
        elif phi == 'based':
            from fla.modules.feature_map import TaylorFeatureMap
            self.taylor = TaylorFeatureMap(head_dim=d_qk)
            self.feat_dim = 1 + 2 * d_qk + d_qk * (d_qk - 1) // 2
        elif phi == 'rebased':
            from fla.modules.feature_map import RebasedFeatureMap
            self.rebased_q = RebasedFeatureMap(head_dim=d_qk)
            self.rebased_k = RebasedFeatureMap(head_dim=d_qk)
            self.feat_dim = d_qk * (d_qk + 1) // 2
        self._current_epoch = 0      # trainer sets this; used to throttle the rank diag
        self._checked = False
        self._last_rank_epoch = -2

    def _feature_map(self, q, k):
        if self.phi == 'hedgehog': return self.hh_q(q), self.hh_k(k)
        if self.phi == 'based':    return self.taylor(q), self.taylor(k)
        if self.phi == 'rebased':  return self.rebased_q(q), self.rebased_k(k)
        return F.elu(q) + 1.0, F.elu(k) + 1.0

    def forward(self, x, q, k, v, write_gates, read_gates):
        B, L = x.shape[0], x.shape[1]
        qf, kf = self._feature_map(q, k)
        if self.state_norm in ('kappa', 'per_state'):
            # rescale read gates by the per-state mass, then run the standard GLOBAL combine on
            # the modified gates (kernel unchanged, incl. Triton). per_state ≡ kappa=1 exactly
            # (read gates sum to 1 ⇒ the outer divide collapses). The mass d comes from the fork's
            # Triton den kernel (scan + chunk-parallel bwd) — the eager helper retains its chunk
            # grams for backward (12GB VRAM blowup at LM scale) and is kept as the ORACLE only.
            if _fla_routed is not None and qf.is_cuda and qf.shape[-1] <= 64:
                from fla_rola.ops.simple_gla.rola import rola_perstate_den_triton
                H = self.n_heads
                foldd = lambda t: t.permute(0, 2, 1, 3).reshape(B * H, L, t.shape[-1])
                with torch.autocast(device_type='cuda', enabled=False):
                    dt = _triton_compute_dtype(qf.dtype)
                    d = rola_perstate_den_triton(foldd(qf).to(dt), foldd(kf).to(dt),
                                                 foldd(write_gates).to(dt))
                d = d.view(B, H, L, -1).permute(0, 2, 1, 3)
            else:
                d = _rola_perstate_den(qf, kf, write_gates)
            if self.state_norm == 'kappa':
                kap = torch.sigmoid(self.w_kappa(x)).view(B, L, self.n_heads, 1)
                read_gates = read_gates * (d + 1e-5).pow(-kap)
            else:
                read_gates = read_gates / (d + 1e-5)
        # φ-agnostic: the kernel only sees post-feature-map q,k (G=φ(q)·φ(k)ᵀ), so hedgehog/based/
        # rebased work too — gate on state_norm (global-form combine only) and the feature dim
        # fitting SRAM, NOT on the φ identity. ('kappa' uses the global combine on modified gates.)
        if (_USE_TRITON_RLA and _fla_routed is not None and qf.is_cuda
                and qf.shape[-1] <= 64):
            dv = self.d_v
            # Cast ALL kernel inputs to the autocast compute dtype (bf16) — fixes both the
            # router-softmax-is-fp32 mismatch AND the hedgehog-softmax-is-fp32 slowdown. tl.dot
            # still accumulates fp32. Autocast OFF so intermediates aren't silently re-cast.
            # Model tensors are already FLA's [B,T,H,*] layout — no folding. The fork returns the
            # un-normalized readout; global norm = ones-column augmentation + divide, done here.
            dt = _triton_compute_dtype(qf.dtype)
            c = lambda t: t.to(dt)
            with torch.autocast(device_type='cuda', enabled=False):
                v1 = torch.cat([v, torch.ones_like(v[..., :1])], dim=-1)
                oa, _ = _fla_routed(c(qf), c(kf), c(v1), scale=1.0,
                                    r=c(read_gates), w=c(write_gates))
                out = (oa[..., :dv] / (oa[..., dv:dv + 1] + 1e-5)).to(v.dtype)
        else:
            out = self._kfn(qf, kf, v, write_gates, read_gates)
        if not self._checked:
            self._checked = True
            # Validate the ALGORITHM in true fp32: cast inputs AND disable autocast
            # (autocast would otherwise re-cast the einsums to bf16 -> false 1e-2 mismatch).
            with torch.no_grad(), torch.autocast(device_type='cuda', enabled=False):
                bb, ll = min(2, B), min(256, L)
                s = (slice(0, bb), slice(0, ll))
                f = lambda t: t[s].float()
                ref = self._ref(f(qf), f(kf), f(v), f(write_gates), f(read_gates))
                chk = self._kfn(f(qf), f(kf), f(v), f(write_gates), f(read_gates))
                rel = (chk - ref).abs().max().item() / (ref.abs().max().item() + 1e-6)
                assert rel < 1e-2, f"fused RoLA ({self.state_norm}) vs ref mismatch: rel={rel:.2e}"
                if self.state_norm in ('kappa', 'per_state'):
                    # endpoint check: κ≡1 gates through the global combine must equal the
                    # per-state reference (read gates sum to 1 ⇒ outer divide collapses).
                    # Run on SYNTHETIC well-conditioned inputs: the identity is algebraic, and the
                    # ε-placement deviation (r/(d+ε) vs num/(den+ε), O(ε/den)) is unbounded on
                    # live activations once trained routing is peaked (tiny per-state masses) —
                    # a trained checkpoint's data must not fail a math check. ε=0 identical.
                    gen = torch.Generator(device=qf.device).manual_seed(7)
                    rnd = lambda *sh: torch.randn(*sh, generator=gen, device=qf.device)
                    qs = F.elu(rnd(2, 128, self.n_heads, self.d_qk)) + 1.0
                    ks = F.elu(rnd(2, 128, self.n_heads, self.d_qk)) + 1.0
                    vs = rnd(2, 128, self.n_heads, self.d_v)
                    ws = torch.softmax(rnd(2, 128, self.n_heads, self.num_chunks), -1)
                    raw = torch.softmax(rnd(2, 128, self.n_heads, self.num_chunks), -1)
                    d_s = _rola_perstate_den(qs, ks, ws)
                    r1 = raw * (d_s + 1e-5).pow(-1.0)                    # κ≡1
                    chk1 = _rola_chunked_parallel(qs, ks, vs, ws, r1)
                    ref1 = _rola_perstate_ref(qs, ks, vs, ws, raw)
                    rel1 = (chk1 - ref1).abs().max().item() / (ref1.abs().max().item() + 1e-6)
                    assert rel1 < 3e-2, f"kappa endpoint (κ=1) vs per-state ref mismatch: rel={rel1:.2e}"
        _ep = self._current_epoch
        # Measure rank once per (epoch, sequence-length): MQAR slices have different L
        # (kv=1024 → longest), so this captures rank PER SLICE — does rank grow on the
        # hard/long slice where the task actually demands it, past d_model?
        if not self.training and os.environ.get('CLA_MEASURE_RANK') and L >= 256:
            if not hasattr(self, '_rank_seen'):
                self._rank_seen = set()
            key = (_ep, L)
            if key not in self._rank_seen:
                self._rank_seen.add(key)
                try:
                    import json as _json
                    _r = _effective_attention_rank(qf, kf, read_gates, write_gates)
                    _r.update(nc=self.num_chunks, d_qk=self.d_qk, d_model=self.d_model,
                              nc_dqk=self.num_chunks * self.d_qk, epoch=_ep, seqlen=L)
                    print(f"RANK_JSON {_json.dumps(_r)}", flush=True)
                except Exception:
                    pass
        return out


class ScalarGLAKernel(RoutedKernel):
    """Optimized RoLA-GLA: per-state SCALAR forget gate -> the decay factors out of
    the content contraction, so the shared-gram fused path applies (_rola_gla_chunked,
    decay absorbed chunk-locally + decayed cross-chunk state). Avoids the virtual-head
    [B,L,H,nc,d_qk] blowup (~4x faster, ~9x less mem at the real training batch).
    normalized=False (default, GLA convention): raw gated sum. normalized=True: global
    V+1 partition fn (adds +feat_dim to state). First-use correctness gate."""

    def __init__(self, d_model, n_heads, num_chunks, d_qk, d_v, normalized=False, state_norm=None):
        super().__init__(d_model, n_heads, num_chunks, d_qk, d_v)
        # state_norm supersedes the legacy bool: 'raw' (GLA convention), 'global' (V+1 partition
        # fn), 'per_state', 'kappa'. kappa/per_state rescale the read gates by the per-state mass
        # UNDER DECAY (fork's GLA den kernel, verified incl. dld) then run the global combine —
        # same construction as AdditiveKernel's kappa, with d now the decayed mass.
        if state_norm is None:
            state_norm = 'global' if normalized else 'raw'
        assert state_norm in ('raw', 'global', 'per_state', 'kappa'), state_norm
        self.state_norm = state_norm
        self.normalized = state_norm != 'raw'
        self.uses_v_plus_one = self.normalized
        self.w_g = nn.Linear(d_model, n_heads, bias=False)    # per-head scalar forget gate
        if state_norm == 'kappa':
            self.w_kappa = nn.Linear(d_model, n_heads)
            nn.init.zeros_(self.w_kappa.weight)
            nn.init.constant_(self.w_kappa.bias, -4.0)   # start ≈ global (κ≈0.018), learn upward
        self._checked = False

    def _log_decay(self, x, write_gates):
        B, L = x.shape[0], x.shape[1]; H = self.n_heads
        alpha = F.logsigmoid(self.w_g(x).view(B, L, H)).exp()           # [B,L,H]
        alpha_chunk = 1.0 - write_gates * (1.0 - alpha.unsqueeze(-1))    # [B,L,H,C]
        return alpha_chunk.clamp(min=1e-8).log()

    def forward(self, x, q, k, v, write_gates, read_gates):
        B, L = x.shape[0], x.shape[1]
        qg = F.elu(q) + 1.0; kg = F.elu(k) + 1.0
        ld = self._log_decay(x, write_gates)
        if self.state_norm in ('kappa', 'per_state'):
            # rescale read gates by the DECAYED per-state mass, then run the standard global
            # combine on the modified gates (kernel unchanged). per_state ≡ kappa=1 exactly.
            if _fla_routed is not None and qg.is_cuda and qg.shape[-1] <= 64:
                from fla_rola.ops.simple_gla.rola import rola_perstate_den_gla_triton
                H = self.n_heads
                foldd = lambda t: t.permute(0, 2, 1, 3).reshape(B * H, L, t.shape[-1])
                with torch.autocast(device_type='cuda', enabled=False):
                    dt = _triton_compute_dtype(qg.dtype)
                    d = rola_perstate_den_gla_triton(foldd(qg).to(dt), foldd(kg).to(dt),
                                                     foldd(write_gates).to(dt), foldd(ld).to(dt))
                d = d.view(B, H, L, -1).permute(0, 2, 1, 3)
            else:
                d = _rola_gla_perstate_den(qg, kg, write_gates, ld)
            if self.state_norm == 'kappa':
                kap = torch.sigmoid(self.w_kappa(x)).view(B, L, self.n_heads, 1)
                read_gates = read_gates * (d + 1e-5).pow(-kap)
            else:
                read_gates = read_gates / (d + 1e-5)
        if _USE_TRITON and _fla_routed is not None and qg.is_cuda and qg.shape[-1] <= 64:
            dv = self.d_v
            # Cast all inputs to one dtype (see AdditiveKernel — softmax routers are fp32 under
            # autocast, q/k/v bf16). Model tensors are already [B,T,H,*]; the fork returns the
            # un-normalized readout (GLA convention) — normalized=True adds the ones-column here.
            dt = _triton_compute_dtype(qg.dtype)
            c = lambda t: t.to(dt)
            with torch.autocast(device_type='cuda', enabled=False):
                v_in = torch.cat([v, torch.ones_like(v[..., :1])], dim=-1) if self.normalized else v
                oa, _ = _fla_routed(c(qg), c(kg), c(v_in), g=c(ld), scale=1.0,
                                    r=c(read_gates), w=c(write_gates))
                out = (oa[..., :dv] / (oa[..., dv:dv + 1] + 1e-5)).to(v.dtype) if self.normalized \
                    else oa.to(v.dtype)
        else:
            out = _rola_gla_chunked(qg, kg, v, write_gates, read_gates, ld, normalized=self.normalized)
        if not self._checked:
            self._checked = True
            with torch.no_grad(), torch.autocast(device_type='cuda', enabled=False):
                bb, ll = min(2, B), min(192, L)
                s = (slice(0, bb), slice(0, ll))
                f = lambda t: t[s].float()   # fp32 + autocast off: validate the algorithm, not bf16 roundoff
                ref = _rola_gla_ref(f(qg), f(kg), f(v), f(write_gates), f(read_gates), f(ld), normalized=self.normalized)
                chk = _rola_gla_chunked(f(qg), f(kg), f(v), f(write_gates), f(read_gates), f(ld), normalized=self.normalized)
                rel = (chk - ref).abs().max().item() / (ref.abs().max().item() + 1e-6)
                assert rel < 1e-2, f"fused GLA vs ref mismatch: rel={rel:.2e}"
                # DEEP-DECAY gate: init gates are mild (chunk-total decay ~10 nats), so the
                # check above never exercises the regime where a factored/clamped decay form
                # silently corrupts the intra-chunk gram. Force strong decay (ld≡-3 → ~190
                # nats/chunk) and re-verify chunked==ref — guards against that class of bug.
                ld_deep = torch.full_like(f(ld), -3.0)
                rd = _rola_gla_ref(f(qg), f(kg), f(v), f(write_gates), f(read_gates), ld_deep, normalized=self.normalized)
                cd = _rola_gla_chunked(f(qg), f(kg), f(v), f(write_gates), f(read_gates), ld_deep, normalized=self.normalized)
                rel_d = (cd - rd).abs().max().item() / (rd.abs().max().item() + 1e-6)
                assert rel_d < 1e-2, f"fused GLA deep-decay mismatch: rel={rel_d:.2e}"
                if self.state_norm in ('kappa', 'per_state'):
                    # DEN gate on SYNTHETIC well-massed inputs (live peaked routing is
                    # ε-sensitive): dense oracle vs chunked torch vs (CUDA) Triton den.
                    H, nc = self.n_heads, self.num_chunks
                    qs = torch.rand(2, 128, H, self.d_qk, device=qg.device) + 0.1
                    ws = torch.softmax(torch.randn(2, 128, H, nc, device=qg.device), -1)
                    lds = -torch.rand(2, 128, H, nc, device=qg.device) * 0.5
                    Lam = lds.cumsum(1)
                    dec = torch.exp(Lam[:, :, None] - Lam[:, None, :, :, :])      # [B,i,j,H,nc]
                    Gd = torch.einsum('bihd,bjhd->bijh', qs, qs)
                    m = torch.tril(torch.ones(128, 128, device=qg.device, dtype=torch.bool))
                    d_dense = (Gd[..., None] * dec * ws[:, None] * m[None, :, :, None, None]).sum(2)
                    d_chk = _rola_gla_perstate_den(qs, qs, ws, lds)
                    r1 = (d_chk - d_dense).abs().max().item() / (d_dense.abs().max().item() + 1e-6)
                    assert r1 < 1e-2, f"GLA den chunked vs dense oracle mismatch: rel={r1:.2e}"
                    if _fla_routed is not None and qg.is_cuda and self.d_qk <= 64:
                        from fla_rola.ops.simple_gla.rola import rola_perstate_den_gla_triton
                        foldd = lambda t: t.permute(0, 2, 1, 3).reshape(2 * H, 128, t.shape[-1])
                        d_tri = rola_perstate_den_gla_triton(foldd(qs), foldd(qs), foldd(ws), foldd(lds))
                        d_tri = d_tri.view(2, H, 128, -1).permute(0, 2, 1, 3)
                        r2 = (d_tri - d_dense).abs().max().item() / (d_dense.abs().max().item() + 1e-6)
                        assert r2 < 1e-2, f"GLA den Triton vs dense oracle mismatch: rel={r2:.2e}"
        return out


class VirtualHeadGLAKernel(RoutedKernel):
    """RoLA-GLA with PER-CHANNEL forget gate. The vector decay is entangled inside the
    content contraction (does NOT factor), so there is no shared-gram form — it runs as
    nc virtual heads through FLA's chunk_gla. The nc-scaling cost is fundamental here
    (the paper's 'RoLA+GLA needs virtual heads, doesn't scale' point). normalized:
    per-state V+1 division then read-combine; else raw gated sum then read-combine."""

    def __init__(self, d_model, n_heads, num_chunks, d_qk, d_v, normalized=False):
        super().__init__(d_model, n_heads, num_chunks, d_qk, d_v)
        self.normalized = normalized
        self.uses_v_plus_one = normalized
        self.w_g = nn.Linear(d_model, n_heads * d_qk, bias=False)   # per-channel forget gate

    def forward(self, x, q, k, v, write_gates, read_gates):
        B, L = x.shape[0], x.shape[1]
        H, C = self.n_heads, self.num_chunks
        q = F.elu(q) + 1.0; k = F.elu(k) + 1.0
        alpha = F.logsigmoid(self.w_g(x).view(B, L, H, self.d_qk)).exp()
        alpha_chunk = 1.0 - write_gates.unsqueeze(-1) * (1.0 - alpha.unsqueeze(3))
        q_flat = q.unsqueeze(3).expand(-1, -1, -1, C, -1).reshape(B, L, H * C, self.d_qk)
        k_flat = k.unsqueeze(3).expand(-1, -1, -1, C, -1).reshape(B, L, H * C, self.d_qk)
        g_flat = alpha_chunk.clamp(min=1e-8).log().reshape(B, L, H * C, self.d_qk)
        if self.normalized:
            v_aug = torch.cat([v, torch.ones_like(v[..., :1])], dim=-1)
            v_flat = (v_aug.unsqueeze(3) * write_gates.unsqueeze(-1)).reshape(B, L, H * C, self.d_v + 1)
            raw = _headchunked(chunk_gla, [q_flat, k_flat, v_flat, g_flat], scale=1.0)
            co = (raw[..., :-1] / (raw[..., -1:] + 1e-5)).view(B, L, H, C, self.d_v)
        else:
            v_flat = (v.unsqueeze(3) * write_gates.unsqueeze(-1)).reshape(B, L, H * C, self.d_v)
            co = _headchunked(chunk_gla, [q_flat, k_flat, v_flat, g_flat], scale=1.0).view(B, L, H, C, self.d_v)
        return torch.sum(read_gates.unsqueeze(-1) * co, dim=3)


class GDNKernel(RoutedKernel):
    """RoLA-GDN: gated delta-rule per state via virtual heads (chunk_gated_delta_rule).
    The delta rule (in-chunk triangular solve) does not factor — virtual heads only.
    Scalar forget gate + beta, both routed per state. Un-normalized (delta rule)."""
    uses_v_plus_one = False

    def __init__(self, d_model, n_heads, num_chunks, d_qk, d_v):
        super().__init__(d_model, n_heads, num_chunks, d_qk, d_v)
        self.w_g = nn.Linear(d_model, n_heads, bias=False)
        self.w_beta = nn.Linear(d_model, n_heads, bias=False)

    def forward(self, x, q, k, v, write_gates, read_gates):
        B, L = x.shape[0], x.shape[1]
        H, C = self.n_heads, self.num_chunks
        g = F.logsigmoid(self.w_g(x).view(B, L, H))
        beta = torch.sigmoid(self.w_beta(x).view(B, L, H))
        alpha = g.exp().unsqueeze(3)                          # [B,L,H,1]
        alpha_chunk = 1.0 - write_gates * (1.0 - alpha)       # [B,L,H,C]
        beta_chunk = write_gates * beta.unsqueeze(3)          # [B,L,H,C]
        q_flat = q.unsqueeze(3).expand(-1, -1, -1, C, -1).reshape(B, L, H * C, self.d_qk)
        k_flat = k.unsqueeze(3).expand(-1, -1, -1, C, -1).reshape(B, L, H * C, self.d_qk)
        v_flat = v.unsqueeze(3).expand(-1, -1, -1, C, -1).reshape(B, L, H * C, self.d_v)
        g_flat = alpha_chunk.clamp(min=1e-8).log().reshape(B, L, H * C)
        beta_flat = beta_chunk.reshape(B, L, H * C)
        out = _headchunked(chunk_gated_delta_rule, [q_flat, k_flat, v_flat, g_flat, beta_flat],
                           use_qk_l2norm_in_kernel=True)
        co = out.view(B, L, H, C, self.d_v)
        return torch.sum(read_gates.unsqueeze(-1) * co, dim=3)


KERNEL_REGISTRY = {
    'rla': AdditiveKernel,                # phi-parameterized: elu/hedgehog/based/rebased
    'gla_scalar': ScalarGLAKernel,        # optimized shared-gram GLA (scalar gate)
    'gla_virtual': VirtualHeadGLAKernel,  # per-channel GLA (virtual heads, no shared gram)
    'gdn': GDNKernel,
}


class RoLA(nn.Module):
    """Routed Linear Attention.

    Shared Q/K/V projections + learned read/write routing over `num_chunks`
    recurrent states, with a pluggable inner kernel (composition — no branches):
      x -> project (q,k,v)  -> route (write_gates, read_gates)
        -> kernel(x, q, k, v, write_gates, read_gates)  -> [B,L,H,d_v]  -> out proj.

    The inner kernel is selected by `kernel` (KERNEL_REGISTRY): 'rla' (AdditiveKernel,
    phi=elu/hedgehog/based/rebased), 'gla_scalar' (optimized shared-gram GLA),
    'gla_virtual' (per-channel GLA), 'gdn'. `tie_routers=True` shares one router for
    read+write (symmetric). NOTE: `num_chunks` is the number of routed STATES (nc) —
    NOT the kernel's internal sequence chunk-blocking (a fixed 64). Legacy name.
    """
    def __init__(self, d_model, d_qk, d_v, num_chunks, n_heads=4, tie_routers=False,
                 tie_router_init=False, kernel='rla', phi='elu', kernel_kwargs=None,
                 use_short_conv=False, conv_size=4, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.d_qk = d_qk
        self.d_v = d_v
        self.n_heads = n_heads
        self.num_chunks = num_chunks
        self.tie_routers = tie_routers

        # Build the inner kernel first — it reports proj_qk (q/k projection width;
        # hedgehog halves it, its +/- map doubles back) and feat_dim (state dim). The
        # orchestrator is kernel-agnostic from here (composition, no per-kernel branches).
        kkw = dict(kernel_kwargs or {})
        if kernel == 'rla':
            kkw.setdefault('phi', phi)
        self.kernel = KERNEL_REGISTRY[kernel](
            d_model=d_model, n_heads=n_heads, num_chunks=num_chunks, d_qk=d_qk, d_v=d_v, **kkw)
        self.proj_qk = self.kernel.proj_qk
        self.dim_inner_qk = n_heads * self.proj_qk
        self.dim_inner_v = n_heads * d_v

        # Shared projections (always present, identical across variants).
        self.w_q = nn.Linear(d_model, self.dim_inner_qk, bias=False)
        self.w_k = nn.Linear(d_model, self.dim_inner_qk, bias=False)
        self.w_v = nn.Linear(d_model, self.dim_inner_v, bias=False)
        self.w_o = nn.Linear(self.dim_inner_v, d_model, bias=False)

        # Optional short causal 1D conv over q/k/v (matching FLA's GDN inductive bias).
        # Kernel size `conv_size` (default 4); causal padding to preserve sequence length.
        self.use_short_conv = use_short_conv
        if use_short_conv:
            self.conv_size = conv_size
            self.conv_pad = conv_size - 1
            self.q_conv = nn.Conv1d(self.dim_inner_qk, self.dim_inner_qk,
                                    kernel_size=conv_size, groups=self.dim_inner_qk, bias=False)
            self.k_conv = nn.Conv1d(self.dim_inner_qk, self.dim_inner_qk,
                                    kernel_size=conv_size, groups=self.dim_inner_qk, bias=False)
            self.v_conv = nn.Conv1d(self.dim_inner_v,  self.dim_inner_v,
                                    kernel_size=conv_size, groups=self.dim_inner_v,  bias=False)

        # Routing lives in the orchestrator (orthogonal to the kernel). Linear routers
        # on the residual stream -> dense softmax over states (default init). tie_routers
        # shares one router for read+write (symmetric).
        self.write_router = nn.Linear(d_model, n_heads * num_chunks, bias=False)
        # sym (tie_routers): read_router stays None and _route reuses write_router. This avoids
        # registering a SECOND module that aliases the same weight — which would put two keys
        # (read_router.weight, write_router.weight) for one tensor in the state_dict and crash
        # HF's safetensors save ("shared tensors ... not properly defined"). Functionally
        # identical to read==write; zero numerical change.
        if tie_routers:
            self.read_router = None
        else:
            self.read_router = nn.Linear(d_model, n_heads * num_chunks, bias=False)
            # tie_router_init: untied routers, but read STARTS == write so asym begins in the
            # sym basin (then is free to specialize). Cheap fix for asym's bad conditioning.
            if tie_router_init:
                self.read_router.weight.data.copy_(self.write_router.weight.data)

        # Print actual instantiated state size (parseable) so runners log the real state.
        if os.environ.get('CLA_PRINT_STATE_JSON', '1') != '0':
            try:
                import json as _json
                _stats = self.get_stats(); _stats['kernel'] = kernel
                print(f"STATE_FLOATS_JSON {_json.dumps(_stats)}", flush=True)
            except Exception:
                pass

    def _route(self, x):
        """Dense softmax read/write gates [B,L,H,C]."""
        B, L = x.shape[0], x.shape[1]; H, C = self.n_heads, self.num_chunks
        write_gates = F.softmax(self.write_router(x).view(B, L, H, C), dim=-1)
        _rr = self.read_router if self.read_router is not None else self.write_router  # sym reuses write
        read_gates = F.softmax(_rr(x).view(B, L, H, C), dim=-1)
        return write_gates, read_gates

    def get_auxiliary_loss(self):
        return 0.0   # dense routing -> no load-balance auxiliary loss

    def forward(self, x):
        B, L, _ = x.shape
        H = self.n_heads
        q = self.w_q(x); k = self.w_k(x); v = self.w_v(x)
        if self.use_short_conv:
            # Causal depthwise conv + SiLU (FLA's ShortConvolution); [B,C,L], pad left.
            def _cconv(xx, conv):
                xx = xx.transpose(1, 2)
                xx = F.pad(xx, (self.conv_pad, 0))
                xx = conv(xx)
                return F.silu(xx.transpose(1, 2))
            q = _cconv(q, self.q_conv); k = _cconv(k, self.k_conv); v = _cconv(v, self.v_conv)
        q = q.view(B, L, H, self.proj_qk)   # proj_qk == d_qk except hedgehog (d_qk//2)
        k = k.view(B, L, H, self.proj_qk)
        v = v.view(B, L, H, self.d_v)
        write_gates, read_gates = self._route(x)
        out = self.kernel(x, q, k, v, write_gates, read_gates)   # [B,L,H,d_v] — polymorphic, no branches
        return self.w_o(out.reshape(B, L, H * self.d_v))

    def get_stats(self):
        """Key hyperparameters + recurrent state size. State is governed by the kernel's
        FEATURE dim (feat_dim == d_qk for elu; hedgehog/based/rebased expand it), with a
        +feat_dim term when the kernel uses the global-partition (V+1) trick."""
        fd = self.kernel.feat_dim
        per_entry = fd * self.d_v + (fd if self.kernel.uses_v_plus_one else 0)
        state_floats = self.n_heads * self.num_chunks * per_entry
        return {
            'd_qk': self.d_qk,
            'feat_dim': fd,
            'd_v': self.d_v,
            'n_heads': self.n_heads,
            'num_chunks': self.num_chunks,
            'state_floats': state_floats
        }

    def state_size(self, sequence_length: int = None, **kwargs) -> int:
        # Zoology's Hybrid.state_size / model.state_size_total call this on
        # every mixer. Recurrent state is independent of sequence_length.
        return self.get_stats()['state_floats']


# ----------------------------------------------------------------------------
# RoLA named instances. Dense linear routing on the residual stream; the INNER
# KERNEL is the only parameterized axis. Instances differ along:
#   - read/write symmetry : tied (tie_routers=True) vs untied (asym)
#   - inner kernel        : 'rla' (phi=elu/hedgehog/based/rebased) / 'gla_scalar'
#                           (optimized) / 'gla_virtual' (per-channel) / 'gdn'
# Main model: rola-rla-asym (dense, untied, RLA additive kernel).
# ----------------------------------------------------------------------------

def rola_instance(name, d_qk, d_v, num_chunks, n_heads=4):
    """Return RoLA kwargs for a named instance. Dense linear
    routing on the residual stream; the INNER KERNEL is the only parameterized axis
    (kernel + optional kernel_kwargs). tie_routers => symmetric (one router)."""
    common = dict(d_qk=d_qk, d_v=d_v, num_chunks=num_chunks, n_heads=n_heads,
                  use_short_conv=False)
    # RLA family (AdditiveKernel, global-norm shared-gram; phi = feature map).
    if name == 'rola-rla-asym':
        return dict(kernel='rla', phi='elu', tie_routers=False, **common)
    if name == 'rola-rla-sym':
        return dict(kernel='rla', phi='elu', tie_routers=True, **common)
    # PER-STATE-norm variants (A/B against global norm above): each routed state
    # self-normalizes before the read-combine, instead of one joint partition fn.
    if name == 'rola-rla-asym-ps':
        return dict(kernel='rla', phi='elu', tie_routers=False,
                    kernel_kwargs={'state_norm': 'per_state'}, **common)
    if name == 'rola-rla-sym-ps':
        return dict(kernel='rla', phi='elu', tie_routers=True,
                    kernel_kwargs={'state_norm': 'per_state'}, **common)
    # KAPPA normalization: learned input-dependent interpolation global↔per-state via
    # r̃ = r·(d+ε)^{-κ(x)} (κ per head, init ≈ global). The per-token recall↔aggregate gate.
    # (κ, not α: the paper uses α for the read routing.)
    if name == 'rola-rla-kappa-asym':
        return dict(kernel='rla', phi='elu', tie_routers=False,
                    kernel_kwargs={'state_norm': 'kappa'}, **common)
    if name == 'rola-rla-kappa-sym':
        return dict(kernel='rla', phi='elu', tie_routers=True,
                    kernel_kwargs={'state_norm': 'kappa'}, **common)
    # asym with tied INIT (read==write at start, untied training) — global norm, fused.
    if name == 'rola-rla-asym-tieinit':
        return dict(kernel='rla', phi='elu', tie_routers=False, tie_router_init=True, **common)
    if name in ('rola-hedgehog-sym', 'rola-hedgehog-asym'):
        # Hedgehog softmax-mimic feature map (feat_dim 2*(d_qk//2)=d_qk, state matched).
        return dict(kernel='rla', phi='hedgehog', tie_routers=(name.endswith('-sym')), **common)
    if name in ('rola-based-sym', 'rola-based-asym'):
        # Based / Taylor [1,x,x⊗x] (Arora 2024); feat_dim = 1+2d+d(d-1)/2 governs state.
        return dict(kernel='rla', phi='based', tie_routers=(name.endswith('-sym')), **common)
    if name in ('rola-rebased-sym', 'rola-rebased-asym'):
        # ReBased learnable affine+LN then x^2 (Aksenov 2024); feat_dim = d(d+1)/2.
        return dict(kernel='rla', phi='rebased', tie_routers=(name.endswith('-sym')), **common)
    if name.startswith('rola-gla-scalar'):
        # OPTIMIZED RoLA-GLA: per-state SCALAR forget gate -> shared-gram fused path
        # (ScalarGLAKernel / _rola_gla_chunked), no virtual-head blowup. Axes encoded in name:
        # '-norm-' => global V+1 partition fn (else un-normalized GLA convention);
        # '-sym' => tied routers, '-asym' => untied.
        return dict(kernel='gla_scalar', tie_routers=name.endswith('-sym'),
                    kernel_kwargs={'normalized': '-norm-' in name}, **common)
    if name.startswith('rola-gla-kappa'):
        # KAPPA on the scalar-gated GLA cell: read gates rescaled by the DECAYED per-state
        # mass (fork GLA den kernel), then the normalized global combine. '-sym' tied routers.
        return dict(kernel='gla_scalar', tie_routers=name.endswith('-sym'),
                    kernel_kwargs={'state_norm': 'kappa'}, **common)
    if name.startswith('rola-gla-ps'):
        # Per-state norm on the scalar-gated GLA cell (kappa≡1 endpoint, same den path).
        return dict(kernel='gla_scalar', tie_routers=name.endswith('-sym'),
                    kernel_kwargs={'state_norm': 'per_state'}, **common)
    if name in ('rola-gla-sym', 'rola-gla-norm-sym'):
        # Per-channel (vector) GLA via virtual heads (VirtualHeadGLAKernel). The decay
        # is entangled in the content contraction -> no shared-gram form (the paper's
        # "RoLA+GLA needs virtual heads, doesn't scale" point). '-sym' un-normalized
        # (GLA convention); '-norm-sym' global V+1 partition fn.
        return dict(kernel='gla_virtual', tie_routers=True,
                    kernel_kwargs={'normalized': name.endswith('-norm-sym')}, **common)
    if name == 'rola-gdn-sym':
        # GDN delta-rule per state via virtual heads (GDNKernel); scalar gate + beta
        # routed per state. Un-normalized. Delta rule doesn't factor -> virtual heads only.
        return dict(kernel='gdn', tie_routers=True, **common)
    # NOTE: SSE is deliberately NOT a RoLA instance (row-sparse softmax classification
    # into a state codebook — a different mechanism). Cite the SSE paper (arXiv
    # 2507.16577) directly; do not reintroduce a 'rola-sse' instance.
    raise ValueError(f"unknown RoLA instance: {name!r}")


ROLA_INSTANCES = ('rola-rla-asym', 'rola-rla-sym', 'rola-rla-asym-ps', 'rola-rla-sym-ps',
                  'rola-rla-kappa-asym', 'rola-rla-kappa-sym', 'rola-rla-asym-tieinit',
                  'rola-gla-sym', 'rola-gla-norm-sym',                  # per-channel GLA (virtual heads)
                  'rola-gla-scalar-sym', 'rola-gla-scalar-norm-sym',    # optimized scalar GLA (shared-gram)
                  'rola-gla-scalar-asym', 'rola-gla-scalar-norm-asym',  # asym variants
                  'rola-gdn-sym', 'rola-hedgehog-sym', 'rola-hedgehog-asym',
                  'rola-based-sym', 'rola-based-asym', 'rola-rebased-sym', 'rola-rebased-asym')


# Backward-compat alias: the model is RoLA now, but the zoology mixer wrapper and
# existing config strings still reference the legacy name. New code should use RoLA.
ChunkedLinearAttention = RoLA

