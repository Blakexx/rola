"""First-class RoLA kernel — implementations + correctness/bench harness.

RoLA-RLA (GLOBAL normalization, the paper's Eq. rola_output) =
    O_i = num_i / den_i,
    num_i = sum_{s<=t} (q_t . k_s) (r_t . w_s) v_s,
    den_i = sum_{s<=t} (q_t . k_s) (r_t . w_s),
with q,k POST-feature-map (elu+1, positive) and r,w the read/write routing
distributions (softmax; possibly top-k sparse). A single global partition
function per query, summed over the routed states (top-k zeros the rest, so the
denominator ranges only over routed states for free).

Equivalently linear attention on Kronecker features Q'=q⊗r, K'=k⊗w with V
augmented by a ones-column (the denominator), then a single divide. This is
what makes the first-class kernel clean: it emits the final fused output
[B,L,H,d_v] directly — no per-state materialization, no separate reader pass.

Implementations (all must match `ref_global` forward + every input grad):
  ref_global       : O(L^2) global-norm reference (ground truth).
  virtual_head_global: the EXISTING replicate-to-nc-heads path, global-norm
                       variant (weight per-state [num,den] by read gate, SUM
                       over states, THEN divide). What cla_bench should compute.
  kron_grouped_global: BACKSTOP — Kronecker features in state-groups <= FLA
                       head-dim limit, FLA per group, summed; agnostic to inner
                       kernel (swap the FLA call). Bounded memory, any nc.
  chunked_shared   : the ALGORITHM the Triton kernel mirrors — shared content
                       gram computed ONCE per chunk, routing gram per chunk,
                       Kronecker state carried inter-chunk. Pure PyTorch, so
                       autograd validates fwd+grads before porting to Triton.
  triton_rola      : MILESTONE 2 (TODO) — custom Triton of chunked_shared.

SUCCESS CRITERION: for random inputs across (B,L,dqk,dv,nc), dense AND top-k
gates, forward AND every input grad (dq,dk,dv,dwg,drg) match ref_global within
atol/rtol ~1e-2 (fp reorder noise on these kernels is ~1e-2).
"""
import os
import torch
import torch.nn.functional as F
from fla.ops.linear_attn import fused_chunk_linear_attn
from fla.ops.gla import chunk_gla as _fla_chunk_gla

FLA_HEADDIM_LIMIT = 256  # conservative; group so d_qk*group_states <= this
EPS = 1e-5               # matches cla_bench's den + 1e-5


def _kron(a, b):
    # a:[...,da], b:[...,db] -> [...,da*db]  (outer product flattened)
    return (a.unsqueeze(-1) * b.unsqueeze(-2)).flatten(-2)


# ----------------------------------------------------------------------------
# Reference (ground truth) + the two existing/backstop paths, all GLOBAL-norm.
# ----------------------------------------------------------------------------
def ref_global(q, k, v, wg, rg, eps=EPS):
    """O(L^2) global-norm reference. q,k:[B,L,dqk] v:[B,L,dv] wg,rg:[B,L,nc]."""
    L = q.shape[1]
    G = torch.einsum('btd,bsd->bts', q, k)          # content gram (shared)
    R = torch.einsum('btc,bsc->bts', rg, wg)         # routing gram
    causal = torch.tril(torch.ones(L, L, device=q.device, dtype=q.dtype))
    W = G * R * causal
    num = torch.einsum('bts,bsv->btv', W, v)
    den = W.sum(-1, keepdim=True)
    return num / (den + eps)


def virtual_head_global(q, k, v, wg, rg, eps=EPS):
    """Existing path, global-norm: nc heads via FLA, per-state [num,den], weight
    by read gate, SUM over states, THEN divide. This is what cla_bench's
    writer+reader should compute under global normalization."""
    nc = wg.shape[-1]
    qv = q.unsqueeze(2).expand(-1, -1, nc, -1).contiguous()
    kv = k.unsqueeze(2).expand(-1, -1, nc, -1).contiguous()
    v_aug = torch.cat([v, torch.ones_like(v[..., :1])], dim=-1)        # [B,L,dv+1]
    vv = (v_aug.unsqueeze(2) * wg.unsqueeze(-1)).contiguous()          # [B,L,nc,dv+1]
    o, _ = fused_chunk_linear_attn(qv, kv, vv, normalize=False, scale=1.0)  # [B,L,nc,dv+1]
    o = (o * rg.unsqueeze(-1)).sum(2)                                  # global combine
    num, den = o[..., :-1], o[..., -1:]
    return num / (den + eps)


def kron_grouped_global(q, k, v, wg, rg, eps=EPS, group_states=None):
    """BACKSTOP: Kronecker features in state-groups <= FLA head-dim limit, summed.
    sum over groups of (q.k)(r_g.w_g) = (q.k)(r.w), exact. v_aug shared (the
    ones-column gives the global denominator). Agnostic: swap the FLA call for
    chunk_gla / chunk_gated_delta_rule to route GLA / GDN."""
    B, L, dqk = q.shape
    nc = wg.shape[-1]
    g = group_states or max(1, FLA_HEADDIM_LIMIT // dqk)
    v_aug = torch.cat([v, torch.ones_like(v[..., :1])], dim=-1)
    out = None
    for s in range(0, nc, g):
        Q = _kron(q, rg[..., s:s + g]).unsqueeze(2)    # [B,L,1,dqk*gsize]
        K = _kron(k, wg[..., s:s + g]).unsqueeze(2)
        o, _ = fused_chunk_linear_attn(Q, K, v_aug.unsqueeze(2), normalize=False, scale=1.0)
        o = o.squeeze(2)
        out = o if out is None else out + o
    num, den = out[..., :-1], out[..., -1:]
    return num / (den + eps)


# ----------------------------------------------------------------------------
# The chunked shared-gram ALGORITHM (PyTorch) — what Triton will mirror.
# ----------------------------------------------------------------------------
def chunked_shared(q, k, v, wg, rg, eps=EPS, chunk=64):
    """Chunked, shared content gram. Per chunk: G=q_c k_c^T computed ONCE (the
    FLOP win), routing gram R=rg_c wg_c^T, intra = (G∘R∘causal)@[v,1]; inter =
    read the carried Kronecker state S[dqk,nc,dv+1]. Never materializes L×nc.
    Pure torch ops so autograd gives exact grads to validate against Triton."""
    B, L, dqk = q.shape
    dv = v.shape[-1]
    nc = wg.shape[-1]
    v1 = torch.cat([v, torch.ones_like(v[..., :1])], dim=-1)   # [B,L,dv+1]
    dvp = dv + 1
    S = q.new_zeros(B, dqk, nc, dvp)                           # Kronecker state
    outs = []
    for c0 in range(0, L, chunk):
        c1 = min(c0 + chunk, L)
        qc, kc, vc = q[:, c0:c1], k[:, c0:c1], v1[:, c0:c1]    # [B,Cc,*]
        rgc, wgc = rg[:, c0:c1], wg[:, c0:c1]                  # [B,Cc,nc]
        Cc = c1 - c0
        # intra-chunk: shared content gram, routing gram, causal mask
        G = torch.einsum('bid,bjd->bij', qc, kc)              # [B,Cc,Cc] SHARED
        R = torch.einsum('bic,bjc->bij', rgc, wgc)            # [B,Cc,Cc]
        causal = torch.tril(torch.ones(Cc, Cc, device=q.device, dtype=q.dtype))
        A = G * R * causal
        o_intra = torch.einsum('bij,bjv->biv', A, vc)         # [B,Cc,dv+1]
        # inter-chunk: read past state via Kronecker contraction
        #   o_inter_i = sum_{d,b} qc_i[d] rgc_i[b] S[d,b,:]
        M = torch.einsum('bic,bdcv->bidv', rgc, S)            # contract nc
        o_inter = torch.einsum('bid,bidv->biv', qc, M)        # contract dqk
        outs.append(o_intra + o_inter)
        # state update: S[d,b,v] += sum_j kc_j[d] wgc_j[b] vc_j[v]
        S = S + torch.einsum('bjd,bjc,bjv->bdcv', kc, wgc, vc)
    O = torch.cat(outs, dim=1)                                # [B,L,dv+1]
    num, den = O[..., :-1], O[..., -1:]
    return num / (den + eps)


def chunked_parallel(q, k, v, wg, rg, eps=EPS, chunk=64):
    """Chunk-PARALLEL pure torch: same algorithm as chunked_shared but the chunk
    loop is vectorized (grams batched over chunks; inter-chunk recurrence is an
    exclusive cumsum of per-chunk Kronecker states). No Python loop → cuBLAS-fast,
    and memory stays bounded (materializes [B,nch,nc,dqk,dv+1] chunk states, BT×
    smaller in L than the virtual-head L×nc replication, and no q/k replication)."""
    B, L, dqk = q.shape
    dv = v.shape[-1]
    nc = wg.shape[-1]
    pad = (-L) % chunk
    if pad:
        q = F.pad(q, (0, 0, 0, pad)); k = F.pad(k, (0, 0, 0, pad)); v = F.pad(v, (0, 0, 0, pad))
        wg = F.pad(wg, (0, 0, 0, pad)); rg = F.pad(rg, (0, 0, 0, pad))
    Lp = L + pad
    nch = Lp // chunk
    v1 = torch.cat([v, torch.ones_like(v[..., :1])], dim=-1)
    qc = q.view(B, nch, chunk, dqk); kc = k.view(B, nch, chunk, dqk); vc = v1.view(B, nch, chunk, dv + 1)
    rgc = rg.view(B, nch, chunk, nc); wgc = wg.view(B, nch, chunk, nc)
    # intra-chunk (shared content gram, batched over chunks)
    G = torch.einsum('bnid,bnjd->bnij', qc, kc)
    R = torch.einsum('bnic,bnjc->bnij', rgc, wgc)
    causal = torch.tril(torch.ones(chunk, chunk, device=q.device, dtype=q.dtype))
    A = G * R * causal
    o_intra = torch.einsum('bnij,bnjv->bniv', A, vc)
    # inter-chunk: per-chunk Kronecker states, exclusive prefix-sum over chunks
    KV = torch.einsum('bnjc,bnjd,bnjv->bncdv', wgc, kc, vc)   # [B,nch,nc,dqk,dv+1]
    S_before = torch.cumsum(KV, dim=1) - KV                   # exclusive prefix
    M = torch.einsum('bnic,bncdv->bnidv', rgc, S_before)      # contract nc
    o_inter = torch.einsum('bnid,bnidv->bniv', qc, M)         # contract dqk
    O = (o_intra + o_inter).reshape(B, Lp, dv + 1)[:, :L]
    num, den = O[..., :-1], O[..., -1:]
    return num / (den + eps)


# ============================================================================
# Scalar-gated RoLA-GLA (the OPTIMIZED gated variant).
#
# Per-state SCALAR forget gate -> cumulative log-decay A^c_t factors OUT of the
# content contraction, so the effective weight keeps the shared-gram form:
#   W_tj = (q_t.k_j) * sum_c r_t^c w_j^c exp(A^c_t - A^c_j)
#        = (q_t.k_j) * sum_c (r_t^c e^{A^c_t})(w_j^c e^{-A^c_j})  =  G_tj (r~_t . w~_j).
# i.e. ADDITIVE RoLA with decay-absorbed routing gates. The chunked path mirrors
# chunked_shared but (a) absorbs decay chunk-LOCALLY (bounded exponents) and
# (b) carries the Kronecker state across chunks with a per-chunk decay (a decayed
# scan, sequential over the few chunks). Inputs: ld[B,L,nc] = log of the per-state
# decay alpha_chunk = 1 - w*(1-alpha_base). Recurrence (per state, matches FLA
# chunk_gla + cla_bench's writer): S_t = alpha_t S_{t-1} + w_t (k_t (x) v1_t),
# o_t = sum_c r_t^c (q_t . S_t^c), global-normalized via v1's ones-column.
# ============================================================================
def ref_global_gla(q, k, v, wg, rg, ld, eps=EPS):
    """O(L^2) ground truth for scalar-gated RoLA-GLA. ld:[B,L,nc] log-decay/state."""
    L = q.shape[1]
    A = torch.cumsum(ld, dim=1)                          # [B,L,nc] cumulative log-decay
    G = torch.einsum('btd,bsd->bts', q, k)               # content gram (shared)
    decay = torch.exp(A[:, :, None, :] - A[:, None, :, :])   # [B,L(t),L(j),nc] exp(A_t-A_j)
    D = torch.einsum('btc,bsc,btsc->bts', rg, wg, decay)     # decayed routing gram
    causal = torch.tril(torch.ones(L, L, device=q.device, dtype=q.dtype))
    W = G * D * causal
    v1 = torch.cat([v, torch.ones_like(v[..., :1])], dim=-1)
    O = torch.einsum('bts,bsv->btv', W, v1)
    num, den = O[..., :-1], O[..., -1:]
    return num / (den + eps)


def recurrent_gla(q, k, v, wg, rg, ld, eps=EPS):
    """Independent O(L) sequential check (no chunking). Definitely correct."""
    B, L, dqk = q.shape; dv = v.shape[-1]; nc = wg.shape[-1]
    v1 = torch.cat([v, torch.ones_like(v[..., :1])], dim=-1)   # [B,L,dv+1]
    S = q.new_zeros(B, nc, dqk, dv + 1)
    outs = []
    for t in range(L):
        alpha = torch.exp(ld[:, t])                       # [B,nc]
        write = wg[:, t][:, :, None, None] * (k[:, t][:, None, :, None] * v1[:, t][:, None, None, :])
        S = alpha[:, :, None, None] * S + write           # [B,nc,dqk,dv+1]
        o_c = torch.einsum('bd,bcdv->bcv', q[:, t], S)    # [B,nc,dv+1]
        O_t = torch.einsum('bc,bcv->bv', rg[:, t], o_c)   # [B,dv+1]
        outs.append(O_t)
    O = torch.stack(outs, dim=1)
    num, den = O[..., :-1], O[..., -1:]
    return num / (den + eps)


def chunked_gla(q, k, v, wg, rg, ld, eps=EPS, chunk=64):
    """Shared-gram chunked scalar-gated RoLA-GLA. Pure torch (autograd-friendly).
    Decay absorbed chunk-locally; Kronecker state carried across chunks (decayed
    scan, sequential over the few chunks). Mirrors chunked_shared + decay."""
    B, L, dqk = q.shape; dv = v.shape[-1]; nc = wg.shape[-1]
    dvp = dv + 1
    v1 = torch.cat([v, torch.ones_like(v[..., :1])], dim=-1)
    S = q.new_zeros(B, nc, dqk, dvp)                      # carried Kronecker state
    outs = []
    for c0 in range(0, L, chunk):
        c1 = min(c0 + chunk, L); Cc = c1 - c0
        qc, kc, vc = q[:, c0:c1], k[:, c0:c1], v1[:, c0:c1]
        rgc, wgc, ldc = rg[:, c0:c1], wg[:, c0:c1], ld[:, c0:c1]    # [B,Cc,nc]
        a = torch.cumsum(ldc, dim=1)                      # chunk-local cumulative log-decay [B,Cc,nc]
        rt = rgc * torch.exp(a)                           # r~ = r e^{a}   [B,Cc,nc]
        wt = wgc * torch.exp(-a)                          # w~ = w e^{-a}
        # intra: shared content gram x decayed routing gram, causal
        G = torch.einsum('bid,bjd->bij', qc, kc)          # [B,Cc,Cc]
        D = torch.einsum('bic,bjc->bij', rt, wt)          # [B,Cc,Cc] decayed routing
        causal = torch.tril(torch.ones(Cc, Cc, device=q.device, dtype=q.dtype))
        A = G * D * causal
        o_intra = torch.einsum('bij,bjv->biv', A, vc)     # [B,Cc,dvp]
        # inter: read carried state, decayed to position i via e^{a_i}, gated by r
        M = torch.einsum('bic,bcdv->bidv', rt, S)         # contract nc (rt already has e^{a_i})
        o_inter = torch.einsum('bid,bidv->biv', qc, M)
        outs.append(o_intra + o_inter)
        # state update: decay carried state by chunk-total Lambda, add this chunk's
        # writes referenced to chunk END: KV = sum_j (w_j e^{Lambda - a_j}) k_j (x) v_j
        Lam = a[:, -1, :]                                 # [B,nc] chunk-total log-decay
        w_end = wgc * torch.exp(Lam[:, None, :] - a)      # [B,Cc,nc]
        KV = torch.einsum('bjc,bjd,bjv->bcdv', w_end, kc, vc)     # [B,nc,dqk,dvp]
        S = torch.exp(Lam)[:, :, None, None] * S + KV
    O = torch.cat(outs, dim=1)
    num, den = O[..., :-1], O[..., -1:]
    return num / (den + eps)


# ----------------------------------------------------------------------------
# Triton kernel (MILESTONE 2) — custom fused forward mirroring chunked_shared.
# One program per (batch); state Sflat[BD, BNC*BV] (the [dqk,nc,dv+1] Kronecker
# state, padded) is carried in SRAM across the chunk loop. Shares the content
# gram (G computed once per chunk, not per-state). v1: requires nc that fits SRAM.
# ----------------------------------------------------------------------------
import triton
import triton.language as tl


@triton.jit
def _rola_fwd_kernel(q_ptr, k_ptr, v_ptr, wg_ptr, rg_ptr, out_ptr,
                     L, dqk, dv, nc, eps,
                     sq_b, sq_l, sq_d, sv_b, sv_l, sv_d, sg_b, sg_l, sg_c,
                     so_b, so_l, so_d,
                     BT: tl.constexpr, BD: tl.constexpr, BV: tl.constexpr,
                     BNC: tl.constexpr, NCH: tl.constexpr):
    b = tl.program_id(0)
    dvp = dv + 1
    offs_t = tl.arange(0, BT)
    offs_d = tl.arange(0, BD)
    offs_v = tl.arange(0, BV)
    offs_c = tl.arange(0, BNC)
    dmask = offs_d < dqk
    cmask = offs_c < nc
    Sflat = tl.zeros([BD, BNC * BV], dtype=tl.float32)        # carried state
    for t in range(NCH):
        rows = t * BT + offs_t
        rmask = rows < L
        # --- loads (q,k:[BT,BD]; v_aug:[BT,BV]; rg,wg:[BT,BNC]) ---
        qc = tl.load(q_ptr + b * sq_b + rows[:, None] * sq_l + offs_d[None, :] * sq_d,
                     mask=rmask[:, None] & dmask[None, :], other=0.0)
        kc = tl.load(k_ptr + b * sq_b + rows[:, None] * sq_l + offs_d[None, :] * sq_d,
                     mask=rmask[:, None] & dmask[None, :], other=0.0)
        vc = tl.load(v_ptr + b * sv_b + rows[:, None] * sv_l + offs_v[None, :] * sv_d,
                     mask=rmask[:, None] & (offs_v[None, :] < dv), other=0.0)
        vc += tl.where((offs_v[None, :] == dv) & rmask[:, None], 1.0, 0.0)  # ones-col
        rgc = tl.load(rg_ptr + b * sg_b + rows[:, None] * sg_l + offs_c[None, :] * sg_c,
                      mask=rmask[:, None] & cmask[None, :], other=0.0)
        wgc = tl.load(wg_ptr + b * sg_b + rows[:, None] * sg_l + offs_c[None, :] * sg_c,
                      mask=rmask[:, None] & cmask[None, :], other=0.0)
        # --- intra: shared content gram G, routing gram R, causal ---
        G = tl.dot(qc, tl.trans(kc))                          # [BT,BT]
        R = tl.dot(rgc, tl.trans(wgc))                        # [BT,BT]
        causal = (offs_t[:, None] >= offs_t[None, :]) & rmask[:, None] & rmask[None, :]
        A = G * R * causal
        o_intra = tl.dot(A.to(vc.dtype), vc)                  # [BT,BV]
        # --- inter: read carried state, weight by read gate, sum over states ---
        P = tl.dot(qc, Sflat.to(qc.dtype))                    # [BT, BNC*BV]
        P3 = tl.reshape(P, [BT, BNC, BV])
        o_inter = tl.sum(P3 * rgc[:, :, None], axis=1)        # [BT,BV]
        o = o_intra + o_inter                                 # [BT,BV]: [num.., den, 0..]
        den = tl.sum(tl.where(offs_v[None, :] == dv, o, 0.0), axis=1)   # [BT] partition fn
        outv = o / (den[:, None] + eps)                       # divide all cols; store <dv
        tl.store(out_ptr + b * so_b + rows[:, None] * so_l + offs_v[None, :] * so_d,
                 outv, mask=rmask[:, None] & (offs_v[None, :] < dv))
        # --- state update: Sflat += kc^T @ kron(wgc, vc) ---
        WV = tl.reshape(wgc[:, :, None] * vc[:, None, :], [BT, BNC * BV])
        Sflat += tl.dot(tl.trans(kc), WV.to(kc.dtype))        # [BD, BNC*BV]


# Autotune over launch params (num_warps/num_stages) — closes the occupancy/scheduling gap to FLA's
# hand-tuned kernels (the reason FLA-vh used to win at low occupancy). Tile sizes (BT/BD/BV/BG) stay
# wrapper-set; autotune only varies launch config and prunes any that overflow SRAM. Keyed on shape.
_AT_CFGS = [triton.Config({}, num_warps=w, num_stages=s) for w in (2, 4, 8) for s in (1, 2, 3)]
_AT_KEY = ['dqk', 'dv', 'nc']


@triton.autotune(configs=_AT_CFGS, key=_AT_KEY)
@triton.jit
def _rola_fwd_tiled(q_ptr, k_ptr, v_ptr, wg_ptr, rg_ptr, outa_ptr,
                    L, dqk, dv, nc,
                    sq_b, sq_l, sq_d, sv_b, sv_l, sv_d, sg_b, sg_l, sg_c,
                    soa_b, soa_n, soa_l, soa_v,
                    BT: tl.constexpr, BD: tl.constexpr, BV: tl.constexpr,
                    BG: tl.constexpr, NCH: tl.constexpr):
    """One program per (batch, STATE-BLOCK of BG states). Carries only this block's slice
    of the Kronecker state [BD, BG*BV] in SRAM → scales to any nc. Emits the AUGMENTED
    partial output [.., dv|den] (no divide); the wrapper sums blocks then divides once
    (exact: num & den are linear sums over states)."""
    b = tl.program_id(0)
    sb = tl.program_id(1)
    dvp = dv + 1
    offs_t = tl.arange(0, BT); offs_d = tl.arange(0, BD); offs_v = tl.arange(0, BV)
    offs_c = sb * BG + tl.arange(0, BG)
    dmask = offs_d < dqk; cmask = offs_c < nc
    Sflat = tl.zeros([BD, BG * BV], dtype=tl.float32)
    for t in range(NCH):
        rows = t * BT + offs_t; rmask = rows < L
        qc = tl.load(q_ptr + b * sq_b + rows[:, None] * sq_l + offs_d[None, :] * sq_d,
                     mask=rmask[:, None] & dmask[None, :], other=0.0)
        kc = tl.load(k_ptr + b * sq_b + rows[:, None] * sq_l + offs_d[None, :] * sq_d,
                     mask=rmask[:, None] & dmask[None, :], other=0.0)
        vc = tl.load(v_ptr + b * sv_b + rows[:, None] * sv_l + offs_v[None, :] * sv_d,
                     mask=rmask[:, None] & (offs_v[None, :] < dv), other=0.0)
        vc += tl.where((offs_v[None, :] == dv) & rmask[:, None], 1.0, 0.0)
        rgc = tl.load(rg_ptr + b * sg_b + rows[:, None] * sg_l + offs_c[None, :] * sg_c,
                      mask=rmask[:, None] & cmask[None, :], other=0.0)
        wgc = tl.load(wg_ptr + b * sg_b + rows[:, None] * sg_l + offs_c[None, :] * sg_c,
                      mask=rmask[:, None] & cmask[None, :], other=0.0)
        G = tl.dot(qc, tl.trans(kc))
        R = tl.dot(rgc, tl.trans(wgc))                        # routing gram over THIS block
        causal = (offs_t[:, None] >= offs_t[None, :]) & rmask[:, None] & rmask[None, :]
        A = G * R * causal
        o_intra = tl.dot(A.to(vc.dtype), vc)
        P = tl.dot(qc, Sflat.to(qc.dtype))
        P3 = tl.reshape(P, [BT, BG, BV])
        o_inter = tl.sum(P3 * rgc[:, :, None], axis=1)
        o = o_intra + o_inter                                 # augmented partial (NOT divided)
        tl.store(outa_ptr + b * soa_b + sb * soa_n + rows[:, None] * soa_l + offs_v[None, :] * soa_v,
                 o, mask=rmask[:, None] & (offs_v[None, :] < dvp))
        WV = tl.reshape(wgc[:, :, None] * vc[:, None, :], [BT, BG * BV])
        Sflat += tl.dot(tl.trans(kc), WV.to(kc.dtype))


@triton.autotune(configs=_AT_CFGS, key=_AT_KEY)
@triton.jit
def _rola_bwd_qr(q_ptr, k_ptr, v_ptr, wg_ptr, rg_ptr, g_ptr, dq_ptr, dr_ptr,
                L, dqk, dv, nc,
                sq_b, sq_l, sq_d, sv_b, sv_l, sv_d, sg_b, sg_l, sg_c, sgr_b, sgr_l, sgr_d,
                sdq_b, sdq_n, sdq_l, sdq_d, sdr_b, sdr_l, sdr_c,
                BT: tl.constexpr, BD: tl.constexpr, BV: tl.constexpr, BG: tl.constexpr, NCH: tl.constexpr):
    """Backward (dq, dr) via FORWARD state scan. g = grad of the AUGMENTED output [.., dv|den].
    Mirrors the forward kernel's chunk/state loop. dq accumulates over state-blocks (partial,
    summed by the wrapper); dr is per-state (this block writes its slice)."""
    b = tl.program_id(0); sb = tl.program_id(1); dvp = dv + 1
    offs_t = tl.arange(0, BT); offs_d = tl.arange(0, BD); offs_v = tl.arange(0, BV)
    offs_c = sb * BG + tl.arange(0, BG)
    dmask = offs_d < dqk; cmask = offs_c < nc; vmask = offs_v < dvp
    Sflat = tl.zeros([BD, BG * BV], dtype=tl.float32)
    for t in range(NCH):
        rows = t * BT + offs_t; rmask = rows < L
        qc = tl.load(q_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d, mask=rmask[:, None] & dmask[None, :], other=0.0)
        kc = tl.load(k_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d, mask=rmask[:, None] & dmask[None, :], other=0.0)
        v1 = tl.load(v_ptr + b*sv_b + rows[:, None]*sv_l + offs_v[None, :]*sv_d, mask=rmask[:, None] & (offs_v[None, :] < dv), other=0.0)
        v1 += tl.where((offs_v[None, :] == dv) & rmask[:, None], 1.0, 0.0)
        rgc = tl.load(rg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0)
        wgc = tl.load(wg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0)
        gc = tl.load(g_ptr + b*sgr_b + rows[:, None]*sgr_l + offs_v[None, :]*sgr_d, mask=rmask[:, None] & vmask[None, :], other=0.0)
        causal = (offs_t[:, None] >= offs_t[None, :]) & rmask[:, None] & rmask[None, :]
        G = tl.dot(qc, tl.trans(kc)); Rg = tl.dot(rgc, tl.trans(wgc)); P = tl.dot(gc, tl.trans(v1))
        dq_intra = tl.dot((causal * Rg * P).to(kc.dtype), kc)
        dr_intra = tl.dot((causal * G * P).to(wgc.dtype), wgc)
        rg_g = tl.reshape(rgc[:, :, None] * gc[:, None, :], [BT, BG * BV])
        dq_inter = tl.dot(rg_g.to(Sflat.dtype), tl.trans(Sflat))
        QS = tl.dot(qc, Sflat.to(qc.dtype))
        dr_inter = tl.sum(tl.reshape(QS, [BT, BG, BV]) * gc[:, None, :], axis=2)
        tl.store(dq_ptr + b*sdq_b + sb*sdq_n + rows[:, None]*sdq_l + offs_d[None, :]*sdq_d,
                 dq_intra + dq_inter, mask=rmask[:, None] & dmask[None, :])
        tl.store(dr_ptr + b*sdr_b + rows[:, None]*sdr_l + offs_c[None, :]*sdr_c,
                 dr_intra + dr_inter, mask=rmask[:, None] & cmask[None, :])
        WV = tl.reshape(wgc[:, :, None] * v1[:, None, :], [BT, BG * BV])
        Sflat += tl.dot(tl.trans(kc), WV.to(kc.dtype))


@triton.autotune(configs=_AT_CFGS, key=_AT_KEY)
@triton.jit
def _rola_bwd_kwv(q_ptr, k_ptr, v_ptr, wg_ptr, rg_ptr, g_ptr, dk_ptr, dw_ptr, dv_ptr,
                 L, dqk, dv, nc,
                 sq_b, sq_l, sq_d, sv_b, sv_l, sv_d, sg_b, sg_l, sg_c, sgr_b, sgr_l, sgr_d,
                 sdk_b, sdk_n, sdk_l, sdk_d, sdw_b, sdw_l, sdw_c, sdv_b, sdv_n, sdv_l, sdv_d,
                 BT: tl.constexpr, BD: tl.constexpr, BV: tl.constexpr, BG: tl.constexpr, NCH: tl.constexpr):
    """Backward (dk, dw, dv) via REVERSE state scan (right-to-left), carrying dS_after. dk,dv
    accumulate over state-blocks (partial); dw is per-state. Transposed-causal intra (sum over
    i>=j) + write-grad via dS_after."""
    b = tl.program_id(0); sb = tl.program_id(1); dvp = dv + 1
    offs_t = tl.arange(0, BT); offs_d = tl.arange(0, BD); offs_v = tl.arange(0, BV)
    offs_c = sb * BG + tl.arange(0, BG)
    dmask = offs_d < dqk; cmask = offs_c < nc; vmask = offs_v < dvp
    dS = tl.zeros([BD, BG * BV], dtype=tl.float32)
    for ti in range(NCH):
        t = NCH - 1 - ti
        rows = t * BT + offs_t; rmask = rows < L
        qc = tl.load(q_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d, mask=rmask[:, None] & dmask[None, :], other=0.0)
        kc = tl.load(k_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d, mask=rmask[:, None] & dmask[None, :], other=0.0)
        v1 = tl.load(v_ptr + b*sv_b + rows[:, None]*sv_l + offs_v[None, :]*sv_d, mask=rmask[:, None] & (offs_v[None, :] < dv), other=0.0)
        v1 += tl.where((offs_v[None, :] == dv) & rmask[:, None], 1.0, 0.0)
        rgc = tl.load(rg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0)
        wgc = tl.load(wg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0)
        gc = tl.load(g_ptr + b*sgr_b + rows[:, None]*sgr_l + offs_v[None, :]*sgr_d, mask=rmask[:, None] & vmask[None, :], other=0.0)
        causal = (offs_t[:, None] >= offs_t[None, :]) & rmask[:, None] & rmask[None, :]
        G = tl.dot(qc, tl.trans(kc)); Rg = tl.dot(rgc, tl.trans(wgc)); P = tl.dot(gc, tl.trans(v1))
        A = (G * Rg * causal)            # forward weight (for dv_intra)
        A2 = (Rg * P * causal); B2 = (G * P * causal)
        dk_intra = tl.dot(tl.trans(A2).to(qc.dtype), qc)        # sum_{i>=j} A2_ij q_i
        dw_intra = tl.dot(tl.trans(B2).to(rgc.dtype), rgc)
        dv_intra = tl.dot(tl.trans(A).to(gc.dtype), gc)         # sum_{i>=j} A_ij g_i
        # write-grad via dS_after
        wg_v1 = tl.reshape(wgc[:, :, None] * v1[:, None, :], [BT, BG * BV])
        dk_wr = tl.dot(wg_v1.to(dS.dtype), tl.trans(dS))
        KS = tl.dot(kc, dS.to(kc.dtype))                        # [BT, BG*BV]
        KS3 = tl.reshape(KS, [BT, BG, BV])
        dw_wr = tl.sum(KS3 * v1[:, None, :], axis=2)
        dv_wr = tl.sum(wgc[:, :, None] * KS3, axis=1)
        tl.store(dk_ptr + b*sdk_b + sb*sdk_n + rows[:, None]*sdk_l + offs_d[None, :]*sdk_d,
                 dk_intra + dk_wr, mask=rmask[:, None] & dmask[None, :])
        tl.store(dw_ptr + b*sdw_b + rows[:, None]*sdw_l + offs_c[None, :]*sdw_c,
                 dw_intra + dw_wr, mask=rmask[:, None] & cmask[None, :])
        tl.store(dv_ptr + b*sdv_b + sb*sdv_n + rows[:, None]*sdv_l + offs_v[None, :]*sdv_d,
                 dv_intra + dv_wr, mask=rmask[:, None] & (offs_v[None, :] < dv))
        rg_g = tl.reshape(rgc[:, :, None] * gc[:, None, :], [BT, BG * BV])
        dS += tl.dot(tl.trans(qc), rg_g.to(qc.dtype))           # exclusive: add AFTER use


def triton_rola_bwd_kwv(q, k, v, wg, rg, g, chunk=64, BG=16):
    B, L, dqk = q.shape; dv = v.shape[-1]; nc = wg.shape[-1]
    BD = max(16, triton.next_power_of_2(dqk)); BV = max(16, triton.next_power_of_2(dv + 1))
    NB = triton.cdiv(nc, BG); NCH = triton.cdiv(L, chunk)
    q, k, v, wg, rg, g = [x.contiguous() for x in (q, k, v, wg, rg, g)]
    dk = torch.zeros(B, NB, L, dqk, device=q.device, dtype=torch.float32)
    dw = torch.zeros(B, L, nc, device=q.device, dtype=torch.float32)
    dvo = torch.zeros(B, NB, L, BV, device=q.device, dtype=torch.float32)
    _rola_bwd_kwv[(B, NB)](q, k, v, wg, rg, g, dk, dw, dvo, L, dqk, dv, nc,
        q.stride(0), q.stride(1), q.stride(2), v.stride(0), v.stride(1), v.stride(2),
        wg.stride(0), wg.stride(1), wg.stride(2), g.stride(0), g.stride(1), g.stride(2),
        dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3), dw.stride(0), dw.stride(1), dw.stride(2),
        dvo.stride(0), dvo.stride(1), dvo.stride(2), dvo.stride(3),
        BT=chunk, BD=BD, BV=BV, BG=BG, NCH=NCH)
    return dk.sum(1), dw, dvo.sum(1)[..., :dv]


def triton_rola_bwd_qr(q, k, v, wg, rg, g, chunk=64, BG=16):
    B, L, dqk = q.shape; dv = v.shape[-1]; nc = wg.shape[-1]
    BD = max(16, triton.next_power_of_2(dqk)); BV = max(16, triton.next_power_of_2(dv + 1))
    NB = triton.cdiv(nc, BG); NCH = triton.cdiv(L, chunk)
    q, k, v, wg, rg, g = [x.contiguous() for x in (q, k, v, wg, rg, g)]
    dq = torch.zeros(B, NB, L, dqk, device=q.device, dtype=torch.float32)
    dr = torch.zeros(B, L, nc, device=q.device, dtype=torch.float32)
    _rola_bwd_qr[(B, NB)](q, k, v, wg, rg, g, dq, dr, L, dqk, dv, nc,
        q.stride(0), q.stride(1), q.stride(2), v.stride(0), v.stride(1), v.stride(2),
        wg.stride(0), wg.stride(1), wg.stride(2), g.stride(0), g.stride(1), g.stride(2),
        dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3), dr.stride(0), dr.stride(1), dr.stride(2),
        BT=chunk, BD=BD, BV=BV, BG=BG, NCH=NCH)
    return dq.sum(1), dr


def _triton_rola_aug(q, k, v, wg, rg, chunk=64, BG=16):
    """Triton forward returning the AUGMENTED output [B,L,dv+1] (num | den), tiled over states."""
    B, L, dqk = q.shape; dv = v.shape[-1]; nc = wg.shape[-1]
    BD = max(16, triton.next_power_of_2(dqk)); BV = max(16, triton.next_power_of_2(dv + 1))
    NB = triton.cdiv(nc, BG); NCH = triton.cdiv(L, chunk)
    q, k, v, wg, rg = [x.contiguous() for x in (q, k, v, wg, rg)]
    out_aug = torch.zeros(B, NB, L, BV, device=q.device, dtype=torch.float32)
    _rola_fwd_tiled[(B, NB)](
        q, k, v, wg, rg, out_aug, L, dqk, dv, nc,
        q.stride(0), q.stride(1), q.stride(2), v.stride(0), v.stride(1), v.stride(2),
        wg.stride(0), wg.stride(1), wg.stride(2),
        out_aug.stride(0), out_aug.stride(1), out_aug.stride(2), out_aug.stride(3),
        BT=chunk, BD=BD, BV=BV, BG=BG, NCH=NCH)
    return out_aug[..., :dv + 1].sum(1)


def triton_rola(q, k, v, wg, rg, eps=EPS, chunk=64, BG=16):
    """Triton forward for global-norm RoLA-RLA, tiled over states (scales to any nc)."""
    Oa = _triton_rola_aug(q, k, v, wg, rg, chunk=chunk, BG=BG)
    dv = v.shape[-1]
    return (Oa[..., :dv] / (Oa[..., dv:dv + 1] + eps)).to(q.dtype)


class _TritonRoLAFn(torch.autograd.Function):
    """Fully-fused Triton RoLA-RLA: Triton forward + a fused Triton BACKWARD (two chunked scans —
    _rola_bwd_qr forward-scan for dq/dr, _rola_bwd_kwv reverse-scan for dk/dw/dv — implementing the
    analytic gradients, verified vs ref_global to fp noise). No torch reference, no L² blowup."""
    @staticmethod
    def forward(ctx, q, k, v, wg, rg, eps, chunk):
        Oa = _triton_rola_aug(q, k, v, wg, rg, chunk=chunk)
        dv = v.shape[-1]; den = Oa[..., dv:dv + 1]
        O = (Oa[..., :dv] / (den + eps)).to(q.dtype)
        ctx.save_for_backward(q, k, v, wg, rg, O, den)
        ctx.eps, ctx.chunk = eps, chunk
        return O

    @staticmethod
    def backward(ctx, dO):
        q, k, v, wg, rg, O, den = ctx.saved_tensors; eps, chunk = ctx.eps, ctx.chunk
        dOf = dO.float()
        g_num = dOf / (den + eps)
        g_den = -(dOf * O.float()).sum(-1, keepdim=True) / (den + eps)
        g = torch.cat([g_num, g_den], dim=-1)                 # augmented-output grad [B,L,dv+1]
        # backward tiles are SRAM-heavier than the forward (7 [BT,BT] grams + [BT,BG,BV]); use
        # smaller chunk/BG so it fits the 100KB shared-memory cap (math is chunk-size-independent).
        qf, kf, vf, wgf, rgf = q.float(), k.float(), v.float(), wg.float(), rg.float()
        dq, dr = triton_rola_bwd_qr(qf, kf, vf, wgf, rgf, g, chunk=16, BG=16)
        dk, dw, dvv = triton_rola_bwd_kwv(qf, kf, vf, wgf, rgf, g, chunk=16, BG=16)
        cast = lambda t: t.to(q.dtype)
        return cast(dq), cast(dk), cast(dvv), cast(dw), cast(dr), None, None


class _TritonRoLAFnTorchBwd(torch.autograd.Function):
    """RLA: Triton forward + autograd-of-chunked_parallel backward (kept as a reference path)."""
    @staticmethod
    def forward(ctx, q, k, v, wg, rg, eps, chunk):
        ctx.save_for_backward(q, k, v, wg, rg)
        ctx.eps, ctx.chunk = eps, chunk
        return triton_rola(q, k, v, wg, rg, eps=eps, chunk=chunk)

    @staticmethod
    def backward(ctx, dO):
        # Backward = autograd of the CHUNK-PARALLEL torch reference (vectorized, no Python loop,
        # O(bh·L·chunk) — not O(bh·L²)). The explicit full-L analytic backward is correct but
        # materializes [bh,L,L] → loses badly to chunk-parallel at model scale (bh=B·H≈192). This
        # is the fastest correct RLA backward; the Triton FORWARD still gives the low-mem/inference
        # win. (A fully-FUSED chunked Triton backward — reverse-state scan in-kernel — would beat
        # even this; that's the remaining kernel work.)
        q, k, v, wg, rg = ctx.saved_tensors
        with torch.enable_grad():
            ins = [t.detach().requires_grad_(True) for t in (q, k, v, wg, rg)]
            o = chunked_parallel(*ins, eps=ctx.eps, chunk=ctx.chunk)
            grads = torch.autograd.grad(o, ins, dO)
        return (*grads, None, None)


def triton_rola_ag(q, k, v, wg, rg, eps=EPS, chunk=64):
    """Autograd-capable Triton RoLA-RLA: Triton fused FORWARD + bespoke fully-fused Triton BACKWARD
    (_TritonRoLAFn: dq/dr fwd-scan + dk/dw/dv reverse-scan, grads analytic from saved tensors, NO
    forward recompute). Verified vs ref_global. (cuBLAS chunk-parallel is faster on the RLA backward
    alone — RLA is matmul-heavy/not kernel-bound — but the single bespoke path avoids the recompute
    and the gating mess; the kernel isn't the layer bottleneck so the difference is immaterial.
    _TritonRoLAFnTorchBwd remains available for the cuBLAS-backward path.)"""
    return _TritonRoLAFn.apply(q, k, v, wg, rg, eps, chunk)


# ----------------------------------------------------------------------------
# Scalar-gated RoLA-GLA Triton (tiled over states, chunk-local decay). Mirrors the
# RLA tiled kernel + the GLA decay machinery: a = cumsum(ld) within chunk; decay-
# absorbed gates rt=rg·e^a, wt=wg·e^-a; decayed routing gram R = rt·wtᵀ; cross-chunk
# state decayed by e^Lam (chunk-total), writes referenced to chunk end. Per-token decay
# is FLOORED so the in-kernel factored exp stays fp32-safe over the BT-row chunk
# (no-op for realistic gates). Backward = autograd of the safe [Cc,Cc,C] torch form.
# ----------------------------------------------------------------------------
_GLA_FLOOR = -2.5   # per-token log-decay floor (retention ≥ 8.2%/tok); fp32-safe for BT≤32


def _chunked_gla_safe(q, k, v, wg, rg, ld, eps=EPS, chunk=32, normalized=True):
    """Safe torch chunked GLA: intra decayed gram via the [Cc,Cc,C] clamp-before-exp form
    (exact + overflow-free in ALL decay regimes), cross-chunk decayed state carry. Autograd-
    able → the exact backward source for the Triton GLA forward. q,k:[B,L,dqk] v:[B,L,dv]."""
    B, L, dqk = q.shape; dv = v.shape[-1]; nc = wg.shape[-1]
    v1 = torch.cat([v, torch.ones_like(v[..., :1])], dim=-1) if normalized else v
    dvp = v1.shape[-1]
    S = q.new_zeros(B, nc, dqk, dvp); outs = []
    for c0 in range(0, L, chunk):
        c1 = min(c0 + chunk, L); Cc = c1 - c0
        qc, kc, vc = q[:, c0:c1], k[:, c0:c1], v1[:, c0:c1]
        rgc, wgc, ldc = rg[:, c0:c1], wg[:, c0:c1], ld[:, c0:c1]
        a = torch.cumsum(ldc, dim=1); Lam = a[:, -1, :]
        G = torch.einsum('bid,bjd->bij', qc, kc)
        dec = torch.exp((a[:, :, None, :] - a[:, None, :, :]).clamp(max=0.0))
        D = torch.einsum('bic,bjc,bijc->bij', rgc, wgc, dec)
        causal = torch.tril(torch.ones(Cc, Cc, device=q.device, dtype=q.dtype))
        o_intra = torch.einsum('bij,bjv->biv', G * D * causal, vc)
        M = torch.einsum('bic,bcdv->bidv', rgc * torch.exp(a), S)
        o_inter = torch.einsum('bid,bidv->biv', qc, M)
        outs.append(o_intra + o_inter)
        w_end = wgc * torch.exp(Lam[:, None, :] - a)
        S = torch.exp(Lam)[:, :, None, None] * S + torch.einsum('bjc,bjd,bjv->bcdv', w_end, kc, vc)
    O = torch.cat(outs, dim=1)
    if not normalized:
        return O
    num, den = O[..., :-1], O[..., -1:]
    return num / (den + eps)


@triton.autotune(configs=_AT_CFGS, key=_AT_KEY)
@triton.jit
def _rola_gla_fwd_tiled(q_ptr, k_ptr, v_ptr, wg_ptr, rg_ptr, ld_ptr, outa_ptr,
                        L, dqk, dv, nc,
                        sq_b, sq_l, sq_d, sv_b, sv_l, sv_d, sg_b, sg_l, sg_c,
                        soa_b, soa_n, soa_l, soa_v,
                        BT: tl.constexpr, BD: tl.constexpr, BV: tl.constexpr,
                        BG: tl.constexpr, NCH: tl.constexpr):
    b = tl.program_id(0); sb = tl.program_id(1); dvp = dv + 1
    offs_t = tl.arange(0, BT); offs_d = tl.arange(0, BD); offs_v = tl.arange(0, BV)
    offs_c = sb * BG + tl.arange(0, BG)
    dmask = offs_d < dqk; cmask = offs_c < nc
    Sflat = tl.zeros([BD, BG * BV], dtype=tl.float32)
    for t in range(NCH):
        rows = t * BT + offs_t; rmask = rows < L
        qc = tl.load(q_ptr + b * sq_b + rows[:, None] * sq_l + offs_d[None, :] * sq_d,
                     mask=rmask[:, None] & dmask[None, :], other=0.0)
        kc = tl.load(k_ptr + b * sq_b + rows[:, None] * sq_l + offs_d[None, :] * sq_d,
                     mask=rmask[:, None] & dmask[None, :], other=0.0)
        vc = tl.load(v_ptr + b * sv_b + rows[:, None] * sv_l + offs_v[None, :] * sv_d,
                     mask=rmask[:, None] & (offs_v[None, :] < dv), other=0.0)
        vc += tl.where((offs_v[None, :] == dv) & rmask[:, None], 1.0, 0.0)
        rgc = tl.load(rg_ptr + b * sg_b + rows[:, None] * sg_l + offs_c[None, :] * sg_c,
                      mask=rmask[:, None] & cmask[None, :], other=0.0)
        wgc = tl.load(wg_ptr + b * sg_b + rows[:, None] * sg_l + offs_c[None, :] * sg_c,
                      mask=rmask[:, None] & cmask[None, :], other=0.0)
        ldc = tl.load(ld_ptr + b * sg_b + rows[:, None] * sg_l + offs_c[None, :] * sg_c,
                      mask=rmask[:, None] & cmask[None, :], other=0.0)
        a = tl.cumsum(ldc, axis=0)                            # [BT,BG] chunk-local cumulative decay
        rt = rgc * tl.exp(a); wt = wgc * tl.exp(-a)           # decay-absorbed gates (floored → fp32-safe)
        G = tl.dot(qc, tl.trans(kc))                          # SHARED content gram (once per chunk)
        R = tl.dot(rt, tl.trans(wt))                          # decayed routing gram (this state block)
        causal = (offs_t[:, None] >= offs_t[None, :]) & rmask[:, None] & rmask[None, :]
        A = G * R * causal
        o_intra = tl.dot(A.to(vc.dtype), vc)
        P = tl.dot(qc, Sflat.to(qc.dtype))
        P3 = tl.reshape(P, [BT, BG, BV])
        o_inter = tl.sum(P3 * rt[:, :, None], axis=1)         # decayed read gate
        o = o_intra + o_inter
        tl.store(outa_ptr + b * soa_b + sb * soa_n + rows[:, None] * soa_l + offs_v[None, :] * soa_v,
                 o, mask=rmask[:, None] & (offs_v[None, :] < dvp))
        Lam = tl.sum(tl.where(offs_t[:, None] == (BT - 1), a, 0.0), axis=0)   # [BG] chunk-total decay
        w_end = wgc * tl.exp(Lam[None, :] - a)
        WV = tl.reshape(w_end[:, :, None] * vc[:, None, :], [BT, BG * BV])
        decvec = tl.reshape(tl.exp(Lam)[:, None] * tl.full([BG, BV], 1.0, tl.float32), [BG * BV])
        Sflat = decvec[None, :] * Sflat + tl.dot(tl.trans(kc), WV.to(kc.dtype))


@triton.autotune(configs=_AT_CFGS, key=_AT_KEY)
@triton.jit
def _gla_bwd_qr(q_ptr, k_ptr, v_ptr, wg_ptr, rg_ptr, ld_ptr, g_ptr, dq_ptr, drg_ptr, dart_ptr, Sb_ptr,
               L, dqk, dv, nc,
               sq_b, sq_l, sq_d, sv_b, sv_l, sv_d, sg_b, sg_l, sg_c, sgr_b, sgr_l, sgr_d,
               sdq_b, sdq_n, sdq_l, sdq_d, sdr_b, sdr_l, sdr_c, sda_b, sda_l, sda_c,
               ssb_b, ssb_n, ssb_t, ssb_d, ssb_e,
               BT: tl.constexpr, BD: tl.constexpr, BV: tl.constexpr, BG: tl.constexpr, NCH: tl.constexpr):
    """GLA backward, FORWARD scan: dq, drg, da_rt (decay grad from the read gate rt=rg·e^a), and
    stores S_before per chunk (for the reverse kernel). Decay-aware (rt/wt, decayed state carry)."""
    b = tl.program_id(0); sb = tl.program_id(1)
    offs_t = tl.arange(0, BT); offs_d = tl.arange(0, BD); offs_v = tl.arange(0, BV); offs_e = tl.arange(0, BG * BV)
    offs_c = sb * BG + tl.arange(0, BG)
    dmask = offs_d < dqk; cmask = offs_c < nc; vmask = offs_v < (dv + 1)
    Sflat = tl.zeros([BD, BG * BV], dtype=tl.float32)
    for t in range(NCH):
        rows = t * BT + offs_t; rmask = rows < L
        qc = tl.load(q_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d, mask=rmask[:, None] & dmask[None, :], other=0.0)
        kc = tl.load(k_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d, mask=rmask[:, None] & dmask[None, :], other=0.0)
        v1 = tl.load(v_ptr + b*sv_b + rows[:, None]*sv_l + offs_v[None, :]*sv_d, mask=rmask[:, None] & (offs_v[None, :] < dv), other=0.0)
        v1 += tl.where((offs_v[None, :] == dv) & rmask[:, None], 1.0, 0.0)
        rgc = tl.load(rg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0)
        wgc = tl.load(wg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0)
        ldc = tl.load(ld_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0)
        gc = tl.load(g_ptr + b*sgr_b + rows[:, None]*sgr_l + offs_v[None, :]*sgr_d, mask=rmask[:, None] & vmask[None, :], other=0.0)
        a = tl.cumsum(ldc, axis=0); ea = tl.exp(a); rt = rgc * ea; wt = wgc * tl.exp(-a)
        G = tl.dot(qc, tl.trans(kc)); D = tl.dot(rt, tl.trans(wt)); P = tl.dot(gc, tl.trans(v1))
        caus = (offs_t[:, None] >= offs_t[None, :]) & rmask[:, None] & rmask[None, :]
        dG = P * D * caus; dD = P * G * caus
        dq_intra = tl.dot(dG.to(kc.dtype), kc); drt_intra = tl.dot(dD.to(wt.dtype), wt)
        tl.store(Sb_ptr + b*ssb_b + sb*ssb_n + t*ssb_t + offs_d[:, None]*ssb_d + offs_e[None, :]*ssb_e,
                 Sflat, mask=dmask[:, None])
        rt_g = tl.reshape(rt[:, :, None] * gc[:, None, :], [BT, BG * BV])
        dq_inter = tl.dot(rt_g.to(Sflat.dtype), tl.trans(Sflat))
        QS = tl.dot(qc, Sflat.to(qc.dtype))
        drt_inter = tl.sum(tl.reshape(QS, [BT, BG, BV]) * gc[:, None, :], axis=2)
        drt = drt_intra + drt_inter
        tl.store(dq_ptr + b*sdq_b + sb*sdq_n + rows[:, None]*sdq_l + offs_d[None, :]*sdq_d,
                 dq_intra + dq_inter, mask=rmask[:, None] & dmask[None, :])
        tl.store(drg_ptr + b*sdr_b + rows[:, None]*sdr_l + offs_c[None, :]*sdr_c, drt * ea, mask=rmask[:, None] & cmask[None, :])
        tl.store(dart_ptr + b*sda_b + rows[:, None]*sda_l + offs_c[None, :]*sda_c, drt * rt, mask=rmask[:, None] & cmask[None, :])
        Lam = tl.sum(tl.where(offs_t[:, None] == (BT - 1), a, 0.0), axis=0)
        w_end = wgc * tl.exp(Lam[None, :] - a)
        WV = tl.reshape(w_end[:, :, None] * v1[:, None, :], [BT, BG * BV])
        decvec = tl.reshape(tl.exp(Lam)[:, None] * tl.full([BG, BV], 1.0, tl.float32), [BG * BV])
        Sflat = decvec[None, :] * Sflat + tl.dot(tl.trans(kc), WV.to(kc.dtype))


def triton_gla_bwd_qr(q, k, v, wg, rg, ld, g, chunk=32, BG=16):
    B, L, dqk = q.shape; dv = v.shape[-1]; nc = wg.shape[-1]
    BD = max(16, triton.next_power_of_2(dqk)); BV = max(16, triton.next_power_of_2(dv + 1))
    NB = triton.cdiv(nc, BG); NCH = triton.cdiv(L, chunk)
    q, k, v, wg, rg, ld, g = [x.contiguous() for x in (q, k, v, wg, rg, ld, g)]
    dq = torch.zeros(B, NB, L, dqk, device=q.device, dtype=torch.float32)
    drg = torch.zeros(B, L, nc, device=q.device, dtype=torch.float32)
    dart = torch.zeros(B, L, nc, device=q.device, dtype=torch.float32)
    Sb = torch.zeros(B, NB, NCH, BD, BG * BV, device=q.device, dtype=torch.float32)
    _gla_bwd_qr[(B, NB)](q, k, v, wg, rg, ld, g, dq, drg, dart, Sb, L, dqk, dv, nc,
        q.stride(0), q.stride(1), q.stride(2), v.stride(0), v.stride(1), v.stride(2),
        wg.stride(0), wg.stride(1), wg.stride(2), g.stride(0), g.stride(1), g.stride(2),
        dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3), drg.stride(0), drg.stride(1), drg.stride(2),
        dart.stride(0), dart.stride(1), dart.stride(2),
        Sb.stride(0), Sb.stride(1), Sb.stride(2), Sb.stride(3), Sb.stride(4),
        BT=chunk, BD=BD, BV=BV, BG=BG, NCH=NCH)
    return dq.sum(1), drg, dart, Sb


@triton.autotune(configs=_AT_CFGS, key=_AT_KEY)
@triton.jit
def _gla_bwd_kwv(q_ptr, k_ptr, v_ptr, wg_ptr, rg_ptr, ld_ptr, g_ptr, Sb_ptr,
                 dk_ptr, dwg_ptr, dv_ptr, dakwv_ptr, dlam_ptr,
                 L, dqk, dv, nc,
                 sq_b, sq_l, sq_d, sv_b, sv_l, sv_d, sg_b, sg_l, sg_c, sgr_b, sgr_l, sgr_d,
                 ssb_b, ssb_n, ssb_t, ssb_d, ssb_e,
                 sdk_b, sdk_n, sdk_l, sdk_d, sdw_b, sdw_l, sdw_c, sdv_b, sdv_n, sdv_l, sdv_d,
                 sda_b, sda_l, sda_c, sdl_b, sdl_t, sdl_c,
                 BT: tl.constexpr, BD: tl.constexpr, BV: tl.constexpr, BG: tl.constexpr, NCH: tl.constexpr):
    """GLA backward, REVERSE scan: dk, dwg, dv, da_kwv (=da_wt+da_wend), dLam. Carries decayed dS;
    loads S_before (from the fwd kernel) for dLam_carry. Decay-aware."""
    b = tl.program_id(0); sb = tl.program_id(1); dvp = dv + 1
    offs_t = tl.arange(0, BT); offs_d = tl.arange(0, BD); offs_v = tl.arange(0, BV); offs_e = tl.arange(0, BG * BV)
    offs_c = sb * BG + tl.arange(0, BG)
    dmask = offs_d < dqk; cmask = offs_c < nc
    dS = tl.zeros([BD, BG * BV], dtype=tl.float32)
    for ti in range(NCH):
        t = NCH - 1 - ti
        rows = t * BT + offs_t; rmask = rows < L
        qc = tl.load(q_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d, mask=rmask[:, None] & dmask[None, :], other=0.0)
        kc = tl.load(k_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d, mask=rmask[:, None] & dmask[None, :], other=0.0)
        v1 = tl.load(v_ptr + b*sv_b + rows[:, None]*sv_l + offs_v[None, :]*sv_d, mask=rmask[:, None] & (offs_v[None, :] < dv), other=0.0)
        v1 += tl.where((offs_v[None, :] == dv) & rmask[:, None], 1.0, 0.0)
        rgc = tl.load(rg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0)
        wgc = tl.load(wg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0)
        ldc = tl.load(ld_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0)
        gc = tl.load(g_ptr + b*sgr_b + rows[:, None]*sgr_l + offs_v[None, :]*sgr_d, mask=rmask[:, None] & (offs_v[None, :] < dvp), other=0.0)
        Sb = tl.load(Sb_ptr + b*ssb_b + sb*ssb_n + t*ssb_t + offs_d[:, None]*ssb_d + offs_e[None, :]*ssb_e, mask=dmask[:, None], other=0.0)
        a = tl.cumsum(ldc, axis=0); ena = tl.exp(-a); rt = rgc * tl.exp(a); wt = wgc * ena
        Lam = tl.sum(tl.where(offs_t[:, None] == (BT - 1), a, 0.0), axis=0)
        w_end = wgc * tl.exp(Lam[None, :] - a)
        G = tl.dot(qc, tl.trans(kc)); P = tl.dot(gc, tl.trans(v1)); D = tl.dot(rt, tl.trans(wt))
        caus = (offs_t[:, None] >= offs_t[None, :]) & rmask[:, None] & rmask[None, :]
        A = G * D * caus; dG = P * D * caus; dD = P * G * caus
        dk_intra = tl.dot(tl.trans(dG).to(qc.dtype), qc)
        dwt = tl.dot(tl.trans(dD).to(rt.dtype), rt)
        dv_intra = tl.dot(tl.trans(A).to(gc.dtype), gc)
        # write-grad via dS
        KS = tl.dot(kc, dS.to(kc.dtype)); KS3 = tl.reshape(KS, [BT, BG, BV])
        dw_end = tl.sum(KS3 * v1[:, None, :], axis=2)
        wv1 = tl.reshape(w_end[:, :, None] * v1[:, None, :], [BT, BG * BV])
        dk_KV = tl.dot(wv1.to(dS.dtype), tl.trans(dS))
        dv_KV = tl.sum(w_end[:, :, None] * KS3, axis=1)
        # dLam = carry (needs Sb) + w_end's Lam part
        dLam = tl.sum(tl.reshape(dS * Sb, [BD, BG, BV]), axis=2)         # [BD,BG] sum over v
        dLam = tl.sum(dLam, axis=0) * tl.exp(Lam)                        # [BG] sum over d, ×e^Lam
        dLam = dLam + tl.sum(dw_end * w_end, axis=0)
        da_wend = -dw_end * w_end; dwg_wend = dw_end * tl.exp(Lam[None, :] - a)
        dwgc = dwt * ena + dwg_wend; da_wt = -dwt * wt
        tl.store(dk_ptr + b*sdk_b + sb*sdk_n + rows[:, None]*sdk_l + offs_d[None, :]*sdk_d,
                 dk_intra + dk_KV, mask=rmask[:, None] & dmask[None, :])
        tl.store(dwg_ptr + b*sdw_b + rows[:, None]*sdw_l + offs_c[None, :]*sdw_c, dwgc, mask=rmask[:, None] & cmask[None, :])
        tl.store(dv_ptr + b*sdv_b + sb*sdv_n + rows[:, None]*sdv_l + offs_v[None, :]*sdv_d,
                 dv_intra + dv_KV, mask=rmask[:, None] & (offs_v[None, :] < dv))
        tl.store(dakwv_ptr + b*sda_b + rows[:, None]*sda_l + offs_c[None, :]*sda_c, da_wt + da_wend, mask=rmask[:, None] & cmask[None, :])
        tl.store(dlam_ptr + b*sdl_b + t*sdl_t + offs_c*sdl_c, dLam, mask=cmask)
        rt_g = tl.reshape(rt[:, :, None] * gc[:, None, :], [BT, BG * BV])
        decvec = tl.reshape(tl.exp(Lam)[:, None] * tl.full([BG, BV], 1.0, tl.float32), [BG * BV])
        dS = decvec[None, :] * dS + tl.dot(tl.trans(qc), rt_g.to(qc.dtype))


def triton_gla_bwd(q, k, v, wg, rg, ld, g, chunk=32, BG=16):
    """Full bespoke fused-shared-gram GLA backward. Returns dq,dk,dv,dwg,drg,dld."""
    B, L, dqk = q.shape; dv = v.shape[-1]; nc = wg.shape[-1]
    BD = max(16, triton.next_power_of_2(dqk)); BV = max(16, triton.next_power_of_2(dv + 1))
    NB = triton.cdiv(nc, BG); NCH = triton.cdiv(L, chunk)
    dq, drg, dart, Sb = triton_gla_bwd_qr(q, k, v, wg, rg, ld, g, chunk=chunk, BG=BG)
    q, k, v, wg, rg, ld, g = [x.contiguous() for x in (q, k, v, wg, rg, ld, g)]
    dk = torch.zeros(B, NB, L, dqk, device=q.device, dtype=torch.float32)
    dwg = torch.zeros(B, L, nc, device=q.device, dtype=torch.float32)
    dvo = torch.zeros(B, NB, L, BV, device=q.device, dtype=torch.float32)
    dakwv = torch.zeros(B, L, nc, device=q.device, dtype=torch.float32)
    dlam = torch.zeros(B, NCH, nc, device=q.device, dtype=torch.float32)
    _gla_bwd_kwv[(B, NB)](q, k, v, wg, rg, ld, g, Sb, dk, dwg, dvo, dakwv, dlam, L, dqk, dv, nc,
        q.stride(0), q.stride(1), q.stride(2), v.stride(0), v.stride(1), v.stride(2),
        wg.stride(0), wg.stride(1), wg.stride(2), g.stride(0), g.stride(1), g.stride(2),
        Sb.stride(0), Sb.stride(1), Sb.stride(2), Sb.stride(3), Sb.stride(4),
        dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3), dwg.stride(0), dwg.stride(1), dwg.stride(2),
        dvo.stride(0), dvo.stride(1), dvo.stride(2), dvo.stride(3), dakwv.stride(0), dakwv.stride(1), dakwv.stride(2),
        dlam.stride(0), dlam.stride(1), dlam.stride(2),
        BT=chunk, BD=BD, BV=BV, BG=BG, NCH=NCH)
    # dld = reverse-cumsum(da) per chunk, da = da_rt + da_kwv, + dLam at each chunk's last position
    da = dart + dakwv                                       # [B,L,nc]
    pad = (-L) % chunk; Lp = L + pad
    if pad: da = torch.nn.functional.pad(da, (0, 0, 0, pad))
    da = da.view(B, NCH, chunk, nc)
    da[:, :, -1, :] = da[:, :, -1, :] + dlam                # Lam = a[last]
    dld = torch.flip(torch.cumsum(torch.flip(da, [2]), 2), [2]).reshape(B, Lp, nc)[:, :L]
    return dq, dk.sum(1), dvo.sum(1)[..., :dv], dwg, drg, dld


def triton_gla(q, k, v, wg, rg, ld, eps=EPS, chunk=32, BG=16, normalized=True):
    """Triton forward for scalar-gated RoLA-GLA, tiled over states. Floors per-token decay
    (fp32-safe factored gram). q,k:[B,L,dqk] v:[B,L,dv] wg,rg,ld:[B,L,nc]."""
    B, L, dqk = q.shape; dv = v.shape[-1]; nc = wg.shape[-1]
    ld = ld.clamp(min=_GLA_FLOOR).contiguous()
    BT = chunk; BD = max(16, triton.next_power_of_2(dqk)); BV = max(16, triton.next_power_of_2(dv + 1))
    NB = triton.cdiv(nc, BG); NCH = triton.cdiv(L, BT)
    q, k, v, wg, rg = [x.contiguous() for x in (q, k, v, wg, rg)]
    out_aug = torch.zeros(B, NB, L, BV, device=q.device, dtype=torch.float32)
    _rola_gla_fwd_tiled[(B, NB)](
        q, k, v, wg, rg, ld, out_aug, L, dqk, dv, nc,
        q.stride(0), q.stride(1), q.stride(2), v.stride(0), v.stride(1), v.stride(2),
        wg.stride(0), wg.stride(1), wg.stride(2),
        out_aug.stride(0), out_aug.stride(1), out_aug.stride(2), out_aug.stride(3),
        BT=BT, BD=BD, BV=BV, BG=BG, NCH=NCH)
    O = out_aug[..., :dv + 1].sum(1)
    if not normalized:
        return O[..., :dv].to(q.dtype)
    num, den = O[..., :dv], O[..., dv:dv + 1]
    return (num / (den + eps)).to(q.dtype)


def _triton_gla_aug(q, k, v, wg, rg, ld, chunk=32, BG=16):
    """GLA Triton forward returning the AUGMENTED output [B,L,dv+1] (num | den)."""
    B, L, dqk = q.shape; dv = v.shape[-1]; nc = wg.shape[-1]
    ld = ld.clamp(min=_GLA_FLOOR).contiguous()
    BD = max(16, triton.next_power_of_2(dqk)); BV = max(16, triton.next_power_of_2(dv + 1))
    NB = triton.cdiv(nc, BG); NCH = triton.cdiv(L, chunk)
    q, k, v, wg, rg = [x.contiguous() for x in (q, k, v, wg, rg)]
    out_aug = torch.zeros(B, NB, L, BV, device=q.device, dtype=torch.float32)
    _rola_gla_fwd_tiled[(B, NB)](
        q, k, v, wg, rg, ld, out_aug, L, dqk, dv, nc,
        q.stride(0), q.stride(1), q.stride(2), v.stride(0), v.stride(1), v.stride(2),
        wg.stride(0), wg.stride(1), wg.stride(2),
        out_aug.stride(0), out_aug.stride(1), out_aug.stride(2), out_aug.stride(3),
        BT=chunk, BD=BD, BV=BV, BG=BG, NCH=NCH)
    return out_aug[..., :dv + 1].sum(1)


class _TritonGLAFn(torch.autograd.Function):
    """Fully-fused bespoke shared-gram GLA: Triton forward + bespoke fused Triton BACKWARD
    (`triton_gla_bwd`: dq/dr fwd-scan + dk/dw/dv reverse-scan + dld, decay-aware, nc-flat, NO
    virtual-head replication, NO forward recompute — grads computed analytically from saved
    tensors). Verified vs the torch GLA reference (all 6 grads incl dld) to fp noise."""
    @staticmethod
    def forward(ctx, q, k, v, wg, rg, ld, eps, chunk, normalized):
        Oa = _triton_gla_aug(q, k, v, wg, rg, ld, chunk=chunk); dv = v.shape[-1]
        den = Oa[..., dv:dv + 1]; num = Oa[..., :dv]
        O = (num / (den + eps)).to(q.dtype) if normalized else num.to(q.dtype)
        ctx.save_for_backward(q, k, v, wg, rg, ld, O, den)
        ctx.eps, ctx.chunk, ctx.normalized = eps, chunk, normalized
        return O

    @staticmethod
    def backward(ctx, dO):
        q, k, v, wg, rg, ld, O, den = ctx.saved_tensors; eps = ctx.eps
        dOf = dO.float()
        if ctx.normalized:
            g_num = dOf / (den + eps); g_den = -(dOf * O.float()).sum(-1, keepdim=True) / (den + eps)
        else:                                  # raw: output IS num; ones-column (den) unused → 0 grad
            g_num = dOf; g_den = torch.zeros_like(den)
        g = torch.cat([g_num, g_den], dim=-1)
        gr = triton_gla_bwd(q.float(), k.float(), v.float(), wg.float(), rg.float(), ld.float(), g, chunk=ctx.chunk)
        return (*[x.to(q.dtype) for x in gr], None, None, None)


def _vh_gla(q, k, v, wg, rg, ld, normalized=True):
    """Virtual-head RoLA-GLA via FLA's fused chunk_gla (per-state scalar decay broadcast over
    d_qk). Autograd-capable (FLA's Triton fwd+bwd). q,k:[B,L,dqk] v:[B,L,dv] wg,rg,ld:[B,L,nc]."""
    B, L, dqk = q.shape; nc = wg.shape[-1]
    v1 = torch.cat([v, torch.ones_like(v[..., :1])], dim=-1) if normalized else v
    qv = q.unsqueeze(2).expand(-1, -1, nc, -1).contiguous()
    kv = k.unsqueeze(2).expand(-1, -1, nc, -1).contiguous()
    vv = (v1.unsqueeze(2) * wg.unsqueeze(-1)).contiguous()
    gg = ld.unsqueeze(-1).expand(-1, -1, -1, dqk).contiguous()
    o, _ = _fla_chunk_gla(qv, kv, vv, gg, scale=1.0)
    o = (o * rg.unsqueeze(-1)).sum(2)
    return o[..., :-1] / (o[..., -1:] + EPS) if normalized else o


def triton_gla_ag(q, k, v, wg, rg, ld, eps=EPS, chunk=32, normalized=True):
    """Autograd-capable Triton RoLA-GLA (Triton fused fwd + exact safe-torch bwd)."""
    return _TritonGLAFn.apply(q, k, v, wg, rg, ld, eps, chunk, normalized)


# ----------------------------------------------------------------------------
# Test harness
# ----------------------------------------------------------------------------
def _phi(x):
    return F.elu(x) + 1.0


def _mk(B, L, dqk, dv, nc, topk=None, seed=0, dev='cuda', dtype=torch.float32):
    """Realistic inputs: q,k post-φ (positive), gates softmax (optionally top-k
    sparse). All leaves so autograd gives input grads."""
    torch.manual_seed(seed)
    q = _phi(torch.randn(B, L, dqk, device=dev, dtype=dtype)).detach().requires_grad_(True)
    k = _phi(torch.randn(B, L, dqk, device=dev, dtype=dtype)).detach().requires_grad_(True)
    v = torch.randn(B, L, dv, device=dev, dtype=dtype).detach().requires_grad_(True)

    def gates(seed_off):
        torch.manual_seed(seed + seed_off)
        logits = torch.randn(B, L, nc, device=dev, dtype=dtype)
        if topk is not None and topk < nc:
            thresh = logits.topk(topk, dim=-1).values[..., -1:]
            logits = logits.masked_fill(logits < thresh, float('-inf'))
        return F.softmax(logits, dim=-1).detach().requires_grad_(True)

    return q, k, v, gates(100), gates(200)   # q,k,v,wg,rg


def _grad_check(fn, ref, **mk):
    q, k, v, wg, rg = _mk(**mk)
    ins = [q, k, v, wg, rg]
    o_ref = ref(q, k, v, wg, rg)
    gref = torch.autograd.grad(o_ref.sum(), ins)
    o = fn(q, k, v, wg, rg)
    g = torch.autograd.grad(o.sum(), ins)
    diffs = {'fwd': (o - o_ref).abs().max().item()}
    for name, a, b in zip(['dq', 'dk', 'dv', 'dwg', 'drg'], g, gref):
        diffs[name] = (a - b).abs().max().item()
    return diffs


if __name__ == '__main__':
    dev = 'cuda'
    fmt = lambda d: " ".join(f"{kk}={vv:.1e}" for kk, vv in d.items())
    cands = [('virtual_head', virtual_head_global),
             ('kron_grouped', kron_grouped_global),
             ('chunked_shared', chunked_shared)]
    for topk in (None, 4):
        tag = "dense" if topk is None else f"top{topk}"
        print(f"\n=== global-norm correctness ({tag}), fwd + grads vs ref_global ===")
        for nc in (16, 64, 256):
            for name, fn in cands:
                d = _grad_check(fn, ref_global, B=2, L=256, dqk=12, dv=12, nc=nc, topk=topk)
                worst = max(d.values())
                flag = "OK " if worst < 2e-2 else "!! "
                print(f"{flag}nc={nc:3d} {name:14s}: {fmt(d)}")
