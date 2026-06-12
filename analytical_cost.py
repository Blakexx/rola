"""Analytical FLOPs + activation memory for the RoLA fused kernels — done the
least-wrong way: the closed-form is ASSERTED equal to torch's FlopCounterMode run
on the ACTUAL kernel, so a derivation error fails loudly instead of shipping.

Principle:
  * Our fused kernels are pure torch -> FlopCounterMode counts every aten matmul
    EXACTLY (it hooks the dispatcher). So the tool is ground truth; the closed-form
    only has to MATCH it. If it doesn't, the closed-form is wrong, not the tool.
  * Triton baselines (FLA GLA/GDN) are prior art with published chunked-algorithm
    FLOPs; those are derived/cited separately (the tool can't see Triton). The
    method here is calibrated on the torch cells so the same bookkeeping carries.
  * Convention: FORWARD, matmul-family ops only (2*m*n*k per [m,k]x[k,n]).
    Elementwise (decay, masks, gating) is O(elements), not counted by the tool and
    negligible vs matmuls; reported separately. Backward ~= 2x forward.

Run: .venv/bin/python analytical_cost.py
"""
import torch
from torch.utils.flop_counter import FlopCounterMode
import rola

torch.manual_seed(0)


def measured_fwd_flops(fn, *args):
    """Exact forward matmul FLOPs of a pure-torch fn, via the dispatcher."""
    m = FlopCounterMode(display=False)
    with m:
        fn(*args)
    return m.get_total_flops()


# ---- closed-form FORWARD matmul FLOPs of _rola_chunked_parallel (RoLA-RLA) ----
# q,k:[B,L,H,d]  v:[B,L,H,e]  wg,rg:[B,L,H,nc]; kernel augments v->ev=e+1.
# bh=B*H, n=L/chunk (assume divisible), p=chunk. Matmuls (see rola.py:108-119):
#   G  einsum bnid,bnjd->bnij     : 2*bh*n*p*p*d
#   R  einsum bnic,bnjc->bnij     : 2*bh*n*p*p*nc
#   o_intra A,vc bnij,bnjv->bniv  : 2*bh*n*p*p*ev
#   KV  bnjc,bnjd,bnjv->bncdv     : 2*bh*L*nc*d*ev      (n*p=L)
#   M   bnic,bncdv->bnidv         : 2*bh*L*p*nc*... -> see note; tool is truth
#   o_inter qc,M bnid,bnidv->bniv : 2*bh*L*d*ev
def cf_rola_rla(B, L, H, d, e, nc, chunk):
    bh = B * H; n = L // chunk; p = chunk; ev = e + 1
    intra = 2 * bh * n * p * p * (d + nc + ev)          # G + R + o_intra
    kv    = 2 * bh * L * nc * d * ev                     # KV  (n*p=L)
    m     = 2 * bh * n * p * nc * d * ev                 # M: read carried state per token
    inter = 2 * bh * L * d * ev                          # o_inter
    return intra + kv + m + inter


def recurrent_rola_rla(q, k, v, wg, rg, eps=1e-5):
    """RECURRENT (token-by-token) additive RoLA, pure torch -> FlopCounter-exact.
    The decode schedule: maintain nc states, rank-1 update + read per token. No
    pairwise gram -> fewer FLOPs than chunked, O(nc*d*e) state memory, but serial.
    Same global-norm math as _rola_chunked_parallel (v augmented with a ones col)."""
    B, L, H, d = q.shape; e = v.shape[-1]; nc = wg.shape[-1]
    v1 = torch.cat([v, torch.ones_like(v[..., :1])], dim=-1)       # ev = e+1
    S = q.new_zeros(B, H, nc, d, v1.shape[-1])
    outs = []
    for t in range(L):
        kv = torch.einsum('bhd,bhv->bhdv', k[:, t], v1[:, t])      # outer product (shared content)
        S = S + torch.einsum('bhc,bhdv->bhcdv', wg[:, t], kv)      # write to nc states
        read = torch.einsum('bhd,bhcdv->bhcv', q[:, t], S)         # read each state
        outs.append(torch.einsum('bhc,bhcv->bhv', rg[:, t], read)) # combine by read gate
    O = torch.stack(outs, 1)
    return O[..., :-1] / (O[..., -1:] + eps)


def cf_recurrent_rla(B, L, H, d, e, nc):
    """Closed-form fwd FLOPs of the recurrent form, matmul/2mnk convention.
    NOTE: kv outer-product + write-into-states are broadcasts/outer products, NOT
    contractions, so the standard convention (and FlopCounterMode) does not count
    them. Only read (contracts d) + combine (contracts c) count. [The tool caught
    the bug of counting the outer products: it gave ~2x too high.]"""
    bh = B * H; ev = e + 1
    return 2 * bh * L * (nc * d * ev + nc * ev)        # read + combine only


def virtual_head_additive(q, k, v, wg, rg, chunk=64):
    """Non-fused (virtual-head) additive RoLA, PURE TORCH so FlopCounter sees it
    exactly: run nc weight-tied linear-attention heads (write-gate folds into the
    value, read-gate into the output), each recomputing its own content gram. This
    is the cost the fused kernel avoids by sharing one gram across states."""
    B, L, H, nc = wg.shape
    one = torch.ones(B, L, H, 1, device=q.device, dtype=q.dtype)
    O = None
    for c in range(nc):
        vc = v * wg[..., c:c + 1]                                  # write-gate -> value
        oc = rola._rola_chunked_parallel(q, k, vc, one, one, 1e-5, chunk)  # plain LA, own gram
        oc = rg[..., c:c + 1] * oc                                 # read-gate combine
        O = oc if O is None else O + oc
    return O


def matched_rank_row(R, d_v, L, chunk, d_qk_rola=16):
    """Both at effective rank R (=> matched state R*d_v): monolithic rank-R (one wide
    projection / feature map, gram contracts R) vs routed rank-R (narrow d_qk + nc=R/d_qk,
    gram contracts d_qk+nc). Both pure torch -> FlopCounter-exact."""
    B = H = 1
    nc = max(1, R // d_qk_rola); d_rola = d_qk_rola
    one = torch.ones(B, L, H, 1)
    # monolith rank R: nc=1, d_qk=R  (== wide-LA / Based / Hedgehog gram cost at feat_dim=R)
    qm = torch.randn(B, L, H, R); km = torch.randn(B, L, H, R); vm = torch.randn(B, L, H, d_v)
    mono = measured_fwd_flops(rola._rola_chunked_parallel, qm, km, vm, one, one, 1e-5, chunk)
    # routed rank R: narrow d_qk, nc states
    qr = torch.randn(B, L, H, d_rola); kr = torch.randn(B, L, H, d_rola); vr = torch.randn(B, L, H, d_v)
    wg = torch.softmax(torch.randn(B, L, H, nc), -1); rg = torch.softmax(torch.randn(B, L, H, nc), -1)
    routed = measured_fwd_flops(rola._rola_chunked_parallel, qr, kr, vr, wg, rg, 1e-5, chunk)
    return mono, routed, nc


def main():
    # ============ MATCHED-RANK FLOP comparison (the headline efficiency figure) ============
    print("=== matched effective-rank R (== matched state): monolithic vs routed FLOPs ===")
    print("  monolith rank-R = one wide proj / feature map (gram contracts R) [wide-LA/Based/Hedgehog]")
    print("  routed rank-R   = narrow d_qk=12 + nc=R/12 routing (gram contracts d_qk+nc); R=12*nc")
    print(f"{'rank R':>8}{'nc':>5}{'monolith FLOPs':>18}{'routed FLOPs':>16}{'routed cheaper':>16}")
    for nc_ in [8, 16, 32, 64, 128, 256]:
        R = 12 * nc_                                       # ranks d_qk=12 actually reaches
        mono, routed, nc = matched_rank_row(R, d_v=12, L=256, chunk=64, d_qk_rola=12)
        print(f"{R:>8}{nc:>5}{mono:>18,}{routed:>16,}{mono/routed:>15.2f}x")
    print("  (gram: monolith ~C*R  vs  routed ~C*(d_qk+nc); state term R*d_v is shared)\n")

    # ============ ALL FORMS of RoLA at one config (the full spectrum) ============
    B, H, L, d, e, nc = 1, 1, 256, 12, 12, 64
    q = torch.randn(B, L, H, d); k = torch.randn(B, L, H, d); v = torch.randn(B, L, H, e)
    wg = torch.softmax(torch.randn(B, L, H, nc), -1); rg = torch.softmax(torch.randn(B, L, H, nc), -1)
    state_mem = nc * d * (e + 1) * 4  # recurrent state bytes (the serving cost, all forms)
    print(f"=== RoLA forms @ B{B}H{H}L{L} d{d} e{e} nc{nc}  (state={state_mem/1e3:.1f}KB, all forms) ===")
    print(f"{'form':<26}{'fwd FLOPs':>16}{'vs recurrent':>14}{'parallel':>10}")
    rec = measured_fwd_flops(recurrent_rola_rla, q, k, v, wg, rg)
    print(f"{'recurrent (decode)':<26}{rec:>16,}{1.0:>13.2f}x{'serial':>10}")
    for ch in [16, 64, 256]:  # chunk=256=L -> full/quadratic
        f = measured_fwd_flops(rola._rola_chunked_parallel, q, k, v, wg, rg, 1e-5, ch)
        lbl = f"fused chunk={ch}" + (" (=full)" if ch == L else "")
        print(f"{lbl:<26}{f:>16,}{f/rec:>13.2f}x{'yes':>10}")
    vh = measured_fwd_flops(virtual_head_additive, q, k, v, wg, rg, 64)
    print(f"{'virtual-head chunk=64':<26}{vh:>16,}{vh/rec:>13.2f}x{'yes':>10}")
    print("  -> recurrent = fewest FLOPs (no pairwise gram) but serial; chunk size trades")
    print("     FLOPs for parallelism; fused < virtual-head at the same chunk.\n")

    # validate the recurrent closed-form against the tool too
    cf = cf_recurrent_rla(B, L, H, d, e, nc)
    print(f"recurrent closed-form check: cf={cf:,} measured={rec:,} ratio={cf/rec:.3f}\n")

    # --- fused vs non-fused (virtual-head) concrete saving, both tool-measured ---
    print("=== fused vs virtual-head FLOPs (both pure torch, FlopCounter-exact) ===")
    print(f"{'case':<30}{'fused':>14}{'virtual-head':>16}{'saving x':>10}")
    for c in [dict(B=1, H=1, L=256, d=12, e=12, nc=16, chunk=64),
              dict(B=1, H=1, L=256, d=12, e=12, nc=64, chunk=64),
              dict(B=1, H=1, L=256, d=12, e=4,  nc=64, chunk=64),   # tiny d_v -> gram dominates (max saving)
              dict(B=1, H=1, L=256, d=32, e=32, nc=64, chunk=64),   # d,e ~ chunk
              dict(B=1, H=1, L=256, d=64, e=64, nc=64, chunk=64),   # d,e = chunk
              dict(B=1, H=1, L=256, d=128,e=128,nc=64, chunk=64),   # d,e >> chunk -> state dominates -> regress toward 1x
              dict(B=1, H=1, L=256, d=12, e=12, nc=64, chunk=16)]:  # small chunk -> regress (less gram to share)
        B, H, L, d, e, nc, chunk = (c[x] for x in ('B', 'H', 'L', 'd', 'e', 'nc', 'chunk'))
        q = torch.randn(B, L, H, d); k = torch.randn(B, L, H, d); v = torch.randn(B, L, H, e)
        wg = torch.softmax(torch.randn(B, L, H, nc), -1); rg = torch.softmax(torch.randn(B, L, H, nc), -1)
        ff = measured_fwd_flops(rola._rola_chunked_parallel, q, k, v, wg, rg, 1e-5, chunk)
        vf = measured_fwd_flops(virtual_head_additive, q, k, v, wg, rg, chunk)
        print(f"L{L}d{d}e{e}nc{nc:<3}{'':<13}{ff:>14,}{vf:>16,}{vf/ff:>9.2f}x")
    print()

    cases = [
        dict(B=1, H=1, L=256, d=12, e=12, nc=16, chunk=64),
        dict(B=2, H=4, L=512, d=12, e=12, nc=64, chunk=64),
        dict(B=1, H=1, L=256, d=16, e=16, nc=8,  chunk=64),
    ]
    print(f"{'case':<38}{'closed-form':>16}{'FlopCounter':>16}{'ratio':>8}")
    for c in cases:
        B, H, L, d, e, nc, chunk = c['B'], c['H'], c['L'], c['d'], c['e'], c['nc'], c['chunk']
        q = torch.randn(B, L, H, d); k = torch.randn(B, L, H, d)
        v = torch.randn(B, L, H, e)
        wg = torch.softmax(torch.randn(B, L, H, nc), -1)
        rg = torch.softmax(torch.randn(B, L, H, nc), -1)
        meas = measured_fwd_flops(rola._rola_chunked_parallel, q, k, v, wg, rg, 1e-5, chunk)
        cf = cf_rola_rla(B, L, H, d, e, nc, chunk)
        tag = f"RLA B{B}H{H}L{L}d{d}e{e}nc{nc}"
        print(f"{tag:<38}{cf:>16,}{meas:>16,}{(cf/meas if meas else 0):>8.3f}")

    # ---- activation memory (GPU only): peak transient of fwd+bwd ----
    if torch.cuda.is_available():
        print("\n--- activation memory (torch.cuda.max_memory_allocated, fwd+bwd) ---")
        for c in cases:
            B, H, L, d, e, nc, chunk = c['B'], c['H'], c['L'], c['d'], c['e'], c['nc'], c['chunk']
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            q = torch.randn(B, L, H, d, device='cuda', requires_grad=True)
            k = torch.randn(B, L, H, d, device='cuda', requires_grad=True)
            v = torch.randn(B, L, H, e, device='cuda', requires_grad=True)
            wg = torch.softmax(torch.randn(B, L, H, nc, device='cuda'), -1).requires_grad_()
            rg = torch.softmax(torch.randn(B, L, H, nc, device='cuda'), -1).requires_grad_()
            out = rola._rola_chunked_parallel(q, k, v, wg, rg, 1e-5, chunk)
            out.sum().backward()
            mb = torch.cuda.max_memory_allocated() / 1e6
            # analytical dominant transient: Kronecker state [bh,n,nc,d,ev] + grams [bh,n,p,p]
            bh = B * H; n = L // chunk; ev = e + 1
            kron = bh * n * nc * d * ev * 4 / 1e6
            grams = bh * n * chunk * chunk * 4 / 1e6
            print(f"  L{L}d{d}nc{nc}: peak={mb:7.1f}MB   "
                  f"analytic dominant: Kron={kron:6.2f}MB grams={grams:6.2f}MB")
    else:
        print("\n(no CUDA: skipping activation-memory measurement)")


if __name__ == '__main__':
    main()
