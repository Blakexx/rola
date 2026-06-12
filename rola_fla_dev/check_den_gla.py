"""Verification: GLA per-state denominator kernel (_DenGLAFn) vs O(L^2) oracle.

d[i,c] = sum_{j<=i} (q_i.k_j) w_j^c exp(Lam_i^c - Lam_j^c),  Lam = inclusive cumsum(ld).

Checks: (1) additive den regression (USE_G=False path unchanged), (2) GLA den forward
vs oracle (fp32+bf16, nc/L/dqk grid, ragged L, mild + deep decay), (3) all four grads
(dq, dk, dw, dld) vs fp64 oracle autograd through a random-gd scalar loss.
"""
import sys, torch
sys.path.insert(0, '/home/blake/flash-linear-attention-rola')
from fla_rola.ops.simple_gla.rola import rola_perstate_den_triton, rola_perstate_den_gla_triton

torch.manual_seed(0)
dev = 'cuda'


def den_ref(q, k, w, ld=None):
    B, L, D = q.shape
    G = torch.einsum('bid,bjd->bij', q, k)
    mask = torch.tril(torch.ones(L, L, device=q.device, dtype=torch.bool))
    if ld is None:
        return (G[..., None] * w[:, None, :, :] * mask[None, :, :, None]).sum(2)
    Lam = ld.cumsum(1)                                        # [B,L,nc]
    dec = torch.exp(Lam[:, :, None, :] - Lam[:, None, :, :])  # [B,i,j,c]
    return (G[..., None] * dec * w[:, None, :, :] * mask[None, :, :, None]).sum(2)


def rel(a, b):
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-12)


fails = 0
def report(name, r, tol):
    global fails
    ok = r < tol
    fails += (not ok)
    print(f"{name:58s} rel {r:.2e}  {'PASS' if ok else 'FAIL'}")


# ---- 1) additive regression ----
for nc, L in ((16, 256), (48, 200)):
    q = torch.randn(4, L, 12, device=dev).abs(); k = torch.randn(4, L, 12, device=dev).abs()
    w = torch.softmax(torch.randn(4, L, nc, device=dev), -1)
    d = rola_perstate_den_triton(q, k, w)
    report(f"additive regression nc={nc} L={L}", rel(d, den_ref(q.double(), k.double(), w.double()).float()), 2e-3)

# ---- 2) GLA den forward ----
for dt, dtol in ((torch.float32, 2e-3), (torch.bfloat16, 3e-2)):
    for nc, L, dqk in ((8, 256, 12), (16, 200, 12), (48, 256, 16)):
        for ldlo, tag in ((-0.3, 'mild'), (-2.4, 'deep')):
            q = torch.randn(4, L, dqk, device=dev).abs().to(dt)
            k = torch.randn(4, L, dqk, device=dev).abs().to(dt)
            w = torch.softmax(torch.randn(4, L, nc, device=dev), -1).to(dt)
            ld = (torch.rand(4, L, nc, device=dev) * (-ldlo) + ldlo).to(dt)  # in [ldlo, 0)
            d = rola_perstate_den_gla_triton(q, k, w, ld)
            ref = den_ref(q.double(), k.double(), w.double(), ld.double()).float()
            report(f"gla den fwd {str(dt)[6:]} nc={nc} L={L} dqk={dqk} {tag}", rel(d, ref), dtol)

# ---- 3) grads vs fp64 oracle autograd ----
for nc, L, dqk in ((8, 256, 12), (16, 200, 12)):
    base = dict(device=dev, dtype=torch.float32)
    q = torch.randn(2, L, dqk, **base).abs().requires_grad_()
    k = torch.randn(2, L, dqk, **base).abs().requires_grad_()
    w = torch.softmax(torch.randn(2, L, nc, **base), -1).detach().requires_grad_()
    ld = (torch.rand(2, L, nc, **base) * 0.3 - 0.3).detach().requires_grad_()
    gd = torch.randn(2, L, nc, **base)

    d = rola_perstate_den_gla_triton(q, k, w, ld)
    (d * gd).sum().backward()
    g_tri = [t.grad.clone() for t in (q, k, w, ld)]
    for t in (q, k, w, ld):
        t.grad = None

    q64, k64, w64, ld64 = (t.detach().double().requires_grad_() for t in (q, k, w, ld))
    (den_ref(q64, k64, w64, ld64) * gd.double()).sum().backward()
    g_ref = [t.grad.float() for t in (q64, k64, w64, ld64)]
    for name, gt, gr in zip(('dq', 'dk', 'dw', 'dld'), g_tri, g_ref):
        report(f"gla den grad {name} nc={nc} L={L}", rel(gt, gr), 5e-3)

print("ALL PASS" if fails == 0 else f"{fails} FAILURES")
sys.exit(1 if fails else 0)
