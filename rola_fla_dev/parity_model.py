#!/usr/bin/env python
"""Phase D2 — full-model parity: rola.py kernels on the fla_rola Triton path vs the eager torch
path. Same module, same weights, same inputs; flip the path by monkeypatching the module flags.
PASS = fwd + all input grads match to TF32 noise. Run after any rola.py wiring change."""
import sys
sys.path.insert(0, '/mnt/c/Users/Blake/Documents/VSCode/CLA')
import torch
import rola

DEV, TOL = 'cuda', 2e-2


def run(kernel, x, q, k, v, wg, rg, use_triton):
    rola._USE_TRITON = rola._USE_TRITON_RLA = use_triton
    leaves = [t.detach().clone().requires_grad_() for t in (q, k, v, wg, rg)]
    out = kernel(x, *leaves)
    torch.manual_seed(99)
    dO = torch.randn_like(out)
    grads = torch.autograd.grad(out, leaves, dO)
    return out.detach(), grads


def rel(a, b):
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-9)


def case(name, kernel, nc, d_model=64, H=2, d_qk=16, d_v=16, B=2, L=256):
    torch.manual_seed(1234)
    x = torch.randn(B, L, d_model, device=DEV)
    q = torch.randn(B, L, H, d_qk, device=DEV)
    k = torch.randn(B, L, H, d_qk, device=DEV)
    v = torch.randn(B, L, H, d_v, device=DEV)
    wg = torch.softmax(torch.randn(B, L, H, nc, device=DEV), -1)
    rg = torch.softmax(torch.randn(B, L, H, nc, device=DEV), -1)
    o_t, g_t = run(kernel, x, q, k, v, wg, rg, True)
    kernel._checked = True   # skip re-running the first-use self-check on the eager pass
    o_e, g_e = run(kernel, x, q, k, v, wg, rg, False)
    rf = rel(o_t, o_e)
    rg_ = max(rel(a, b) for a, b in zip(g_t, g_e))
    ok = max(rf, rg_) < TOL
    print(f"  {name:34s} fwd={rf:.1e} grad={rg_:.1e}   {'PASS' if ok else '*** FAIL ***'}")
    return ok


if __name__ == '__main__':
    print("=" * 78 + "\nMODEL PARITY — rola.py: fla_rola Triton path vs eager torch path\n" + "=" * 78)
    allok = True
    for nc in (16, 64):
        ak = rola.AdditiveKernel(64, 2, nc, 16, 16, phi='elu', state_norm='global').to(DEV)
        allok &= case(f"RLA  global elu        nc={nc}", ak, nc)
        gk_raw = rola.ScalarGLAKernel(64, 2, nc, 16, 16, normalized=False).to(DEV)
        allok &= case(f"GLA  raw               nc={nc}", gk_raw, nc)
        gk_nrm = rola.ScalarGLAKernel(64, 2, nc, 16, 16, normalized=True).to(DEV)
        allok &= case(f"GLA  normalized        nc={nc}", gk_nrm, nc)
    # bf16 autocast smoke: the LM regime (routers promote to fp32 under autocast; kernel must cast)
    torch.manual_seed(7)
    ak = rola.AdditiveKernel(64, 2, 16, 16, 16, phi='elu', state_norm='global').to(DEV)
    ak._checked = True
    rola._USE_TRITON = rola._USE_TRITON_RLA = True
    x = torch.randn(2, 256, 64, device=DEV)
    q, k, v = (torch.randn(2, 256, 2, 16, device=DEV) for _ in range(3))
    wg = torch.softmax(torch.randn(2, 256, 2, 16, device=DEV), -1)
    rg = torch.softmax(torch.randn(2, 256, 2, 16, device=DEV), -1)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        out = ak(x, q.requires_grad_(), k, v, wg, rg)
        out.sum().backward()
    print(f"  {'bf16 autocast smoke (RLA)':34s} out={tuple(out.shape)} finite={bool(out.isfinite().all())}   "
          f"{'PASS' if out.isfinite().all() else '*** FAIL ***'}")
    allok &= bool(out.isfinite().all())
    print("\n" + ("MODEL PARITY OK." if allok else "MODEL PARITY FAILED."))
    sys.exit(0 if allok else 1)
