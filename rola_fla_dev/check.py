#!/usr/bin/env python
"""Mechanical no-regression + routing-correctness gate for implementing RoLA in the FLA fork.

Run after EVERY edit to the fork (`fla_rola`):
    PYTHONPATH=/mnt/c/Users/Blake/Documents/VSCode/CLA python rola_fla_dev/check.py

Two gates, both live (no stale golden — canonical `fla` is the source of truth):

  (1) REGRESSION  [SAFETY, hard gate]
      For every canonical baseline kernel, `fla_rola`'s UNROUTED path must stay numerically
      identical to canonical `fla` — forward AND all input grads. Any divergence means a routing
      edit broke backward-compatibility (the baselines run pristine `fla`, so the fork's unrouted
      path must match it bit-for-bit-ish). FAIL here => exit 1. This is the safety the user demanded.

  (2) ROUTING  [CORRECTNESS TARGET, soft gate]
      `fla_rola`'s ROUTED path (the new r/w/decay params) must match the VERIFIED oracle in
      `rola_kernels` (recurrent_gla / ref_global / the analytic backward) — forward + all grads.
      SKIP until the routed API exists; PASS once each step lands. Reported, not exit-gated
      (it's the WIP target) unless --strict.

Usage:  check.py [--strict]   (--strict also exits 1 on routing FAIL)
"""
import sys, os
sys.path.insert(0, '/mnt/c/Users/Blake/Documents/VSCode/CLA')   # rola_kernels (oracle)
import torch
import fla            # canonical, pristine — baselines
import fla_rola       # the fork under construction
import rola_kernels   # verified oracle (recurrent_gla, ref_global, _chunked_gla_safe, analytic bwd)

DEV = 'cuda'
STRICT = '--strict' in sys.argv
TOL_REGR = 1e-4       # unrouted fork vs canonical: expect ~0 (same code); TF32 run-to-run slack
TOL_ROUTE = 2e-2      # routed vs oracle: TF32 reduction-order noise


def mk(B=2, H=4, L=256, D=16, seed=0):
    torch.manual_seed(seed)
    g = lambda *s: torch.randn(*s, device=DEV, dtype=torch.float32)
    return dict(q=g(B, L, H, D), k=g(B, L, H, D), v=g(B, L, H, D),
                gs=torch.nn.functional.logsigmoid(g(B, L, H)),        # scalar gate (simple_gla, gdn)
                gk=torch.nn.functional.logsigmoid(g(B, L, H, D)),     # per-channel gate (gla)
                beta=torch.sigmoid(g(B, L, H)))


# --- canonical baseline kernels: (name, call(module,ins)->out, grad_input_keys) ---
def _simple_gla(m, i): return m.ops.simple_gla.chunk_simple_gla(i['q'], i['k'], i['v'], i['gs'], scale=1.0)[0]
def _gla(m, i):        return m.ops.gla.chunk_gla(i['q'], i['k'], i['v'], i['gk'], scale=1.0)[0]
def _linear(m, i):     return m.ops.linear_attn.fused_chunk_linear_attn(i['q'], i['k'], i['v'], normalize=False, scale=1.0)[0]
def _gdn(m, i):        return m.ops.gated_delta_rule.chunk_gated_delta_rule(i['q'], i['k'], i['v'], i['gs'], i['beta'], use_qk_l2norm_in_kernel=True)[0]
BASELINES = [('simple_gla', _simple_gla, ['q', 'k', 'v', 'gs']),
             ('gla', _gla, ['q', 'k', 'v', 'gk']),
             ('linear', _linear, ['q', 'k', 'v']),
             ('gated_delta', _gdn, ['q', 'k', 'v', 'gs', 'beta'])]


def _rel(a, b): return (a - b).abs().max().item() / (b.abs().max().item() + 1e-9)


def _fwd_bwd(call, module, ins, gkeys, dO):
    leaves = {k: (v.detach().requires_grad_() if k in gkeys else v) for k, v in ins.items()}
    o = call(module, leaves)
    grads = torch.autograd.grad(o, [leaves[k] for k in gkeys], dO)
    return o.detach(), grads


def regression():
    print("=" * 78 + "\n(1) REGRESSION — fla_rola(unrouted) must == canonical fla (baselines pristine)\n" + "=" * 78)
    allok = True
    for name, call, gkeys in BASELINES:
        ins = mk(seed=hash(name) % 1000)
        torch.manual_seed(123)
        oc = call(fla, {k: v for k, v in ins.items()})
        dO = torch.randn_like(oc)
        oc, gc = _fwd_bwd(call, fla, ins, gkeys, dO)
        of, gf = _fwd_bwd(call, fla_rola, ins, gkeys, dO)
        rf = _rel(of, oc); rg = max(_rel(a, b) for a, b in zip(gf, gc))
        ok = max(rf, rg) < TOL_REGR; allok &= ok
        print(f"  {name:14s} fwd={rf:.1e} grad={rg:.1e}   {'PASS' if ok else '*** FAIL ***'}")
    return allok


# --- un-normalized routed-numerator oracles (FLA convention: caller normalizes) ---
def _oracle_rla(q, k, v, r, w):
    L = q.shape[1]
    G = torch.einsum('btd,bsd->bts', q, k)
    R = torch.einsum('btc,bsc->bts', r, w)
    causal = torch.tril(torch.ones(L, L, device=q.device, dtype=q.dtype))
    return torch.einsum('bts,bsv->btv', G * R * causal, v)


def _oracle_gla(q, k, v, r, w, ld):
    L = q.shape[1]
    A = torch.cumsum(ld, dim=1)
    G = torch.einsum('btd,bsd->bts', q, k)
    decay = torch.exp(A[:, :, None, :] - A[:, None, :, :])
    D = torch.einsum('btc,bsc,btsc->bts', r, w, decay)
    causal = torch.tril(torch.ones(L, L, device=q.device, dtype=q.dtype))
    return torch.einsum('bts,bsv->btv', G * D * causal, v)


def routing():
    print("=" * 78 + "\n(2) ROUTING — fla_rola(routed) must == rola_kernels oracle (correctness target)\n" + "=" * 78)
    import inspect
    chunk_simple_gla = fla_rola.ops.simple_gla.chunk_simple_gla
    if 'r' not in inspect.signature(chunk_simple_gla).parameters:
        for nm in ('RoLA-RLA', 'RoLA-GLA'):
            print(f"  {nm:24s} SKIP — routing params (r,w) not on chunk_simple_gla yet")
        return True
    allok = True
    for variant in ('RoLA-RLA', 'RoLA-GLA'):
        for nc in (16, 64, 256):
            torch.manual_seed(nc + (0 if variant == 'RoLA-RLA' else 7))
            # K=12 < BD=16 deliberately: tests the buffer-padding rows (a dqk==BD-only gate
            # missed an uninitialized-row reduction bug in the GLA dLam assembly, 2026-06-10).
            B, L, K, V = 2, 128, 12, 12
            g = lambda *s: torch.randn(*s, device=DEV, dtype=torch.float32)
            q, k, v = g(B, L, K), g(B, L, K), g(B, L, V)
            r = torch.softmax(g(B, L, nc), -1)
            w = torch.softmax(g(B, L, nc), -1)
            # ld pre-floored at -2.5 (the Triton path's fp32-safety floor, identity here) so the
            # fork and the unfloored oracle compute the same function and the comparison is exact.
            ld = (1 - w.detach() * (1 - torch.sigmoid(g(B, L, nc)))).clamp(min=1e-6).log().clamp(min=-2.5) if variant == 'RoLA-GLA' else None
            leaves = [q, k, v, r, w] + ([ld] if ld is not None else [])
            for t in leaves:
                t.requires_grad_()
            torch.manual_seed(555)
            dO = g(B, L, V)
            # fork routed (H=1): [B,L,1,*]
            ar = [t.unsqueeze(2) if t.dim() == 3 else t for t in (q, k, v, r, w)]
            gld = ld.unsqueeze(2) if ld is not None else None
            of = chunk_simple_gla(ar[0], ar[1], ar[2], g=gld, scale=1.0, r=ar[3], w=ar[4])[0].squeeze(2)
            gf = torch.autograd.grad(of, leaves, dO, retain_graph=False)
            # oracle
            oo = _oracle_rla(q, k, v, r, w) if ld is None else _oracle_gla(q, k, v, r, w, ld)
            go = torch.autograd.grad(oo, leaves, dO)
            rf = _rel(of, oo); rg = max(_rel(a, b) for a, b in zip(gf, go))
            ok = max(rf, rg) < TOL_ROUTE; allok &= ok
            print(f"  {variant:9s} nc={nc:<4d} fwd={rf:.1e} grad={rg:.1e}   {'PASS' if ok else '*** FAIL ***'}")
    return allok


if __name__ == '__main__':
    torch.cuda.init()
    r_ok = regression()
    rt_ok = routing()
    print("\n" + ("REGRESSION OK — baselines unchanged." if r_ok else "REGRESSION FAILED — a routing edit broke a baseline. REVERT."))
    sys.exit(0 if r_ok and (rt_ok or not STRICT) else 1)
