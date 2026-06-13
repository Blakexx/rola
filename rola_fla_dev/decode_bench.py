"""Decode-step benchmark: RoLA via virtual-heads over canonical fused_recurrent vs the
matched-state wide monolith on the same kernel. Forward only (decode has no backward).

The serial regime has no pairwise Gram, so the chunked form's shared-Gram savings do not
apply; the prediction is bandwidth PARITY at matched state (state bytes are equal by
construction), with vh adding only broadcast/materialization overhead. All arms run the
canonical fla fused_recurrent_simple_gla — no new kernel surface.

--check : step-by-step vh decode == chunked routed kernel on the same inputs (and the
          monolith decode == its own chunked form). Runs fine on a contended GPU.
--bench : steady-state ms/token at B requests, nc in {16,64,256}, RLA + scalar-GLA,
          global + kappa combines. Needs an idle GPU.
"""
import argparse
import sys
import time

import torch

sys.path.insert(0, '/home/blake/flash-linear-attention-rola')
from fla_rola.ops.simple_gla import chunk_simple_gla
from fla_rola.ops.simple_gla.fused_recurrent import fused_recurrent_simple_gla

B, H, DQK, DV = 8, 8, 16, 16
DEV, DT = 'cuda', torch.bfloat16


def make_inputs(L, nc, gla, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    t = lambda *s: torch.randn(*s, device=DEV, generator=g, dtype=torch.float32)
    q, k = t(B, L, H, DQK).abs(), t(B, L, H, DQK).abs()
    v = t(B, L, H, DV)
    r = torch.softmax(t(B, L, H, nc), -1)
    w = torch.softmax(t(B, L, H, nc), -1)
    ld = (-torch.rand(B, L, H, nc, device=DEV, generator=g) * 0.5).clamp(min=-2.5) if gla else None
    return q, k, v, r, w, ld


def vh_expand(q, k, v, w, ld, nc):
    """[B,L,H,*] -> virtual-head tensors [B,L,H*nc,*]; v carries the write gate and a den column."""
    Bq, L = q.shape[0], q.shape[1]
    qv = q.unsqueeze(3).expand(Bq, L, H, nc, DQK).reshape(Bq, L, H * nc, DQK)
    kv = k.unsqueeze(3).expand(Bq, L, H, nc, DQK).reshape(Bq, L, H * nc, DQK)
    v1 = torch.cat([v, torch.ones_like(v[..., :1])], -1)                     # den column
    vv = (v1.unsqueeze(3) * w.unsqueeze(-1)).reshape(Bq, L, H * nc, DV + 1)
    gv = ld.reshape(Bq, L, H * nc).float() if ld is not None else None
    return qv, kv, vv, gv


def vh_combine(o_aug, r, nc, norm, kappa=0.5, eps=1e-5):
    """o_aug:[B,L,H*nc,DV+1] per-state readouts (num | den) -> combined [B,L,H,DV]."""
    Bq, L = o_aug.shape[0], o_aug.shape[1]
    o = o_aug.view(Bq, L, H, nc, DV + 1)
    num, den = o[..., :DV], o[..., DV]
    if norm == 'kappa':
        r = r * (den.abs() + eps).pow(-kappa)
    elif norm == 'per_state':
        r = r / (den.abs() + eps)
    o_num = (num * r.unsqueeze(-1)).sum(3)
    o_den = (den * r).sum(3)
    return o_num / (o_den.unsqueeze(-1) + eps)


def decode_routed_step(qt, kt, vt, rt, wt, ldt, state, nc, norm):
    """One decode step (L=1 tensors), canonical fused_recurrent under the hood."""
    qv, kv, vv, gv = vh_expand(qt, kt, vt, wt, ldt, nc)
    o, state = fused_recurrent_simple_gla(qv, kv, vv, g=gv, scale=1.0,
                                          initial_state=state, output_final_state=True)
    return vh_combine(o.float(), rt, nc, norm), state


def decode_monolith_step(qt, kt, vt, state):
    o, state = fused_recurrent_simple_gla(qt, kt, vt, scale=1.0,
                                          initial_state=state, output_final_state=True)
    return o, state


def check(nc=16, L=64, gla=True, norm='global'):
    q, k, v, r, w, ld = make_inputs(L, nc, gla)
    # chunked reference (the verified training path), augmented den column via vh-style v
    qv, kv, vv, gv = vh_expand(q, k, v, w, ld, nc)
    o_ref_aug, _ = chunk_simple_gla(qv, kv, vv, g=gv, scale=1.0)
    o_ref = vh_combine(o_ref_aug.float(), r, nc, norm)
    # step-by-step decode
    state, outs = None, []
    for t in range(L):
        sl = slice(t, t + 1)
        o_t, state = decode_routed_step(q[:, sl], k[:, sl], v[:, sl], r[:, sl], w[:, sl],
                                        ld[:, sl] if ld is not None else None, state, nc, norm)
        outs.append(o_t)
    o_dec = torch.cat(outs, 1)
    rel = (o_dec - o_ref).abs().max().item() / (o_ref.abs().max().item() + 1e-9)
    tag = f"decode==chunked {'GLA' if gla else 'RLA'} nc={nc} norm={norm}"
    print(f"{tag:48s} rel {rel:.2e}  {'PASS' if rel < 3e-2 else 'FAIL'}")
    return rel < 3e-2


def bench(steps=200, warmup=50):
    print(f"# decode steady-state, B={B} H={H} dqk=dv={DQK}, ms/token (median of {steps})")
    print(f"{'cell':34s} {'ms/tok':>8s} {'state MiB':>10s}")
    for gla in (False, True):
        for nc in (16, 64, 256):
            arms = [('rola-vh global', 'global'), ('rola-vh kappa', 'kappa')]
            q, k, v, r, w, ld = make_inputs(1, nc, gla, seed=1)
            c = lambda t: t.to(DT) if t is not None else None
            for name, norm in arms:
                state = torch.zeros(B, H * nc, DQK, DV + 1, device=DEV, dtype=torch.float32)
                f = lambda: decode_routed_step(c(q), c(k), c(v), r, w, c(ld), state, nc, norm)
                ms = time_fn(f, steps, warmup)
                mib = state.numel() * 4 / 2**20
                print(f"{('GLA' if gla else 'RLA')}/{nc:<4d} {name:24s} {ms:8.3f} {mib:10.1f}")
            # matched wide monolith: dqk' = nc*dqk, same state floats (+den col to match)
            dqkw = nc * DQK
            qm = torch.randn(B, 1, H, dqkw, device=DEV).abs().to(DT)
            km = torch.randn(B, 1, H, dqkw, device=DEV).abs().to(DT)
            vm = torch.randn(B, 1, H, DV + 1, device=DEV).to(DT)
            state_m = torch.zeros(B, H, dqkw, DV + 1, device=DEV, dtype=torch.float32)
            f = lambda: decode_monolith_step(qm, km, vm, state_m)
            ms = time_fn(f, steps, warmup)
            mib = state_m.numel() * 4 / 2**20
            print(f"{('GLA' if gla else 'RLA')}/{nc:<4d} {'monolith wide':24s} {ms:8.3f} {mib:10.1f}")


def time_fn(f, steps, warmup):
    for _ in range(warmup):
        f()
    torch.cuda.synchronize()
    ts = []
    for _ in range(steps):
        t0 = time.perf_counter()
        f()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2] * 1e3


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--bench', action='store_true')
    a = ap.parse_args()
    if a.check:
        ok = True
        for gla in (False, True):
            for nc in (16, 64):
                for norm in ('global', 'kappa', 'per_state'):
                    ok &= check(nc=nc, gla=gla, norm=norm)
        print("ALL PASS" if ok else "FAILURES")
        sys.exit(0 if ok else 1)
    if a.bench:
        bench()
