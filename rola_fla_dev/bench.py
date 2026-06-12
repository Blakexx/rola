#!/usr/bin/env python
"""Performance validation of the integrated fork path (Phase P3, on the final artifact).

Contenders at MATCHED total state (nc states × d_qk × d_v per head), RLA + GLA:
  rola      — fla_rola routed chunk_simple_gla (shared-gram Triton, r/w[/g] params)
  vh        — virtual-head on CANONICAL fla: q/k replicated across nc states folded into the
              head axis, v pre-scaled by the write gate, read-combine outside. The honest
              "what you'd do without the routed kernel" baseline (per-state scalar decay maps
              onto simple_gla's per-head g).
  monolith  — canonical fla simple_gla, ONE state per head with d_qk_mono = nc*d_qk (same
              state size, no routing). Lower bound on kernel cost for that much state.

Measured per contender: fwd ms | fwd+bwd ms (CUDA events, median of reps after warmup),
peak GPU mem over fwd+bwd, activation mem (held between fwd and bwd = saved-for-backward).
Config mirrors LM training: B=8 H=8 L=1024 d_qk=d_v=16, bf16 inputs, nc ∈ {16,64,256}.
Run on an IDLE GPU.
"""
import sys
sys.path.insert(0, '/mnt/c/Users/Blake/Documents/VSCode/CLA')
import torch
import fla.ops.simple_gla as fla_sg
import fla_rola.ops.simple_gla as fork_sg

DEV = 'cuda'
B, H, L, DQK, DV = 8, 8, 1024, 16, 16
REPS, WARMUP = 10, 3


def mk(nc, gla, dtype=torch.bfloat16, seed=0):
    torch.manual_seed(seed)
    g = lambda *s: torch.randn(*s, device=DEV, dtype=dtype)
    t = dict(q=g(B, L, H, DQK), k=g(B, L, H, DQK), v=g(B, L, H, DV),
             r=torch.softmax(g(B, L, H, nc).float(), -1).to(dtype),
             w=torch.softmax(g(B, L, H, nc).float(), -1).to(dtype),
             qm=g(B, L, H, nc * DQK), km=g(B, L, H, nc * DQK))   # monolith's matched-state q/k
    if gla:
        alpha = torch.sigmoid(g(B, L, H, nc).float())
        t['ld'] = (1 - t['w'].float() * (1 - alpha)).clamp(min=1e-6).log().clamp(min=-2.5).to(dtype)
    return t


def run_rola(t, gla):
    o, _ = fork_sg.chunk_simple_gla(t['q'], t['k'], t['v'], g=t.get('ld'), scale=1.0,
                                    r=t['r'], w=t['w'])
    return o


def run_kappa(t, gla):
    """kappa normalization: per-state-denominator pre-pass + gate rescale + the SAME fork kernel.
    Fixed kappa=0.5 (cost is exponent-independent); |d| guard keeps timing numerics finite on
    random (non-positive) bench inputs."""
    Bq, Lq, Hq, _ = t['q'].shape
    fold = lambda x: x.permute(0, 2, 1, 3).reshape(Bq * Hq, Lq, x.shape[-1])
    if gla:
        from fla_rola.ops.simple_gla.rola import rola_perstate_den_gla_triton
        d = rola_perstate_den_gla_triton(fold(t['q']), fold(t['k']), fold(t['w']), fold(t['ld']))
    else:
        from fla_rola.ops.simple_gla.rola import rola_perstate_den_triton
        d = rola_perstate_den_triton(fold(t['q']), fold(t['k']), fold(t['w']))
    d = d.view(Bq, Hq, Lq, -1).permute(0, 2, 1, 3)
    rk = t['r'] * (d.abs() + 1e-5).pow(-0.5).to(t['r'].dtype)
    o, _ = fork_sg.chunk_simple_gla(t['q'], t['k'], t['v'], g=t.get('ld'), scale=1.0,
                                    r=rk, w=t['w'])
    return o


def run_vh(t, gla):
    nc = t['r'].shape[-1]
    qv = t['q'].unsqueeze(3).expand(B, L, H, nc, DQK).reshape(B, L, H * nc, DQK)
    kv = t['k'].unsqueeze(3).expand(B, L, H, nc, DQK).reshape(B, L, H * nc, DQK)
    vv = (t['v'].unsqueeze(3) * t['w'].unsqueeze(-1)).reshape(B, L, H * nc, DV)
    g = t['ld'].reshape(B, L, H * nc).float() if gla else None
    o, _ = fla_sg.chunk_simple_gla(qv, kv, vv, g=g, scale=1.0)
    return (o.view(B, L, H, nc, DV) * t['r'].unsqueeze(-1)).sum(3)


def run_monolith(t, gla):
    g = t['ld'][..., 0].float() if gla else None        # one scalar gate per head
    o, _ = fla_sg.chunk_simple_gla(t['qm'], t['km'], t['v'], g=g, scale=1.0)
    return o


def measure(fn, t, gla, grad_keys):
    leaves = {k: (v.detach().requires_grad_() if k in grad_keys else v) for k, v in t.items()}
    ev = lambda: torch.cuda.Event(enable_timing=True)
    fwd_ms, full_ms, peak, act = [], [], 0, 0
    for i in range(WARMUP + REPS):
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        # honest activation accounting: bytes actually RETAINED for backward (saved tensors,
        # deduped by storage) — catches saved input leaves that a "post-fwd delta" misses.
        saved = {}
        def pack(x):
            if x.is_cuda:
                saved[x.untyped_storage().data_ptr()] = x.untyped_storage().nbytes()
            return x
        s0, s1, s2 = ev(), ev(), ev()
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
            s0.record()
            o = fn(leaves, gla)
            s1.record()
        torch.cuda.synchronize()
        a = sum(saved.values())
        loss = o.float().sum()
        loss.backward()
        s2.record(); torch.cuda.synchronize()
        if i >= WARMUP:
            fwd_ms.append(s0.elapsed_time(s1)); full_ms.append(s0.elapsed_time(s2))
            peak = max(peak, torch.cuda.max_memory_allocated() - base); act = max(act, a)
        for v in leaves.values():
            if v.grad is not None:
                v.grad = None
    med = lambda xs: sorted(xs)[len(xs) // 2]
    return med(fwd_ms), med(full_ms), peak / 2**20, act / 2**20


if __name__ == '__main__':
    print(f"B={B} H={H} L={L} d_qk={DQK} d_v={DV} bf16 | median of {REPS} reps | mem in MiB")
    print("STAGE 2 — post-projection kernel, matched total state (nc*d_qk*d_v per head):")
    hdr = f"{'variant':5s} {'nc':>4s} {'impl':9s} {'fwd ms':>8s} {'fwd+bwd ms':>11s} {'peak MiB':>9s} {'act MiB':>8s}"
    print(hdr); print('-' * len(hdr))
    KER = {}
    for gla in (False, True):
        variant = 'GLA' if gla else 'RLA'
        for nc in (16, 64, 256):
            t = mk(nc, gla)
            gk = ['q', 'k', 'v', 'r', 'w'] + (['ld'] if gla else [])
            for name, fn in (('rola', run_rola), ('vh', run_vh), ('monolith', run_monolith)):
                try:
                    f, fb, pk, ac = measure(fn, t, gla, gk if name != 'monolith' else ['qm', 'km', 'v'])
                    KER[(variant, nc, name)] = fb
                    print(f"{variant:5s} {nc:>4d} {name:9s} {f:8.2f} {fb:11.2f} {pk:9.0f} {ac:8.0f}", flush=True)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"{variant:5s} {nc:>4d} {name:9s} {'OOM':>8s}", flush=True)
    # STAGE 1 — projection cost (timed GEMMs). The monolith pays nc-scaling here TOO: q/k
    # projections of width H*nc*DQK vs rola's H*DQK (+H*nc routers). vh shares rola's projections
    # (it's an implementation of the same model). Total = stage1 + stage2 per impl.
    import torch.nn as nn
    DM = 512
    print("\nSTAGE 1 — projections (timed; bf16 GEMMs, d_model=512), and stage1+stage2 totals:")
    hdr2 = (f"{'nc':>4s} {'impl':9s} {'params M':>9s} {'proj f+b ms':>12s} {'proj act MiB':>13s}"
            f" {'TOTAL f+b ms (RLA/GLA)':>24s}")
    print(hdr2); print('-' * len(hdr2))
    for nc in (16, 64, 256):
        torch.manual_seed(5)
        x = torch.randn(B, L, DM, device=DEV, dtype=torch.bfloat16)
        cfgs = {
            'rola': [DM * H * DQK] * 2 + [DM * H * DV] + [DM * H * nc] * 2,   # q,k,v + r,w routers
            'monolith': [DM * H * nc * DQK] * 2 + [DM * H * DV],              # wide q,k + v
        }
        mods = {n: nn.ModuleList([nn.Linear(DM, w // DM, bias=False) for w in ws]).to(DEV, torch.bfloat16)
                for n, ws in cfgs.items()}
        for name in ('rola', 'monolith'):
            m = mods[name]
            params = sum(p.numel() for p in m.parameters())
            ev = lambda: torch.cuda.Event(enable_timing=True)
            times, act = [], 0
            for i in range(WARMUP + REPS):
                xi = x.detach().requires_grad_()
                saved = {}
                def pack(t):
                    if t.is_cuda:
                        saved[t.untyped_storage().data_ptr()] = t.untyped_storage().nbytes()
                    return t
                torch.cuda.synchronize(); s0, s1 = ev(), ev()
                with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
                    s0.record()
                    outs = [lin(xi) for lin in m]
                    loss = sum(o.float().sum() for o in outs)
                loss.backward(); s1.record(); torch.cuda.synchronize()
                if i >= WARMUP:
                    times.append(s0.elapsed_time(s1)); act = max(act, sum(saved.values()))
                m.zero_grad(set_to_none=True)
            pm = sorted(times)[len(times) // 2]
            kr = KER.get(('RLA', nc, name)); kg = KER.get(('GLA', nc, name))
            tot = f"{(pm + kr):.2f} / {(pm + kg):.2f}" if kr and kg else "-"
            print(f"{nc:>4d} {name:9s} {params/1e6:9.2f} {pm:12.2f} {act/2**20:13.0f} {tot:>24s}", flush=True)
        print(f"{'':4s} (vh uses rola's projections + its kernel numbers above)")
