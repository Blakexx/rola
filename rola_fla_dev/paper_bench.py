#!/usr/bin/env python
"""Paper-grade kernel-efficiency benchmark → rola_paper/.

Protocol (fixes the thermal confound seen in ad-hoc runs):
  * ROUND-ROBIN interleaving: each timing rep cycles through ALL impls before the next rep,
    so clock/thermal drift biases every impl equally (between-impl ratios are drift-free).
  * median + IQR over R reps, CUDA events, warmup excluded; activation bytes via
    saved_tensors_hooks (deduped by storage); peak via reset_peak_memory_stats per rep.
  * Two stages: (1) projections (timed GEMMs at d_model=512), (2) post-projection kernel at
    matched total state nc*d_qk*d_v per head. Stage totals reported.
Emits: rola_paper/kernel_efficiency.csv (raw), .md and .tex tables.
"""
import sys, csv, statistics
sys.path.insert(0, '/mnt/c/Users/Blake/Documents/VSCode/CLA')
sys.path.insert(0, '/mnt/c/Users/Blake/Documents/VSCode/CLA/rola_fla_dev')
import torch

sys.argv = [sys.argv[0]]
import bench as B  # reuse mk/run_* + shape constants from the working bench

OUT = '/home/blake/rola_paper/results/kernel_bench'
R, WARM = 15, 4
IMPLS = [('rola', B.run_rola), ('vh', B.run_vh), ('monolith', B.run_monolith)]
IMPLS_RLA = IMPLS + [('kappa', B.run_kappa)]   # kappa: both variants (GLA den via decayed mass scan)


def one_rep(fn, leaves, gla, dO_seed):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    saved = {}
    def pack(x):
        if x.is_cuda:
            saved[x.untyped_storage().data_ptr()] = x.untyped_storage().nbytes()
        return x
    e0, e1, e2 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
        e0.record()
        o = fn(leaves, gla)
        e1.record()
    torch.cuda.synchronize()
    loss = o.float().sum()
    loss.backward()
    e2.record(); torch.cuda.synchronize()
    for v in leaves.values():
        if v.grad is not None:
            v.grad = None
    return (e0.elapsed_time(e1), e0.elapsed_time(e2),
            (torch.cuda.max_memory_allocated() - base) / 2**20, sum(saved.values()) / 2**20)


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    stage1_only = '--stage1-only' in sys.argv   # kernel data already final → re-measure projections only
    rows = []
    gpu = torch.cuda.get_device_name()
    for gla in (() if stage1_only else (False, True)):
        var = 'GLA' if gla else 'RLA'
        for nc in (16, 64, 256):
            t = B.mk(nc, gla)
            impls = IMPLS_RLA   # kappa benched for BOTH variants (GLA den kernel landed)
            gk = ['q', 'k', 'v', 'r', 'w'] + (['ld'] if gla else [])
            states = {}
            for name, fn in impls:
                keys = gk if name != 'monolith' else ['qm', 'km', 'v']
                states[name] = {k: (v.detach().requires_grad_() if k in keys else v) for k, v in t.items()}
            data = {n: [] for n, _ in impls}
            for r in range(WARM + R):                      # interleaved rounds
                for name, fn in impls:
                    rec = one_rep(fn, states[name], gla, r)
                    if r >= WARM:
                        data[name].append(rec)
            for name, _ in impls:
                f, fb, pk, ac = zip(*data[name])
                med = lambda x: statistics.median(x)
                iqr = lambda x: (statistics.quantiles(x, n=4)[2] - statistics.quantiles(x, n=4)[0])
                rows.append(dict(variant=var, nc=nc, impl=name,
                                 fwd_ms=round(med(f), 3), fwd_iqr=round(iqr(f), 3),
                                 fb_ms=round(med(fb), 3), fb_iqr=round(iqr(fb), 3),
                                 peak_mib=round(max(pk)), act_mib=round(max(ac))))
                print(rows[-1], flush=True)
    # stage 1: projections, same interleaving
    import torch.nn as nn
    DM = 512
    proj_rows = []
    for nc in (16, 64, 256):
        torch.manual_seed(5)
        x = torch.randn(B.B, B.L, DM, device='cuda', dtype=torch.bfloat16)
        cfgs = {'rola': [DM * B.H * B.DQK] * 2 + [DM * B.H * B.DV] + [DM * B.H * nc] * 2,
                'monolith': [DM * B.H * nc * B.DQK] * 2 + [DM * B.H * B.DV]}
        mods = {n: nn.ModuleList([nn.Linear(DM, w // DM, bias=False) for w in ws]).to('cuda', torch.bfloat16)
                for n, ws in cfgs.items()}
        data = {n: [] for n in cfgs}
        for r in range(WARM + R):
            for name, m in mods.items():
                xi = x.detach().requires_grad_()
                torch.cuda.synchronize()
                e0, e1, e2 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
                e0.record()
                outs = [lin(xi) for lin in m]
                e1.record()
                sum(o.float().sum() for o in outs).backward()
                e2.record(); torch.cuda.synchronize()
                m.zero_grad(set_to_none=True)
                if r >= WARM:
                    data[name].append((e0.elapsed_time(e1), e0.elapsed_time(e2)))
        for name, m in mods.items():
            f, fb = zip(*data[name])
            proj_rows.append(dict(nc=nc, impl=name,
                                  params_m=round(sum(p.numel() for p in m.parameters()) / 1e6, 2),
                                  proj_fwd_ms=round(statistics.median(f), 3),
                                  proj_fb_ms=round(statistics.median(fb), 3)))
            print(proj_rows[-1], flush=True)
    # emit CSV
    if stage1_only:
        rows = list(csv.DictReader(open(f'{OUT}/kernel_efficiency.csv')))   # kernel data unchanged
    else:
        with open(f'{OUT}/kernel_efficiency.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    with open(f'{OUT}/projection_cost.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(proj_rows[0].keys()))
        w.writeheader(); w.writerows(proj_rows)
    # emit LaTeX (stage 2 + totals)
    proj = {(p['nc'], p['impl']): p for p in proj_rows}
    get = lambda v, n, i: next(r for r in rows if r['variant'] == v and int(r['nc']) == n and r['impl'] == i)
    with open(f'{OUT}/kernel_efficiency.tex', 'w') as fh:
        fh.write("% generated by rola_fla_dev/paper_bench.py — GPU: " + gpu + "\n")
        fh.write("%% B=%d H=%d L=%d d_qk=%d d_v=%d bf16; median of %d interleaved reps\n"
                 % (B.B, B.H, B.L, B.DQK, B.DV, R))
        fh.write("\\begin{tabular}{llrrrrrrr}\n\\toprule\n")
        fh.write("& & \\multicolumn{2}{c}{kernel (ms)} & \\multicolumn{2}{c}{memory (MiB)} & "
                 "\\multicolumn{3}{c}{incl. projections} \\\\\n")
        fh.write("variant/$n_c$ & impl & fwd & fwd+bwd & peak & activ. & total fwd & total f+b & params (M) \\\\\n\\midrule\n")
        for var in ('RLA', 'GLA'):
            for nc in (16, 64, 256):
                for impl in ('rola', 'kappa', 'vh', 'monolith'):
                    r = get(var, nc, impl)
                    pj = proj.get((nc, 'rola' if impl != 'monolith' else 'monolith'))
                    totf = float(r['fwd_ms']) + pj['proj_fwd_ms']
                    tot = float(r['fb_ms']) + pj['proj_fb_ms']
                    fh.write(f"{var}/{nc} & {impl} & {float(r['fwd_ms']):.2f} & {float(r['fb_ms']):.2f} & "
                             f"{r['peak_mib']} & {r['act_mib']} & {totf:.2f} & {tot:.2f} & {pj['params_m']:.2f} \\\\\n")
            fh.write("\\midrule\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    print(f"\nWROTE {OUT}/kernel_efficiency.csv, projection_cost.csv, kernel_efficiency.tex  [{gpu}]")


if __name__ == '__main__':
    main()
