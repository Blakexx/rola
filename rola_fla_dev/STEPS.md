# Implementing RoLA in the FLA fork — mechanical, auditable steps

The fork lives at `/home/blake/flash-linear-attention-rola` (package `fla_rola`). Canonical `fla` is
installed alongside it and stays **pristine** — baselines import `fla`, RoLA imports `fla_rola`.

**The invariant, enforced after every single edit:**
```
PYTHONPATH=/mnt/c/Users/Blake/Documents/VSCode/CLA python rola_fla_dev/check.py
```
- **(1) REGRESSION must stay `0e+00` PASS for all four baselines** (simple_gla, gla, linear, gated_delta),
  forward + grads. This is the safety rail: it proves the routing additions never touched the unrouted
  path. If a step turns any of these red, the change is wrong — **revert it**, do not proceed.
- **(2) ROUTING** flips from SKIP → PASS one target at a time as the steps below land. PASS = the fork's
  routed path matches the verified oracle (`rola_kernels` math / the Phase-A torch reference) to noise.

Rule: **one step = one small diff + the two gates green.** Never batch steps. Never edit a baseline
kernel's behavior — routing is purely additive (`r`/`w`/`g` default `None` ⇒ identical to canonical).

---

## Phase 0 — fork + gate  ✅ DONE
- [x] Fork cloned, package renamed `fla`→`fla_rola`, installed `-e`; coexists with canonical `fla`.
- [x] `check.py` regression gate green: `fla_rola` unrouted == canonical `fla`, fwd+grad, `0e+00`, all 4 baselines.

## Phase A — shared-gram routed forward, in place  ✅ DONE
Goal: routing added as **additive params on `chunk_simple_gla`** (`r`=read gate, `w`=write gate `[B,T,H,nc]`;
`g` = per-state log-decay `[B,T,H,nc]` for GLA, or `None` ⇒ RLA), routing through the **shared-gram algorithm**
(content gram once per chunk, no q/k replication, no L×nc). `r=None` ⇒ untouched canonical path.
- [x] `chunk.py`: `_rola_chunk_core` (RLA = vectorized chunk-parallel cumsum scan; GLA = per-state decayed
  scan over chunks, mirrors `rola_kernels.chunked_parallel`/`chunked_gla`) + `chunk_rola_fwd` (fold `[B,T,H,*]`→
  `[BH,T,*]`). `chunk_simple_gla(..., r=, w=)` branches to it; returns the un-normalized readout (caller
  normalizes via the ones-column, as rola.py already does). Torch ops ⇒ **autograd gives exact q,k,v,r,w,g grads**.
- [x] Gate green: regression `0e+00` (4 baselines, fwd+grad); routing fwd+all-grads ~1e-7 vs oracle, RLA+GLA, nc=16/64.

**Correctness is now locked.** Everything below is **performance** (replace the torch chunk loop with Triton),
guarded on every edit by: regression `0e+00` AND routing PASS vs the Phase-A torch path.

## Phase B — Triton kernels (shared-gram, gram-once)
Self-contained in the fork (`fla_rola/ops/simple_gla/rola.py` — no dependency on our private `rola_kernels`).
Kernels ported verbatim from the verified reference; they internally augment v with a ones-column, so the
fork returns `[..,:dv]` (numerator) and the backward pads the incoming grad with a zero den-column — exactly
the verified kernel restricted to the real v. `chunk_rola_fwd` dispatches CUDA+`K≤64` → Triton, else torch.
- [x] **B1/C1-RLA — RLA Triton fwd + fused Triton bwd (DONE).** `_rola_fwd_tiled` + `_rola_bwd_qr`/`_rola_bwd_kwv`
  wrapped in `_RoLARLAFn` (un-normalized contract). → gate: routing **RoLA-RLA** PASS fwd+all-grads at
  nc=16/64/**256** (1.8e-3, TF32 noise — confirms the Triton path is live and tiles over states); regression `0e+00`.
- [x] **B2/C1-GLA — GLA Triton fwd + fused Triton bwd (DONE).** `_rola_gla_fwd_tiled` + `_gla_bwd_qr`/`_gla_bwd_kwv`
  + torch dld assembly in `_RoLAGLAFn`; the `g!=None` path dispatches to it (chunk=32, fp32-safe decay floor
  `_GLA_FLOOR=-2.5` — gate inputs pre-floored so the clamp is identity vs the unfloored oracle). → gate:
  routing **RoLA-GLA** PASS fwd+all-grads (incl. dld) nc=16/64/256 (~2e-3 TF32); regression `0e+00`.
  *Audit note:* kernel bodies verified semantically identical to the reference by mechanical diff; one
  transcription bug found+fixed in the hand-written wrapper (`_gla_bwd` missing `dq.sum(1)`) — caught by the gate.

## Phase D — adopt in the model
- [x] **D1 (DONE).** `rola.py` now imports `fla_rola.ops.simple_gla.chunk_simple_gla` (`_fla_routed`) instead of
  `rola_kernels`. Model tensors pass in FLA's native `[B,T,H,*]` layout (no fold round-trip); normalization =
  ones-column augmentation on the fork's un-normalized readout, done in rola.py. Dtype handling
  (`_triton_compute_dtype` + autocast-off) preserved; `-2.5` decay floor matches prior Triton behavior.
  Synced to zoology. `rola_kernels` now referenced ONLY by `check.py` (oracle).
- [x] **D2 parity (DONE).** `rola_fla_dev/parity_model.py`: same model module, Triton-fork vs eager, fwd + all
  input grads — RLA elu-global + GLA raw + GLA normalized, nc=16/64, all PASS at TF32 noise (7e-4–2e-3);
  bf16-autocast smoke finite. Also: 6 routed kernels now use FLA's on-disk autotune cache
  (`autotune_cache_kwargs`) — cold-process runs stop re-paying the ~10-min autotune.
- [x] **D3 LM smoke (DONE).** `ROLA_USE_TRITON=1` train_lm.py, 100 steps each on WikiText-103, real HF Trainer
  stack (bf16 + grad checkpointing): rola-rla-asym AND rola-gla-scalar-norm-asym both train clean — loss
  16.6→13.2, eval 6.62 (ppl ~753), no NaN/dtype issues, exit 0. Routed-branch API hardening: raises
  NotImplementedError on cu_seqlens/initial_state/output_final_state/g_gamma; validates g's per-state shape.

## Phase F — chunk-parallel backward (win everywhere)
**Why:** bench (BENCH.md) shows the bwd is the only loss: each current bwd program serializes ALL gram
work behind a sequential chunk scan (grid (B·H,NB) → 64 programs at nc=16, occupancy cliff), and the
qr/kwv kernels EACH recompute G,P per chunk (2× gram work). FLA's own bwd materializes states then
parallelizes — do the same:
- **K1 scan_S** (sequential over chunks, grid (BH,NB), TINY work): store S_before[t] ∀t.
  RLA carry: S += KᵗᵀW∘V1; GLA: S = e^Λ·S + Kᵀ(w_end∘V1). Buffer Sb [B,NB,NCH,BD,BG·BV] fp32.
- **K2 scan_dS** (sequential reverse, grid (BH,NB), TINY): store dS_after[t] ∀t.
  RLA: dS += Qᵀ(R∘G); GLA: dS = e^Λ·dS + Qᵀ(rt∘G) — store BEFORE the update (current kernels use
  the pre-update value at chunk t). Buffer dSa, same shape.
- **K3 grad** (PARALLEL, grid (BH,NB,NCH)): per chunk load tiles + Sb[t] + dSa[t]; compute G,P,D ONCE;
  emit ALL grads — RLA: dq,dr,dk,dw,dv; GLA: + drg, da_rt, da_kwv, dLam (= Σ(dSa∘Sb)·e^Λ + Σdw_end∘w_end —
  both operands loaded). The per-chunk math is copied VERBATIM from the verified fused kernels with
  carries replaced by loads. dld assembly unchanged.
**Expected:** NCH× more programs on the heavy kernel (64→2048 at nc=16), grams once instead of twice,
RLA chunk can rise from 16→32. Mem cost: +2 state buffers (~36MB at LM nc=16; ~1GB+1GB at nc=256 —
GLA already pays one Sb today). SRAM: use BG=8 (state slices 2×8·BV·4 ≈ 32KB) + autotune warps/stages.
**Discipline:** RLA first → gate green → GLA → gate green → parity → bench. Old fused bwd kernels are
DELETED once the new path is green (no fallback duplication).
- [x] **F1/F2.** K1 `_scan_S` + K2 `_scan_dS` (USE_G constexpr covers RLA+GLA) + combined K3 → gate GREEN
  (all grads, both variants, nc=16/64/256; regression 0e+00). Constraint found: tl.dot needs K≥16 ⇒ BG≥16.
- [x] **F3 (DONE).** Combined K3 regressed on register spill (both state tiles + both intermediate sets
  live) → split into `_par_grad_{rla,gla}_{qr,kwv}` (one state tile each); GLA-kwv additionally needed
  the dLam Σ(dSa∘Sb) reduction moved to TORCH (chunked over NCH to avoid a GB transient) to drop its Sb
  load. RLA chunk=64 exceeded 100KB smem → stays 32. Old fused bwd kernels DELETED (551-line module).
  Gate (10 PASS) + parity green. Bwd speedups vs fused at matched thermals: RLA 5.1×/2.8×@nc16/64,
  GLA 1.7–1.9× everywhere. Full-economics totals: rola > monolith everywhere except GLA@16 (−6%);
  rola ≥ vh everywhere. Results in BENCH.md UPDATE.

## Phase E — cloud
- [ ] **E1.** Rebuild the image with `fla_rola` installed (baselines keep canonical `fla`). Run the golden +
  regression gate inside the image and confirm green before any sweep uses it.

---

### Why this order is safe
- Phase A is pure additive torch in `chunk_simple_gla` behind `r=None` → regression is `0e+00` and the routed
  path's grads come from autograd, so they're correct **by construction**. This is the reference the rest is gated on.
- Phases B–C swap in Triton **one direction at a time**, each gated against (a) regression `0e+00` — proving the
  `None` baseline path is untouched — and (b) routing PASS vs the Phase-A torch reference.
- Nothing (`rola_kernels`, the torch path) is removed until its Triton replacement is gate-green.

## Phase G — Triton per-state-denominator kernel (kappa/per_state without the eager pre-pass)
**Why:** the eager `_rola_perstate_den` retains its einsum grams for backward → blew past 12GB VRAM at
LM batch 16 (WDDM paging, 25 s/it) and forced a non-comparable batch change. The den IS cheap in-kernel.
**Math:** d[i,c] = Σ_{j≤i} (φq_i·φk_j) w_j^c = (G∘causal)@wg |chunk + q·Z, Z[d,c]=Σ_{j<chunk} w_j^c φk_j[d].
**Build (in fla_rola/ops/simple_gla/rola.py, mirrors the proven scan+parallel pattern at BV=1 scale):**
- K1 `_den_fwd` (grid (BH,NB), seq scan): carry Z [BD,BG] fp32; per chunk: G=qkᵀ; d = (G∘caus)@wgc + qc@Z;
  store d rows + Zb checkpoint [B,NB,NCH,BD,BG]; Z += kcᵀ@wgc.
- K2 `_den_bwd_scan` (reverse): carry dZ [BD,BG]; store dZa BEFORE update; dZ += qcᵀ@g_d.
- K3 `_den_grad` (PARALLEL (BH,NB,NCH)): P = g_d@wgcᵀ [BT,BT];
  dq = (caus∘P)@kc + g_d@Zbᵀ; dk = (caus∘P)ᵀ@qc + wgc@dZaᵀ; dw = (G∘caus)ᵀ@g_d + kc@dZa.
  All tl.dot K-dims ≥16 ✓ (BD=BG=16, BT=32). Buffers tiny ([B,NB,NCH,16,16]).
- `_DenFn` autograd wrapper on folded [BH,L,*]; rola.py: kappa/per_state branch calls the fork den fn on
  CUDA (torch helper stays as the verification ORACLE only).
**Verify:** standalone vs `_rola_perstate_den` autograd (fwd + dq,dk,dw) nc=16/64/256; parity_model gains a
kappa cell; then re-bench (kappa rows should drop to ≈global +10-20%) and re-run kappa LM at batch 16
accum 4 (comparable to the other arms).
