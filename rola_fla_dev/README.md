# rola_fla_dev — safe harness for putting RoLA into the FLA fork

This directory is the **verification framework** for implementing Routed Linear Attention inside the
literal FLA fork (`fla_rola` at `/home/blake/flash-linear-attention-rola`), without ever regressing the
canonical baselines.

## The setup
- **`fla`** — canonical FLA, pristine. Baselines (simple_gla / gla / linear / gated_delta) import this.
- **`fla_rola`** — the fork (FLA with the package renamed). RoLA routing gets added here, additively.
- **`rola_kernels`** — our independently-verified kernels (recurrent_gla, ref_global, analytic backward).
  Used here **only as the correctness oracle**; not the shipping path.
- Both `fla` and `fla_rola` are installed `-e` and coexist in one process, so a single `check.py` run can
  diff them directly.

## The two gates (`check.py`)
1. **REGRESSION (safety, hard gate).** `fla_rola`'s *unrouted* kernels must equal canonical `fla` —
   forward + all input grads, every baseline. Live comparison (no stale golden; canonical is truth).
   Any non-zero divergence ⇒ exit 1 ⇒ a routing edit broke backward-compat ⇒ revert.
2. **ROUTING (target, soft gate).** `fla_rola`'s *routed* path must equal the `rola_kernels` oracle —
   forward + all grads, to TF32 noise. SKIP until implemented; PASS as each `STEPS.md` step lands.
   `--strict` makes routing FAIL also exit 1.

## Use
```bash
# run after EVERY edit to fla_rola:
PYTHONPATH=/mnt/c/Users/Blake/Documents/VSCode/CLA python rola_fla_dev/check.py
# strict (gate on routing too, once routing exists):
PYTHONPATH=/mnt/c/Users/Blake/Documents/VSCode/CLA python rola_fla_dev/check.py --strict
```
Current state: regression `0e+00` PASS (all 4 baselines, fwd+grad); routing SKIP (not yet implemented).

## Files
- `check.py` — the gate (regression + routing), prints a PASS/FAIL table, exit-codes for CI/scripted loops.
- `STEPS.md` — the ordered, mechanical implementation plan; each step = one small diff + which gate must stay/go green.
- `README.md` — this file.

## Discipline
- One step at a time (`STEPS.md`). Never batch.
- Routing is purely additive: `g`/`r`/`w` default `None` ⇒ behavior identical to canonical. The regression
  gate is what *proves* that on every edit.
- Nothing (`rola_kernels`, torch refs) is deleted until its fork replacement is gate-green.
