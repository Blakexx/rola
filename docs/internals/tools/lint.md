# `tools/lint` — every lint, its rule, and whether it gates

Mirrors `tools/lint/` file for file. `docs/build.md`'s "Standards lint" section covers
the first generation of these checks (the ast-grep rules, `lint_standards.py`,
`run_clang_tidy.sh`); this page carries the whole roster and states, per lint, how it gates.

**A GATE IS A CHECK SOMEBODY CAN SATISFY.** The gating set below is at zero on the tree. A lint the tree does not pass
yet is not thereby wrong -- its findings are the work -- but refusing every commit until that work is done teaches
contributors to bypass the gate. So such a lint is a RATCHET: it refuses only what a commit adds, and the findings the
tree carried when it began gating are a baseline that only shrinks.

## The gating set

| lint | rule |
|---|---|
| `ast_grep_gate.py` + `rules/*.yml` | the standards' shapes an AST can see (§12 fences and collectives, a torch header in a per-arm translation unit, an anonymous-namespace kernel, a width-shaped `if`) |
| `lint_standards.py` | unroll discipline (§6), the arm-TU torch include, the decode-prologue barrier law, the whole `docs/internals/` mirror contract (including that a mirror page names no identifier the tree does not carry), the flock spelling, the supported matrix, the shard partition, the vendored pin, the comment-density mandate, repository portability |
| `run_clang_tidy.sh` | clang-tidy on the host translation units |
| `run_clang_tidy_device.sh` + `clang-tidy-device.yaml` | dead stores, unused parameters and redundant expressions on the TORCH-FREE device translation units, through clang's CUDA front end. Toolchain noise — clang's bundled CUDA wrapper headers disagreeing with the installed toolkit — is printed and never counted |
| `host_warnings.py` | `-Wall -Wextra -Wunused` dry compile of the host translation units, ON-TARGET findings only; a vendored header's warning is not this codebase's to fix |
| `unused_instantiations.py` | every `__global__` function template must demangle-match a symbol in a committed manifest. `__device__` templates are found, counted and explicitly NOT checked: manifests record entry functions only, and a wrapped-once helper has no standalone symbol |
| `run_ruff_e3.sh` | pycodestyle's blank-line family. It keeps its own invocation because ruff preview-gates that family, and turning preview on in `pyproject.toml` would gate every other family's preview rules too |
| `readability_spacing.py` | a function or kernel body of forty lines or more with ZERO blank lines in it, and a run of blank lines in `csrc` (Python's own two-line convention is ruff's `E303` territory). The rule asks for segmentation and does not say where |
| `python -m rola_devtools.mirror --check` | every tracked path is declared ships or private for the public mirror in `.github/mirror/declarations.json`, so a new file never ships, and never silently stays behind, undecided (the tool is rola-devtools', linked into the venvs; its README describes the export) |
| `clang-format` (root `.clang-format`) | the formatting floor, including `MaxEmptyLinesToKeep: 1` and `SeparateDefinitionBlocks: Always` — the formatter's half of the readability rule above |

<a id="ratchets"></a>
## The ratchets

`tools/lint/ratchet.py <lint>` runs a lint and compares its findings to `tools/lint/baselines/<lint>.json`. A finding is
known by its path, the first sentence of its message and a digest of the line it names, so an edit above it does not
move it. The gate fails on a finding the baseline does not hold, on a baseline entry the lint no longer finds (run
`--shrink` and commit the smaller baseline with the fix), and on a baseline holding more of an entry than HEAD's. Editing
a line that carries a finding changes its digest, so the commit that touches the line fixes the finding. A lint that
cannot run fails; it never reads as zero. `--init` records a new lint's baseline once.

A finding that is not a defect is exempted the lint's own way, with its reason: vulture's whitelist
(`vulture_whitelist.py`), a `reserved:` marker for a constant parameter. "A ruling per hit" is that work: the lint cannot
tell a real finding from a false one, so a human decides each, and the decision lands as a fix or a stated exemption.

| lint | baseline (2026-09-14) | what burns it down |
|---|---|---|
| `work_codes.py` | 558 | prose: `DELETIONS.md` 162 and `open-work.md` 59 need rows that say what each change did; the rest are test and tool docstrings |
| `constant_parameters.py` | 134 | a ruling per finding: inline the constant, or mark the parameter `reserved:` with its reason. Callers outside the tree and positional calls are invisible, so it over-reports |
| `run_vulture.sh` | 60 | a ruling per finding: delete the dead code, or whitelist a demonstrated dynamic use (name-dispatched methods, fixtures, `setup.py` users) |
| `drift_guards.py` | 20 over 11 rules | the table below |
| `r9_enforcement.py` | 1 | `decode_lattice.cuh`'s `switch (kind)` folds its fourth case into a bare default; the fix is a `uniform_switch`, a decode kernel change |

<a id="drift-guards"></a>
## The drift guards, and the inventory they exist to size

Each rule is a law that a stage ratified by DELETING something, and each is a law
because the deleted thing is cheap to put back: a template axis that "only needs one more
parameter", a second kernel body for the case the first does not fit, a `float*` view of
a page, an `if` on the depth in a host entry, a comment explaining what the code used to
be. A law that lives only in a document is re-broken by the next person who has not read
it.

Every rule is a heuristic over text and says so in its own docstring: a finding is a
question for a reviewer, never a proof. Four rules were tightened after their first
inventory was read, each with its reason in the source — an inventory nobody trusts is an
inventory nobody flips.

**Measured 2026-09-14, 20 findings over 11 rules** (`python3 tools/lint/drift_guards.py` prints each):

| rule | findings |
|---|---|
| `arm_axes` | 1 |
| `bytes_equal` | 5: decode's batteries assert state identity with `torch.equal` |
| `comments_describe_the_present` | 3 |
| `device_switch_is_uniform` | 1: `decode_lattice.cuh`'s `switch (kind)`, the site `r9_enforcement.py` finds |
| `host_dispatch_is_arm_switch` | 0 |
| `one_admission_law` | 0 |
| `one_body_per_family` | 1 |
| `one_derivation_site` | 1 |
| `second_arm_table` | 5: test-local arm tables |
| `single_value_axis` | 0 |
| `state_float_view` | 3 |

`device_switch_is_uniform` and `r9_enforcement.py` overlap on purpose and are not the
same rule: `r9_enforcement.py` asks whether a structure branch has a trapping default and
a case-set marker, and this one asks whether it is a `uniform_switch` at all — which is
the form the constitution now requires. They find the same site from two directions,
which is the corroboration that both are connected; reconciling them into one rule waits
on the fix, so that the flip can decide which message a finding carries.

## Fixtures

Every lint owns a fixture pair under `tools/lint/fixtures/` (a "must fire" case and a
"must not fire" case, most often in one file) and a `--test-fixtures` mode on its own
driver. A rule that has never been shown to fire is a rule nobody knows is connected; one
that has never been shown to stay quiet is a rule nobody can afford to make gating.

```bash
python3 tools/lint/host_warnings.py --test-fixtures
python3 tools/lint/unused_instantiations.py --test-fixtures
python3 tools/lint/readability_spacing.py --test-fixtures
python3 tools/lint/r9_enforcement.py --test-fixtures
python3 tools/lint/constant_parameters.py --test-fixtures
python3 tools/lint/drift_guards.py --test-fixtures
tools/lint/run_clang_tidy_device.sh --test-fixtures
tools/lint/run_vulture.sh --test-fixtures      # needs vulture on PATH
```

Fixtures are EXCLUDED from every lint's normal-mode scan (`readability_spacing.py`'s
`py_files()`, `run_vulture.sh`'s `--exclude`, `drift_guards.py`'s `_walk`). The lesson —
`ast_grep_gate.py`'s own header records it, and it has since bitten three lints in this
directory — is that a fixture's deliberate violation re-triggering the real report is
exactly how a hook drifts into being silently disabled.

## Running the roster by hand

```bash
pre-commit run --all-files          # the whole roster, gates and reports together

python3 tools/lint/ratchet.py work_codes           # a ratchet's gate; the lint alone prints its inventory
python3 tools/lint/drift_guards.py
python3 tools/lint/work_codes.py
python3 tools/lint/constant_parameters.py
python3 tools/lint/r9_enforcement.py
tools/lint/run_vulture.sh
```
