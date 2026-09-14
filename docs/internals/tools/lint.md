# `tools/lint` — every lint, its rule, and whether it gates

Mirrors `tools/lint/` file for file. `docs/build.md`'s "Standards lint" section covers
the first generation of these checks (the ast-grep rules, `lint_standards.py`,
`run_clang_tidy.sh`); this page carries the whole roster and states, per lint, whether it
refuses a commit today or only reports.

**A GATE IS A CHECK SOMEBODY CAN SATISFY.** Every gating lint below is at ZERO on the tip
its flip landed on. A check that fires on the shipped tree is not thereby wrong — it is
the work — but wiring it to refuse commits before that work is done teaches contributors
to bypass the gate, which costs more than the check was worth. So a lint that is not
clean yet prints its findings, exits 0, and says here what it is waiting for.

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
| `.github/mirror/mirror.py --check` | every tracked path is declared ships or private for the public mirror in `.github/mirror/declarations.json`, so a new file never ships, and never silently stays behind, undecided ([`.github/mirror/README.md`](../../../.github/mirror/README.md)) |
| `clang-format` (root `.clang-format`) | the formatting floor, including `MaxEmptyLinesToKeep: 1` and `SeparateDefinitionBlocks: Always` — the formatter's half of the readability rule above |

## The report-only set, and what each is waiting for

| lint | findings on this tree | what it is waiting for |
|---|---|---|
| `drift_guards.py` | 11 over 11 rules | five `torch.equal` storage claims and two structure findings in the decode family's own files, plus four test-local arm tables. Sized below |
| `work_codes.py` | 611 | the documentation sweep. `csrc/`, `rola/`, `tools/`, `tests/` and `benchmarks/` are clean; what remains is prose, most of it in the two documents whose subject IS the campaign's history |
| `r9_enforcement.py` | 1 | `decode_lattice.cuh`'s `switch (kind)` folds its fourth case into a bare default. The fix is a `uniform_switch` and it belongs to the decode family's own stage |
| `constant_parameters.py` | 144 | a ruling per finding. A knob that only ever takes one value is a constant, and the ruling that made it a rule came from a parameter documented as "stays None" -- the class vulture does not catch, because the name IS used, always with the same value. It over-reports by construction (a positional call site and a caller outside this tree are both invisible), which is why the `reserved:` marker carries a REASON |
| `run_vulture.sh` | 51 | a ruling per finding. Vulture cannot see cross-module use, so the list mixes genuine dead code with entry points, fixtures and public API; a whitelist entry is earned by a demonstrated dynamic use and never added to silence a real hit |

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

**Measured 2026-08-30, 11 findings over 11 rules:**

| rule | findings | what they are |
|---|---|---|
| `arm_axes` | 0 | no carry kernel exists to carry an axis. The producer's and decode's own axes are deliberately out of scope: whether they fold into `(D, DV, warps_per_cta)` is an open question in `docs/open-work.md`, not a finding |
| `state_float_view` | 1 | `decode.cuh`'s `float* state` member |
| `bytes_equal` | 5 | decode's own batteries assert state identity with `torch.equal`. The rule no longer fires on an INTEGER comparison: `torch.equal` is exact on an integer dtype, and the rule is about a float view of storage, which is the one place the two differ |
| `one_derivation_site` | 0 | the rule names the CARRY block's derivations; decode's lattice fill is derived from decode's own geometry |
| `one_admission_law` | 0 | — |
| `single_value_axis` | 0 | — |
| `second_arm_table` | 4 | three test-local `_ARMS`/`CHUNK_ARMS` tables and `test_carry_geometry.py`'s `ARM_PARAMS`. That last one is a deliberately PINNED transcription of a parent tip and is the case that says whether this rule wants an exemption or a different home |
| `one_body_per_family` | 0 | a reverse pass is its own op and is not read as a second body |
| `device_switch_is_uniform` | 1 | `decode_lattice.cuh`'s `switch (kind)` — the same site `r9_enforcement.py` finds, from the other direction |
| `host_dispatch_is_arm_switch` | 0 | — |
| `comments_describe_the_present` | 0 | it imports the tree-wide definition of a work code rather than restating it: two regexes for one law is the drift this file exists to catch, one level up |

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

python3 tools/lint/drift_guards.py  # the report-only four, on their own
python3 tools/lint/work_codes.py
python3 tools/lint/constant_parameters.py
python3 tools/lint/r9_enforcement.py
tools/lint/run_vulture.sh
```
