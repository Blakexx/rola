# `tools/lint` — every lint, its rule, and whether it gates

Mirrors `tools/lint/` file for file: the whole roster, and, per lint, how it gates.

**A GATE IS A CHECK SOMEBODY CAN SATISFY.** The gating set below is at zero on the tree. A lint the tree does not pass
yet is not thereby wrong -- its findings are the work -- but refusing every commit until that work is done teaches
contributors to bypass the gate. So such a lint is a RATCHET: it refuses only what a commit adds, and the findings the
tree carried when it began gating are a baseline that only shrinks.

## The gating set

| lint | rule |
|---|---|
| `ast_grep_gate.py` + `rules/*.yml` | the standards' shapes an AST can see (§12 fences and collectives, a torch header in a per-arm translation unit, an anonymous-namespace kernel, a width-shaped `if`) |
| `lint_standards.py` | unroll discipline (§6), the arm-TU torch include, the barrier-free regions (no CTA barrier between a file's `//: @barrier-free` and `//: @barrier-free-end`), the whole `docs/internals/` mirror contract (including that a mirror page names no identifier the tree does not carry), the flock spelling, the supported matrix, the shard partition, the vendored pin, the comment-density mandate, repository portability |
| `unit_lints.py` + `.clang-tidy`, `clang-tidy-device.yaml` | the pinned clang-tidy (`tools/clang_tidy_pin.json`) on every unit the library links, for each ratified architecture: `.clang-tidy` on the host units (every finding fails), `clang-tidy-device.yaml` on the device units through clang's CUDA front end (dead stores, unused parameters, redundant expressions); and the host units' `-Wall -Wextra -Wunused` front-end compile. A finding in the unit's own package fails, and so does any compiler error -- a unit the compiler cannot parse is a unit no lint read ([below](#unit-lints)) |
| `unused_instantiations.py` | every `__global__` function template must demangle-match a symbol in a committed manifest. `__device__` templates are found, counted and explicitly NOT checked: manifests record entry functions only, and a wrapped-once helper has no standalone symbol |
| `run_ruff_e3.sh` | pycodestyle's blank-line family. It keeps its own invocation because ruff preview-gates that family, and turning preview on in `pyproject.toml` would gate every other family's preview rules too |
| `readability_spacing.py` | a function or kernel body of forty lines or more with ZERO blank lines in it, and a run of blank lines in `csrc` (Python's own two-line convention is ruff's `E303` territory). The rule asks for segmentation and does not say where |
| `python -m rola_devtools.mirror --check` | every tracked path is declared ships or private for the public mirror in `.github/mirror/declarations.json`, so a new file never ships, and never silently stays behind, undecided (the tool is rola-devtools', linked into the venvs; its README describes the export) |
| `clang-format` (root `.clang-format`) | the formatting floor, including `MaxEmptyLinesToKeep: 1` and `SeparateDefinitionBlocks: Always` — the formatter's half of the readability rule above |

<a id="unit-lints"></a>
## The unit lints in the build graph

The clang-tidy profiles and the host warnings read each unit's own compile: `bazel/defs.bzl`'s `unit_lints` aspect hands
`tools/lint/unit_lints.py` the compile action's command line, and the lint's inputs are the compile's, so it reads
what the build compiled and reruns only when that compile's inputs change. The units are the library's own, and each
ratified architecture is its own configuration, whose generated headers are that architecture's build set. A verdict
counts a warning in the unit's own package and any compiler error, and a run that fails with no located diagnostic
is red too; a fixture's verdict must hold exactly its expected findings ([below](#fixtures)).

<a id="portable-agent-files"></a>
## The agent files are portable

`tools/lint/portable_claude.py` (gating, KERNEL_STANDARDS §31) fails a commit whose agent files -- everything under
`.claude/` and every `CLAUDE.md` -- name a machine's own path: a home directory, a mount, a drive. The repo's skills,
hooks and routes ship with the code to every clone and worktree, so they cite repo-relative paths, hooks resolve
through `$CLAUDE_PROJECT_DIR`, and a site's own locations come from the dev config or the environment
(`ROLA_SCRATCH`).

<a id="ratchets"></a>
## The ratchets

`tools/lint/ratchet.py <lint>` runs a lint and compares its findings to `tools/lint/baselines/<lint>.json`. A finding is
known by its path, the first sentence of its message and a digest of the line it names, so an edit above it does not
move it. The gate fails on a finding the baseline does not hold, on a baseline entry the lint no longer finds (run
`--shrink` and commit the smaller baseline with the fix), and on a baseline holding more of an entry than HEAD's.
`--grown` makes that last comparison, every baseline at once, in the checkout; each lint's own ratchet runs over a copy
of the tree, which has no HEAD ([`build-graph.md`](../../build-graph.md#checks)). A HEAD git cannot read fails it. Editing
a line that carries a finding changes its digest, so the commit that touches the line fixes the finding. A lint that
cannot run fails; it never reads as zero. `--init` records a new lint's baseline once.

A finding that is not a defect is exempted the lint's own way, with its reason: vulture's whitelist
(`vulture_whitelist.py`), a `reserved:` marker for a constant parameter. "A ruling per hit" is that work: the lint cannot
tell a real finding from a false one, so a human decides each, and the decision lands as a fix or a stated exemption.

| lint | baseline (2026-09-14) | what burns it down |
|---|---|---|
| `work_codes.py` | 558 | prose: `DELETIONS.md` 162 and `open-work.md` 59 need rows that say what each change did; the rest are test and tool docstrings |
| `constant_parameters.py` | 134; 63 (2026-09-29) | a ruling per finding: inline the constant, or mark the parameter `reserved:` with its reason. The tests count as callers of the public surface (the names `rola` and `rola.expert` export), and a constructor's parameters are keyed by its class -- keyed `__init__`, they matched no call and every class shared one key, which filed 17 exercised options as dead. Callers outside the tree and positional calls are invisible, so it over-reports |
| `run_vulture.sh` | 60 | a ruling per finding: delete the dead code, or whitelist a demonstrated dynamic use (name-dispatched methods, fixtures, a library's reads) |
| `drift_guards.py` | 20 over 11 rules | the table below |
| `r9_enforcement.py` | 1 | `decode_lattice.cuh`'s `switch (kind)` folds its fourth case into a bare default; the fix is a `uniform_switch`, a decode kernel change |
| `coverage.py` | 0 (2026-09-29) | nothing: it gates the undeclared gaps (a kernel not in `tools/coverage.json`, a pointer to nothing); the owed obligations -- budgets, phase ledgers, sanitizer rows and oracle tests for the entmax solves, three kernels awaiting the milestone ratify -- are the backlog it prints ([`coverage.md`](coverage.md)) |
| `burst_tier.py` | 0 (2026-11-09) | KERNEL_STANDARDS §24: an MMA issued outside a `//: @burst` function or a vote, sync, barrier, `while`, data `if` or `unroll 1` inside one, over every device function (a `static` or `__host__` one among them); `//: @burst-exempt <reason>` names a debt (the dense fold's and the dense readout's loops over a chunk's slots, `steps_dense`, `read_dense`, `dense_den`). It reads only a function that issues an MMA itself: a loop above a burst function that hand-rolls what `Burst` units do is a debt it cannot see, named on that kernel's page (the readout's `read_pass`, `readout.md#chunk`) |

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

**Measured 2026-09-14, 20 findings over 11 rules** (`bazel run //tools/lint:drift_guards` prints each):

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

## <a id="standards"></a>The standards lint, in detail

**`ast_grep_gate.py`** scopes `ast-grep scan --config sgconfig.yml` to `csrc/`: a bare scan defaults to the whole repo
and re-fires every rule's must-fire fixture under `tools/lint/fixtures/` as a gating error. Each rule's `message` names
its standards section and incident; `--test-fixtures` is the separate self-test that each rule still fires on its own
fixture.

**`lint_standards.py`** carries what ast-grep cannot express: the unroll discipline (§6: a dynamic-bound loop in device
code carries `#pragma unroll 1`, scoped to `__global__`/`__device__` bodies by brace counting), the torch include as a
text belt to the ast-grep rule, a CTA or grid barrier inside a declared barrier-free region (`__syncwarp` is intra-warp
and excluded), a committed doc, skill or script wrapping a command in an external `flock` on the GPU lock path (scoped
to `docs/`, `.claude/skills/`, `measure/`, `tools/`, `CLAUDE.md` and `README.md`, the standards among them), the
file-reading checks
(`check_supported_table`, `check_shard_partition`, `check_arch_coverage`, `check_vendored_pin`, each a function of the
tree's files, never of an imported `rola`), and
`check_repo_portability` (§17): every tracked file scanned for an absolute home-directory path, a citation of an
out-of-repo journal by path, a real worktree path or a session's scratch path, with no allowlist (its own source is
excluded by name, because its patterns spell what they catch). Every finding prints its section, `file:line` and the
incident.

<a id="compile-commands"></a>**The compile database.** `bazel run //:compile_commands` writes `compile_commands.json`
into the checkout for an editor: each unit's clang command, as the unit lints run it, for the first ratified
architecture, every path absolute.

<a id="device-profile"></a>The device profile runs clang's CUDA frontend `--cuda-host-only` on the compile's own include roots (the
toolkit's `include/cccl` among them, which nvcc's profile adds and clang does not), and a compiler error anywhere
fails it: the parse is part of the gate, which is why the
analyzer is pinned (a distro clang could not parse a torch-bearing unit and reported used parameters as unused). Both
profiles were validated against planted findings, so a clean pass is the tool running, not an inert config.

Two settings keep a unit's time on its own code. THE ANALYZER RUNS WITHOUT INLINING (`-analyzer-config ipa=none`): the
profile's one analyzer check, `deadcode.DeadStores`, reads each function's own liveness, so inlining callees adds
nothing to it -- it was ~80% of a template-heavy unit's time (`entmax/factor.cu`: 69 s without, 389 s with, the same
findings), and the self-test's fixture plants a dead store the check must still find. THE LINT READS ITS ONE
ARCHITECTURE'S BUILD SET: the host pass has every carried build in the image (`common/select.cuh`'s `in_image`), so a
`build_switch` would instantiate every kernel once a carried build; each architecture is the lint's own
configuration, whose generated `builds_header` (`measure/hardware.py`'s) carries that architecture's builds alone.

## Fixtures

Every lint owns a fixture pair under `tools/lint/fixtures/` (a "must fire" case and a
"must not fire" case, most often in one file). A rule that has never been shown to fire is a rule nobody knows is
connected; one that has never been shown to stay quiet is a rule nobody can afford to make gating. The self-tests are
the commit gate's `//tools/lint:lint_fixtures`: each Python lint's `--test-fixtures` mode, and the unit lints' fixtures
linted as units, each verdict holding exactly the fixture's expected findings.

Fixtures are EXCLUDED from every lint's normal-mode scan (`readability_spacing.py`'s
`py_files()`, `run_vulture.sh`'s `--exclude`, `drift_guards.py`'s `_walk`). The lesson —
`ast_grep_gate.py`'s own header records it, and it has since bitten three lints in this
directory — is that a fixture's deliberate violation re-triggering the real report is
exactly how a hook drifts into being silently disabled.

## Running the roster by hand

```bash
bazel test //:gate                  # the whole roster: every check the commit gate runs
bazel test //:lints                 # the checks over files alone

bazel test //:ratchet_work_codes    # a ratchet's gate
bazel run //tools/lint:work_codes   # the lint alone: its inventory
bazel run //tools/lint:drift_guards
bazel run //tools/lint:constant_parameters
bazel run //tools/lint:r9_enforcement
bazel test //:ratchet_vulture
```
