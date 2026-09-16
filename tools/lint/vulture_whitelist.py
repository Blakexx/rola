#!/usr/bin/env python3
"""VULTURE WHITELIST, read by `tools/lint/run_vulture.sh` (which `tools/lint/ratchet.py vulture` gates).

vulture's own convention: a whitelist is Python source that REFERENCES every
name vulture would otherwise flag, so it disappears from the report; it is
never executed, only parsed. Every entry below is a GENUINE DYNAMIC USE --
a name some FRAMEWORK reads by convention, never by an explicit call vulture's
static analysis can see -- with a one-line reason, per the brief. Anything
NOT listed here that vulture still flags is a real candidate to delete, never
silenced by omission.

Run `tools/lint/run_vulture.sh`, which owns the scan: the Python packages, the tools, the batteries, the benches
and `setup.py` (a CALLER -- it runs the build's own gates), with this file as the last argument.
"""


class _Whitelist:
    """vulture's own idiom (see its README "Handling false positives"): a
    referenced ATTRIBUTE never triggers "unused", so every genuinely-dynamic
    name is listed here as `_.name` -- a bare module-level assignment
    (`pytestmark = None`) does NOT work, it is itself a new "unused variable"
    (measured: tried first, vulture flagged this very file)."""

    # pytest reads a module-level `pytestmark` attribute automatically to apply
    # markers (e.g. `pytest.mark.skipif(...)`) to every test in the file; it is
    # never referenced by name anywhere in the file itself, which is exactly
    # what vulture's "unused variable" heuristic flags. Present in every
    # tests/{oracle,integration,unit}/test_*.py file that needs a module-wide
    # skip or GPU marker (27 files, measured 2026-08-29).
    pytestmark = None

    # pytest FIXTURE INJECTION: a test function's parameter name is matched by
    # NAME against a `@pytest.fixture` of the same name and the framework calls
    # it for you -- the parameter is "unused" only to a static analyzer that
    # does not know about fixture injection. `tests/unit/test_ratify_argv.py`'s
    # `ninja` fixture (patches `ratify._ninja_cuda_template` for every test in
    # the file) is the one vulture flags at 60%+100% confidence (the fixture
    # function itself, and each test parameter named after it).
    ninja = None

    # pytest HOOKS: a conftest's `pytest_addoption` and `pytest_configure` are called by the framework, found by
    # their names. `tests/conftest.py` registers `--oracle-margins` with them.
    pytest_addoption = None
    pytest_configure = None

    # rola_devtools.build EXECUTORS: a declared target's worker calls its executor and verify functions from their
    # `module:function` strings in `declare.py` (`measure/executors.py`'s `compile_kernel`, `binary_present`,
    # `probe_environment`, `run_tool`, `timed`, `read_clock`).
    compile_kernel = None
    binary_present = None
    probe_environment = None
    run_tool = None
    timed = None
    read_clock = None

    # THE DIFF SIDES (`tests/oracle/sides.py`) are executors a declaration names by `module:function` string
    # (`declare.py`'s SURFACES and KERNEL_VS_ORACLE), called by the build's side worker and never by name in code.
    carry_reference = None
    carry_kernel = None

    # torch calls `forward` through `Module.__call__`, never by name: every `nn.Module` in the tree defines one and
    # none of them is called explicitly. 13 sites, measured 2026-09-15 (`rola/layer.py`, `rola/routing/`'s decay,
    # producer, side_gain and entmax bodies, and the module stubs three tests define).
    forward = None

    # pytest FIXTURE FUNCTIONS in the root conftest: an `autouse` fixture is called by the framework for every test
    # and never named anywhere, and a non-autouse one is reached through `request.getfixturevalue(NAME)` -- a string.
    poison_torch_memory = None
    _gpu_lock_held = None
    _host_budget_worker_slot = None
    _take_gpu_lock = None

    # THE FACT SYSTEM ADDRESSES BY STRING: a node declares `gives=("union_table", ...)` / `needs=(...)` and reads
    # `f["union_table"]` (`rola/engine/facts/nodes.py`), so the plan field of the same name has no attribute read.
    union_table = None

    # `ChunkPlan` IS A CONTRACT, and its consumer is the absent carry launch (card C): the field list is the claim
    # that no field is `O(N * L)`, so a field with no reader TODAY is the contract the rebuild fills, not dead weight.
    v_row_bytes = None
    y_dtype = None

    # THE MIRROR IS COMPLETE BY CONSTRUCTION: `LivenessLayout` mirrors `csrc/rola/src/facts/liveness_contract.cuh`
    # member for member, and the C++ member is the one a `static_assert` gates (the liveness contract gate).
    token_bit = None

    # DOCUMENTED PUBLIC SURFACE, read by a user rather than by this tree: the paging facade's `backing_kind` and
    # `closed` (`docs/internals/rola_api.md`), the arena's `owners_total` and `free`, the registry's
    # `support_differentiable` column, and `Activation`, the spec-facing type alias (`docs/api.md`).
    backing_kind = None
    closed = None
    owners_total = None
    free = None
    support_differentiable = None
    Activation = None

    # zoology's `Hybrid` reads a layer's `state_size(sequence_length=...)` by name when it sizes a model.
    state_size = None
    sequence_length = None

    # setuptools READS `build_temp` off the command object it owns: `setup.py`'s `finalize_options` pins it to a
    # fixed repo-relative directory so the artifact's identity is a function of the tree, not of pip's temp dir.
    build_temp = None

    # `zipfile.ZipInfo` FIELDS ARE READ BY THE WRITER, not by us: `tools/wheels.py` sets `external_attr` (the POSIX
    # mode) and `compress_type` on each member and `zipfile` reads them when it writes the entry.
    external_attr = None
    compress_type = None


_ = _Whitelist()
_referenced = (_.pytestmark, _.ninja, _.pytest_addoption, _.pytest_configure,  # ruff B018: a bare attribute is "useless"
               _.compile_kernel, _.binary_present, _.probe_environment, _.run_tool, _.timed, _.read_clock,
               _.forward, _.poison_torch_memory, _._gpu_lock_held, _._host_budget_worker_slot, _._take_gpu_lock,
               _.union_table, _.v_row_bytes, _.y_dtype, _.token_bit, _.backing_kind, _.closed, _.owners_total,
               _.free, _.support_differentiable, _.Activation, _.state_size, _.sequence_length,
               _.build_temp, _.external_attr, _.compress_type, _.carry_reference, _.carry_kernel)
