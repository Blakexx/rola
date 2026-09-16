"""rola's declarations (`declare.py`): the root declares this checkout's build, machine facts, instruments, timing
entries and clock reader, one session and one memory pass over the cells, and a store for each result; every declared
code digest names files that exist; a layer arm is registered per construction on the cells declared for it, and a
decode step only on a cell that decodes; loading the file imports neither rola nor torch. No GPU: nothing is built."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from rola_devtools.build.declare import Graph

ROOT = Path(__file__).resolve().parents[2]


def declared() -> dict:
    from rola_devtools.build.declare import Graph, load

    g = Graph()
    public = load(ROOT / "declare.py")["root"](g, python=sys.executable)
    return {"public": public, "targets": g.targets}


def test_the_root_declares_the_checkout_its_session_its_memory_pass_and_their_stores():
    out = declared()
    labels = set(out["targets"])
    assert {"rola/binary", "rola/environment", "rola/sass", "rola/registers", "rola/phases", "rola/roofline",
            "rola/carry_forward", "rola/carry_intra", "rola/clock", "session", "memory", "store/session", "store/memory",
            "store/sass", "timing-server", "timing-server-stop"} <= labels
    assert {"rola/entmax_solve@layer=chunk-sparse-gain8", "rola/entmax_solve@layer=chunk-decode-w16",
            "rola/decode_step@layer=chunk-decode-w16"} <= labels
    assert "rola/decode_step@layer=chunk-sparse-gain8" not in labels
    #: a cell is a NODE, and an instrument takes the nodes of EVERY carry cell as its data inputs: what a build runs
    #: on is pruned by label at the CLI, never chosen here
    from rola_devtools.cells import central

    #: ... and those are the cells SIZED FOR MEASUREMENT (`tier` probe or both); the oracle tier's are the diff
    #: surfaces' and the tier node's, never an instrument's
    carry = {f"cells/{n}" for n, r in central().cells.items()
             if r["data"].endswith("carry_cell") and r["params"].get("tier") in ("probe", "both")}
    assert {t.label for t in out["targets"]["rola/phases"].inputs} == carry
    assert out["targets"]["cells/flagship-dense"].params == {"cell": "flagship-dense"}
    assert {t.label for t in out["targets"]["rola/carry_forward"].inputs} == carry
    assert out["targets"]["rola/binary"].holds == {"host_cpu": "all"}
    assert out["targets"]["rola/phases"].holds == {"gpu": "all"}
    assert out["targets"]["timing-server-stop"].always_run
    assert set(out["public"]) == {"all"}


def test_every_declared_code_digest_names_files_that_exist():
    for target in declared()["targets"].values():
        if target.code is None:
            continue
        cwd = Path(target.env.cwd)
        paths = target.code.get("files") or [target.code["entry"], *target.code.get("data", ())]
        missing = [p for p in paths if not (cwd / p).exists()]
        assert not missing, f"{target.label} declares code it cannot digest: {missing}"


def test_every_node_declares_the_cells_it_runs_on_and_the_root_takes_no_cell_list():
    """Selection is the build CLI's, by label; a node's cells are its own declaration. The sides and the in-checkout
    kernel-vs-oracle diff are declared beside the instruments, and the kernel's side reads the binary while the pure
    reference does not."""
    import inspect

    from rola_devtools.build.declare import load

    declared = load(ROOT / "declare.py")
    assert "cells" not in inspect.signature(declared["root"]).parameters
    g = Graph()
    declared["root"](g, python=sys.executable)
    kernel, reference = g.targets["rola/side/carry-kernel"], g.targets["rola/side/carry-reference"]
    assert "binary" in kernel.deps and "binary" not in reference.deps
    assert kernel.holds == {"gpu": "all"} and reference.holds == {"host_cpu": "all"}
    assert [t.label for t in kernel.inputs] == [t.label for t in reference.inputs]
    diff = g.targets["rola/diff/carry-vs-oracle"]
    assert diff.params["strategy"] == "per-slot" and set(diff.params["params"]["per_quantity"]) == {"num", "den", "state"}
    assert diff.params["minimum"] == len(kernel.inputs)
    assert {t.label for t in g.targets["rola/side/producer"].inputs} == {
        f"cells/{n}" for n in declared["surface_cells"]("producer")}


def test_a_surface_states_what_it_is_and_its_numeric_facts_but_chooses_no_comparison():
    """rola-bench owns the rule a surface is compared under across checkouts, as it owns the timing methodology; rola
    states what a surface IS and, where it matters, a FACT about its numerics -- the readout fan-in reassociates."""
    from rola_devtools.build.declare import load

    d = load(ROOT / "declare.py")
    for name, (executor, kind_, tier, holds) in d["SURFACES"].items():
        assert executor.startswith("tests.oracle.sides:") and kind_ in ("carry", "producer") and holds
    facts = d["NUMERICS"]["carry-kernel"]
    assert set(facts["reassociation"]) == {"num", "den"} and facts["exact"] == ["state"]
    assert all(0 < f["rtol"] < 1e-5 for f in facts["reassociation"].values())


def test_the_oracle_tier_is_one_target_keyed_on_its_code_and_the_binary_and_stored():
    """The whole pytest tier as a node: a red cell becomes a stored fact, and an unchanged tree is a cache hit."""
    from rola_devtools.build.declare import load

    g = Graph()
    load(ROOT / "declare.py")["root"](g, python=sys.executable)
    tier = g.targets["rola/oracle-tier"]
    assert tier.executor.endswith(":pytest_tier") and tier.cache and tier.holds == {"gpu": 1}
    assert tier.params["paths"] == ["tests/oracle"] and {"binary", "environment"} <= set(tier.deps)
    assert "tests/oracle" in tier.code["files"] and "rola" in tier.code["files"]
    assert g.targets["store/oracle-tier"].deps["source"] is tier


def test_a_backward_surface_admits_only_the_cells_under_the_retained_state_budget():
    """The rule is STATED, so the excluded cells are a consequence a reader can compute and not a run that died: the
    oracle side retains the fp64 state at every token and its peak is measured at over nine times that."""
    from rola_devtools.build.declare import load
    from rola_devtools.cells import central

    d = load(ROOT / "declare.py")
    admitted = d["surface_cells"]("oracle")
    assert admitted and "one-box-short" in admitted and "flat-small-dense" not in admitted
    for name, record in central().cells.items():
        if d["kind"](name) == "carry" and record["params"].get("tier") == "oracle":
            assert (name in admitted) == (d["retained_state_bytes"](record) < d["RETAINED_STATE_BUDGET"])
    #: the kernel's forward-only side takes every oracle-tier cell; the budget is the backward's alone
    assert "flat-small-dense" in d["surface_cells"]("carry-kernel")


def test_loading_the_declarations_imports_neither_rola_nor_torch():
    probe = (f"import json, sys; from rola_devtools.build.declare import Graph, load; "
             f"load({str(ROOT / 'declare.py')!r})['root'](Graph()); "
             f"print(json.dumps(sorted(m for m in sys.modules if m.split('.')[0] in ('rola', 'torch'))))")
    done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, cwd="/", timeout=120, check=True)
    assert json.loads(done.stdout.strip().splitlines()[-1]) == []
