"""rola's declarations (`declare.py`): the root declares this checkout's build, machine facts, instruments, timing
entries and clock reader, one session and one memory pass over the cells, and a store for each result; every declared
code digest names files that exist; a layer arm is registered per construction on the cells declared for it, and a
decode step only on a cell that decodes; loading the file imports neither rola nor torch. No GPU: nothing is built."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPARSE = "layer-B2-T512-H8-h512-dv64-bf16-s1"
DECODE = "layer-B2-T128-H2-h128-dv64-fp32-s4-dec8"


def declared(cells: str) -> dict:
    from rola_devtools.build.declare import Graph, load

    g = Graph()
    public = load(ROOT / "declare.py")["root"](g, python=sys.executable, cells=cells)
    return {"public": public, "targets": g.targets}


def test_the_root_declares_the_checkout_its_session_its_memory_pass_and_their_stores():
    out = declared(f"flagship-dense,flagship-alt-k4,{SPARSE},{DECODE}")
    labels = set(out["targets"])
    assert {"rola/binary", "rola/environment", "rola/sass", "rola/registers", "rola/phases", "rola/roofline",
            "rola/carry_forward", "rola/carry_intra", "rola/clock", "session", "memory", "store/session", "store/memory",
            "store/sass", "timing-server", "timing-server-stop"} <= labels
    assert {"rola/entmax_solve@layer=chunk-sparse-gain8", "rola/entmax_solve@layer=chunk-decode-w16",
            "rola/decode_step@layer=chunk-decode-w16"} <= labels
    assert "rola/decode_step@layer=chunk-sparse-gain8" not in labels
    assert not any("chunk-p73-pinned" in label for label in labels)
    #: a cell is a NODE, and an instrument takes the nodes of the cells it runs on as its data inputs
    assert [t.label for t in out["targets"]["rola/phases"].inputs] == ["cells/flagship-dense", "cells/flagship-alt-k4"]
    assert out["targets"]["cells/flagship-dense"].params == {"cell": "flagship-dense"}
    assert [t.label for t in out["targets"]["rola/carry_forward"].inputs] == ["cells/flagship-dense",
                                                                             "cells/flagship-alt-k4"]
    assert out["targets"]["rola/binary"].holds == {"host_cpu": "all"}
    assert out["targets"]["rola/phases"].holds == {"gpu": "all"}
    assert out["targets"]["timing-server-stop"].always_run
    assert set(out["public"]) == {"all"}


def test_every_declared_code_digest_names_files_that_exist():
    for target in declared(f"flagship-alt-k4,{DECODE}")["targets"].values():
        if target.code is None:
            continue
        cwd = Path(target.env.cwd)
        paths = target.code.get("files") or [target.code["entry"], *target.code.get("data", ())]
        missing = [p for p in paths if not (cwd / p).exists()]
        assert not missing, f"{target.label} declares code it cannot digest: {missing}"


def test_loading_the_declarations_imports_neither_rola_nor_torch():
    probe = (f"import json, sys; from rola_devtools.build.declare import Graph, load; "
             f"load({str(ROOT / 'declare.py')!r})['root'](Graph(), cells='flagship-alt-k4,{DECODE}'); "
             f"print(json.dumps(sorted(m for m in sys.modules if m.split('.')[0] in ('rola', 'torch'))))")
    done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, cwd="/", timeout=120, check=True)
    assert json.loads(done.stdout.strip().splitlines()[-1]) == []
