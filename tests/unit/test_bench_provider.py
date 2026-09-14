"""The bench provider (benchmarks/bench/provider.py) and the comparison's arms (tools/compare.py): an arm is named by its
subject and the dials it reads, a point is a registered cell and only its own facts, and arms parse as their flags say.
No GPU: nothing here builds an arm."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "benchmarks", ROOT / "tools"):
    sys.path.insert(0, str(path))

import compare  # noqa: E402

from bench import provider  # noqa: E402
from bench.subjects import SUBJECTS  # noqa: E402


def test_an_arm_is_named_by_its_subject_and_its_non_default_dials():
    assert provider.arm_name("carry_forward") == "carry_forward"
    assert provider.arm_name("prefill_op", calls=4, state="continuation") == "prefill_op@calls=4@state=continuation"
    assert provider.arm_name("carry_forward", schedule="identity") == "carry_forward@schedule=identity"


def test_a_cells_arms_are_its_subjects_under_the_dials_each_reads():
    names = set(provider.arms({"cell": "flat-small-alt-k16"}))
    assert {"carry_forward", "carry_forward@schedule=identity", "liveness_pass", "prefill_op@state=paged"} <= names
    assert not any("@schedule=" in name for name in names if not name.startswith("carry_forward"))
    assert all("state" in SUBJECTS[name.split("@")[0]].dials for name in names if "@state=" in name)


def test_a_point_is_a_registered_cell_and_its_own_facts():
    assert provider.arms({"cell": "flat-small-alt-k16", "tokens": 256, "d_v": 64})
    with pytest.raises(ValueError, match="flat-small-alt-k16 is"):
        provider.arms({"cell": "flat-small-alt-k16", "tokens": 128})
    with pytest.raises(KeyError, match="registered cell"):
        provider.arms({"cell": "no-such-cell"})
    with pytest.raises(KeyError, match="registered cell"):
        provider.arms({"cell": "flat-small-alt-k16", "calls": 4})


def test_the_comparisons_arms_parse_as_their_flags_say():
    arm = compare.rola_arm("label:tip,arm:carry_forward@schedule=identity,worktree:/w/tip,venv:/w/venv-tip")
    assert arm == {"label": "tip", "arm": "carry_forward@schedule=identity", "worktree": "/w/tip", "venv": "/w/venv-tip"}
    foreign = compare.foreign_arm("label:attention,provider:pkg.attention:arms,arm:flash,python:/p/python,cwd:/c")
    assert foreign["provider"] == "pkg.attention:arms"
    with pytest.raises(argparse.ArgumentTypeError, match="lacks"):
        compare.foreign_arm("label:attention,arm:flash")
