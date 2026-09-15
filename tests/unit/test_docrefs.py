"""`tools/docrefs.py`: what a doc reference is, what its target's digest moves on, and the gate's verdict.

No GPU and no repository: each test plants a small tree under `tmp_path` and points the tool's root at it.
"""
from __future__ import annotations

import docrefs
import pytest


@pytest.fixture
def tree(tmp_path, monkeypatch):
    monkeypatch.setattr(docrefs, "ROOT", tmp_path)

    def plant(files: dict[str, str]) -> set[str]:
        for name, text in files.items():
            (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / name).write_text(text)
        docrefs._text.cache_clear()  #: the tool reads each file once per run; a test rewrites them within one
        return set(files)
    return plant


def test_a_docstring_a_comment_or_a_reformat_moves_no_pin_and_a_code_change_does(tree):
    files = tree({"rola/m.py": 'def f(x):\n    """Old words."""\n    return x + 1\n'})
    before = docrefs.read(docrefs.Target("rola/m.py", "f"), files)
    tree({"rola/m.py": 'def f(x):  # a comment\n    """New words."""\n\n    return (x + 1)\n'})
    assert docrefs.read(docrefs.Target("rola/m.py", "f"), files) == before
    tree({"rola/m.py": "def f(x):\n    return x + 2\n"})
    assert docrefs.read(docrefs.Target("rola/m.py", "f"), files) not in (before, None)


def test_a_method_a_constant_and_a_re_export_resolve_and_a_missing_name_does_not(tree):
    files = tree({"rola/__init__.py": "", "rola/a.py": "LIMIT = 3\n\nclass C:\n    def go(self):\n        return 1\n",
                  "rola/b.py": "from rola.a import C\n"})
    assert docrefs.read(docrefs.Target("rola/a.py", "LIMIT"), files)
    assert docrefs.read(docrefs.Target("rola/b.py", "C.go"), files) == docrefs.read(
        docrefs.Target("rola/a.py", "C.go"), files)
    assert docrefs.read(docrefs.Target("rola/a.py", "C.stop"), files) is None


def test_the_four_citation_forms_are_read_and_a_fenced_block_is_not(tree):
    files = tree({
        "rola/__init__.py": "", "rola/ops.py": "def carry():\n    pass\n", "tools/t.py": "def main():\n    pass\n",
        "docs/other.md": "# Title\n\n## Wheels\n\ntext\n",
        "docs/page.md": ("`tools/t.py`'s `main` and `rola.ops.carry` beside `docs/other.md` and [w](other.md#wheels)\n"
                         "```\n`rola.ops.fenced`\n```\n`torch.ops.x` and `a phrase`\n"),
    })
    got = {str(t) for t in docrefs.references("docs/page.md", files)}
    assert got == {"tools/t.py::main", "rola/ops.py::carry", "docs/other.md", "docs/other.md#wheels"}


def test_a_mirror_page_cites_its_sources_names_bare_and_keeps_a_pinned_one_that_left(tree):
    files = tree({"tools/t.py": "def main():\n    pass\n",
                  "docs/internals/tools/t.md": "# `tools/t.py`\n\n`main` calls `helper`, and `gone` went.\n"})
    got = {str(t) for t in docrefs.references("docs/internals/tools/t.md", files)}
    assert got == {"tools/t.py", "tools/t.py::main"}
    kept = docrefs.references("docs/internals/tools/t.md", files, pinned=frozenset({"tools/t.py::gone"}))
    assert docrefs.Target("tools/t.py", "gone") in kept
    assert docrefs.read(docrefs.Target("tools/t.py", "gone"), files) is None


def test_a_section_runs_to_the_next_heading_at_its_level(tree):
    files = tree({"docs/p.md": "# Top\n\n## Wheels\n\none\n\n### Detail\n\ntwo\n\n## Next\n\nthree\n"})
    wheels = docrefs.read(docrefs.Target("docs/p.md", "wheels"), files)
    tree({"docs/p.md": "# Top\n\n## Wheels\n\none\n\n### Detail\n\ntwo\n\n## Next\n\nchanged\n"})
    assert docrefs.read(docrefs.Target("docs/p.md", "wheels"), files) == wheels
    tree({"docs/p.md": "# Top\n\n## Wheels\n\none\n\n### Detail\n\nchanged\n\n## Next\n\nchanged\n"})
    assert docrefs.read(docrefs.Target("docs/p.md", "wheels"), files) != wheels


def test_device_code_pins_a_function_body_or_a_named_region(tree):
    files = tree({"csrc/k.cu": ("static int step(int x) {\n  // why\n  return x + 1;\n}\n"
                                "//: @ref fold\nint y = 2;\n//: @end\n")})
    body, region = (docrefs.read(docrefs.Target("csrc/k.cu", n), files) for n in ("step", "fold"))
    tree({"csrc/k.cu": "static int step(int x) {\n  return x + 1;  /* reworded */\n}\n//: @ref fold\nint y = 3;\n//: @end\n"})
    assert docrefs.read(docrefs.Target("csrc/k.cu", "step"), files) == body
    assert docrefs.read(docrefs.Target("csrc/k.cu", "fold"), files) != region


def test_the_verdict_fails_what_is_new_or_stale_and_only_reports_what_changed():
    target = docrefs.Target("rola/a.py", "f")
    pin = {"doc": "docs/p.md", "target": str(target), "sha256": "aaaa", "pinned_at": "0", "reviewed": True}
    lock = {("docs/p.md", str(target)): pin,
            ("docs/p.md", "rola/gone.py"): {**pin, "target": "rola/gone.py"},
            ("docs/p.md", "rola/broken.py"): {**pin, "target": "rola/broken.py", "sha256": None}}
    found = {("docs/p.md", str(target)): (target, 3, "bbbb"),
             ("docs/p.md", "rola/broken.py"): (docrefs.Target("rola/broken.py"), 4, None),
             ("docs/q.md", str(target)): (target, 1, "bbbb")}
    new, stale, grown, pending = docrefs.verdict(found, lock, head={})
    assert set(new) == {("docs/q.md", str(target))}
    assert stale == [("docs/p.md", "rola/gone.py")]
    assert grown == [("docs/p.md", "rola/broken.py")]
    assert set(pending) == {("docs/p.md", str(target))}
    assert docrefs.verdict(found, lock, head=lock)[2] == []
