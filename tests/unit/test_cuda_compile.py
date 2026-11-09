"""The graph's CUDA compile action (bazel/cuda_compile.py): the dependency file read as Make writes it, the unread
headers listed and nothing else (a header read through a link counts as read, a directory is never listed), ptxas's
report kept whole and kept off the console, and a failed or unnamed compile refused. A stand-in nvcc; no CUDA."""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bazel"))

import cuda_compile  # noqa: E402

SCRIPT = ROOT / "bazel" / "cuda_compile.py"

REPORT = """\
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z1kPi' for 'sm_86'
ptxas info    : Function properties for _Z1kPi
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 8 registers, used 0 barriers, 360 bytes cmem[0]
ptxas warning : Registers are spilled to local memory in function '_Z1kPi', 8 bytes spill stores
a.cu(3): warning #177-D: variable "x" was declared but never referenced
      int x;
          ^
"""


def test_the_dependency_file_is_read_as_make_writes_it():
    depfile = "unit.o : unit.cu \\\n    src/a.cuh \\\n    src/with\\ space.cuh \\\n    src/dollar$$.cuh\n"
    assert cuda_compile.prerequisites(depfile) == ["unit.cu", "src/a.cuh", "src/with space.cuh", "src/dollar$.cuh"]


def test_every_rule_of_the_dependency_file_counts():
    assert cuda_compile.prerequisites("a.o: a.cu b.h\nb.h:\n") == ["a.cu", "b.h"]


def test_only_unread_files_are_listed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("read.cuh", "unread.cuh", "real.cuh"):
        (tmp_path / name).write_text("")
    #: the sandbox's links: a candidate that resolves to a file nvcc named by another path was read
    (tmp_path / "linked.cuh").symlink_to(tmp_path / "real.cuh")
    (tmp_path / "tree").mkdir()
    listed = cuda_compile.unused(["read.cuh", "unread.cuh", "linked.cuh", "tree"], ["unit.cu", "read.cuh", "real.cuh"])
    assert listed == ["unread.cuh"]


def test_the_console_drops_only_ptxas_s_report():
    inside, shown = False, []
    for line in REPORT.splitlines():
        inside = cuda_compile.ptxas_info(line, inside)
        shown += [] if inside else [line]
    assert shown == REPORT.splitlines()[5:]


def _nvcc(tmp_path: Path, *, names_source: bool = True, fails: bool = False) -> Path:
    """A stand-in nvcc: writes its object and the dependency file `-MF` names, and prints ptxas's report."""
    stand_in = tmp_path / "nvcc"
    read = ["unit.cu"] if names_source else []
    stand_in.write_text(textwrap.dedent(f"""\
        #!{sys.executable}
        import sys
        argv = sys.argv[1:]
        open(argv[argv.index("-o") + 1], "w").write("object")
        open(argv[argv.index("-MF") + 1], "w").write("unit.o : " + " ".join({read!r} + ["read.cuh"]) + "\\n")
        sys.stderr.write({REPORT!r})
        sys.exit({1 if fails else 0})
        """))
    stand_in.chmod(0o755)
    return stand_in


def _compile(tmp_path: Path, nvcc: Path) -> subprocess.CompletedProcess:
    for name in ("unit.cu", "read.cuh", "unread.cuh"):
        (tmp_path / name).write_text("")
    (tmp_path / "candidates.txt").write_text("read.cuh\nunread.cuh\n")
    return subprocess.run([sys.executable, str(SCRIPT), "--source", "unit.cu", "--candidates", "candidates.txt",
                           "--unused", "unused.txt", "--report", "report.txt", "--", str(nvcc), "-c", "unit.cu",
                           "-o", "unit.o"], cwd=tmp_path, capture_output=True, text=True)


def test_a_compile_lists_its_unread_headers_and_keeps_its_report(tmp_path):
    done = _compile(tmp_path, _nvcc(tmp_path))
    assert done.returncode == 0, done.stderr
    assert (tmp_path / "unused.txt").read_text() == "unread.cuh\n"
    assert (tmp_path / "report.txt").read_text() == REPORT
    assert done.stderr == "".join(REPORT.splitlines(keepends=True)[5:])


def test_a_failed_compile_shows_everything_and_fails(tmp_path):
    done = _compile(tmp_path, _nvcc(tmp_path, fails=True))
    assert done.returncode == 1
    assert done.stderr == REPORT
    assert not (tmp_path / "unused.txt").exists()


def test_a_dependency_file_that_does_not_name_its_source_is_refused(tmp_path):
    done = _compile(tmp_path, _nvcc(tmp_path, names_source=False))
    assert done.returncode != 0
    assert "does not name its source unit.cu" in done.stderr
    assert not (tmp_path / "unused.txt").exists()
