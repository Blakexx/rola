# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE DISASSEMBLER SEAM: how this tree reads SASS, in one place.

Four instruments read the assembled code -- the ratification's body hashes (`tools/ratify.py`), the ptxas signature gate
(`tools/sass_gate.py`), the per-component region ledger (`tools/region_ledger.py`) and the register life ranges
(`tools/life_ranges.py`) -- and each ran its own `cuobjdump` and `nvdisasm` with its own spelling of an instruction
line. Four spellings of one format is the drift shape this repository has been bitten by before (`tools/gen_shards.py`
says it for the arm list, `tools/build_flags.py` for the nvcc flags), so the INVOCATION and the LINE GRAMMAR are here
and the analysis stays with the instrument that owns it.

    cubins(path)                  every ELF cubin of a built extension, extracted once into a temporary directory
    cubin(path, member)           the one image a ledger reads, by the name it carries
    sass_lines(obj)               `cuobjdump -sass`, streamed (the ratification reads a whole binary this way)
    disassemble(cubin, *flags)    `nvdisasm`, as text

`INSTRUCTION` matches an instruction line -- `/*0a10*/ @!P0 IMAD.WIDE R4, R6, ...` -- with the address, the predicate
and the opcode as groups; `LINE_INFO` matches the `--print-line-info-inline` frame comment above one. A tool that wants
the encoding lines too (a body hash covers them) matches `ENCODING`.
"""
from __future__ import annotations

import re
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import dev_config

#: `/*ADDR*/ [@PRED] OPCODE ...` -- one assembled instruction, as both disassemblers print it
INSTRUCTION = re.compile(r"^\s*/\*([0-9a-f]{4,5})\*/\s+(@!?U?P\w+\s+)?([A-Z0-9_.$]+)")
#: the encoding line that follows an instruction in `cuobjdump -sass` output
ENCODING = re.compile(r"^\s*/\*\s*0x[0-9a-f]+\s*\*/\s*$")
#: `//## File "path", line N` -- one frame of the inline chain under `--print-line-info-inline`
LINE_INFO = re.compile(r'//## File "([^"]+)", line (\d+)')
#: `Function : NAME` and `arch = sm_XX`, the section headers of `cuobjdump -sass`
FUNCTION = re.compile(r"^\s*Function : (\S+)")
ARCH = re.compile(r"^\s*arch = sm_(\d+)\s*$")


def cubins(path: Path) -> list[Path]:
    """The cubins of `path`: itself when it is one, else every ELF image `cuobjdump -xelf` writes out of the extension.
    The directory is temporary and the caller may read it for its lifetime."""
    path = Path(path)
    if path.suffix == ".cubin":
        return [path]
    out = Path(tempfile.mkdtemp(prefix="rola_sass_"))
    cuobjdump = dev_config.cuda_bin("cuobjdump")
    names = subprocess.run([cuobjdump, "-lelf", str(path)], capture_output=True, text=True, check=True).stdout
    picked = [m.group(1) for line in names.splitlines() if (m := re.search(r"ELF file\s+\d+:\s+(\S+)", line))]
    subprocess.run([cuobjdump, "-xelf", "all", str(path.resolve())], cwd=out, capture_output=True, check=True)
    return [out / name for name in picked]


def cubin(path: Path, member: str) -> Path:
    """The one cubin of `path` whose name holds `member`; refuses where that is not exactly one image."""
    found = [image for image in cubins(path) if member in image.name]
    if len(found) != 1:
        raise SystemExit(f"{member} names {len(found)} ELF images in {Path(path).name}; the instruments read one")
    return found[0]


@contextmanager
def sass_lines(obj: str | Path) -> Iterator[Iterator[str]]:
    """`cuobjdump -sass` over a whole object, STREAMED: the ratification scans megabytes of it and never holds the text.
    A disassembler that fails is fatal -- a gate cannot pass a binary it could not read."""
    proc = subprocess.Popen([dev_config.cuda_bin("cuobjdump"), "-sass", str(obj)], stdout=subprocess.PIPE, text=True)
    try:
        yield proc.stdout
    finally:
        rc = proc.wait()
    if rc != 0:
        raise SystemExit(f"cuobjdump -sass FAILED (rc={rc}) on {obj}; the SASS gates cannot pass a binary they "
                         "cannot read")


def disassemble(cubin_path: Path, *flags: str) -> str:
    """`nvdisasm [flags] cubin`, as text. `--print-line-info-inline -gi` is what the ledgers pass for the frame chain."""
    return subprocess.run([dev_config.cuda_bin("nvdisasm"), *flags, str(cubin_path)],
                          capture_output=True, text=True, check=True).stdout


def frames(text: str) -> dict[int, list[tuple[str, int]]]:
    """SASS offset -> the inline frame chain at that instruction, innermost first, as (file name, line) pairs, off
    `--print-line-info-inline` output: the comments precede the instruction they belong to, and an instruction with no
    comment of its own inherits the chain above it."""
    pending: list[tuple[str, int]] = []
    chain: list[tuple[str, int]] = []
    out: dict[int, list[tuple[str, int]]] = {}
    for line in text.splitlines():
        info = LINE_INFO.search(line)
        if info:
            pending.append((info.group(1).split("/")[-1], int(info.group(2))))
            continue
        instruction = INSTRUCTION.match(line)
        if instruction:
            if pending:
                chain = pending
            pending = []
            out[int(instruction.group(1), 16)] = chain
    return out


__all__ = ["ARCH", "ENCODING", "FUNCTION", "INSTRUCTION", "LINE_INFO", "cubin", "cubins", "disassemble", "frames",
           "sass_lines"]
