# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE DISASSEMBLER SEAM: how this tree reads SASS, in one place.

Four instruments read the assembled code -- the ratification's body hashes (`tools/ratify.py`), the ptxas signature gate
(`tools/sass_gate.py`), the per-component region ledger (`tools/region_ledger.py`) and the register life ranges
(`tools/life_ranges.py`) -- and each ran its own `cuobjdump` and `nvdisasm` with its own spelling of an instruction
line. Four spellings of one format is the drift shape this repository has been bitten by before (`tools/gen_shards.py`
says it for the arm list, `tools/build_flags.py` for the nvcc flags), so the INVOCATION and the LINE GRAMMAR are here
and the analysis stays with the instrument that owns it.

    cubins(path)                  every ELF cubin of a built extension, extracted once a build into the scratch cache
    cubin(path, member)           the one image a ledger reads, by the name it carries
    sass_lines(obj)               `cuobjdump -sass`, streamed (the ratification reads a whole binary this way)
    disassemble(cubin, *flags)    `nvdisasm`, as text
    section(listing, kernel)      one function's code section of a listing (offsets restart in every function)

`INSTRUCTION` matches an instruction line -- `/*0a10*/ @!P0 IMAD.WIDE R4, R6, ...` -- with the address, the predicate
and the opcode as groups; `LINE_INFO` matches the `--print-line-info-inline` frame comment above one. A tool that wants
the encoding lines too (a body hash covers them) matches `ENCODING`.
"""
from __future__ import annotations

import functools
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

#: THE DISASSEMBLERS AND SCRATCH THIS PROCESS READS: the pinned toolkit's and the dev config's scratch
#: (`dev_config.cuda_home()`, `host.scratch`), or the graph's -- its hermetic toolkit and the action's own temporary directory (`use_graph`).
#: -- see docs/internals/tools/sass_gate.md#graph
_GRAPH: dict[str, Path] = {}


def use_graph(*, cuobjdump: str, nvdisasm: str) -> None:
    """Read the graph's disassemblers, and keep scratch in a directory of this process's own."""
    _GRAPH.update(cuobjdump=Path(cuobjdump).absolute(), nvdisasm=Path(nvdisasm).absolute(),
                  scratch=Path(tempfile.mkdtemp(prefix="rola_sass_")))


def tool(name: str) -> str:
    """The toolkit binary `name` (`cuobjdump`, `nvdisasm`)."""
    if _GRAPH:
        return str(_GRAPH[name])
    import dev_config

    return dev_config.cuda_bin(name)


def scratch(name: str) -> Path:
    """A scratch directory for `name`, never read back as a record."""
    if _GRAPH:
        path = _GRAPH["scratch"] / name
        path.mkdir(parents=True, exist_ok=True)
        return path
    import dev_config

    return dev_config.scratch(name)


#: `/*ADDR*/ [@PRED] OPCODE ...` -- one assembled instruction, as both disassemblers print it
INSTRUCTION = re.compile(r"^\s*/\*([0-9a-f]{4,5})\*/\s+(@!?U?P\w+\s+)?([A-Z0-9_.$]+)")
#: the encoding line that follows an instruction in `cuobjdump -sass` output
ENCODING = re.compile(r"^\s*/\*\s*0x[0-9a-f]+\s*\*/\s*$")
#: THE OPCODE CLASSES an instruction budget is stated in (KERNEL_STANDARDS section 29): each class one pipe or one kind
#: of overhead, matched by opcode prefix (the first match wins, so `F2FP`'s pack is matched before the conversion unit,
#: `REDUX`, the warp reduction, is a shuffle before `RED` would make it global, and `LDGDEPBAR`, the `cp.async` commit,
#: is control before `LDG` would)
OPCODE_CLASSES = {
    "mma": ("HMMA", "IMMA"),
    "shared load": ("LDSM", "LDS"),
    "shared store": ("STS",),
    "shared atomic": ("ATOMS",),
    "control": ("BRA", "BSSY", "BSYNC", "BAR", "WARPSYNC", "EXIT", "CALL", "RET", "NOP", "DEPBAR", "YIELD",
                "LDGDEPBAR"),
    "shuffle": ("SHFL", "REDUX", "MATCH"),
    "global": ("LDGSTS", "LDG", "STG", "RED", "ATOMG", "ATOM"),
    "pack": ("PRMT", "F2FP"),
    "conversion unit": ("F2I", "I2F", "F2F", "MUFU"),
    "float": ("FFMA", "FMUL", "FADD", "FMNMX", "HFMA2", "HMUL2", "HADD2", "HMNMX2"),
    "select/compare": ("SEL", "FSEL", "ISETP", "FSETP", "HSETP2", "PLOP3", "VOTE"),
    "integer/address": ("IMAD", "IADD3", "LEA", "LOP3", "SHF", "IMNMX", "IABS", "MOV", "S2R", "CS2R", "POPC", "FLO",
                        "LDC", "S2UR", "R2UR"),
}


def opcode_class(opcode: str) -> str:
    """The class an opcode's budget counts it in, `other` for none; a uniform-datapath op (`UIADD3`, `ULDC`) its vector
    op's class."""
    found = next((name for name, prefixes in OPCODE_CLASSES.items() if opcode.startswith(prefixes)), "other")
    return opcode_class(opcode[1:]) if found == "other" and opcode.startswith("U") and len(opcode) > 1 else found


#: `//## File "path", line N` -- one frame of the inline chain under `--print-line-info-inline`
LINE_INFO = re.compile(r'//## File "([^"]+)", line (\d+)')
#: `.section .text.MANGLED,` -- a function's code section in an `nvdisasm` listing; offsets restart in every one
SECTION = re.compile(r"^\s*\.section\s+\.text\.([^,\s]+),")
#: `Function : NAME` and `arch = sm_XX`, the section headers of `cuobjdump -sass`
FUNCTION = re.compile(r"^\s*Function : (\S+)")
ARCH = re.compile(r"^\s*arch = sm_(\d+a?)\s*$")


#: the extracted builds the cache keeps, the one just used among them, by last use
KEEP_BUILDS = 4
#: a build (or a crashed extraction's staging) unused this long may be pruned: a reader of it has finished
PRUNE_AFTER_S = 3600


def cubins(path: Path) -> list[Path]:
    """The cubins of `path`: itself when it is one, else every ELF image `cuobjdump -xelf` writes out of the extension,
    in the scratch cache under the binary's content hash -- extracted once a build, so the directory names the binary
    it came from and the cache holds a few builds, not one copy a run."""
    path = Path(path)
    if path.suffix == ".cubin":
        return [path]
    root = scratch("cubins")
    with path.open("rb") as f:
        out = root / hashlib.file_digest(f, "sha256").hexdigest()[:16]
    if not out.is_dir():
        staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=root))
        subprocess.run([tool("cuobjdump"), "-xelf", "all", str(path.resolve())], cwd=staging,
                       capture_output=True, check=True)
        try:
            staging.rename(out)
        except OSError:
            if not out.is_dir():
                raise
            shutil.rmtree(staging)
    os.utime(out)
    _prune(root, out)
    return sorted(out.iterdir())


def _prune(root: Path, used: Path) -> None:
    """Every build past the newest `KEEP_BUILDS` and every staging directory, once unused for `PRUNE_AFTER_S`."""
    idle = time.time() - PRUNE_AFTER_S
    seen = []
    for d in root.iterdir():
        try:
            if d.is_dir() and d != used:
                seen.append((d.stat().st_mtime, d))
        except FileNotFoundError:
            continue
    kept = 0
    for mtime, d in sorted(seen, reverse=True):
        if not d.name.startswith(".") and kept < KEEP_BUILDS - 1:
            kept += 1
        elif mtime < idle:
            shutil.rmtree(d, ignore_errors=True)


def device_arch() -> str:
    """The running device's architecture as a cubin names it (`sm_86`): what a launch profiled on this card ran."""
    import torch

    sys.path.insert(1, str(Path(__file__).resolve().parents[1]))
    from measure import hardware

    return hardware.device_arch(*torch.cuda.get_device_capability())


def cubin(path: Path, member: str, arch: str) -> Path:
    """The one cubin of `path` whose name holds `member` and `arch`; refuses where that is not exactly one image. A
    fatbin holds one image an architecture, and an instrument attributes a launch to the image the DEVICE ran -- the
    first match by name was the sm_80 image on an sm_86 card, and every census read it before this seam refused."""
    found = [image for image in cubins(path) if member in image.name and arch in image.name]
    if len(found) != 1:
        raise SystemExit(f"{member} for {arch} names {len(found)} ELF images in {Path(path).name}; "
                         "the instruments read one")
    return found[0]


@contextmanager
def sass_lines(obj: str | Path) -> Iterator[Iterator[str]]:
    """`cuobjdump -sass` over a whole object, STREAMED: the ratification scans megabytes of it and never holds the text.
    A disassembler that fails is fatal -- a gate cannot pass a binary it could not read."""
    proc = subprocess.Popen([tool("cuobjdump"), "-sass", str(obj)], stdout=subprocess.PIPE, text=True)
    try:
        yield proc.stdout
    finally:
        rc = proc.wait()
    if rc != 0:
        raise SystemExit(f"cuobjdump -sass FAILED (rc={rc}) on {obj}; the SASS gates cannot pass a binary they "
                         "cannot read")


def disassemble(cubin_path: Path, *flags: str) -> str:
    """`nvdisasm [flags] cubin`, as text. `--print-line-info-inline -gi` is what the ledgers pass for the frame chain.
    Read once a process for each version of the file (its path, size and mtime) and flag set: the gate asked for the
    same disassembly once a function. -- see docs/internals/tools/sass_gate.md#cost"""
    st = Path(cubin_path).stat()
    return _disassemble(str(cubin_path), st.st_size, st.st_mtime_ns, flags)


@functools.lru_cache(maxsize=64)
def _disassemble(path: str, _size: int, _mtime_ns: int, flags: tuple[str, ...]) -> str:
    return subprocess.run([tool("nvdisasm"), *flags, path],
                          capture_output=True, text=True, check=True).stdout


def one_kernel(names: list[str], kernel: str | None, where: str) -> str:
    """The one name of `names` matching the regex `kernel` -- or the only name, with none given; refuses otherwise, so an
    instrument never merges two functions' offsets (they restart in every function)."""
    found = [n for n in names if kernel is None or re.search(kernel, n)]
    if len(found) != 1:
        raise SystemExit(f"{where}: {len(found)} functions match {kernel!r} among {names}; name one (`--kernel`)")
    return found[0]


def sections(listing: str) -> dict[str, str]:
    """An `nvdisasm` listing's code sections by mangled name, each from its `.section .text.NAME` line to the next
    section."""
    out: dict[str, list[str]] = {}
    name = None
    for line in listing.splitlines():
        m = SECTION.match(line)
        if m:
            name = m.group(1)
            out[name] = [line]
        elif name is not None and re.match(r"\s*\.section\s", line):
            name = None
        elif name is not None:
            out[name].append(line)
    return {k: "\n".join(v) for k, v in out.items()}


def section(listing: str, kernel: str | None, where: str = "the listing") -> str:
    """The code section of the one function matching `kernel` (`one_kernel`)."""
    by_name = sections(listing)
    return by_name[one_kernel(list(by_name), kernel, where)]


def frames(text: str) -> dict[int, list[tuple[str, int]]]:
    """SASS offset -> the inline frame chain at that instruction, innermost first, as (file name, line) pairs, off
    `--print-line-info-inline` output of ONE function (`section`): the comments precede the instruction they belong to,
    and an instruction with no comment of its own inherits the chain above it. Refuses a listing of several functions,
    whose offsets would overwrite each other's."""
    if sum(1 for line in text.splitlines() if SECTION.match(line)) > 1:
        raise SystemExit("sass.frames: a listing of several functions; map one function's section (`sass.section`)")
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


__all__ = ["ARCH", "ENCODING", "FUNCTION", "INSTRUCTION", "LINE_INFO", "SECTION", "cubin", "cubins", "device_arch",
           "disassemble", "frames", "one_kernel", "sass_lines", "section", "sections"]
