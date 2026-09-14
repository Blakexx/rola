# Bringing up a new architecture

An architecture this package has not been ratified on **will refuse to build and
refuse to load**. That is the product statement, not a defect. Bringing one up is a
measurement, and this is the procedure.

You need: the target GPU, a declared toolchain's `nvcc`/`ptxas` (or a new toolchain —
see the last section), and the ability
to open a pull request.

## 1. Tabulate the architecture's resources

`csrc/rola/src/chunk/arch_caps.cuh` carries, per compute capability, the capability rows and the numbers
the launch-bound formula needs:

```cpp
struct SmResources {
  int smem_per_sm;         // cudaDevAttrMaxSharedMemoryPerMultiprocessor
  int max_threads_per_sm;  // cudaDeviceProp::maxThreadsPerMultiProcessor
  int regs_per_sm;         // cudaDeviceProp::regsPerMultiprocessor
};
```

Take them from the CUDA C Programming Guide's *Compute Capabilities* table and add
the row to `sm_arch_tabulated()` and `sm_resources()`. **Do not guess**: an
overstated resource makes `ptxas` silently discard the launch bound, which is a
silent occupancy loss on the architecture nobody checked. `check_arch_table()`
(`csrc/rola/src/common/arch_runtime.cu`, called from every family's launching entry,
carry included) checks every row against the driver at run time, so a wrong number
is caught — but it is caught by a user unless you check it first.

## 2. Probe and ratify

```bash
python tools/ratify.py --arch 89 --write
```

This compiles the whole dispatch closure for `sm_89`, reads `ptxas -v`, and writes
`tools/manifests/<toolchain>/sm_89.json` — **and touches no other architecture's file.** No GPU
is required: ratification is a compile-time measurement.

Read the output before committing it. Two things are not automatic:

- **Unhonored bounds.** Any *"Value of threads per SM … is out of range"* means
  that instantiation compiled with no launch bound at all. Fix step 1 rather than
  recording the numbers.
- **New spills.** If instantiations spill on this architecture and not on the
  ratified ones, that is a real finding about the register file, and it belongs in
  the pull-request description with the numbers. A manifest records what was
  measured; it does not excuse it.

## 3. Gate

```bash
python tools/ratify.py --arch 80 --arch 86 --arch 89
python tools/ratify.py --self-test
python tools/supported.py --write     # the README table is generated
```

`--self-test` must still fail on all four things it is required to reject. The
supported-configurations table will move `sm_89` from *not ratified* to *ratified*
on its own — it is derived from the arch table and the manifest directory, and CI
fails if it is edited by hand.

## 4. Build and test on the device

```bash
ROLA_CUDA_ARCHS="80;86;89" pip install -e . --no-build-isolation
pytest tests/oracle
pytest tests/integration
pytest tests/unit
```

The ORACLE tier says the kernel computes the right function on this hardware. The
INTEGRATION tier says it computes the same function at every geometry the built
matrix carries there, and
under both state backings — which
is the part a new register file is most likely to disturb, because it is the part
that depends on how the compiler laid the kernel out.

## 5. Open a pull request

**A manifest change is always a reviewed PR, never a CI auto-commit.** The diff
should be: the arch-table row, one new `tools/manifests/<toolchain>/sm_XX.json`, the
regenerated README block, and a description carrying the measured spill and
register profile with any anomaly explained.

## Changing the toolchain

Adding a CUDA version is not a CI matrix cell; it is a **re-ratification of every
architecture**, because every number in every manifest is a property of one
assembler. The wheel matrix axis and the ratification axis are the same axis — that
is the whole closed-world point. Budget it as a measurement run, not as a config
change.

1. **Install the toolkit** and point the dev config's `toolchain.cuda_home` at it. NVIDIA's
   redistributable archives install without root; they place libraries in `lib/`, where `nvcc`
   and torch look in `lib64/`, so add `lib64 -> lib`.
2. **Write the record**, `tools/toolchains/<name>.json`: the whole `ptxas --version` output, the
   CUDA major, the PyTorch wheel index for that major. A torch of another major is refused.
3. **Check the compiler cache.** The pinned sccache must compile under the toolkit (sccache 0.17
   silently produced no objects under nvcc 13.4); otherwise build with
   `toolchain.sccache_enabled: false` until the pin moves.
4. **Census before writing.** `python tools/ratify.py --census --json <file>` under the new toolkit
   against the current toolchain's census: every entry's spill, registers and in-loop locals. A
   worse entry is a kernel finding, not a verdict on the toolkit.
5. **Ratify** (`--write`, every shipped arch) into `tools/manifests/<name>/`, build, and run the
   suite's timing A/B against the current toolchain's build of the same source in one interleaved
   session, repeated; isolate any regression (compiler against library) before choosing.
6. **Retire the old toolchain** in a deletion-only commit: its record and its manifests.
