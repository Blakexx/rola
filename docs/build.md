# Build — ahead of time, closed world

What the build is and what it guarantees: the closed-world rules it enforces, the iteration builds it allows, the arm
lists, the flags' measured decisions, the stable ABI and the two wheels. The build is the graph's; its targets,
parameters, rules and gates are [`build-graph.md`](build-graph.md). rola does not build from source: `pip install .`
refuses, naming the targets below (`pyproject.toml`'s backend, `tools/no_source_build.py`).

## The governing constraint

The kernels are compiled **ahead of time** into a fatbin containing
`code=sm_XX` images only. There is no JIT, no PTX for the driver to compile, and no
runtime autotuning. Everything the binary can execute has been assembled by a
`ptxas` whose output was measured and committed
([`ratification.md`](ratification.md)).

That is a deliberate trade. A JIT-and-autotune build (Triton's model) adapts to
hardware nobody tested, at the cost of shipping codegen nobody looked at; the
supported matrix then means "we expect it to work". Here the supported matrix
**is** the ratified set, by construction, and the build refuses to widen it
silently.

## <a id="building"></a>Building

```bash
bazel run //:library                                   # every ratified architecture and arm, into rola_cu13/
bazel run //:library --//:archs=86 --//:carry_arms=3   # an iteration build (below)
bazel run //:wheels                                    # the two wheels of a clean, ratified commit (below)
```

A build's PARAMETERS are the graph's settings -- `--//:archs`, `--//:carry_arms`, `--//:decode_arms`,
`--//:carry_parts` ([`build-graph.md`](build-graph.md#compile)) -- and everything about the MACHINE is the dev config.
`host.nvcc_threads` is `nvcc --threads`, default 2, the number of `-gencode` targets: beyond the target count it buys
nothing and multiplies the resident set, since the two parallelism axes multiply. Every build runs under the
machine-wide host budget, which the front door takes for it ([`build-graph.md`](build-graph.md#front-door)); any other
CPU-heavy command runs under the same pool with `python -m rola_devtools.locks.host [--exclusive] -- <cmd>`.

## What the build checks

The library's record action and gates ([`build-graph.md`](build-graph.md#gates)) refuse a build that breaks a
closed-world rule ([`ratification.md`](ratification.md#the-four-closed-world-rules)):

1. **R2 — cubins only.** Each architecture is compiled `code=sm_XX` alone (the library's transition writes rules_cuda's
   spec `compute_XX:sm_XX`), and the SASS gate refuses a library carrying a PTX image. A `code=compute_XX` entry would
   embed PTX and let the driver just-in-time compile for an architecture whose register and spill behaviour was never
   measured.
2. **R3 — ratified archs.** `tools/manifests/<toolchain>/sm_XX.json` must exist and be non-empty for every arch being
   built. A file-existence question: the cheapest check, which must never be tempting to skip.
3. **R1 — toolchain.** The graph's `ptxas --version` selects the toolchain: the record in `tools/toolchains/` that
   names it, whitespace-normalized ([`internals/tools/toolchains.md`](internals/tools/toolchains.md)). A toolkit no
   record names refuses; that toolchain's manifests must pin the same string; the disassemblers the gates read with
   (`cuobjdump`, `nvdisasm`) must report the record's `readers` byte for byte; torch must be built for the toolchain's
   CUDA major (`tools/toolchains.py`'s `rule1`). The whole output, not a parsed release number: the build string
   distinguishes two shipments of one nominal release, and it is exactly those that can move an allocation while the
   version says nothing changed. In the same pass, R5: the vendored subtree under `csrc/third_party/` must hash to its
   `PIN.json` digest.
4. **Shards.** The checked-in instantiation shards must be the ones `tools/gen_shards.py` produces. Generated sources
   are committed so that what is reviewed is what compiles; a stale one fails here rather than shipping a matrix
   nobody wrote.
5. **The arm table.** The generator's table still matches the declaration, the selection header names exactly the arms
   this build selected, and the built `_C` carries exactly those
   ([`ratification.md#arm-switch`](ratification.md#arm-switch)).
6. **The SASS gate.** `tools/sass_gate.py` on the linked library (KERNEL_STANDARDS §20): local memory, convergence
   subroutines, predicated HMMA, the step lints, a PTX image (R2 on the binary itself), and a kernel the source
   defines that no cubin carries. It compares
   against no manifest, so it runs on every build, iteration builds included.
7. **`rola_cu13/_build_config.py`**, the record: the archs, the assembler, the torch version and ABI flag, the
   manifest digest and per-arch entry counts, the vendored pin, whether it is an iteration build, and
   `ptx_jit_fallback: False`.

Ratify is not among them: the committed manifests go stale between milestones by design (KERNEL_STANDARDS §16), so
its verdict on a build is a target of its own, `bazel test //:ratified`, which a milestone and every wheel require
([`build-graph.md`](build-graph.md#ratify)).

<a id="generated-headers"></a>The build's GENERATED HEADERS (the arm selections, the carry parts, the builds, the
csrc stamp) are actions over their generators, and `bazel run //:library` copies them into `build/generated/`, where
a compile outside the graph reads them, and ratify's compile path writes the same files there for its own run. Only `common/build_stamp.cu` reads the csrc stamp, whose text
changes with every `csrc/` edit.

## <a id="iteration"></a><a id="arm-subset"></a>An iteration build compiles a subset

Instantiation is the dominant build resource, so a developer testing ONE arm may compile one arm on one
architecture:

```bash
bazel run //:library --//:archs=86 --//:decode_arms=5
```

Any parameter but a default makes an ITERATION build (`tools/gen_shards.py`'s `is_iteration`, which the record
states), and three properties make it safe to have:

* **The default is every declared arm and every ratified architecture.** An empty arm setting expands to the whole
  list, so a plain build and every wheel are the closed-world one. There is no configuration in which a subset is the
  default.
* **A subset binary is SELF-IDENTIFYING and refuses what it lacks.** `rola.ops.carry.arms()` and
  `rola.ops.decode.arms()` return the rows the fatbin actually carries, read off the same list the build compiled
  from; asking a launch for any other shape is a `STD_TORCH_CHECK` refusal naming the arm, never a mis-launch.
* **It is NOT SHIPPABLE, and its record says so.** Ratify proves the fatbin carries EVERY ratified instantiation in its
  assigned shard; a subset carries fewer by construction, so it is never ratified and never a wheel. Ratification is
  a milestone's, from a full build (KERNEL_STANDARDS section 16).

**AND THE RATIFICATION MUST NOT INHERIT IT, WHICH IS THE SUBTLE HALF.** The graph's ratify reads the build's
parameters with its census and refuses an iteration build (`tools/ratify.py`'s `graph_parameters`), so no manifest
records a subset. And no arm subset is a compile flag: each family's selection is a generated header
(`carry_selection.inc`, `decode_selection.inc`;
[`internals/tools/gen_shards.md#families`](internals/tools/gen_shards.md#families)) written for the build's own
scope, so a flag naming WHAT WAS COMPILED, rather than HOW, never reaches a unit's command line, where a ratification
reading the compile could inherit it. The one such flag is `-gencode`, the architectures, and ratify's compile path
takes them from its own run, never from a build. Measured when a decode subset was a `-D`: the next ratification
inherited it and wrote a manifest of 393 entries carrying ONE decode arm, while the library carried all sixteen --
and it PASSED. A gate that certifies numbers no shipped binary has is worse than no gate.

The same rule governs `--//:archs=86`: iterate single-arch, gate and measure on the full both-arch build.
`--//:carry_parts` <a id="carry-parts"></a> builds some of the carry kernel's parts (`readout.stream`, `readout.loads`,
`readout.drain`, `fold.pool`, `fold.ring`, `fold.loads`) REAL and the rest as STUBS that keep their HMMAs and drop
their other work. With
every part real the arm's instructions are identical to a build without the switch. Never shipped.

## <a id="compile-ledger"></a>Why the arm units are torch-free

Instantiation is the dominant build resource, and with per-arm units it is not the kernels that dominate but the
tensor library's headers: a translation unit that includes them parses longer, and holds a larger resident set, than an
arm's own codegen. So the carry family's per-arm ABI (`csrc/rola/src/carry/carry_arm_abi.cuh`) speaks POINTERS and not
tensors, and only the dispatch unit includes the torch seam: a tensor-typed seam would re-pay the headers once per arm
and pin every arm unit's resident set at theirs. The ast-grep rule `tools/lint/rules/torch_include_in_arm_tu.yml`
refuses a torch header in a per-arm unit. MEASURED per `cicc` (sm_86, `--threads=1`):

| translation unit | wall | peak a `cicc` |
|---|---|---|
| a bare CUDA unit (one trivial kernel) | 0.9 s | 0.19 GB |
| the same, plus `torch/extension.h` | 33.7 s | 2.86 GB |
| one carry arm (the torch-free ABI) | 4.7 s | 0.33 GB |
| `carry/carry.cu` (the dispatch alone) | 36.2 s | 2.88 GB |

Over a full both-arch build every unit that parses the extension header lands in one narrow band (`factor.cu` 2.92 GB,
`entmax.cu` 2.89, `decode.cu` 2.81, `carry/carry.cu` 2.75), and no generated arm unit is in the top twelve; the carry
family's single unit, before the per-arm split, peaked at 6.14 GB. A build's peak is the header's footprint times the
`cicc` the host budget runs at once, and it stops being a function of the arm count.

Per-arm units are what let the arms compile in parallel across the jobs the host budget grants, and what make `ratify`
per-unit incremental, keyed on the argv and the content of every input the compiler reported reading.

## <a id="carry-arm-lists"></a>The carry arm list

`tools/manifests/shipped_set.json` declares the carry family's arms and `tools/gen_shards.py` reads it. An arm is the
carve's operation-plan pair, keyed `fold_rows, fold_factors, read_rows, read_factors, channels, same_split`
([`internals/engine/rules.md#carve`](internals/engine/rules.md#carve)); the box's per-level extents, the depth and the
widths are runtime addressing. The declaration has one `shipped` list: every declared row is built by a default build,
ratified in the manifests and shipped. Each row is one generated `carry_arm_<i>.cu`, `i` its index in the
canonical-sorted
list.

A binary answers which arms it carries itself (`rola.ops.carry.arms()`), never the source tree. A launch whose carve
selects an arm the binary does not carry is refused by name, with the key field that has no built member.

## <a id="kernel-identity"></a>A kernel's code is a function of its source and toolchain

**The graph's cache is content-addressed.** A unit's object is cached under its command line and the content of every
input its compile read (nvcc's dependency file, [`build-graph.md`](build-graph.md#compile)), shared by every checkout
on the machine, so a hit is the object a cold compile of the same inputs makes. The graph compiles with no other
compiler cache.

**A written hash map carries the hasher's version and is comparable only with its own.** What goes into a digest is a
property of `tools/sass_bodies.py`, not of the binary, so maps are stamped with `HASH_VERSION` and `--compare` refuses
a mismatched or unstamped pair rather than reporting the whole matrix as differing.

**Worktree builds and cross-path SASS comparison are ordinary operations.** Every kernel lives in a named namespace,
so a ratified name is a function of source content and toolchain only: an anonymous-namespace kernel's mangled name
embeds a discriminator derived from the source directory, and the ast-grep rule refuses one
([`internals/tools/lint.md`](internals/tools/lint.md)).

**`--split-compile` is REJECTED, measured.** It changes the assembled code of most entries (1,073 of 1,228 when
measured), which under the closed-world policy is a kernel change requiring re-ratification, and it is slower besides:
the two parallelism axes multiply, collapsing the job count.

## <a id="fatbin-compression"></a>The fatbin is compressed explicitly

`-Xfatbin -compress-all` is in the flag list (`tools/build_flags.py`, which both the
graph's compile and ratify's compile path take). `fatbinary` compresses by default, but that
default covers PTX and debug images only, and for the cubins it applies a SIZE
HEURISTIC (every cubin it stored raw was <= 9,076,960 B, every one it compressed >= 10,075,232 B when measured),
which would make the shipped artifact a function of how the instantiations are partitioned into translation units
rather than of the code. The explicit flag compresses every cubin, so the artifact's size is the code's: measured on
one 256-instantiation matrix, `-compress-all` beat both partitions the heuristic was given (61.3 MB against 73.7 and
141.9 MB).

Compression is applied to the image AFTER `ptxas` has assembled it, so it cannot move
codegen — proven rather than asserted, by comparing every entry point's SASS across
the flag change: **0 differing, 0 absent either way**. Taking the
ratification again under the new flag list reproduces every register and spill number,
so the manifests come back byte-identical.

THE COST IS PAID AT MODULE LOAD, AND IT IS MEASURED. Three runs each, fresh process,
CUDA context already up so the number is the module and not the context:

| | before | after |
|---|---|---|
| extension import | 2.01 / 2.04 / 2.13 s | 2.08 / 2.18 / 2.60 s |
| FIRST driver touch of a kernel | 0.0050 s (x3) | 0.0086 / 0.0088 / 0.0088 s |
| a warm query | 3-4 us | 4-6 us |

**+3.7 ms once per process, at the first launch** — that is the decompression, and it
is where it should be. Import is noise-dominated and shows no separable effect.

The manifest's `toolchain` block records the codegen flag list (`nvcc_flags`), so a
codegen-affecting flag change is a fingerprint the ratification gate refuses
(`tools/ratify.py`'s `check_flags`, one of
[`ratification.md`'s closed-world rules](ratification.md#the-four-closed-world-rules)).

## Verifying a built binary

```bash
python -c "from rola_cu13 import _build_config; print(_build_config.show())"
bazel run //tools:sass_gate -- rola_cu13/_C.abi3.so    # its signatures, and no PTX image (R2)
bazel test //:ratified                                  # must PASS
```

The shipped `.so` carries one ELF image per (translation unit, architecture) and **0 PTX images**. That is not a
side effect; it is R2, checked by the SASS gate on every build (`cuobjdump --list-ptx` lists nothing).

## <a id="stable-abi"></a>The binary targets torch's stable ABI

The extension is built against torch's **stable** C shim and header-only types and against nothing of torch's C++ ABI:
every kernel's host code goes through `csrc/rola/src/common/torch_seam.cuh`, and the registration TU declares operators
(`torch.ops.rola.*`) instead of binding a pybind module ([`internals/rola_api.md`](internals/rola_api.md)).
`tools/build_flags.py` pins the targeted minimum (`TORCH_MIN = (2, 13)`, passed as `-DTORCH_TARGET_VERSION`, so any
stable call newer than the minimum fails to compile), and every unit compiles against Python's limited API
(`-DPy_LIMITED_API`, torch's minimum CPython; the `_C.abi3.so` name). One binary therefore loads under every torch from 2.13 on and every CPython from 3.10 on.

2.13 is the minimum because it is the first torch whose stable surface has everything the host code uses; the newest
piece is the native handle of the current CUDA stream. Raising the minimum is a one-line change in `tools/build_flags.py`; lowering
it needs every call above the new minimum replaced first, and the compiler names each one. The gate for a change here is
a build against the minimum torch and the battery on the minimum and on the newest.

## <a id="wheels"></a>Two wheels: `rola` and its binary plugin `rola-cu13`

```bash
pip install "rola[cu13]"
```

installs two distributions at one version. **`rola`** is pure Python (`py3-none-any`): the layer, the routing, the
decay, the refusals. **`rola-cu13`** (`cp310-abi3-linux_x86_64`) is the binary plugin of the cu13 toolchain: the package
`rola_cu13` holding the fatbin `_C.abi3.so` for every ratified architecture, the build record `_build_config.py`, and
its `manifests/` record, the ratification that record's `manifest_sha256` hashes. A future toolchain is a second plugin
beside it, never a rebuild of `rola`.

`rola` loads the plugin when a kernel is first called (`rola/ops/_ext.py`). It reads the plugin's build record BEFORE
its library, and a record of another version refuses without registering an operator: the two are one version. With no
plugin, or a mismatched one, the first kernel call raises, naming `pip install "rola[cu13]==<version>"`; the pure-Python
parts (the oracle, the configuration objects) still import. The `cu13` extra is the wheels tool's, not
`pyproject.toml`'s, because it pins `rola`'s own version; the other extras and every other field of the metadata are
`pyproject.toml`'s `[project]`.

A checkout's binary sits where a wheel puts it: `bazel run //:library` copies it to `rola_cu13/_C.abi3.so`, so a
checkout and an installed wheel load the binary by one path. `tools/toolchains.py`'s `built_extension` is how the tools find it.

**The wheels package the graph's library.** `bazel run //:wheels` refuses a tree with changes, then requires ratify's
verdict on the build to pass (which refuses an iteration build), then refuses the build unless its record names this
version, the toolchain and the manifest digest of the files it then packages. It writes
`dist/rola-<v>-py3-none-any.whl` and `dist/rola_cu13-<v>-cp310-abi3-linux_x86_64.whl`, byte-reproducible for a commit
(fixed member order and timestamps). `bazel run //tools:wheels -- --check dist` installs them into a scratch venv over
the environment's torch and requires `rola` alone to refuse naming the extra, and both together to load and name
their carry arms ([`internals/tools/wheels.md`](internals/tools/wheels.md)).

The platform tag is the build host's, `linux_x86_64`: the wheels install on a machine whose glibc is at least the build
host's. A `manylinux` tag, which PyPI requires, is not claimed.

## Dependencies

`torch` (2.13 or newer) and `einops` (`pyproject.toml`); nothing else. The kernels are CUDA, so there is no `triton`
or `fla` dependency.

## Standards lint

`KERNEL_STANDARDS.md` is enforced by the lint family under `tools/lint/` — the ast-grep rules, `lint_standards.py`,
the two pinned clang-tidy profiles and the ratchets — run by the commit gate (`bazel test //:gate`), and by CI through
the graph (`.github/workflows/lint.yml`). Its roster, how each gates, how the clang-tidy
profiles read each unit's own compile and how to run them by hand are
[`internals/tools/lint.md`](internals/tools/lint.md).
