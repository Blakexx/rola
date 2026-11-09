# `csrc/rola/src/common/hardware.cuh` — the hardware configuration

The hardware a kernel runs on, mapped to the constants its code and its choices read, in one file:
`csrc/rola/src/common/hardware.json`. `hardware.cuh` is generated from it (`python -m measure.hardware --write`)
and must never be edited by hand; `measure/hardware.py` reads it, resolves it and checks it, and
`tools/lint/lint_standards.py` gates the check on every commit.

<a id="layers"></a>

## Architecture defaults, product overrides

A compute capability fixes the SM's design -- its capabilities, its shared memory and registers, its schedulers,
its MMA instructions, its shared atomic unit -- but not every rate: products of one architecture differ. GeForce
cards run fp16 x fp16 -> fp32 tensor math at half the fp16-accumulate rate, the professional and datacenter
parts of the same chip at full rate (the GA102 whitepaper: Table 3, the RTX A6000 and A40 at 154.8 and 149.7
dense TFLOPS either way; Table 9, the RTX 3090 at 71 against 142). So the file holds:

- `arch.<sm_XX>`: the architecture's defaults -- its `caps` row and its `rates`, each rate at what the SM design
  does (sm_86's fp32-accumulate HMMA at 16 cycles, the full rate).
- `product.<slug>`: a product's overrides over its architecture's defaults (the RTX 3080 Ti's fp32-accumulate
  HMMA at 32 cycles, and every rate measured on it). The slug is `measure/hw_profile.py`'s: the device name with
  the vendor words and the memory suffix dropped (`NVIDIA GeForce RTX 3080 Ti` -> `rtx3080ti`).

A product's rates are its architecture's under its overrides (`hardware.resolve(product=...)`); an
architecture's alone are `hardware.resolve(arch)`. A device with no product row runs on its architecture's
defaults, which are what the SM design does.

<a id="families"></a>

## Families, builds and products

Every architecture row names its FAMILY in its `build` block -- `ampere` (sm_80 to sm_89), `hopper` (sm_90),
`blackwell` (reserved) -- the name a kernel's schedule is selected by (`Family`; the constitution's R8). Instruction
differences stay at the atom, read at the compiling architecture.

<a id="builds"></a>
A BUILD is the key a kernel compiles at: `Build{cc, variant}`. Variant 0 is the architecture's own `build` block; a
product whose `build` block overrides a trait (`hardware.BUILD_TRAITS`) runs the next variant, and products whose
overrides agree share one (`hardware.builds()`), so a kernel is compiled once a distinct set of traits, never once a
product. Every kernel that reads a build trait compiles at each carried build, and an image carries every variant of
its architecture. A variant is NOT GATED yet: the oracle set and the SASS gate run one build an architecture, so the
check refuses a product override until the first product that needs one brings its variant under both.

<a id="builds-renamed"></a>
A kernel templated on its build is the same code under a new name, so a refactor compares by signature:
`tools/sass_bodies.py --compare A B --drop-builds` keys both maps by the demangled signature with each kernel's
`BuildTag<Build{...}>` argument dropped -- refused where the argument is not its own image's, since a build's kernels compile in its image
alone -- and a kernel whose only template argument was its build keyed as the untemplated kernel it replaced.
Every comparison, with the flag or without, reports the source stamp (`csrc_stamp_kernel`, which moves with
every source edit) apart, never as a verdict.

<a id="build-tag"></a>
A kernel takes its build as a TYPE, `BuildTag<B>` (its `build` and `traits` as constants): nvcc 13.0's host stub
cannot name a kernel's class-type template argument whose type sits in a namespace (it prints the type's mangled name),
while a class template carrying it compiles -- so host templates and plans take `Build` itself, kernels its tag.

<a id="traits"></a>
`traits_of(Build)` is what a kernel's plan reads: the family, the window W ([`window.md`](window.md): a card's,
chosen per architecture) and the architecture's `Caps`. A tabulated architecture that is not built may leave its
window unchosen; a built one must choose it. sm_90's is 1,024, the A100's, provisional until the sm_90 carry's design
picks it (Blake, 2026-11-09). The check
(`measure/hardware.py --check`) refuses an architecture whose build names no family of `FAMILIES` or a trait outside
the family and `BUILD_TRAITS`, a built architecture whose build chooses no window, and a product overriding a trait
outside `BUILD_TRAITS`.

<a id="products"></a>
`kProducts` names each product row's slug, architecture and variant: the host finds a device's product by the slug
of its name and runs the product's variant, or its architecture's variant 0 when it has no row.

<a id="consumers"></a>

## Two consumers

- COMPILE TIME reads a BUILD's traits (below): its architecture's capabilities and the build choices. `hardware.cuh`
  carries every tabulated architecture's `Caps` row, `tabulated(cc)` and `caps_of(cc)`, and every build's
  `traits_of(Build)`; [`arch_caps.cuh`](arch_caps.md) includes it and names the compiling target's row, and
  [`arch_runtime.cu`](arch_runtime.md) checks the row against the driver.
- RUN TIME reads rates: the readout's accumulation compares the device's fp16- and fp32-accumulate MMA rates on the
  host (`readout_f16`) -- the binary carries every architecture's and product's MMA rates (`kMmaRows`), found by the
  device's name ([`arch_runtime.md#mma-quanta`](arch_runtime.md#mma-quanta)): the same device always gets the same
  form.

`hardware.cuh` also carries the MMA kinds (`Mma`), the order a product's quanta carry their MMA rates in.

<a id="sources"></a>

## Every rate has a source

A rate is `{"cycles": ..., <source>}` with exactly one source:

- `measured` -- a lab (`measure/lab/results/<lab>/sm_XX.json`), its result `row`, and the product it was measured
  `on` (a product's own overrides are measured on it). The check refuses a value off its record's median by more
  than its rounding, or a record taken on another device.
- `documented` -- the public document and the figure in it.
- `announced` -- a value no source supports there yet, with why (sm_80's atomics are the RTX 3080 Ti's until a
  rented A100 measures them): a known gap in plain sight, never a silent default.

The rates: `schedulers` (an SM), `issue` (an instruction a scheduler), `wavefront` (a shared-memory load
wavefront an SM), `atomic` and `atomic_wavefront` (a shared atomic instruction's least and each of its
wavefronts an SM), `mma.<kind>` (an MMA a scheduler), `tax.<kind>.<class>` (what an instruction of the
class adds to the kind's MMA chain, read by the issue-hiding gate, not by a choice), `latency.<class>` (issue to a
dependent's issue: an FFMA, a 32-bit shared load, an `ldmatrix` of four, an HMMA on its accumulator; the schema's
`latency` list), and `shared.<kind>`, `global.<kind>` and `pipe.<name>` (below; the schema's lists). NVIDIA documents
issue, the shared-memory bank rate, peak tensor rates and the arithmetic pipes' throughputs; a shared atomic's cost,
the shared pipe's and the global path's other prices, the taxes and the latencies are measured
([`smem_rates`](../../lab/smem_rates.md), [`drain`](../../lab/drain.md),
[`tensor_tax`](../../lab/tensor_tax.md), [`issue_latency`](../../lab/issue_latency.md)).

<a id="shared"></a>

### The shared-memory pipe's prices (`shared.*`)

What the pipe takes, an SM, beyond a load's wavefront (`wavefront`) and a shared atomic's (`atomic`,
`atomic_wavefront`), all measured in [`smem_rates`](../../lab/smem_rates.md) on the RTX 3080 Ti (sm_80's announced
the same until a rented A100 measures them): `store` and `store_wavefront` (a plain shared store takes the larger of 2
cycles an instruction and ~1.14 a wavefront: 4 bytes a lane costs twice what 8 or 16 do a byte); `copy_bypass` and
`copy_access` (a `cp.async.cg` copy's write into shared memory, 0.5 cycles a 128 bytes; a `.ca` copy's, 5.4);
`global_read` and `global_fill` (a global load's data read out of the banks, 1.19 a 128 bytes, and what a line from L2
adds by being written into them first, 0.38). [`tools/pipes.py`](../tools/pipes.md) prices a launch's traffic by these.

<a id="global"></a>

### The SM's global path (`global.*`)

Measured with every SM busy, on the RTX 3080 Ti (sm_80's announced the same until a rented A100 measures them):
`l2_line`, the cycles an SM's share of L2 takes to deliver a 128-byte line (`smem_rates`' `cp.async.cg` stream alone:
~18 bytes a cycle an SM), and `red_request`, a global atomic's line request (the [`drain`](../../lab/drain.md) lab's
skipping form: ~5 cycles). [`tools/pipes.py`](../tools/pipes.md) prices a launch's L2 lines and atomic requests by
them.

<a id="pipes"></a>

### The arithmetic pipes (`pipe.*`)

Cycles a warp instruction holds its pipe a scheduler, from the CUDA C++ Programming Guide's throughput table (results
a clock an SM, over four schedulers and 32 lanes): `alu` 2 (integer add, compare, logic, shift: 64 a clock on 8.0 and
8.6), `fma` 2 on sm_80 and 1 on sm_86 (fp32: 64 and 128), `fma_fp16` 1 (a half2 instruction), `fmaheavy` 2 (integer
multiply and multiply-add), `xu` 8 (special functions, conversions: 16). Checked on sm_86 against ncu's busy counters
to the percent ([pipes.md#checked](../tools/pipes.md#checked)).

<a id="check"></a>

## The check

`python -m measure.hardware --check`: the header is what the file generates; every architecture's capabilities
are the schema's; every built architecture (one with a manifest under `tools/manifests/`) has a row whose
defaults can price a choice (`schedulers`, `issue`, `wavefront`, `atomic`, `atomic_wavefront`, every MMA kind);
every product names a tabulated architecture and its device is that product; every rate is a field the schema
names, with exactly one source that holds.

<a id="extend"></a>

## Extending it

- A product: its block under `product`, with the overrides that differ from its architecture (and a
  confirmation lab on the card).
- An architecture: its `caps` row and its default rates (documented where a document gives them, the rest
  measured), then `--write`; [`bringup.md`](../../bringup.md) walks it.
- A rate: a field in the schema's vocabulary, a value with its source in each row that has it.
