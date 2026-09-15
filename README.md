# RoLA — Routed Linear Attention

RoLA is a linear-attention layer whose recurrent state is a **set of addressable
slots**, and whose per-token routing decides which slots each token reads from and
writes to. The state is a fixed size chosen by the topology, not a cache that grows
with the sequence — so the memory footprint at token *T* is the same as at token 1.
This repository ships the CUDA consumer kernel that executes that routed
recurrence, the layer around it, and the evidence for both.

Paper: *(link on publication)* · Citation: [below](#citation)

---

## What this repository ships that a kernel repository usually does not

**A ratified supported matrix, not a supported-arch aspiration.** Every
architecture this package will run on has a committed manifest of measured
per-instantiation register and spill counts, pinned to the exact assembler that
produced them. The build refuses to compile an architecture without one; the binary
refuses to launch on a device without one. There is **no PTX JIT fallback** — an
unmeasured architecture fails loudly rather than silently running codegen nobody
looked at. → [`tools/manifests/`](tools/manifests/), [`docs/bringup.md`](docs/bringup.md)

**A test taxonomy in three tiers, where each tier ASSUMES the one below it.**
`tests/oracle` gates the kernels against an fp64 oracle (and, once, against
torch's own linear-attention loop) across the full semantic matrix, at tolerances
that are derived and documented rather than tuned until green.
`tests/integration` gates the COMPOSITION — the layer, the dispatch walks, the
prefill/decode handoff, and the two-branches-agree class where the kernel is
gated against **itself** across every geometry the built matrix carries and under
both state backings; engineered identities are tested as **byte equality**, and
legitimate reassociations against the oracle plus a **derived** error bound, with
a pair changing class itself a test failure. `tests/unit` gates each file's own
public contract. A tier never re-grades the arithmetic the tier below it proved.
→ [`docs/testing.md`](docs/testing.md),
[`tests/oracle/`](tests/oracle/), [`tests/integration/`](tests/integration/)

**Committed measured cells, with provenance.** The benchmark results in this
README are files in this repository, each carrying the machine, driver, toolchain,
and the SHA of the ratification manifest the binary was built from. Regression
thresholds are derived from each cell's own measured spread, never an invented
percentage, and the kernel's geometry is read out of the *built binary's own* arm
list rather than modelled.
→ [`benchmarks/cells/`](benchmarks/cells/), [`docs/build.md`](docs/build.md)

**A documentation tree the code is checked against.** The engine's design reasons
live in `docs/internals/`, one document per source file, with one-line hazard
stubs left in the code at the sites they explain; the decisions that outlive any
particular kernel live in `docs/architecture/`. A test fails if a source file has
no mirror document or a stub cites an anchor that no longer exists, so the two
cannot drift apart.
→ [`docs/architecture/`](docs/architecture/README.md),
[`docs/internals/`](docs/internals/README.md)

---

## Installation

### Requirements

| | |
|---|---|
| Python | 3.10 – 3.13 |
| PyTorch | built for the toolchain's CUDA major (`tools/toolchains/<name>.json`, `torch_index`) |
| CUDA toolkit | a declared toolchain's exact `ptxas` — the table below names each ratified one |
| GPU | a **ratified** compute capability (see the table below) |
| Build | `ninja`; source installs need `--no-build-isolation` |

`--no-build-isolation` is not optional and not a workaround: pip's build isolation
would fetch a CPU-only torch from PyPI and build the extension against the wrong
ABI. Flash-attention and mamba-ssm require the same thing for the same reason.

### Install modes

| Mode | Command | Notes |
|---|---|---|
| Wheel | `pip install "rola[cu13]"` | `rola` (pure Python) and `rola-cu13` (the fatbin for the ratified matrix, and its manifests), at one version ([docs/build.md#wheels](docs/build.md#wheels)) |
| Source | `pip install . --no-build-isolation` | From a checkout: builds the fatbin ahead of time, no JIT, both packages in one install |
| Development | `pip install -e . --no-build-isolation` | Runs the R1/R3 gates and the post-build ratification |

The build is **ahead-of-time and closed-world**. It emits `code=sm_XX` only, never
`code=compute_XX`, so the fatbin carries no PTX for a driver to JIT; the shipped
`.so` is asserted to contain zero PTX images. Setting `ROLA_CUDA_ARCHS` to an
architecture with no ratified manifest fails the build before it compiles anything.

The consumer's 74 kernel instantiations are compiled as **three generated
translation units** (`csrc/rola/src/instantiations/`, generated per carry arm),
partitioned as contiguous blocks of the declared arm list, so the matrix parallelizes
across `MAX_JOBS` instead of serializing inside one compiler process. They are
checked in — what is reviewed is what compiles — and `python tools/gen_shards.py
--check` is the gate that keeps them the files their generator produces. `MAX_JOBS`
defaults to a value derived from **available RAM**, which is the resource that
actually binds ([`docs/build.md`](docs/build.md)).

## Supported configurations

This table is **generated** by `tools/supported.py` from the kernel's arch table and
the committed manifests, and CI fails if it drifts. A hand-maintained support table
is a claim; a generated one is a report.

<!-- BEGIN GENERATED: tools/supported.py -->
| Compute capability | GPUs | Toolchain | Ratified under | Instantiations | Status |
|---|---|---|---|---|---|
| `sm_80` | A100, A30 | `cu13` | ptxas 13.0.48 | 345 | ratified |
| `sm_86` | A10, A40, RTX 30xx | `cu13` | ptxas 13.0.48 | 345 | ratified |
| `sm_87` | Jetson Orin | — | — | — | **not ratified — will refuse to load** |
| `sm_89` | L40S, RTX 40xx | — | — | — | **not ratified — will refuse to load** |
| `sm_90` | — | — | — | — | **not ratified — will refuse to load** |
<!-- END GENERATED: tools/supported.py -->

The unratified rows are not an apology; they are the product statement. The first
claim above is worth nothing if this table quietly implied coverage that was never
measured. Bringing up a new architecture is a measurement, and
[`docs/bringup.md`](docs/bringup.md) is the procedure.

### Routing activations

Also generated, from the one registry every validator reads
(`rola/routing/activations.py`). `kerneled` is what decides dispatch: an
unratified activation routes to the differentiable reference path carrying an
explicit `not_kerneled` flag, and the kernel entry refuses it outright — a labelled
research surface, never a silent fallback.

<!-- BEGIN GENERATED: rola.routing.activations.capability_table -->
| activation | normalized | simplex_output | exact_zeros | support_differentiable | kerneled |
| --- | --- | --- | --- | --- | --- |
| entmax(alpha=1.5) | yes | yes | yes | no | yes |
| entmax(alpha=2) | yes | yes | yes | no | yes |
| opaque | yes | no | yes | no | yes |
| softmax | yes | yes | no | yes | yes |
<!-- END GENERATED: rola.routing.activations.capability_table -->

## Usage

The API is the reduction theorem: linear attention's contract, plus ONE new concept —
the feature map. `docs/api.md` is the contract in full.

### The op

```python
import rola

routes   = rola.RouteProducer(rola.uniform(2, rola.union_routing(8, alpha=1.5)),
                              hidden_size=512, num_heads=8)

factors = routes(hidden_states)          # the (q, k) pair, after the feature map
y, state = rola.rola_op(factors, v)
```

`rola_op` runs the BUILT ENVELOPE and refuses everything else, by name — including
every call that carries a gradient, because the backward arrives with the native
backward. There is no second arm behind it: the fp64 oracle is the executable spec,
called directly (below), never dispatched to, since a silent ~1000x-slower route is
exactly what a fallback would be. It takes no plan and no configuration: those are how
the op runs rather than what the model is, and the op decides how it runs, per call,
from the routing it was actually given.

### The layer

```python
from rola import LayerContinuation, LearnedDecay, RoLA

layer = RoLA(routes, d_v=64,
             decay=LearnedDecay(2 ** -8, widths=routes.widths, num_heads=8),
             layer_idx=0).cuda()
y, continuation = layer(x, continuation=LayerContinuation(state=rola.state()))
```

One positional argument, because everything else was either a fact about the feature
map (which lives on the feature map, `hidden_size` and `num_heads` included) or a
launch decision (which the op makes). `d_v` is the layer's: it sizes
`v_proj`/`o_proj`, and the routing carries no value geometry.

### The state

`rola.state()` builds one, and `rola.RoLAState` is a **fixed-size recurrent state,
not a KV cache**: a PAGED TENSOR — storage, page table, `[B, H, N, cols]`, device,
dtype, strategy — and nothing else, paged by default so what stays resident scales
with the leaves the routing REALIZES. Continuation is LAYERED: the layer returns
`(y, LayerContinuation)`, a frozen container the layer owns (`.state` is the op's paged
tensor, its only field today). Every `transformers` cache abstraction is built for the
GROWING case, which this is not, so RoLA owns its object outright
(docs/internals/state.md).

### The launch geometry

There is no planner and no cost model. The runtime geometry — `BC`, the CTA's
resident state tile — belongs to the BUILT ARM: the op looks the topology up in the
binary's own arm list and runs it, or names the clause that refuses. `rola.expert` is
the marked door for a caller that must ask whether a call has an arm at all
(`envelope_refusal`) or pin the state's backing; the axes that are
not host-selectable refuse by name rather than being ignored. The kernel's
correctness does not depend on which arm a topology lands on, and that is what Tier
2 exists to prove.

### The oracle

`rola.naive_rola` is the fp64 definition, and it **ships in the wheel**. A user who
suspects a numerical problem can run the definition against the kernel on their own
shapes, on their own hardware, with no repository checkout. That is what makes the
tolerance claims auditable rather than asserted.

## Performance

Benchmark cells are committed data, not scripts that might reproduce it:
[`benchmarks/cells/topology_manifest.json`](benchmarks/cells/topology_manifest.json)
carries each cell's realized statistics, and
[`benchmarks/cells/latency_baseline.json`](benchmarks/cells/latency_baseline.json)
carries per-arm latencies with normalized provenance (GPU, driver, toolchain,
manifest SHA — no host identifiers). **Those cells are the retired tiled consumer's
and are parked as a record**; no chunk-arm number may be compared against them
([`benchmarks/cells/README.md`](benchmarks/cells/README.md)). The methodology —
interleaved paired arms, median of per-pair ratios, IQR reported alongside — is
documented in [`docs/measurement.md`](docs/measurement.md), which also states what
the successor harness owes.

## Evidence

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — the foundation: the recurrence,
  the lattice, the state's format, the grains, the axis law and the shipped set.
- [`docs/architecture/`](docs/architecture/README.md) — the permanent decision
  record: the self-masking theorem, the affine-recurrence rule, the factored
  representation, fp32 accumulation, closed-world codegen, the launch-geometry
  rule, and the guarantees/choices/preferences taxonomy.
- [`docs/internals/`](docs/internals/README.md) — one document per source file,
  carrying the reasons attached to a line, plus the deletion ledger.
- [`docs/testing.md`](docs/testing.md) — the tiers, the tolerance discipline, the
  geometry pairs and their contract classes, and the conformance data-regime box.
- [`docs/measurement.md`](docs/measurement.md) — the one stopwatch, the discipline a
  harness must acquire rather than document, and the append-only ledger.
- [`docs/ratification.md`](docs/ratification.md) — what a manifest is, what the
  closed-world rules refuse, and why the gate self-tests.
- [`docs/build.md`](docs/build.md) — the ahead-of-time build and the
  self-describing binary.
- [`docs/bringup.md`](docs/bringup.md) — bringing up an unratified architecture.
- [`docs/provenance.md`](docs/provenance.md) — where this code came from.
- [`docs/open-work.md`](docs/open-work.md) — what is knowingly unfinished.

## Tests

```bash
pytest tests/unit tests/oracle -m "not cuda"    # CPU: no device, no build
pytest tests/oracle                             # kernels vs the fp64/torch baselines
pytest tests/integration                        # composition, dispatch, two-branches-agree
pytest tests/unit                               # each file's public contract
python tools/ratify.py --arch 80 --arch 86              # ratification, no GPU needed
python tools/ratify.py --self-test                      # prove the gate can fail
```

Never `pytest tests/` unscoped. The `cuda` marker exists so that `-m "not cuda"`
deselects at *collection* time, before torch initializes a CUDA context.

## Bringing up new hardware

Probe → manifest → gate, and the manifest change is always a reviewed diff, never a
CI auto-commit. See [`docs/bringup.md`](docs/bringup.md).

## Related repositories

| Repository | What lives there |
|---|---|
| `flash-linear-attention` | The fla fork: its `rola` branch holds the RoLA layer and HF model over this package |
| `rola-zoology` | Synthetic capability evaluations (MQAR and friends) |
| `rola-bench` | The benchmarks and the local measurement suite |
| `rola-results` | The measurement records every reported number is read from |
| `rola-devtools` | The development tools every RoLA repository shares: the public mirror's export, the interleaving driver |
| `rola-paper` | The paper source |

Each is developed in a private `-dev` repository and published here by a mirror job: a push to the development
default branch publishes its declared files as one snapshot commit (`.github/mirror/declarations.json`; the export is [rola-devtools](https://github.com/Blakexx/rola-devtools)').

## Changelog

Semantic changes are listed per minor version. **A loosened tolerance is a semantic
change** and requires a changelog entry, the derivation that justifies the new
number, and reviewer sign-off; tightening one requires none of that.

`0.1.0` — first extracted release. See [`docs/open-work.md`](docs/open-work.md) for
what is deliberately not in it.

## Citation

```bibtex
@misc{rola,
  title  = {Routed Linear Attention},
  author = {Bottum, Blake},
  year   = {2026},
}
```

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
