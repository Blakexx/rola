# Ratification — the closed-world codegen policy

A **ratification manifest** is a measured, committed record of what the assembler
did to every instantiation the extension ships, on one architecture, under one
exact `ptxas`, filed under that assembler's toolchain ([`internals/tools/toolchains.md`](internals/tools/toolchains.md)):

```
tools/manifests/cu13/sm_86.json
  { "toolchain": { "ptxas": "<the whole --version string>", "defines": {},
                   "nvcc_flags": [...], "sass_hash_version": 2,
                   "mold": {"version": "mold 2.42.0", "sha256": "..."} },
    "source":   { "csrc_sha256": "...", "date": "..." },
    "shards":   { "carry_arm_0": { "members_sha256": "...", "count": 1 }, ... },
    "entries":  { "sm_86:<mangled name>": { "regs": 64, "spill_stores": 8,
                                            "spill_loads": 24,
                                            "hot_locals": 0, "local_ops": 12,
                                            "sass_sha256": "..." }, ... } }
```

One entry per instantiation the source defines: the entmax solves, the decode step's arms, the intra kernel's, the
carry family's declared arms (`tools/manifests/shipped_set.json`) and the device build stamps. The manifest is the
count; `bazel run //:ratify -- --census` prints the per-key table for the current tree.
`defines` is the empty tuple — `RATIFIED_DEFINES = ()` in `tools/ratify.py`, since
the closed-world codegen surface carries no `#define` this gate has to pin. The mangled
name is the key because it is exact, stable, and encodes every template argument
— an entry cannot be silently re-pointed at a different instantiation by a rename
or a reordering.

## What the shipped set is, and where it is declared

The carry family's arms are a **declaration**, not a literal in a code module:
`tools/manifests/shipped_set.json` names the rows. An arm IS a carve's operation-plan
pair, keyed `fold_rows, fold_factors, read_rows, read_factors, channels, same_split`
([`internals/engine/rules.md#carve`](internals/engine/rules.md#carve)); the box's
per-level extents, the depth and the widths are runtime addressing, so they never
multiply the arm set.

`tools/gen_shards.py` owns the *enumeration* — what a row means, which translation
unit carries it, what the selection header and the `shards` block say about it — and
reads the *membership* from that file. Keeping the two apart buys three things:

* a shipped set is a statement about the product, so it is reviewable on its own,
  as a diff, without executing anything;
* a tool that must not import the package can still read it (a repo tool reads the
  repo's files; importing the package answers a question about whatever build the
  running environment happens to point at);
* The build's post-build check can **re-read the declaration and compare it to the
  generator's table**, which is a second statement rather than a restatement.

The declaration has one `shipped` list: every declared row is built, ratified and
shipped. The generator refuses a declaration that is malformed, and names the field: an
unknown axis value, a duplicate row, a missing list, or rows out of canonical order —
the last because an arm's INDEX is what the selection header, the per-arm translation
units and `--//:carry_arms` name, so it must be a property of the declaration and not
of where a row was typed. `bazel run //tools:gen_shards -- --self-test` requires the reader
to refuse every malformed declaration by name, and renders a synthetic three-row table
at a subset scope ([`internals/tools/gen_shards.md`](internals/tools/gen_shards.md)).

<a id="arm-switch"></a>
## `arm_switch`, and what the arm gate proves about it

`csrc/rola/src/common/arm_switch.cuh` is the host twin of `uniform_switch`: one
branch over a GENERATED arm set, with the same four layers
([`internals/common/arm_switch.md`](internals/common/arm_switch.md)). The set comes
from the generated selection header, so the set the host dispatches over and the set
the translation units compiled against are one enumeration; `ROLA_CARRY_ARM_COUNT`
and `ROLA_CARRY_ARM_DECLARED` are both emitted, because the refusal has to tell *this
wheel does not carry that arm* from *nothing builds that arm in any configuration*.

`tools/gen_shards.py`'s `check_built_arm_table` and `sweep_binary` are where the layers become checked facts -- the
library's arm gate runs them (`bazel/gates.py`) -- and they run **after** the link because what they check is what the
build produced:

1. The generator's table still matches the declaration, re-read from disk;
2. The selection header the build compiled against names exactly the arms the build
   selected — count, declared count, and the set's own members;
3. The coverage sweep: the freshly built `_C`, **loaded by path**, is asked what it
   carries, and the answer must be exactly those rows.

A failure is fatal on every build, iteration builds included. Ratify's own verdict is a
separate target because §16's milestone rule is about manifest *staleness* — measurements
that go out of date between milestones by design — and none of these three is a measurement.

<a id="the-derivation"></a>
## The derivation, ratified as code hash and golden outputs

The addressing block is host code: one function, `derive_carry_geom`, run once per
launch, whose output every kernel of the family then addresses through. An off-by-one
in a span or a shift is a wrong answer nothing else in this gate would see, because
the SASS it produces belongs to somebody else's kernel. So it is ratified with the
same two independent statements the codegen gets, filed in
`tools/manifests/derivation.json`:

* a **code hash** per declared source file (`common/geom.cuh` today) — the wholesale
  trigger, and deliberately *not* `csrc_digest()`: a change to a kernel body is not a
  change to the derivation, and a record that expires on every unrelated edit is a
  record people re-write without reading;
* **golden outputs** — the derived block, byte for byte, on the named cases declared
  in `tools/manifests/derivation_cases.json`, plus the REFUSAL each declared-illegal
  case produces.

The refusals are ratified beside the outputs because R13 makes the kernel's strict
shape half a contract whose other half is what it refuses; a derivation that quietly
began admitting a non-power-of-two width would pass a gate that only checked the
shapes it admits.

The record reads the two ways the `sass_sha256` ratchet does: with the code hash
matching, no golden may move; with it moved, the gate names which cases moved and
which did not. A separate `cases_sha256` covers the declaration itself, so a golden
cannot stop failing because somebody deleted the case that failed.

Each geometry case names a real `(D, DV, warps_per_cta)` point through the `BC` that
pair implies — `BC = leaves_per_warp(DV) × warps_per_cta`. The derived block does not
depend on `DV` directly; `DV` enters through `BC`. Two arms whose `(DV,
warps_per_cta)` land on the same `BC` therefore derive the same block, and the
goldens record that rather than hiding it.

The harness compiles the shipped header and links no extension, so this check cannot
be answered by a stale `.so`: there is no binary in the loop to be stale.
`--write-derivation` is its own mode — the derivation is arch-independent, so it is
neither measured per arch nor worth a whole codegen ratification to refresh — and a
full `--write` refreshes it too, so a re-ratification can never file codegen numbers
from one tree beside derivation goldens from another.

<a id="the-device-stamp"></a>
## The device stamp keys on the csrc content hash

A build stamp is the one check a stale binary cannot pass — but only if its value is
a function of the SOURCE. A stamp folded from struct sizes and launch widths is a
function of a few declarations, so a `csrc/` change that leaves those alone leaves the
stamp alone, and a binary built before the change reads back as fresh.

`rola::csrc_build_stamp()` (`csrc/rola/src/common/build_stamp.cu`) is a kernel that
returns 60 bits of `csrc_digest()`, compiled in through the generated header
`build/generated/csrc_stamp.inc`. `tests/unit/test_build_stamp.py` requires the value
the loaded fatbin returns to EQUAL what this tree hashes to right now — an equality
against a tree fact, not merely a number that changes.

It is a generated header rather than a `-D` for the reason the arm selection is: a
define reaches every translation unit's argv, so every `csrc/` edit would rebuild the
whole extension, while a header included only by the stamp site rebuilds one unit, the
one action that depends on it ([`build-graph.md`](build-graph.md#compile)). (A flag a
build does *not* key on — `NVCC_PREPEND_FLAGS` — is the failure one level over: the
build reports success and the binary silently stays the previous variant.) The graph's
stamp action writes the header before anything compiles and `tools/ratify.py`'s compile
path writes it before it measures anything, from the same function (`tools/csrc_stamp.py`).

The footprint stamps each family already carries are untouched and answer a different
question: whether the loaded binary's struct sizes are the ones this source declares.

## Why it exists

The kernel's `__launch_bounds__` is computed from a three-term residency formula,
and a spill accepted as a measured optimum against worse alternatives is a
judgement that has to live SOMEWHERE nothing else checks — a document that says so
is not enforced, and a later codegen change can grow the spill silently. The
manifest is what makes that judgement an asserted invariant instead: a property
the project believes is a property the project checks. **The ratchet is a
ratchet, not a spill ban** (the recorded adjudication): a spill is admissible
when it is priced, recorded here, and worse than every alternative in hand — not
when it is absent.

The manifest records each entry's spill stores and loads beside its registers, so a
priced spill is visible in it as a number, and every carry, intra and decode
instantiation ratifies zero-spill.

## What it compiles, and why that is the shipped source

Ratify's compile path (`python tools/ratify.py`, an architecture's first manifest) compiles **the translation units
the build compiles**, taken
from the one list both read (`tools/kernels.py`): every hand-written `.cu` under
`csrc/rola/src`, plus the generated per-arm carry units under `instantiations/` that the arm
selection names -- under the flag list the graph compiles with, `tools/build_flags.py`'s, rather
than restated. A kernel is shipped codegen the moment its file is under `src/`, so it
is measured like every other; `bazel run //tools:kernels -- --check-manifest` names any kernel the
source defines and no manifest records. Two properties follow:

- **the numbers describe the shipped binary**: the keys name the entry points the
  binary carries, compiled from the files that ship them.
- **there is one enumeration of the matrix.** The per-arm units, the dispatch's own
  X-macro walk and this gate's source list all come out of `tools/gen_shards.py`;
  there is no second file that could enumerate a different arm set.

Every family's keys are unaffected by which translation unit expands them, by
construction: `rola::carry::carry_kernel`, `rola::entmax::detail`'s kernels and
`rola::decode::detail::decode_step_kernel` all live in NAMED namespaces (the anonymous-
namespace path-trap fix, `internals/common/build_stamp.md`), so a mangled name is a
function of source content and toolchain only, whichever shard or per-arm unit compiled
it.

On the graph's build it compiles nothing (`--graph`, [`build-graph.md`](build-graph.md#ratify)): the census is the
build's own compile of the same units under the same flags, read from each unit's ptxas report and object.

`--shard NAME` scopes the compile to one shard's slice for triage. It is a
measurement scope only: `--write` refuses it, because a manifest written from a
subset would silently drop every instantiation it did not compile.

## What the gate refuses

The gate reads `ptxas -v` and the disassembly of the same objects. Its outcomes,
and the difference between them, are deliberate:

| Outcome | Meaning |
|---|---|
| **FAIL — toolchain drift** | The invoked `ptxas` is not the one the manifest was ratified under. A spill/register manifest is a property of the *assembler* as much as of the source, so these numbers say nothing about this assembler — in **both** directions. Re-ratification is required. |
| **FAIL — spill growth** | An instantiation spills more than its manifest entry. Spills are the thing being bounded. |
| **FAIL — an unhonored bound** | `ptxas` reported *"Value of threads per SM … is out of range"*. That entry compiled with **no launch bound at all**, so the assembler was free to spend registers as it liked. A hard failure even at zero spills. |
| **FAIL — source drift** | The tree's `csrc/rola/src` digest is not the one the manifest was measured against. |
| **FAIL — hot-loop growth** | An instantiation executes more `LDL`/`STL` *inside a loop* than it ratified. In-loop local traffic is a memory round-trip per iteration, disqualifying whatever the byte count. |
| **FAIL — local-memory growth** | An instantiation executes more `LDL`/`STL` *anywhere* than it ratified. A cheap complement to the two above: `ptxas`'s spill bytes miss local traffic that is not a spill (a dynamically indexed local array), and the in-loop count only sees a backward-branch interval. |
| **FAIL — SASS body drift** | An instantiation assembles to different code than was ratified. |
| **REPORT — register drift** | A changed register count with no spill growth is printed, not failed. Register allocation is an outcome that legitimately moves; what must not move is the spilling. |

### The body hash is the incremental half of the freshness trigger

The source digest is **wholesale**: any edit under `csrc/rola/src` moves it and
invalidates all ratified entries at once. That is right as a trigger and useless as
a diagnosis. `sass_sha256` — the sha256 of an instantiation's assembled instruction
text, whitespace-normalised, encoding words included, listing padding excluded — is
the per-entry answer to what the digest can only ask globally, and it reads two ways:

* **the digest matches** → every ratified body must still hash the same. The digest
  says the *inputs* did not move; this says the *output* did not either, which is an
  independent statement and the one the spill numbers depend on.
* **the digest moved** → the numbers must be re-measured, but the entries whose
  bodies are unchanged are provably still described by their ratified numbers, and
  the gate names only the moved ones. That is what makes a scoped `--shard`
  re-measurement a checkable claim rather than an assertion.

**Maps compare across build paths.** Every kernel lives in a named namespace, so its
mangled name is a function of source content and toolchain only and a worktree build
ratifies clean (`internals/common/build_stamp.md`).

**And from the same SCANNER.** What reaches the digest is decided by
`ratify.sass_scan`, so widening the capture moves every hash in an unchanged binary
— a false verdict with the path trap's shape. `ratify.SASS_HASH_VERSION` is
the one definition of that decision; it is stamped into the manifest's `toolchain`
block and into every map `tools/sass_bodies.py` writes, and both refuse a comparison
that does not agree on it. A record carrying no stamp is refused rather than assumed
to be version 1, because guessing produces a confident wrong answer.

## <a id="the-four-closed-world-rules"></a>The closed-world rules

| Rule | Refused | Enforced by |
|---|---|---|
| **R1** | building under an assembler the manifest was not ratified under | `tools/toolchains.py::rule1` (the graph's record action), `ratify.py::check_toolchain` |
| **R1b** | building under CODEGEN FLAGS the manifest was not ratified under | `ratify.py::check_flags` |
| **R2** | emitting `code=compute_XX` — any PTX a driver could JIT | the library's architecture spec (`compute_XX:sm_XX`, `bazel/defs.bzl`'s transition), asserted on the linked library by the SASS gate (`cuobjdump --list-ptx` lists **0 PTX images**) |
| **R3** | an architecture in the fatbin with no ratified manifest | `tools/build_record.py::ratified_manifest` (the graph's record action, whose refusal hands over no library) — a **file-existence** question |
| **R4** | launching on a device the kernel's arch table does not describe | `common/arch_runtime.cu`'s `check_arch_table()` at run time (first statement of every family's launching entry, carry included), `static_assert` at compile time |
| **R5** | building from a vendored codegen input (the CUTLASS/CuTe headers under `csrc/third_party/`) that does not match its recorded pin | `tools/build_record.py::vendored_pin` (the graph's record action); the pin is stamped into `_build_config.py` |

There is no environment variable that turns any of these off.

## Re-ratification is a measurement, not an edit

The manifest is regenerated only by `--write`, which re-runs the same compile and
records what it reports. Hand-editing an entry to make the gate pass would be
asserting a spill nobody measured — exactly the failure the manifest replaces. When
a kernel change legitimately moves the numbers: re-run `--write`, and record why
they moved.

## One file per architecture

`--arch` selects what is measured, checked, and written, so bringing up `sm_89`
writes one reviewable file and cannot restate a measurement it did not take. The
split is **filing only**, and that is checked rather than claimed: the digest
`_build_config.py` records is taken over the merged canonical
`{toolchain, entries, shards}` record with sorted keys, so N files hash to exactly
what one file hashed to. Two manifests pinning different assemblers is a hard
failure in every reader, and so is two manifests disagreeing about shard
membership — a split filing may not become a split toolchain or a split matrix.

## The `shards` block, and what it makes provable

`shards` records WHICH instantiations this ratification covers, as one membership
hash per shard: the sha256 of the shard's sorted list of template tuples, not of
its file's bytes. A comment edit does not move it; adding, dropping or moving one
instantiation does.

It enters the digest, and that is the point. "Only `carry_arm_0` was
re-measured" is otherwise an unprovable sentence; with membership in the ratified
record it is a diff. Because the block carries membership and nothing else — no
paths, no ordering — re-filing the same measurements cannot move the digest, which
`--self-test` checks in both directions (a re-filing must not move it, a membership
change must).

## The gate self-tests

```bash
bazel run //:ratify -- --self-test
```

requires the gate to **fail** on every mutant it must reject — a manifest tightened at one entry, a perturbed
assembler string, an unratified architecture, a split filing that does not rejoin or disagrees about membership, a
decorative shard block, a perturbed source digest, in-loop and total local-access growth, a moved or missing body
hash — and its parsers and hash to behave on synthetic SASS. The list is `tools/ratify.py`'s, which runs it
([`internals/tools/ratify.md`](internals/tools/ratify.md)). A gate whose green run is unfalsifiable says nothing, which
is why the self-test is part of the gate rather than a note about it.

## Where the record lives after the build

`rola_cu13/_build_config.py` is generated by the graph's record action, ships in the binary plugin beside the `.so`
(`rola-cu13`, [build.md#wheels](build.md#wheels)) and answers *"which measured evidence backs this exact
`.so`?"* from an installed wheel with no repository checkout:

```python
>>> from rola_cu13 import _build_config; _build_config.show()
{'version': ..., 'archs': ['sm_80', 'sm_86'], 'ptxas': ...,
 'manifest_sha256': '...', 'manifest_entries': {'sm_80': ..., 'sm_86': ...},
 'ptx_jit_fallback': False, 'vendored': {'cutlass': {...}}}
```

The tier-2 census independently recomputes `manifest_sha256` from the committed
manifests and requires it to match, so the tests and the binary cannot be looking
at different evidence.
