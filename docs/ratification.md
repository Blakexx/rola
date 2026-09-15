# Ratification — the closed-world codegen policy

A **ratification manifest** is a measured, committed record of what the assembler
did to every instantiation the extension ships, on one architecture, under one
exact `ptxas`, filed under that assembler's toolchain ([`internals/tools/toolchains.md`](internals/tools/toolchains.md)):

```
tools/manifests/cu13/sm_86.json
  { "toolchain": { "ptxas": "<the whole --version string>", "defines": {},
                   "nvcc_flags": [...], "sass_hash_version": 2,
                   "sccache": {"version": "sccache 0.17.0", "sha256": "..."} },
    "source":   { "csrc_sha256": "...", "date": "..." },
    "shards":   { "chunk_s0": { "members_sha256": "...", "count": 16 }, ... },
    "entries":  { "sm_86:<mangled name>": { "regs": 64, "spill_stores": 8,
                                            "spill_loads": 24,
                                            "hot_locals": 0, "local_ops": 12,
                                            "sass_sha256": "..." }, ... } }
```

**347 entries per architecture** on this line, measured on the current tree: 320 entmax
solves, 17 `decode_step_kernel` and the decode path's device build stamp, and 10
`intra_kernel` arms.

THE CLEAN SLATE REMOVED 103 PER ARCH (450 -> 347): the carry family's 43 (`carry_kernel`
`<D, B, K, M, W, C, STATE>`, `carry_split_kernel`, the `scan_a_kernel`/`scan_b_kernel`
and the build stamps) and the facts family's 60 (`chunk_facts_tables<PLAN, MASK>` over
the lex `(D, B, BC)` plans crossed with the two prefill call classes plus the GIVEN-SPAN
owner-row plans, `block_bitmap_kernel`, `atom_bitmap_kernel`). Neither family is built
here (`docs/internals/DELETIONS.md`); the rebuilt families bring their successors back
and re-ratify then. The paragraphs below describe the pre-C0 census and are kept as the
record of what was ratified when: plus 16
`decode_step_kernel` (`DV x D x DECAY`), the decode path's own device build stamp, and
the entmax solves' 320. (The lattice walk retired the `IN_SMEM` template axis
and the 32-arm decode matrix it produced, and added the decode build stamp —
[`internals/decode/decode.md` §31](internals/decode/decode.md#instantiation-matrix),
[`internals/decode/decode_api.md` §build-stamp](internals/decode/decode_api.md#build-stamp).) (The
chunk consumer's 44 `compacted_kernel`, the walk probe's 6 and the reverse pass'
14 left with the deletion batch; baseline = tag `baseline/pre-k31`.) The tiled `consumer_forward_kernel<BC, DV, D, DECAY, GT>`
product that four shards once carried retired with the tiled consumer
([`docs/internals/DELETIONS.md`](internals/DELETIONS.md)).
`defines` is the empty tuple — `RATIFIED_DEFINES = ()` in `tools/ratify.py`, since
the closed-world codegen surface carries no `#define` this gate has to pin. Two
ratified architectures (`sm_80`, `sm_86`) make **804 entries total**. The manifest is
the authority on the count and no comment restates it independently. The mangled
name is the key because it is exact, stable, and encodes every template argument
— an entry cannot be silently re-pointed at a different instantiation by a rename
or a reordering.

## What the shipped set is, and where it is declared

The carry family's arms are a **declaration**, not a literal in a code module:
`tools/manifests/shipped_set.json` names the rows, keyed
`(D, DV, warps_per_cta)` — the routing depth, the value width and the CTA's warp
count, and nothing else (KERNEL_STANDARDS §R11: the mode word, the stream counts and
the park bit are runtime; the window and the chunk are the descriptor's).

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

The generator refuses a declaration that is malformed, and names the field: an
unknown axis value, a duplicate row, a row in both the shipped and the test list, or
rows out of canonical order — the last because an arm's INDEX is what the selection
header, the per-arm translation units and `ROLA_CARRY_ARMS` name, so it must be a
property of the declaration and not of where a row was typed.
`python tools/gen_shards.py --self-test` requires the reader to refuse ten malformed
declarations by name, and renders a synthetic three-row table so the selection header,
the arm set and the shard block are shown producing rows — which the live declaration,
being empty, cannot show.

**The declaration is empty on this line.** The carry family's kernels are not present,
so no arm is declared, no per-arm unit is generated and no build carries a carry arm.
The mechanism is the build's contract, and it is exercised at zero rows.

<a id="arm-switch"></a>
## `arm_switch`, and what the post-build check proves about it

`csrc/rola/src/common/arm_switch.cuh` is the host twin of `uniform_switch`: one
branch over a GENERATED arm set, with the same four layers
([`internals/common/arm_switch.md`](internals/common/arm_switch.md)). The set comes
from the generated selection header, so the set the host dispatches over and the set
the translation units compiled against are one enumeration; `ROLA_CARRY_ARM_COUNT`
and `ROLA_CARRY_ARM_DECLARED` are both emitted, because the refusal has to tell *this
wheel does not carry that arm* from *nothing builds that arm in any configuration*.

`setup.py::_post_build_arm_table_check` is where the layers become checked facts, and
it runs **after** the compile because what it checks is what the build produced:

1. The generator's table still matches the declaration, re-read from disk;
2. The selection header the build compiled against names exactly the arms the build
   selected — count, declared count, and the set's own members;
3. The coverage sweep: the freshly built `_C`, **loaded by path**, is asked what it
   carries, and the answer must be exactly those rows.

A failure is fatal whatever `ROLA_STRICT_MANIFEST` says. The milestone/CI split
(§16) is about manifest *staleness* — measurements that go out of date between
milestones by design — and none of these three is a measurement.

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
whole extension, while a header included only by the stamp site rebuilds one file and
is hashed by ninja through the depfile. (A flag ninja does *not* hash —
`NVCC_PREPEND_FLAGS` — is the failure one level over: the build reports success and
the binary silently stays the previous variant.) `setup.py` writes the header before
it compiles anything and `tools/ratify.py` writes it, from the same function, before
it measures anything.

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

**On the current tree, 804 instantiations measured, 0 with an unhonored bound, and
4 that spill at all** — all four the same entmax kernel, `union_backward_kernel`
`<32, 8, false>` in both its bf16 and fp32 forms, at 16 stores / 16 loads on
`sm_86` and 24/24 on `sm_80`, 64 registers. **Every chunk and decode instantiation
is zero-spill.**

The tiled consumer's two priced-spill arms (`<BC=16, DV=32|64, D=1, nodecay,
GT=16>`, carried deliberately because a 72-register budget removed the spill and
cost a CTA on exactly those two arms) left the manifest with that kernel.
The adjudication is kept because the RULE it settled is what governs the next such
trade: a spill is admissible when it is priced, recorded, and worse than every
alternative in hand (`workflows/perf/p36_quantum.md`, "THE ONE RATCHET ITEM,
RECORDED AND FLAGGED RATHER THAN LAUNDERED").

## What it compiles, and why that is the shipped source

`python tools/ratify.py` compiles **the translation units the build compiles** —
the generated per-arm carry units under `csrc/rola/src/instantiations/` (none on this
line: the arm list is empty), plus `decode.cu`, `entmax.cu`, `factor.cu`, the
intra family and its reverse translation unit — under the flag list `setup.py` uses,
imported from
`tools/build_flags.py` rather than restated. A family is shipped codegen the moment it
is in `SOURCES`, so it is measured like every other shipped translation unit. Two properties follow, and
neither held before:

- **the numbers describe the shipped binary.** `nvcc` derives an
  anonymous-namespace kernel's mangled discriminator from the file that
  instantiates it, so while the gate compiled a probe under `tools/`, its 80
  decode/entmax keys named symbols that existed in no shipped `.so`. They now name
  the entry points the binary carries.
- **there is one enumeration of the matrix.** The per-arm units, the dispatch's own
  X-macro walk and this gate's source list all come out of `tools/gen_shards.py`;
  there is no second file that could enumerate a different arm set.

The chunk keys are unaffected by the shard split by construction:
`rola::chunk::compacted_kernel` lives in a NAMED namespace, so its mangled name
does not depend on which translation unit expands it.

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
| **REPORT — sccache drift** | The recorded `toolchain.sccache` (`{version, sha256}` or `null`) differs from this run's. Printed, not failed: `sccache` is a measured-transparent pass-through (`docs/build.md#sccache`), so its version cannot move a ratified register/spill number the way a `ptxas` swap can. What IS gated hard, before it is used at all, is that a resolved `sccache` matches `tools/sccache_pin.json` (`tools/sccache_toolchain.py`). |

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

**Maps may now come from binaries built at different paths.** `decode.cu` and
`entmax.cu` used to place their kernels in an anonymous namespace, so the mangled
name embedded a discriminator derived from the source path; a manifest written in a
worktree named 192 kernels that no build at any other path would ever define. Fixed
by moving those kernels into a named internal namespace (
`workflows/refactor/p19_symbols.md`) — the ratified names are a function of source
content and toolchain only, and a worktree build now ratifies clean. History:
`docs/build.md`'s path-trap section.

**And from the same SCANNER.** What reaches the digest is decided by
`ratify.sass_scan`, so widening the capture moves every hash in an unchanged binary
— a false verdict with exactly the (now-fixed) path trap's shape. `ratify.SASS_HASH_VERSION` is
the one definition of that decision; it is stamped into the manifest's `toolchain`
block and into every map `tools/sass_bodies.py` writes, and both refuse a comparison
that does not agree on it. A record carrying no stamp is refused rather than assumed
to be version 1, because guessing produces a confident wrong answer.

## The four closed-world rules

| Rule | Refused | Enforced by |
|---|---|---|
| **R1** | building under an assembler the manifest was not ratified under | `setup.py::_gate_toolchain`, `ratify.py::check_toolchain` |
| **R1b** | building under CODEGEN FLAGS the manifest was not ratified under | `ratify.py::check_flags` |
| **R2** | emitting `code=compute_XX` — any PTX a driver could JIT | `setup.py::gencodes`, asserted on the shipped `.so` (`cuobjdump --list-ptx` reports **0 PTX images**) |
| **R3** | an architecture in the fatbin with no ratified manifest | `setup.py::_gate_ratified` — a **file-existence** question, asked before anything compiles |
| **R4** | launching on a device the kernel's arch table does not describe | `common/arch_runtime.cu`'s `check_arch_table()` at run time (first statement of every family's launching entry, carry included), `static_assert` at compile time |

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

It enters the digest, and that is the point. "Only `chunk_s0` was
re-measured" is otherwise an unprovable sentence; with membership in the ratified
record it is a diff. Because the block carries membership and nothing else — no
paths, no ordering — re-filing the same measurements cannot move the digest, which
`--self-test` checks in both directions (a re-filing must not move it, a membership
change must).

## The gate self-tests

```bash
python tools/ratify.py --self-test
```

requires the gate to **fail** on the things it must reject:

1. a manifest tightened at one entry,
2. a perturbed `ptxas` version string,
3. an architecture with no committed manifest (so rule 3 has a *no* answer),
4. a per-arch split that does not rejoin to one digest,
5. per-arch files that disagree about shard membership,
6. a shard block that is decorative (a membership change that moves nothing) or one
   that is a property of the filing (a re-filing that moves the digest),
7. a perturbed source digest,
8. in-loop local-access growth past a ratified count — plus the hot-loop parser
   proving, on synthetic SASS, that it counts an in-loop `LDL` and clears a cold
   one,
9. local-access growth on the `LDL` the hot-loop gate is right to ignore, which is
   what shows the cheap ratchet is not a restatement of the expensive one,
10. a ratified `sass_sha256` that no longer matches, and an entry carrying none at
    all — plus the hash itself proving, on synthetic SASS, that two different
    instruction texts hash differently and that re-padding one listing does not
    move its hash.

A gate whose green run is unfalsifiable says nothing. This project has already
shipped two gates that turned out to be vacuous before they were fixed, which is
why the self-test is part of the gate rather than a note about it.

## Where the record lives after the build

`rola_cu13/_build_config.py` is generated by `setup.py`, ships in the binary plugin beside the `.so`
(`rola-cu13`, [build.md#wheels](build.md#wheels)) and answers *"which measured evidence backs this exact
`.so`?"* from an installed wheel with no repository checkout:

```python
>>> from rola_cu13 import _build_config; _build_config.show()
{'version': ..., 'archs': ['sm_80', 'sm_86'], 'ptxas': ...,
 'manifest_sha256': '...', 'manifest_entries': {'sm_80': 112, 'sm_86': 112},
 'ptx_jit_fallback': False, 'vendored': {'cutlass': {...}}}
```

The tier-2 census independently recomputes `manifest_sha256` from the committed
manifests and requires it to match, so the tests and the binary cannot be looking
at different evidence.
