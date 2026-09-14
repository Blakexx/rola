# `docs/internals/` — the source mirror

One markdown file here per source file under `csrc/rola/`, at the same relative
path minus the `src/` component the mirror does not repeat
(`csrc/rola/src/decode/decode.cuh` → `docs/internals/decode/decode.md`;
`csrc/rola/src/paging/vmm_owner.cu` → `docs/internals/paging/vmm_owner.md`). A `.cu`
and its `.cuh` share one doc, since they are one component. The source families —
`chunk/`, `decode/`, `entmax/`, `paging/` — therefore appear here as the same
directories.

The mirror rule runs SOURCE → DOC, so a deleted source leaves an orphan doc that
no test catches. Deleting the doc with the source is therefore part of deleting
the source, and the removal is recorded in `DELETIONS.md` like any other.

## Why this directory exists

Every measured constraint has to be recorded where a reader meets it before
editing — a forgotten constraint is a rung failure — and a kernel body is not
that place: reasons written at the line they constrain invert the file, and a
kernel body carrying them runs to 61% comment lines. The two artifacts are
therefore separate. **Code carries structure; this tree carries reasons.**

What stays in the source:

- a file-header block: what the file is, what it owns, where its reasons live;
- a block before an important declaration: what it is and what it means;
- a **hazard stub** at a site where a mechanical edit silently changes results.

A hazard stub is one line, no prose, no numbers beyond its anchor id:

```cpp
//: HAZARD update-then-read -- docs/internals/decode/decode.md#update-then-read
```

Stubs are for **measured** hazards only — a place where moving, reordering or
retyping the code changes the numbers and the compiler will not say so. They are
never used for general design notes; those are here, in full.

## How to read a doc in this tree

Each file opens with what the source file is and the invariants it must keep,
then a numbered section per anchored site. Anchor ids are stable: they are cited
from the source and from `tools/lint/lint_standards.py`, which fails if a
source file has no mirror or a stub cites an anchor no doc defines.

`DELETIONS.md` is the ledger of code removed from this tree — what, when, and
why it was safe.

## Naming something that is not there

A page may only name identifiers its source still carries. `lint_standards.py` reads
every name a page writes in code font and fails on one that exists nowhere in the tree:
a mirror page can otherwise outlive the mechanism it documents, and every earlier check
was about the page's SHAPE — that it exists, sits at the right path and resolves its
anchors — rather than about whether its subject was still there.

Naming what is GONE is often the fact worth keeping: an entry that was unbound, a
mechanism that retired, another project's file the convention was borrowed from. Declare
those names on one line beside the prose that explains them,

    <!-- ABSENT: rola_consumer_forward rola_page_admit_mask -->

and the check reads the declaration in BOTH directions — a declared-absent name the tree
actually carries fails too, because the passage is then telling a reader the opposite of
the truth.

**A commit hash cited in this tree resolves in this repository, unless it predates the
extraction (2026-08-01): those name commits of the retired pre-extraction fork and
resolve in no published repository ([`docs/provenance.md`](../provenance.md)).**

## What this tree does not hold, and where it is

The mirror rule runs source → doc, so a family with no sources here has no pages
here. The carry (window body) and stats families were deleted from this branch by
the clean-slate line: their pages went with them, and the bodies are readable on
the `k35-final` branch tip, which is never edited again and is the parity
reference the rebuild is graded against. What those kernels ARE — what gets
built, in what order, and which file or record each mechanism was proven in — is
stated in [`docs/ARCHITECTURE.md`](../ARCHITECTURE.md) §6.1 and §6.2, and what is
knowingly missing in the meantime is a row in [`docs/open-work.md`](../open-work.md).
Their mirror pages come back with their stages, one page per source file as ever.

## Where the other docs live

| Directory | Holds |
|---|---|
| `docs/internals/` | per-source-file reasons (this tree) |
| `docs/architecture/` | the permanent decision record: the choices that shaped the design and why they were made |
| `docs/` (top level) | operator-facing: build, bringup, testing, ratification, provenance |
