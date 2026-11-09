# `tools/lint/coverage.py` + `tools/coverage.json` — every kernel, every obligation

THE QUESTION A GATE CANNOT ASK OF ITSELF: which kernels does it cover? Each gate is written against the kernels its
author had in mind, and the inventory that preceded this lint found most of the kernel gates had narrowed to one family
without anyone deciding so (`sass_gate.py` read one cubin; `ratify.py`'s unit list had lost three kernels). This lint
asks the question across all of them, and the ratchet makes the answer hold.

## <a id="declaration"></a>The declaration

`tools/coverage.json` names every kernel the source defines ([`kernels.md`](kernels.md)) and, per OBLIGATION, how it is
met:

| obligation | met by |
|---|---|
| `oracle` | a test grading the kernel against a reference per slot -- an fp64 oracle, or a bit-exact model of a discrete function (the liveness pass) |
| `sanitizer` | a row of `tools/sanitize_oracle.py`'s family table, present, whose test launches the kernel |
| `budget` | a budget line under `tools/budgets/` (KERNEL_STANDARDS §19) |
| `phases` | a phase ledger the kernel books, and the test or tool that reads it (§22) |
| `ratified` | the committed manifests record it (`"manifest"`, checked per arch), or owed until the next milestone ratify (§16) |

A value is a POINTER (a file, or `file::test`, whose `def` must exist; a sanitizer row name), `{"exempt": reason}` --
a stamp kernel has no budget -- or `{"owed": reason}`.

## <a id="findings"></a>The findings, and the ratchet

One finding per UNDECLARED gap: a kernel the declaration does not name (at the kernel's own `file:line`), a declared
kernel the source no longer defines, an obligation left blank, a pointer that resolves to nothing -- a deleted test,
a renamed test in a sanitizer row, a budget file that is gone -- and a kernel declared ratified that a manifest does
not record. `tools/lint/ratchet.py coverage` gates them, so a new kernel fails the commit until it is declared and a
pointer that breaks fails it. OWED entries are the backlog: printed with their reasons and counted, never findings,
because a new kernel necessarily lands owing (no budget yet, not ratified until the milestone) and each owed line is a
reviewed line of the declaration. Paying one replaces it with a pointer.

The first run found one pointer that had already broken: the sanitizer's decode row named a test the work queue's
commit had renamed, so the lane would have failed to launch its cell.

The lint reads the sanitizer table by AST and imports nothing of rola's, so it runs where the other commit hooks run.
