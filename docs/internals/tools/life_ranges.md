# `tools/life_ranges.py` — peak live registers per source region

The report: a kernel's peak live register count and its histogram, the peak and mean per source region (the innermost
function the instruction inlines from, struct methods as `Struct::method`, the naming `tools/region_ledger.py` uses),
and the peak program points. The cubin needs `-lineinfo` for the regions; an extension's cubin (`--so`, `--member`)
reports the peak and histogram without them. `--kernel` names the function where the cubin holds several (a carry arm:
the carry and its plan pass); offsets restart in every function.

Why it exists (KERNEL_STANDARDS §22): a budget argument about registers cites the peak live at the program point it
concerns, never ptxas's allocation.

<a id="budget"></a>
THE REGISTER BUDGET (`--budget tools/budgets/<kernel>.json`, KERNEL_STANDARDS §32.3): each budget component's peak live
registers against its `registers` line, an instruction attributed as the region ledger attributes it
(`region_ledger.owner_of`) -- the innermost inline frame in the `--source` files (repeat the flag: the kernel's file
first, then its components') whose function is one of the budget's `components`, walking outward no further than the
kernel's own body, else that body (its glue). Red where a component passes its line or has
none. The carry at 87240898 (arm 0, sm_86): every component peaks at 204-252 live, the readout at 252 -- about 200
registers stay live through every phase, the room a phase has for a second stage's operands is ~50; its lines are
those readings, a ratchet until the window's design states each phase's (`registers_basis` in the budget). The shipped carry (sm_86, arm 0) is allocated 255 registers; it peaks at 251 live in
the readout's tile loop and holds ~198 where an item starts, the fold's 128-register state among them.

<a id="liveness"></a>
## The liveness

`nvdisasm --print-life-ranges` prints, beside each instruction, a mark a register (`^` written, `v` read, `x` both,
`:` live through) and its own live count. The marks are local facts; the count is not used, because nvdisasm's
control flow is wrong for the kernels this tree builds:

- a `CALL.REL.NOINC` to a local label -- how ptxas leaves a loop through a convergence barrier -- is a jump, which
  nvdisasm takes for a call that returns;
- every call is marked as writing the calling convention's clobbered registers (122 on sm_86), which a collective's
  slow-path subroutine never touches.

Both drop a value live across the call from the count: read with its own counts, the carry's 128-register state
vanished at every item boundary (a fill inside the window loop showed ~40 live).

The tool recomputes liveness over the rows: a call to a local label is a jump that reads and writes nothing; a call to
a subroutine reads and writes what its body (from its label to its `RET`) does and continues; a `BRX` goes to its
`BRANCH_TARGETS`; `EXIT`, `RET` and `BPT` end a path unless predicated. A write under a predicate ends no live range, so
a register counts where it is live AND some write reaches it -- a register written and read under one predicate does
not count from the kernel's entry. The subroutines' bodies are left out of the report.

The check (2026-10-14, the carry at a 248-register cap): peak 244 live; ptxas fits it in 248 (246 usable, two of the
count reserved) and spills 24 bytes at 240 (238 usable). nvdisasm's own count gave 226.

A function at 255 registers is one nvdisasm refuses (`Invalid register count`); the tool reads a scratch copy whose
`EIATTR_REGCOUNT` says 254 (`listable`). The attribute only sizes the listing: its columns still reach R253, past the
R252 such a function reaches (two of the count are reserved). An instruction naming a register past the columns is
refused.

`--arm N` compiles the carry arm's translation unit alone with `-lineinfo` against the build's generated headers
(`build/generated`, so the parts mask of the last build) for this machine's GPU architecture (or `--arch`), then reads
it -- the compile the build ledger and the composer each used to spell. `--json PATH` also writes the instruction count,
the peak live registers, whether line info was present, and every region's peak, mean and instruction count.
