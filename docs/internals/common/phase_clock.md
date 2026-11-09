# `common/phase_clock.cuh` — the phase ledger and trace

Mirrors `csrc/rola/src/common/phase_clock.cuh`.

`PhaseClock<Phases>` is one warp's measurement of itself, shared by every kernel family that
measures its phases: lane 0 accumulates the cycles between laps into the warp's shared row, one
slot a phase, and `publish` adds the row into the launch's ledger, `[cta][warp][Phases]` int64,
at the kernel's end. The row is `kRow = Phases + 1` slots: the last holds the previous lap's
clock, so the instrument keeps no register live across a phase -- a held 64-bit mark spilled
the decode step's deep arms, whose walk fills the register file. When a trace is bound (`bind_trace`), each lap and each `stamp` records
`cycle << 8 | event` into the warp's trace row up to its capacity.

<a id="deferred-barrier"></a>**A lap after a barrier books the release.** `__syncthreads` is
`BAR.SYNC.DEFER_BLOCKING`: a warp arrives, issues one more instruction, then blocks. A clock read
placed right after the barrier is that instruction, so it read the ARRIVAL, and every barrier's
wait landed in the next phase (the decode step's owner-mask phase measured 6.3K cycles that were
the digit masks' wait). `lap` therefore loads the previous mark first and reads the clock with
`ops::cycles_after(mark)`: the load is the instruction issued before blocking, and the read waits
for it. Measured on the decode step (sm_86): the digit masks 0.5K -> 5.8K, the owner masks
6.3K -> 1.7K, the total unchanged.

The carry aliases it at its phase count (`carry_kernel.md#phase-ledger`). The struct is
inlined wherever it is used.

<a id="on"></a>**`on`**, the constructor's third argument (default true), says whether anything will read the
row. Off, lane 0 is not the lead. The compiler if-converts a lap into lane 0's predicated
block, so an off lap still issues its seven instructions predicated off; **`Gated`**, the
template's second argument (default false), makes the lap a real branch on the warp-uniform
`on` (`ops::once`), so an off clock issues nothing. The intra takes `Gated` and passes "a ledger
is bound": its ten-odd laps a tile cost ~3% of an unmeasured launch before (ncu: `L = 64K` at
four heads 1.17 → 1.14 ms over the two steps). The carry and decode keep the default: the
branch's reconvergence changed the carry's register allocation and it spilled 512 bytes.
