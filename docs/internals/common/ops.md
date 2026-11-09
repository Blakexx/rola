# `csrc/rola/src/common/ops.cuh` — the device op helpers

The instruction-level helpers every kernel executes through: the MMA atoms (bf16, f16 and int8) and their fragment
loads, the asynchronous staging and its barriers, the pinned-base addressing discipline the SASS gate patrols, the
lane votes, the reductions and the scheduling controls.

## The baseline operations

HOW EACH STEP IS PERFORMED, as operations with contracts. Every entry is defined by WHAT THE ALGORITHM NEEDS, never by
what an instruction does: "stage this tile and tell me when it is visible", not "issue a cp.async". Get that level
wrong and the abstraction leaks on the first new architecture -- TMA cannot implement "issue a cp.async", but it can
implement "stage this tile" ([`atoms/copy.md`](atoms/copy.md)). A kernel names no instruction: it calls the atoms,
and a family's schedule ([`hardware.md#families`](hardware.md#families)) is where its structure differs.

### Operations and contracts

Every operation is defined by WHAT THE ALGORITHM NEEDS. A helper with no caller in a shipped kernel is marked
(no shipped caller); it is kept or removed with the code.

### The atoms

Each concern of the operations is an atom header under `common/atoms/`, its body chosen per family where it is defined (`docs/KERNEL_STANDARDS.md` section 8); `ops.cuh` includes them all.

- [`atoms/address.cuh`](atoms/address.md): shared addresses, gathered and pinned bases, the state plane's page translation and the facts' per-atom bits.
- [`atoms/copy.cuh`](atoms/copy.md): global to shared without the issuing thread, their groups and waits, and the 16-byte quantum moved synchronously.
- [`atoms/barrier.cuh`](atoms/barrier.md): the CTA's and named rendezvous, the mbarrier's phases, the warp's edge and the publish fence.
- [`atoms/warp.cuh`](atoms/warp.md): lane votes and their readings, the warp-uniform index, the lane prefix and broadcast, and the controls ptxas cannot see through.
- [`atoms/clock.cuh`](atoms/clock.md): the SM's cycle counter for the phase ledgers.
- [`atoms/mma.cuh`](atoms/mma.md): the MMA's shape and unit, its fragment loads and the register transpose, and the bf16, f16 and int8 atoms.
- [`atoms/math.cuh`](atoms/math.md): the special-function unit's roots and powers, byte and halfword selects and packs, the f16 and bf16 pair arithmetic.
- [`atoms/reduce.cuh`](atoms/reduce.md): contributions into an accumulator others also add into, in global or shared memory.
- [`atoms/memory.cuh`](atoms/memory.md): shared and global accesses addressed rather than indexed, plain and predicated in the instruction.

### Operations that are not here

There is no `warp_reduce`, no `compact` and no publish/acquire pair weaker than a full rendezvous: an operation
invented before it has a caller cannot be defined by what the algorithm needs, because no algorithm needs it yet.

### What is deliberately not a capability

The shared-memory layout (pad-8 against an XOR swizzle) is legal on every
tabulated architecture and is chosen by MEASUREMENT, not by the table.  A
layout knob in the capability table would make every future measurement
unfalsifiable.  It is a free axis.
