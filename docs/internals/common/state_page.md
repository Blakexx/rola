# The state page: s24 planes at a group exponent

Mirror doc for `csrc/rola/src/common/state_page.cuh`. `Blocks<DV>` is the ONE definition of the int8 state
format's page (ruled 2026-09-28), and `rola.ops.paging.page_layout` its Python mirror; the device codec below is
the arithmetic every kernel that reads or writes a stored state uses, and `rola.ops.paging.encode_pages` /
`decode_pages` its host statement for the oracle and the tests. `SplitBlocks<DV>` is the unported carry's own page.

## <a id="geometry"></a>The page, in bytes

A page is one atom = 16 leaves:

    [16 x DV hi s16][16 x DV lo u8][16 x DV/32 exponent s8][16 mass fp32]

A leaf-channel is the s24 mantissa `m = hi << 8 | lo` (hi signed, lo unsigned) at its (leaf, 32-channel group)'s
power-of-two exponent `e`: its value is `m * 2^e`. The mass is fp32, one a leaf. The mass block starts on a 16-byte
boundary, so `kBytes` is 1616 / 3168 / 6272 at `DV` 32 / 64 / 128 -- about 3.03 bytes a leaf-channel against the
split fp32 page's 4. The planes are aligned and never packed into 3-byte elements: a hi row is one 128-byte line at
`DV = 64`, and a read that needs only the s16 operand touches the hi plane alone.

Nothing about the arena, the page table, the id space or the VMM commitment depends on what is inside a page: they
address `kBytes`-sized slots.

## <a id="codec"></a>The codec

- **The exponent** of a group whose largest magnitude is `f * 2^k` (`f` in `[0.5, 1)`) is `e = max(k - 23, -126)`:
  the largest mantissa fits under `2^23`, and `2^e` stays a normal float (`pow2` builds it from the exponent field).
  `group_exponent` reads `k - 23` off the float's exponent field (`field - 149`; a subnormal's field is 0, and the
  floor takes it), which is `frexp`'s answer without the call.
  An all-zero group stores `e = 0`, so a zero page is all zero bytes -- the arena's fresh page, byte for byte.
- **The store** rounds each value once, `m = rint(v / 2^e)`, held inside `+-(2^23 - 1)` (`mantissa`); an all-zero
  group stores zeros, so a zeroed page is the zero state.
- **A full read** is `(hi << 8 | lo) * 2^e`, exact in fp32.
- **A hi-only read** is `hi * 2^(e + 8)`: the value truncated to 15 bits of its group, the s16 operand a read-only
  load uses. Truncation keeps an exact zero a zero.

<a id="lane-eight"></a>**A lane's eight channels.** A reader that holds eight consecutive channels of a row a lane
(the decode walk) loads them as one 16-byte hi word quad and one 8-byte lo word pair. `mantissa8` builds channel
`c`'s s24 with one byte permute -- its lo byte, its two hi bytes, and the high byte's sign replicated into the top --
`hi8` its s16 alone, and `pack_hi8` / `pack_lo8` pack eight rounded mantissas back into the two planes (bytes 1-2 of
each a hi word pair, byte 0 of each a lo word quad).

A value is exact to within half a quantum `2^e`, i.e. `2^-24` of its group's largest magnitude; a group's small
values keep that absolute precision, not a relative one, which is why the gates allow the store's quantum
(`tests/oracle/fixtures.py`'s `state_quantum` and `state_rounded`, the declared store mirrored in fp64).

## <a id="split"></a>`SplitBlocks`: the unported carry's page

`[16 x DV hi][16 x DV lo][16 mass hi][16 mass lo]`, an fp32 word as two bf16 halves, `16 * (DV + 1) * 4` bytes. The
struct is read by no kernel on this tree; the carry's int8 port deletes it.
