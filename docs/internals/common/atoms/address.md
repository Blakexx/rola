# `common/atoms/address.cuh` — the addresses

Mirrors `csrc/rola/src/common/atoms/address.cuh`, a concern of the baseline operations ([`../ops.md`](../ops.md)): shared addresses, gathered and pinned bases, the state plane's page translation and the facts' per-atom bits.

- `smem_addr(p)`: the CTA-local address of a shared object, the address type every shared operation consumes.

- <a id="gather-ptr"></a>`gather_ptr(base, off)`: a gathered row's address, a 64-bit base formed once plus a 32-bit
  BYTE displacement ZERO-EXTENDED rather than multiplied out. The arithmetic is the contract: at element type the
  compiler re-derives `base + index` as a widening multiply per access (measured 30x `IMAD.WIDE` in a loop this form
  emits none for). The caller owns the 32-bit bound. The base is asserted GLOBAL, because a generic pointer turns a
  reduction into an atomic read-modify-write and every gathered load into a generic one. (no shipped caller)

- <a id="t"></a>`pin_address(p)`: a base the compiler must keep rather than re-derive. Under register pressure a
  compiler drops a live base and recomputes it in the loop -- the widening multiply moved rather than removed; the
  value becomes the result of an operation ptxas cannot see through, at the cost of the two registers it occupies.

- <a id="page-slot"></a>`page_slot`: the state plane's one base translation, `slot = page_tbl ? page_tbl[atom] :
  atom` ([`paging/paging.md`](../../paging/paging.md)). A negative slot is skipped at both ends of the state I/O, and
  the predicate is CTA-uniform because an atom belongs to exactly one box, which is what licenses skipping the
  barriers inside a sweep with the traffic. (no shipped caller)

- <a id="atom-activity"></a>`atom_activity`: the facts' two orthogonal per-atom bits, `kAtomWritten` and
  `kAtomRead`, never pre-ORed by the producer (the read-only distinction is what the touched-atom census is about, and
  ORing saves nothing). LOAD an atom iff it is resident and `(read | written)`; STORE it iff `written` -- identical
  under both backings, which is what makes the paging gate's `torch.equal` a certificate of the skip logic. A null
  `atom_bits` is the universal fact: every resident atom is read and written. (no shipped caller)
