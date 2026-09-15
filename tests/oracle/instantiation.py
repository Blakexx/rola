"""THE TOPOLOGY VOCABULARY the conformance families are stated over, as DATA.

`CHUNK_ARMS` holds `(C, BC, D, B)` rows, and `BUILT_TOPOLOGIES` the `(D, B)` topologies reached at the dispatch's
`C = CHUNK_TOKENS`. The families (`families.py`) draw their topologies from it, and the decode family's BC-blindness
test sweeps its rows. It used to be read from the stats passes' arm list in `tools/gen_shards.py`; that list was deleted
with the passes, so the rows are declared here: no longer a mirror of a built matrix, but the vocabulary the rebuilt
family is required to reach. The census that joined these axes with the families' regimes, and the kernel-envelope
mirror it used, were deleted with the chunk consumer (docs/internals/DELETIONS.md).

**Pure data and pure CPU.** No torch, no device, no extension.
"""
from __future__ import annotations

#: `(C, BC, D, B)` -- the dispatch vocabulary the family is stated over. Held here
#: since C0; see the module docstring for why it is a declaration and not a mirror.
CHUNK_ARMS: tuple[tuple[int, int, int, int], ...] = (
    (16, 64, 2, 8), (32, 64, 2, 8), (64, 64, 2, 8),
    (16, 64, 2, 64), (32, 64, 2, 64), (64, 64, 2, 64),
    (32, 32, 2, 64), (64, 32, 2, 64),
    (16, 128, 2, 64), (32, 128, 2, 64), (64, 128, 2, 64),
    (32, 256, 2, 64),
    (16, 64, 2, 16), (32, 64, 2, 16), (64, 64, 2, 16),
    (32, 128, 3, 16), (32, 128, 4, 8), (32, 32, 4, 8),
    # the FLAT-ROUTING baseline arms (D = 1; N = B <= 256 by the manifest):
    (32, 64, 1, 64), (32, 64, 1, 256), (16, 64, 1, 256),
    # the SCALE rung (N = 65536):
    (32, 128, 2, 256),
)

#: The `(D, B)` topologies the dispatch can reach. The engine's arm fold pins
#: `C = CHUNK_TOKENS` and picks `BC` from the manifest, so an arm at any other
#: `C` is built but unreachable from the facade, and the census must be stated
#: over what the DISPATCH reaches or it would claim coverage nobody can run.
CHUNK_TOKENS = 32
BUILT_TOPOLOGIES: frozenset[tuple[int, int]] = frozenset(
    (D, B) for (C, _BC, D, B) in CHUNK_ARMS if C == CHUNK_TOKENS)
