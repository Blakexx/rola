# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0

"""THE DECODE FOLD: the declared summation order, held to BIT-IDENTITY.

The step's operand pass lives INSIDE the step kernel
(``csrc/rola/src/decode/decode_fold.cuh``), so what this file gates it gates through the
step's own output. Two of the three things the fold does are bit-exact for free -- the
``[B, 1, H, W] -> [B*H, W]`` fold is the same bytes at ``T = 1``, and a widening cast
rounds nothing. The third, the per-level normalizing sum, has a summation ORDER, and an
order nobody wrote down is an order no gate can hold anything to.

So the order is DECLARED -- lane-strided ascending partials over 32 lanes, folded by the
five ``__shfl_down_sync`` steps -- and this file is that declaration's gate.

**THE READOUT THE GATE IS BUILT ON.** With ``D = 1``, a read level that is nonzero
everywhere, no deposit, and a state whose rows are zero except at ONE leaf ``s0``, the
step's whole arithmetic collapses to

    ``y[c] = a[s0] * e[c] / (a[s0] * m + eps)``,   ``a = row[s0] / row_sum(row)``

-- one multiply per column and one divide, all in fp32 and all in a fixed order. So ``y``
is a function of the DIVISOR's last bits and of nothing else, and the reference below
computes exactly that expression from ``declared_row_sum``.

Cell 4 is the protocol's teeth: on a row built so that summation order is observable, the
declared order and a different one DISAGREE, so the reference is answering a question
about order rather than about arithmetic in general.

Run: ``pytest tests/integration/test_decode_fold.py``
"""
from __future__ import annotations

import pytest
import torch

from rola.ops.constants import READOUT_EPS
from rola.ops.decode import _decode_step, derive_decode_geometry, fold_levels
from rola.ops.lattice import permutation
from rola.ops.paging import to_split_planes
from tests.oracle.oracle_fixtures import _topology

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="the fold is a CUDA kernel"),
]

_DTYPES = (torch.float32, torch.bfloat16)
_DV = 64


def declared_row_sum(row: torch.Tensor) -> torch.Tensor:
    """``[BH, W] -> [BH]`` fp32 in the kernel's DECLARED order, and in no other."""
    BH, width = row.shape
    lanes = torch.zeros(BH, 32, dtype=torch.float32, device=row.device)
    for lane in range(32):
        acc = torch.zeros(BH, dtype=torch.float32, device=row.device)
        for k in range(lane, width, 32):
            acc = acc + row[:, k].float()
        lanes[:, lane] = acc
    off = 16
    while off:
        lanes[:, :off] = lanes[:, :off] + lanes[:, off:2 * off]
        off >>= 1
    return lanes[:, 0]


def _hi_truncate(x: torch.Tensor) -> torch.Tensor:
    """The value a READ-ONLY atom's hi-plane-only load reconstructs: the fp32 word with
    its low 16 bits zeroed (``hi || 0``, docs/internals/common/state_page.md#load) --
    the truncation, not a round. ``s0`` here is touched by no deposit, so every row the
    step reads is read-only and this IS the value the kernel's arithmetic sees."""
    bits = x.contiguous().view(torch.int32)
    return ((bits >> 16) << 16).view(torch.float32).reshape(x.shape)


def _single_leaf_state(BH: int, config, s0: int, device, seed: int):
    """A ``[BH, N, d_v + 1]`` logical page, STORED as split bf16 planes (the contract,
    docs/internals/state.md#split-planes) -- zero everywhere but CANONICAL leaf ``s0``,
    in the plane's own LATTICE order. Returns ``(state, e, m)``: ``state`` in the stored
    form the step's entry sweep reads, and ``e``/``m`` HI-TRUNCATED to the value a
    read-only atom's load actually reconstructs -- the reference must be built from what
    the kernel reads, not from the fp32 draw that never reaches it whole."""
    gen = torch.Generator(device=device).manual_seed(seed)
    pi = permutation(config.widths, config.lattice_k, config.lattice_m, device=device)
    state = torch.zeros(BH, config.N, _DV + 1, dtype=torch.float32, device=device)
    row = torch.rand(BH, _DV + 1, dtype=torch.float32, device=device, generator=gen) + 0.5
    state[:, int(pi[s0])] = row
    row_hi = _hi_truncate(row)
    return to_split_planes(state), row_hi[:, :_DV], row_hi[:, _DV]


def _y_of(read, *, dtype, normalize, seed, s0=0):
    """One step over the collapsing fixture. Returns ``(y[BH, d_v], e, m)``."""
    B, _T, H, width = read[0].shape
    BH, device = B * H, read[0].device
    kw = dict(dtype=dtype, device=device)
    #: NO DEPOSIT: an all-zero write side leaves `supp(W)` empty, so every touched row is
    #: read-only and the state the readout sees is the state that was placed.
    write = tuple(torch.zeros(B, 1, H, width, **kw) for _ in read)
    g_write = torch.ones(B, 1, H, **kw)
    v = torch.zeros(B, 1, H, _DV, **kw)
    config = derive_decode_geometry(_topology((width,)), d_v=_DV, decay=False, BH=BH,
                                    device=device)
    state, e, m = _single_leaf_state(BH, config, s0, device, seed + 1)
    y, _state, _ws = _decode_step(v, read, write, g_write, config,
                                  state.view(B, H, width, _DV + 1),
                                  normalize=normalize)
    return y.reshape(BH, _DV), e, m


def _reference_y(read_level, *, normalize, e, m, s0=0):
    """The collapsed readout, computed from the DECLARED order and nothing else."""
    folded = fold_levels((read_level,))[0]
    a = folded[:, s0] / declared_row_sum(folded) if normalize else folded[:, s0]
    return (a.unsqueeze(-1) * e) / (a * m + READOUT_EPS).unsqueeze(-1)


def _read_row(B, H, width, *, dtype, seed, device="cuda"):
    gen = torch.Generator(device=device).manual_seed(seed)
    #: Every entry is strictly positive, so `supp(R)` is the whole leaf space and the
    #: walk visits every row -- the sum being gated is over the FULL width.
    return (torch.rand(B, 1, H, width, dtype=dtype, device=device, generator=gen) + 0.25,)


# ---------------------------------------------------------------------------
# 1. the cast-only half: the widening and the address fold are exact
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("width", [16, 64, 256])
def test_the_unnormalized_fold_is_the_torch_fold(dtype, width):
    """No divide: `y` is the amplitude the torch fold reads out of the same view."""
    read = _read_row(3, 2, width, dtype=dtype, seed=4)
    y, e, m = _y_of(read, dtype=dtype, normalize=(False,), seed=4)
    assert torch.equal(y, _reference_y(read[0], normalize=False, e=e, m=m))


def test_a_strided_view_folds_by_address():
    """A level is ordinarily a VIEW of one packed plane, and the fold indexes it there."""
    packed = _read_row(3, 2, 32, dtype=torch.float32, seed=3)[0]
    sliced = (packed[..., ::2],)
    dense = (sliced[0].contiguous(),)
    y_view, e, m = _y_of(sliced, dtype=torch.float32, normalize=(True,), seed=8)
    y_dense, e2, m2 = _y_of(dense, dtype=torch.float32, normalize=(True,), seed=8)
    assert torch.equal(e, e2) and torch.equal(m, m2), "the two runs saw different states"
    assert torch.equal(y_view, y_dense)
    assert torch.equal(y_view, _reference_y(dense[0], normalize=True, e=e, m=m))


# ---------------------------------------------------------------------------
# 2. the declared order, held to bit-identity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("width", [16, 32, 64, 256])
def test_the_normalized_read_matches_the_declared_order(dtype, width):
    read = _read_row(3, 2, width, dtype=dtype, seed=7)
    y, e, m = _y_of(read, dtype=dtype, normalize=(True,), seed=7)
    assert torch.equal(y, _reference_y(read[0], normalize=True, e=e, m=m))


def test_the_flag_is_honoured_and_not_assumed():
    """The same row, both flags: normalizing and not normalizing are DIFFERENT answers."""
    read = _read_row(2, 2, 64, dtype=torch.float32, seed=11)
    y_on, e, m = _y_of(read, dtype=torch.float32, normalize=(True,), seed=11)
    y_off, e2, m2 = _y_of(read, dtype=torch.float32, normalize=(False,), seed=11)
    assert torch.equal(e, e2) and torch.equal(m, m2)
    assert not torch.equal(y_on, y_off), "the normalize flag changed nothing"
    assert torch.equal(y_on, _reference_y(read[0], normalize=True, e=e, m=m))
    assert torch.equal(y_off, _reference_y(read[0], normalize=False, e=e, m=m))


# ---------------------------------------------------------------------------
# 3. the fold's refusals
# ---------------------------------------------------------------------------

def test_more_than_one_token_is_refused():
    read = _read_row(2, 2, 16, dtype=torch.float32, seed=5)
    two = tuple(t.repeat(1, 2, 1, 1) for t in read)
    with pytest.raises(RuntimeError, match=r"\[B, 1, H, W\]"):
        _y_of(two, dtype=torch.float32, normalize=(False,), seed=5)


def test_a_dtype_outside_the_instantiated_set_is_refused():
    read = _read_row(2, 2, 16, dtype=torch.float16, seed=5)
    with pytest.raises(RuntimeError, match="float32 or bfloat16"):
        _y_of(read, dtype=torch.float16, normalize=(False,), seed=5)


# ---------------------------------------------------------------------------
# 4. the teeth: the reference is answering a question about ORDER
# ---------------------------------------------------------------------------

def test_the_declared_order_is_distinguishable_from_another_one():
    """A row where summation order is OBSERVABLE, so the gate cannot be vacuous."""
    #: Each addend is half an ulp of 1, so ascending accumulation loses every one of
    #: them; the declared order pairs them off in the lanes first and keeps them.
    tiny = 2.0 ** -24
    row = torch.tensor([[1.0] + [tiny] * 63], dtype=torch.float32, device="cuda")
    ascending = torch.zeros(1, dtype=torch.float32, device="cuda")
    for k in range(row.shape[1]):
        ascending = ascending + row[:, k]
    declared = declared_row_sum(row)
    assert not torch.equal(declared, ascending)
    assert declared.item() > ascending.item() == 1.0


def test_the_kernel_takes_the_declared_order_and_not_the_ascending_one():
    """The same observable row, THROUGH THE STEP: the two orders give different `y`."""
    tiny = 2.0 ** -24
    row = torch.tensor([1.0] + [tiny] * 63, dtype=torch.float32, device="cuda")
    read = (row.view(1, 1, 1, 64).contiguous(),)
    y, e, m = _y_of(read, dtype=torch.float32, normalize=(True,), seed=21)
    folded = fold_levels(read)[0]
    declared = declared_row_sum(folded)
    ascending = torch.zeros(folded.shape[0], dtype=torch.float32, device=folded.device)
    for k in range(folded.shape[1]):
        ascending = ascending + folded[:, k]
    assert not torch.equal(declared, ascending), "the fixture stopped being order-observable"
    a_asc = folded[:, 0] / ascending
    y_asc = (a_asc.unsqueeze(-1) * e) / (a_asc * m + READOUT_EPS).unsqueeze(-1)
    assert torch.equal(y, _reference_y(read[0], normalize=True, e=e, m=m))
    assert not torch.equal(y, y_asc), "the kernel's divisor is order-blind here"
