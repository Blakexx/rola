"""Side-gain positive parameterization and its one configuration boolean.

Routing allocations are normalized per level per side, so without a magnitude the model
can only ever write total mass 1 per token. The learned WRITE magnitude supplies it:

    hidden -> [packed route logits, write gain logit]     # g_read := 1

There is no read gain: the readout is a ratio, so ``g_read := 1`` under the
self-normalizing readout, where a per-token uniform read scale cancels identically in
the ratio.

**THE GAIN LOGIT IS COLUMNS OF THE ROUTER GEMM.** The gain is a function of the read
stream, and so are the routing logits, so a separate ``[hidden -> H]`` projection was a
second GEMM plus a second cast of the whole source stream for one column per head. The
columns live in ``route_W``'s trailing span (``rola/routing/projection.py``); this
module owns what is left, which is the bias, the layout and the positive
parameterization.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

WRITE_GAIN = 0


def softplus_side_gain(logit: torch.Tensor, bias: torch.Tensor | float = 0.0) -> torch.Tensor:
    """Baseline positive parameterization: ``softplus(logit + bias)``.

    **Flagged model decision**: unit gain at zero logits/bias is *not*
    migration-equivalent -- the exact ``ELU+1`` decomposition gives ``g_read = sqrt(K)``,
    ``g_write = K`` for the historical width-``K`` feature map. The caller's choice of
    ``bias`` is a new model-level init decision requiring its own quality/stability gate
    before large training, and must not be presented as equivalent to any earlier
    checkpoint.
    """
    return F.softplus(logit + bias)


def init_side_gain_bias(target_gain: float) -> float:
    """Invert ``softplus`` to place an all-zero-logit gain at ``target_gain``."""
    if target_gain <= 0:
        raise ValueError(f"target_gain must be > 0, got {target_gain}")
    import math

    return math.log(math.expm1(target_gain))


@dataclass(frozen=True, slots=True)
class GainConfig:
    """The INTERNAL gain surface: one boolean, behind the public ``gain: bool``.

    ``False`` REMOVES the projection column rather than pinning it to 1: a parameter
    whose value is fixed is a parameter whose gradient is wasted, and a missing column
    is a constant, which has no gradient to waste.
    """

    write_gain: bool = True

    @classmethod
    def from_public(cls, gain: bool) -> GainConfig:
        if type(gain) is not bool:
            raise TypeError(f"gain must be a bool, got {gain!r}")
        return cls(write_gain=gain)

    @property
    def layout(self) -> tuple[str, ...]:
        """The projection's columns, in order. A column is present iff its gain is."""
        return ("write",) if self.write_gain else ()


def gain_layout(*, write_gain: bool = True) -> tuple[str, ...]:
    """The side-gain projection's columns. THE ONE PLACE it is decided."""
    return GainConfig(write_gain=write_gain).layout


class SideGain(nn.Module):
    """The gain half of the feature map: a bias and a positive parameterization.

    It takes the LOGIT COLUMNS the router GEMM already produced and returns
    ``{'g_write': [B, T, H]}``. ``columns`` is 0 when the gain is ablated, and then
    there is no logit to take and the gain is the CONSTANT 1 -- a missing column has no
    gradient to waste, which is what ``gain=False`` means.

    ``bias_init`` places an all-zero-logit gain at that value (``None`` leaves the bias
    at zero). Why the gain belongs to routing rather than to the layer:
    ``docs/api.md`` section 2.1.
    """

    def __init__(
        self,
        *,
        num_heads: int,
        config: GainConfig,
        bias_init: float | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.config = config
        self.layout = self.config.layout
        bias_value = 0.0 if bias_init is None else init_side_gain_bias(bias_init)
        self.gain_bias = nn.Parameter(
            torch.full((self.num_heads, len(self.layout)), bias_value))

    @property
    def columns(self) -> int:
        """Router-GEMM columns this gain consumes, per head."""
        return len(self.layout)

    def forward(self, logits: torch.Tensor | None, *, batch: int, tokens: int,
                dtype: torch.dtype, device) -> dict[str, torch.Tensor]:
        if not self.columns:
            return {"g_write": torch.ones(batch, tokens, self.num_heads,
                                          dtype=dtype, device=device)}
        if logits.shape[-1] != self.columns:
            raise ValueError(
                f"the side gain takes {self.columns} logit column(s) per head, got "
                f"{logits.shape[-1]}")
        biased = logits + self.gain_bias.to(logits.dtype)[None, None, :, :]
        return {"g_write": softplus_side_gain(biased[..., WRITE_GAIN]).to(dtype)}


__all__ = [
    "WRITE_GAIN",
    "GainConfig",
    "SideGain",
    "gain_layout",
    "init_side_gain_bias",
    "softplus_side_gain",
]
