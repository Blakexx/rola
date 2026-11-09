# `csrc/rola/src/scale/scale_api.cuh` — the scale pass's host entry

The one entry the pybind unit binds (`csrc/rola/rola_api.cpp`'s `scale_forward`); `rola.ops.scale.scale` is its Python
caller. What the pass computes is [`scale.md`](scale.md).

<a id="scale-forward"></a>

## `scale_forward`

`scale_forward(read_rows, write_rows, fm, widths, level_s, heads, inv_factor) -> bytes`: the bundle's two levels of
rows a side (as `RouteFactors.amp_read` / `amp_write` hold them, at their padded widths), the facts pass's `fm`, level
S, the query heads a write head and the reciprocal of the carry's factor. The entry finds its card from the rows'
device and launches at the card's build (`ScalePlan<B>`, [`scale.md`](scale.md#constants)): it allocates there the
`[BH, windows, width_S]` fp32 mass workspace (the windows' prefix masses, read by nothing after the call) and returns the
`[BH heads, windows W]` int8 bytes (the window grid whole, at the card's W). Three launches on the current stream.

<a id="refusals"></a>

## The refusals

Each by name: a depth other than two; level S not 0 or 1; a level's rows not `[B, T, H, width]` u8 or f16 with digits
contiguous and every row and stride on 16 bytes (a lane's eight digits are one 8- or 16-byte load); a width not a
multiple of 8 digits, or level S's masses over 48 KB of shared memory (12,288 digits); more than 65,535 query
batch-heads (the grid's y); `fm` not the facts pass's contiguous fp32 `[BH, L, 2]`. A build's window not a multiple of
32 does not compile (its plan's assertion). `rola.ops.scale.serves` names the depth refusal before the launch.
