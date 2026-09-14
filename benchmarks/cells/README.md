# The parked records — fixtures for a consumer that retired

| file | status | what it is |
|---|---|---|
| `topology_manifest.json` | **PARKED, historical** | the committed statistics manifest of the tiled harness's named cells: per cell the widths, the owner-block size, the seed, the support constructors and the realized `mean_K`/`mean_KI` a run was allowed to see. |
| `latency_baseline.json` | **PARKED, historical** | per `(cell, arm)` median and IQR, with `clock_settled` and the timing source, as measured on the tiled consumer. |

Both are keyed to the TILED consumer and its owner-block ladder, which retired at the
deletion recorded in `docs/internals/DELETIONS.md`. Their arms name launches that no
longer exist, so **no current gate reads either file and no current number may be
compared against one**: a different kernel, a different tiling and a different envelope,
and quoting these medians beside a current one would be comparing two instruments.

They are kept rather than deleted because they are the RECORD of what was measured — the
same reason the measurements store keeps every sample of a retired arm.

The live registry is `carry_cells.json` (kernel cells) and `layer_cells.json` +
`layer_manifest.json` (layer cells); `../README.md` is the map.
