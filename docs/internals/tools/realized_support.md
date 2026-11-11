# `tools/realized_support.py` — what a layer construction routes

A carry cell DECLARES its support (`k_tok`, the draw); a layer construction's weights REALIZE one, through the
producer's solve on the cell's hidden states (`measure.cells.layer.build`). A layer reading's regime is that realized
support, so the tool reads it from the bundle the kernels take, after quantization: per level and side, the nonzero
digits a token over every batch row, token and head -- the mean, the median, the 99th percentile and the fraction of
the level's width -- and whether the side is declared sparse (a dense side's zeros are values that rounded to zero).

    bazel run //tools:realized_support -- [nl64k-alt-h4] [--json support.json]   # one construction, or every one

`declare.py` declares it: every construction once a binary, stored at `rola/support`.

The construction's logit gain is the knob that moves it (`tests/integration/test_producer_sweep.py` holds the
direction); `measure/cells/layer.py` declares the long layer at gains 1, 4 and 16, so a layer-level comparison spans the
support instead of reading one point of it.
