# `read_w1024`

Mirrors `measure/lab/experiments/read_w1024.cu`, a part harness of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

`read_image.cu` built with sm_80's 512-slot readout pass (`READ_IMAGE_SLOTS`; 4 image planes and 32 staged entries a
digit, so it fits sm_86) on nl64k-alt-k4's read lists merged in window pairs (`read_w1024.inputs.py`: 16 windows of
1024 tokens, the second window's union ranks after the first's, one pass, colored and placed as above). A read digit
holds ~16 tokens a window, so each read of its state from the image serves twice W = 512's tokens.

| per token | W = 512, M | W = 512, N | W = 1024, M | W = 1024, N |
|---|---|---|---|---|
| sm_86 measured, free-running | 4.29 | 3.70 | 3.19 | 3.21 |
| A100 predicted, shared-memory bound | 2.93 | 2.61 | 1.91 | 1.91 |

At ~16 tokens a digit the forms tie: M's 16-position steps fill, and N's 0.85x MMAs pay its higher per-MMA cost (its
8-byte entry loads take two wavefronts for 4 MMAs). Record: `measure/lab/results/read_w1024/sm_86.json`.
