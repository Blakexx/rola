// measure/lab/experiments/read_w1024.cu -- QUESTION: what does the readout cost a token at W = 1024 with sm_80's
// 512-slot pass, where a read digit holds ~16 tokens a window, against W = 512's ~8 (each digit's state read from the
// image serving twice the tokens)? METHOD: `read_image.cu` built with a 512-slot pass (`READ_IMAGE_SLOTS`, 8 image
// planes so it fits sm_86 for the counts) on nl64k-alt-k4's read lists merged in window pairs (`read_w1024.inputs.py`):
// the facts-form steps, tokens on M and on N, their outputs checked exactly, their counts priced on both cards.
// see docs/internals/tools/lab.md#read-w1024
#define READ_IMAGE_SLOTS 512
#include "read_image.cu"
