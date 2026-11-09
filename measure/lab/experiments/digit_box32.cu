// measure/lab/experiments/digit_box32.cu -- QUESTION: `digit_box.cu`'s question for the 32 x 32 box, its image in two
// parts (the whole 64 KB does not fit beside the rest on sm_86). METHOD: `digit_box.cu` built for 32 lists x 32
// digits (4 digits a warp), its hand-off the two-part one. see docs/internals/tools/lab.md#digit-box
#define READ_IMAGE_DIGITS 32
#include "digit_box.cu"
