// measure/lab/experiments/list_box.cu -- QUESTION: `digit_box.cu`'s question for a box of 16 lists x 32 digits, its
// whole image (32 KB) resident. METHOD: `digit_box.cu` built for 16 lists (the readout one k-tile on 32-byte read rows,
// each list box adding its partial output; 2 lists a fold warp) x 32 digits. see docs/internals/tools/lab.md#digit-box
#define READ_IMAGE_DIGITS 32
#define READ_IMAGE_LISTS 16
#include "digit_box.cu"
