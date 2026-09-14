// Fixture for tools/lint/r9_enforcement.py -- `bad_switch` must fire (no
// __trap() default, no R9-CASES marker); `ok_switch` must not (both present).
__device__ int bad_switch(int kind) {
  switch (kind) {
    case 0:
      return 1;
    case 1:
      return 2;
    default:
      return 3;
  }
}

// R9-CASES: kKind0, kKind1, kKind2 (fixture)
__device__ int ok_switch(int kind) {
  switch (kind) {
    case 0:
      return 1;
    case 1:
      return 2;
    default:
      __trap();
  }
}
