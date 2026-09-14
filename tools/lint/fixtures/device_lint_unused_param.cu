// Fixture for tools/lint/run_clang_tidy_device.sh -- must fire (misc-unused-parameters).
__device__ int bad(int used, int unused_param) {
  return used;
}

// Must NOT fire: every parameter is used.
__device__ int ok(int a, int b) {
  return a + b;
}
