// Fixture for the device clang-tidy profile (//tools/lint:clang_tidy_device_fixtures) -- must fire (misc-unused-parameters).
__device__ int bad(int used, int unused_param) {
  return used;
}

// Must NOT fire: every parameter is used.
__device__ int ok(int a, int b) {
  return a + b;
}

// must fire (clang-analyzer-deadcode.DeadStores): `twice` is stored and never read
__device__ int dead(int x) {
  int twice = 0;
  twice = x * 2;
  return x;
}
