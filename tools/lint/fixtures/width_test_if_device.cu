// Fixture for tools/lint/rules/width_test_if_device.yml -- must fire.
__device__ void f(int x, int kDv, int DV) {
  if (kDv == 64) { x = 1; }
  if (DV == 128) { x = 2; }
  // Not a width test -- must NOT fire.
  if (other == 64) { x = 3; }
}
