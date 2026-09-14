// Fixture for tools/lint/rules/warp_collective_shortcircuit.yml -- must fire.
__device__ void bad(bool part, unsigned mask, int x) {
  bool r = part && __shfl_sync(mask, x, 0);
  int t = (r ? __ballot_sync(mask, x) : 0);
}
