// Fixture for tools/lint/rules/leader_only_fence.yml -- must fire.
__device__ void bad(int lane, int* flag) {
  if (lane == 0) {
    *flag = 1;
    __threadfence();
  }
}

// A fence NOT guarded by a single-lane predicate must NOT fire.
__device__ void ok(int* flag) {
  *flag = 1;
  __threadfence();
}
