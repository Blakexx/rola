// Fixture for tools/lint/unused_instantiations.py -- paired with
// unused_instantiation_manifest.json. `present_kernel` is mangled-present in
// that fixture manifest (must NOT fire); `absent_kernel` is not (must fire).
template <int D>
__global__ void present_kernel(int* out) {
  *out = D;
}

template <int D>
__global__ void absent_kernel(int* out) {
  *out = D;
}
