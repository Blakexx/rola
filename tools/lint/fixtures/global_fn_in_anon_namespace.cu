// Fixture for tools/lint/rules/global_fn_in_anon_namespace.yml -- must fire.
namespace {
__global__ void kern(int* x) { *x = 1; }
}  // namespace

// A named namespace must NOT fire (this is the shipped pattern).
namespace rola_ok {
__global__ void kern2(int* x) { *x = 1; }
}  // namespace rola_ok
