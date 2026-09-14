// Fixture for tools/lint/rules/torch_include_in_arm_tu.yml -- must fire.
// (The rule itself is scoped to csrc/rola/src/instantiations/**; this fixture
// file lives under tools/lint/fixtures/ only so the rule has something to
// exercise without landing a bad file in the real instantiations directory.)
#include <torch/extension.h>
__global__ void k() {}
