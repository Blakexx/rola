"""The kernel list (`tools/kernels.py`): the parser on the shapes the tree writes, and the tree's own list.

See docs/internals/tools/kernels.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import kernels  # noqa: E402

TEXT = '''
//: __global__ void commented_out(int x) {}
static const char* kMsg = "__global__ void in_a_string(";
__global__ void plain(int* out) {}

template <int D, int DV>
__global__ __launch_bounds__(BoxPlan<D, DV>::kThreads, plan_ctas(D, DV)) void bounded(
    const Params p) {}

template <typename T> struct Holder {};
__global__ __launch_bounds__(256) void after_a_struct(float* y) {}
'''


def test_the_parser_reads_every_definition_and_nothing_in_comments_or_strings():
    assert kernels.parse(TEXT) == [("plain", 4, False), ("bounded", 7, True), ("after_a_struct", 11, False)]


def test_the_symbol_name_is_the_identifier_before_the_argument_list():
    assert kernels.function_name("void rola::carry::carry_kernel<2, 64, 8>(rola::carry::CarryParams)") == "carry_kernel"
    assert kernels.function_name("rola::stampdet::decode_stamp_kernel(long*)") == "decode_stamp_kernel"
    assert kernels.function_name("void ns::k<a::b<1>, 2>(x<3>)") == "k"


def test_the_tree_lists_every_family_and_its_kernels():
    by_family = {}
    for k in kernels.kernels():
        by_family.setdefault(k.family, set()).add(k.name)
    assert {"decode_step_kernel", "decode_stamp_kernel"} <= by_family["decode"]
    assert "carry_kernel" in by_family["carry"]
    assert "intra_kernel" in by_family["intra"]
    assert "facts_pass" in by_family["facts"]
    assert {"union_forward_kernel", "factor_backward_kernel"} <= by_family["entmax"]
    assert {"csrc_stamp_kernel", "sm_clock_kernel"} <= by_family["common"]


def test_every_kernel_is_compiled_by_a_unit():
    for k in kernels.kernels():
        assert kernels.units_of(k), f"{k.where()}: `{k.name}` is defined in a file no translation unit compiles"
