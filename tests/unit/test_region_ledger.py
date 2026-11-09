"""The region ledger's function map (`tools/region_ledger.py`, docs/internals/tools/stall_census.md): every definition
form the kernel sources use starts a function -- a `__host__ __device__ constexpr` helper missed books its lines to the
definition before it -- and no statement inside a body does.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import region_ledger  # noqa: E402

SOURCE = """\
__device__ __forceinline__ void plain(int x) {
  constexpr int kStep = 2;
  constexpr uint32_t kMask = mask_of(3);
  static_assert(sizeof(int) == 4, "");
  static_for<4>([&](auto I) { x += I; });
}
__host__ __device__ __forceinline__ constexpr uint32_t unit(int l) {
  return l;
}
__host__ __device__ constexpr int slot(int d) { return d; }
__device__ __forceinline__ long long wide(int x) {
}
__device__ __forceinline__ const uint16_t* row(const uint16_t* p) {
}
constexpr bool holds() {
}
constexpr int kTable = table_of(4);
template <class P>
struct Sums {
  __device__ __forceinline__ int operator[](int j) const {
  }
  __host__ __device__ __forceinline__ int total() const {
  }
};
__global__ __launch_bounds__(256, 1) void kernel(int* y) {
}
"""


def test_every_definition_form_starts_a_function():
    got = region_ledger.function_map(SOURCE.splitlines())
    assert got == [(1, "plain"), (7, "unit"), (10, "slot"), (11, "wide"), (13, "row"), (15, "holds"), (19, "Sums"),
                   (20, "Sums::operator[]"), (22, "Sums::total"), (25, "kernel")]


def test_a_helper_line_is_its_own_function():
    fmap = region_ledger.function_map(SOURCE.splitlines())
    assert region_ledger.fn_of(fmap, 8) == "unit"
    assert region_ledger.fn_of(fmap, 3) == "plain"


WRAPPED = """\
template <class Wk>
struct IntraAt {
  static constexpr int kNT = Wk::kNT;
  static __device__ __forceinline__ void gram_dense(int x) {
    constexpr int kStep = 2;
  }
  struct Copier {
    __device__ __forceinline__ void values(int x) const {
    }
  };
  template <int D>
  static __device__ __forceinline__ void run(int y) {
  }
};
__device__ __forceinline__ void after(int z) {
}
"""


def test_a_kernel_wrapped_in_a_class_keeps_its_members_and_nested_structs():
    got = region_ledger.function_map(WRAPPED.splitlines())
    assert got == [(2, "IntraAt"), (4, "IntraAt::gram_dense"), (7, "IntraAt::Copier"), (8, "IntraAt::Copier::values"),
                   (12, "IntraAt::run"), (15, "after")]


def test_a_wrapped_member_is_its_own_component_and_a_nested_struct_its_struct():
    comps = {"gram_dense", "Copier"}
    assert region_ledger.component_of("IntraAt::gram_dense", comps) == "gram_dense"
    assert region_ledger.component_of("IntraAt::Copier::values", comps) == "Copier"
    assert region_ledger.component_of("IntraAt::run", comps) is None


def test_a_barrier_file_without_mbar_wait_is_refused(tmp_path):
    """A spin is an `mbar_wait` frame: a `--barrier` file defining none would report no spin, silently."""
    import pytest

    bogus = tmp_path / "barrier.cuh"
    bogus.write_text("__device__ __forceinline__ void rendezvous() {\n}\n")
    with pytest.raises(SystemExit, match="defines no `mbar_wait`"):
        region_ledger.Attribution(tmp_path / "none.so", "", "sm_86", [], bogus)
    real = ROOT / "csrc" / "rola" / "src" / "common" / "atoms" / "barrier.cuh"
    assert any(name == "mbar_wait" for _, name in region_ledger.function_map(real.read_text().splitlines()))
