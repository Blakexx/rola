// measure/harness/carry_parts/carry_parts.cu -- THE PART HARNESS MODULE'S CALIBRATION OPERATOR: `torch.ops.rola_parts
// .calibrate` runs one calibration kernel of carry_calib.cuh on every warp of every CTA (measure/harness/
// bench_carry_calib.py). A bench instrument, the graph's //:parts_library; never shipped.
// See docs/internals/carry/calibration.md
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <torch/csrc/stable/library.h>

#include "common/torch_seam.cuh"

#include "carry_calib.cuh"

STABLE_TORCH_LIBRARY(rola_parts, m) {
  m.def(
      "calibrate(int warps, int mode, int burst, int iters, int owners, Tensor(a!) out, Tensor src) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(rola_parts, CompositeExplicitAutograd, m) {
  m.impl("calibrate", TORCH_BOX(&rola::carry::calib::calibrate));
}

extern "C" PyObject* PyInit__C_parts(void) {
  static PyModuleDef module = {PyModuleDef_HEAD_INIT,
                               "_C_parts",
                               "the part harness's operators, torch.ops.rola_parts",
                               -1,
                               nullptr,
                               nullptr,
                               nullptr,
                               nullptr,
                               nullptr};
  return PyModule_Create(&module);
}
