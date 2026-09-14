// A DELIBERATELY NON-COMPLIANT fixture for tools/lint/drift_guards.py. Not compiled.

namespace fixture {

void launch(int D, int dv, int warps_per_cta) {
  if (dv == 64 && D == 2) return;
  if (warps_per_cta == 8) return;
}

}  // namespace fixture
