// the extension's freshness fact -- see docs/internals/common/build_stamp.md
#pragma once

#include <cstdint>

namespace rola {

int64_t csrc_build_stamp();                // the csrc content hash, read back off the device
double sm_clock_ghz(int64_t spin_cycles);  // the SM's effective clock under load, off the device

}  // namespace rola
