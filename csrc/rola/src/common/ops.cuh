#pragma once

// THE BASELINE OPERATIONS: every atom (`common/atoms/`), one include.
// See docs/internals/common/ops.md

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "common/arch_caps.cuh"

#include "common/atoms/address.cuh"
#include "common/atoms/copy.cuh"
#include "common/atoms/barrier.cuh"
#include "common/atoms/warp.cuh"
#include "common/atoms/clock.cuh"
#include "common/atoms/math.cuh"
#include "common/atoms/reduce.cuh"
#include "common/atoms/memory.cuh"
