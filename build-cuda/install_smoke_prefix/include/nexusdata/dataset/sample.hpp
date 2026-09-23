#pragma once

#include "nexusdata/core/ndarray.hpp"

namespace nexusdata {

/// Single supervised sample. v0.1: input + label only.
/// Thread-safety: same as NDArray (shared storage).
struct Sample {
    NDArray input;
    NDArray label;
};

} // namespace nexusdata
