#pragma once

#include "nexusdata/adapter/dlpack.h"
#include "nexusdata/core/ndarray.hpp"

namespace nexusdata {

/// Export a host (or CUDA) NDArray as DLManagedTensor.
/// Caller must invoke `->deleter(tensor)` (or consume via framework that does).
/// Keeps NDArray storage alive until deleter runs.
[[nodiscard]] DLManagedTensor* ndarray_to_dlpack(const NDArray& array);

/// Import DLManagedTensor into an owning NDArray (deep copy on host).
/// Does not call the source deleter; caller remains responsible for @p tensor lifetime
/// unless @p take_ownership is true (then deleter is called after copy).
[[nodiscard]] NDArray ndarray_from_dlpack(DLManagedTensor* tensor, bool take_ownership = false);

} // namespace nexusdata
