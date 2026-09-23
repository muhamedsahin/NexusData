#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "nexusdata/core/error.hpp"

namespace nexusdata {

/// Rank-N shape as a contiguous list of dimension sizes.
/// Empty shape means a scalar with 1 element (numel == 1) only when
/// explicitly constructed as `{}` for “uninitialized”; use `{1}` for scalar.
/// Convention in v0.1: empty shape => numel 0 (no allocation).
using Shape = std::vector<std::size_t>;

/// Product of all dimensions. Empty shape yields 0.
[[nodiscard]] inline std::size_t numel(const Shape& shape) {
    if (shape.empty()) {
        return 0;
    }
    std::size_t total = 1;
    for (std::size_t dim : shape) {
        if (dim != 0 && total > (SIZE_MAX / dim)) {
            throw ShapeError("shape numel overflow");
        }
        total *= dim;
    }
    return total;
}

[[nodiscard]] inline std::string shape_to_string(const Shape& shape) {
    std::string out = "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            out += ", ";
        }
        out += std::to_string(shape[i]);
    }
    out += "]";
    return out;
}

/// "Component: expected shape E, got G"
[[nodiscard]] inline std::string format_shape_error(const char* what,
                                                    const Shape& expected,
                                                    const Shape& got) {
    return std::string(what) + ": expected shape " + shape_to_string(expected) +
           ", got " + shape_to_string(got);
}

[[nodiscard]] inline bool shapes_equal(const Shape& a, const Shape& b) {
    return a == b;
}

} // namespace nexusdata
