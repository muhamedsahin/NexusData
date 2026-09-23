#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "nexusdata/core/dtype.hpp"
#include "nexusdata/core/error.hpp"
#include "nexusdata/core/ndarray.hpp"

namespace nexusdata {
namespace detail {

/// Read element @p index as double (host contiguous array).
[[nodiscard]] inline double as_double(const NDArray& a, std::size_t index) {
    if (index >= a.numel()) {
        throw IndexError("as_double: index out of range");
    }
    switch (a.dtype()) {
        case DType::Float32:  return static_cast<double>(a.data<float>()[index]);
        case DType::Float64:  return a.data<double>()[index];
        case DType::Int8:     return static_cast<double>(a.data<std::int8_t>()[index]);
        case DType::Int16:    return static_cast<double>(a.data<std::int16_t>()[index]);
        case DType::Int32:    return static_cast<double>(a.data<std::int32_t>()[index]);
        case DType::Int64:    return static_cast<double>(a.data<std::int64_t>()[index]);
        case DType::UInt8:    return static_cast<double>(a.data<std::uint8_t>()[index]);
        case DType::UInt16:   return static_cast<double>(a.data<std::uint16_t>()[index]);
        case DType::UInt32:   return static_cast<double>(a.data<std::uint32_t>()[index]);
        case DType::UInt64:   return static_cast<double>(a.data<std::uint64_t>()[index]);
        case DType::Bool:     return a.data<bool>()[index] ? 1.0 : 0.0;
        default:
            throw InvalidArgumentError("as_double: unsupported dtype " +
                                       std::string(to_string(a.dtype())));
    }
}

inline void set_double(NDArray& a, std::size_t index, double v) {
    if (index >= a.numel()) {
        throw IndexError("set_double: index out of range");
    }
    switch (a.dtype()) {
        case DType::Float32:  a.data<float>()[index] = static_cast<float>(v); break;
        case DType::Float64:  a.data<double>()[index] = v; break;
        case DType::Int8:     a.data<std::int8_t>()[index] = static_cast<std::int8_t>(v); break;
        case DType::Int16:    a.data<std::int16_t>()[index] = static_cast<std::int16_t>(v); break;
        case DType::Int32:    a.data<std::int32_t>()[index] = static_cast<std::int32_t>(v); break;
        case DType::Int64:    a.data<std::int64_t>()[index] = static_cast<std::int64_t>(v); break;
        case DType::UInt8:    a.data<std::uint8_t>()[index] = static_cast<std::uint8_t>(v); break;
        case DType::UInt16:   a.data<std::uint16_t>()[index] = static_cast<std::uint16_t>(v); break;
        case DType::UInt32:   a.data<std::uint32_t>()[index] = static_cast<std::uint32_t>(v); break;
        case DType::UInt64:   a.data<std::uint64_t>()[index] = static_cast<std::uint64_t>(v); break;
        case DType::Bool:     a.data<bool>()[index] = (v != 0.0); break;
        default:
            throw InvalidArgumentError("set_double: unsupported dtype");
    }
}

[[nodiscard]] inline bool is_nan_value(double v) {
    return std::isnan(v);
}

/// Require matrix shape [N, F]; promote rank-1 [N] to [N, 1] conceptually.
inline void require_matrix(const NDArray& X, std::size_t& n, std::size_t& f) {
    if (X.shape().size() == 1) {
        n = X.shape()[0];
        f = 1;
    } else if (X.shape().size() == 2) {
        n = X.shape()[0];
        f = X.shape()[1];
    } else {
        throw ShapeError("expected rank-1 or rank-2 array, got " + shape_to_string(X.shape()));
    }
}

/// Welford online mean/variance accumulator (population or sample).
struct Welford {
    std::size_t n = 0;
    double mean = 0.0;
    double m2 = 0.0;

    void update(double x) {
        if (is_nan_value(x)) {
            return;
        }
        ++n;
        const double delta = x - mean;
        mean += delta / static_cast<double>(n);
        const double delta2 = x - mean;
        m2 += delta * delta2;
    }

    [[nodiscard]] double variance(bool sample = false) const {
        if (n == 0) {
            return 0.0;
        }
        if (sample) {
            return n > 1 ? m2 / static_cast<double>(n - 1) : 0.0;
        }
        return m2 / static_cast<double>(n);
    }

    [[nodiscard]] double stddev(bool sample = false) const {
        return std::sqrt(variance(sample));
    }
};

} // namespace detail
} // namespace nexusdata
