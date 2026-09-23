#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "nexusdata/core/error.hpp"

namespace nexusdata {

/// Supported scalar element types for NDArray.
/// Thread-safety: enum values are immutable.
enum class DType : uint8_t {
    Bool = 0,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float16,   // stored as 2 bytes; no arithmetic in v0.1
    BFloat16,  // stored as 2 bytes; no arithmetic in v0.1
    Float32,
    Float64,
};

/// Byte size of one element of @p dtype.
[[nodiscard]] inline constexpr std::size_t size_of(DType dtype) {
    switch (dtype) {
        case DType::Bool:
        case DType::Int8:
        case DType::UInt8:
            return 1;
        case DType::Int16:
        case DType::UInt16:
        case DType::Float16:
        case DType::BFloat16:
            return 2;
        case DType::Int32:
        case DType::UInt32:
        case DType::Float32:
            return 4;
        case DType::Int64:
        case DType::UInt64:
        case DType::Float64:
            return 8;
    }
    // Unreachable for valid enum values; keep a path for corrupted casts.
    throw InvalidArgumentError("unknown DType value");
}

[[nodiscard]] inline constexpr std::string_view to_string(DType dtype) {
    switch (dtype) {
        case DType::Bool:     return "bool";
        case DType::Int8:     return "int8";
        case DType::Int16:    return "int16";
        case DType::Int32:    return "int32";
        case DType::Int64:    return "int64";
        case DType::UInt8:    return "uint8";
        case DType::UInt16:   return "uint16";
        case DType::UInt32:   return "uint32";
        case DType::UInt64:   return "uint64";
        case DType::Float16:  return "float16";
        case DType::BFloat16: return "bfloat16";
        case DType::Float32:  return "float32";
        case DType::Float64:  return "float64";
    }
    return "unknown";
}

/// Inverse of to_string() (v2.0; used by config and schema readers). Throws
/// InvalidArgumentError for unknown names.
[[nodiscard]] inline DType dtype_from_string(std::string_view name) {
    for (int i = 0; i <= static_cast<int>(DType::Float64); ++i) {
        const auto d = static_cast<DType>(i);
        if (to_string(d) == name) {
            return d;
        }
    }
    throw InvalidArgumentError("unknown dtype name \"" + std::string(name) + "\"");
}

/// Maps a C++ type to DType (compile-time).
template <typename T>
struct dtype_traits;

template <> struct dtype_traits<bool>     { static constexpr DType value = DType::Bool; };
template <> struct dtype_traits<int8_t>   { static constexpr DType value = DType::Int8; };
template <> struct dtype_traits<int16_t>  { static constexpr DType value = DType::Int16; };
template <> struct dtype_traits<int32_t>  { static constexpr DType value = DType::Int32; };
template <> struct dtype_traits<int64_t>  { static constexpr DType value = DType::Int64; };
template <> struct dtype_traits<uint8_t>  { static constexpr DType value = DType::UInt8; };
template <> struct dtype_traits<uint16_t> { static constexpr DType value = DType::UInt16; };
template <> struct dtype_traits<uint32_t> { static constexpr DType value = DType::UInt32; };
template <> struct dtype_traits<uint64_t> { static constexpr DType value = DType::UInt64; };
template <> struct dtype_traits<float>    { static constexpr DType value = DType::Float32; };
template <> struct dtype_traits<double>   { static constexpr DType value = DType::Float64; };

template <typename T>
inline constexpr DType dtype_of = dtype_traits<T>::value;

} // namespace nexusdata
