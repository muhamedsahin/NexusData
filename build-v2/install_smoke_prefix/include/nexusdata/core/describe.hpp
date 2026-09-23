#pragma once

/// Field binding for a user struct. Arithmetic members become extra_tensors
/// (scalar int64 or float64). Strings become metadata. Up to 8 fields.
///
///   struct Rec { int a; std::string c; };
///   NEXUSDATA_DESCRIBE_STRUCT(Rec, a, c)
///   Sample s = to_sample(rec);

#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>

#include "nexusdata/core/ndarray.hpp"
#include "nexusdata/dataset/sample.hpp"

namespace nexusdata {
namespace detail {

inline void bind_field(Sample& s, const char* name, const std::string& value) {
    s.metadata.set(name, value);
}
inline void bind_field(Sample& s, const char* name, const char* value) {
    s.metadata.set(name, value ? value : "");
}

template <class T, typename std::enable_if<std::is_arithmetic<T>::value, int>::type = 0>
void bind_field(Sample& s, const char* name, T value) {
    if constexpr (std::is_floating_point<T>::value) {
        NDArray a(Shape{1}, DType::Float64);
        a.data<double>()[0] = static_cast<double>(value);
        s.extra_tensors.set(name, std::move(a));
    } else {
        NDArray a(Shape{1}, DType::Int64);
        a.data<std::int64_t>()[0] = static_cast<std::int64_t>(value);
        s.extra_tensors.set(name, std::move(a));
    }
}

} // namespace detail
} // namespace nexusdata

#define NEXUSDATA_CAT_(a, b) a##b
#define NEXUSDATA_CAT(a, b) NEXUSDATA_CAT_(a, b)
#define NEXUSDATA_NARGS_(a1, a2, a3, a4, a5, a6, a7, a8, N, ...) N
#define NEXUSDATA_NARGS(...) NEXUSDATA_NARGS_(__VA_ARGS__, 8, 7, 6, 5, 4, 3, 2, 1)

#define NEXUSDATA_BIND(obj, sample, field) \
    ::nexusdata::detail::bind_field((sample), #field, (obj).field);

#define NEXUSDATA_FE1(o, s, a) NEXUSDATA_BIND(o, s, a)
#define NEXUSDATA_FE2(o, s, a, b) NEXUSDATA_BIND(o, s, a) NEXUSDATA_FE1(o, s, b)
#define NEXUSDATA_FE3(o, s, a, b, c) NEXUSDATA_BIND(o, s, a) NEXUSDATA_FE2(o, s, b, c)
#define NEXUSDATA_FE4(o, s, a, b, c, d) NEXUSDATA_BIND(o, s, a) NEXUSDATA_FE3(o, s, b, c, d)
#define NEXUSDATA_FE5(o, s, a, b, c, d, e) NEXUSDATA_BIND(o, s, a) NEXUSDATA_FE4(o, s, b, c, d, e)
#define NEXUSDATA_FE6(o, s, a, b, c, d, e, f) NEXUSDATA_BIND(o, s, a) NEXUSDATA_FE5(o, s, b, c, d, e, f)
#define NEXUSDATA_FE7(o, s, a, b, c, d, e, f, g) \
    NEXUSDATA_BIND(o, s, a) NEXUSDATA_FE6(o, s, b, c, d, e, f, g)
#define NEXUSDATA_FE8(o, s, a, b, c, d, e, f, g, h) \
    NEXUSDATA_BIND(o, s, a) NEXUSDATA_FE7(o, s, b, c, d, e, f, g, h)

#define NEXUSDATA_DESCRIBE_STRUCT(Type, ...)                                                        \
    inline ::nexusdata::Sample to_sample(const Type& nd_obj) {                                     \
        ::nexusdata::Sample nd_sample;                                                              \
        NEXUSDATA_CAT(NEXUSDATA_FE, NEXUSDATA_NARGS(__VA_ARGS__))(nd_obj, nd_sample, __VA_ARGS__) \
        return nd_sample;                                                                           \
    }
