#include "nexusdata/simd/numeric_ops.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <string>
#include <type_traits>

#include "nexusdata/backend/cpu_dispatch.hpp"
#include "nexusdata/core/error.hpp"

namespace nexusdata {
namespace simd {

namespace {

// --- scalar reference kernels -------------------------------------------------

template <typename T>
inline T apply_binary(T v, BinaryOp op, double a) noexcept {
    const double d = static_cast<double>(v);
    switch (op) {
        case BinaryOp::Add: return static_cast<T>(d + a);
        case BinaryOp::Sub: return static_cast<T>(d - a);
        case BinaryOp::Mul: return static_cast<T>(d * a);
        case BinaryOp::Div: return static_cast<T>(d / a);
    }
    return v;
}

template <typename T, BinaryOp OP>
void scalar_op_loop(T* x, std::size_t n, double a) noexcept {
    for (std::size_t i = 0; i < n; ++i) {
        const double d = static_cast<double>(x[i]);
        if constexpr (OP == BinaryOp::Add) {
            x[i] = static_cast<T>(d + a);
        } else if constexpr (OP == BinaryOp::Sub) {
            x[i] = static_cast<T>(d - a);
        } else if constexpr (OP == BinaryOp::Mul) {
            x[i] = static_cast<T>(d * a);
        } else {
            x[i] = static_cast<T>(d / a);
        }
    }
}

template <typename T>
void scalar_op_ref(T* x, std::size_t n, BinaryOp op, double a) noexcept {
    switch (op) {
        case BinaryOp::Add: scalar_op_loop<T, BinaryOp::Add>(x, n, a); break;
        case BinaryOp::Sub: scalar_op_loop<T, BinaryOp::Sub>(x, n, a); break;
        case BinaryOp::Mul: scalar_op_loop<T, BinaryOp::Mul>(x, n, a); break;
        case BinaryOp::Div: scalar_op_loop<T, BinaryOp::Div>(x, n, a); break;
    }
}

template <typename T>
void clip_ref(T* x, std::size_t n, double lo, double hi) noexcept {
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = static_cast<T>(std::clamp(static_cast<double>(x[i]), lo, hi));
    }
}

template <typename T>
void standardize_ref(T* x, std::size_t rows, std::size_t cols, const double* mean,
                     const double* scale) noexcept {
    for (std::size_t r = 0; r < rows; ++r) {
        T* row = x + r * cols;
        for (std::size_t j = 0; j < cols; ++j) {
            row[j] = static_cast<T>((static_cast<double>(row[j]) - mean[j]) / scale[j]);
        }
    }
}

template <typename T>
void minmax_ref(T* x, std::size_t rows, std::size_t cols, const double* sub,
                const double* div, double mul, double add) noexcept {
    for (std::size_t r = 0; r < rows; ++r) {
        T* row = x + r * cols;
        for (std::size_t j = 0; j < cols; ++j) {
            row[j] = static_cast<T>(add + mul * ((static_cast<double>(row[j]) - sub[j]) / div[j]));
        }
    }
}

void round_to_f32_ref(double* x, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = static_cast<double>(static_cast<float>(x[i]));
    }
}

void normalize_planes_ref(float* p, std::size_t planes, std::size_t plane_size,
                          const float* mean, const float* stddev) noexcept {
    for (std::size_t ch = 0; ch < planes; ++ch) {
        float* q = p + ch * plane_size;
        const float m = mean[ch];
        const float s = stddev[ch];
        for (std::size_t i = 0; i < plane_size; ++i) {
            q[i] = (q[i] - m) / s;
        }
    }
}

const detail::KernelTable kScalarTable{
    &scalar_op_ref<float>,    &scalar_op_ref<double>,  &clip_ref<float>,
    &clip_ref<double>,        &standardize_ref<float>, &standardize_ref<double>,
    &minmax_ref<float>,       &minmax_ref<double>,     &round_to_f32_ref,
    &normalize_planes_ref,
};

// --- dispatch -----------------------------------------------------------------

constexpr int kNoOverride = -1;
std::atomic<int> g_override{kNoOverride};

Isa clamp_isa(Isa requested) noexcept {
    const Isa best = detected_isa();
    switch (requested) {
        case Isa::Scalar:
            return Isa::Scalar;
        case Isa::NEON:
            return best == Isa::NEON ? Isa::NEON : Isa::Scalar;
        case Isa::AVX2:
            return (best == Isa::AVX2 || best == Isa::AVX512) ? Isa::AVX2 : Isa::Scalar;
        case Isa::AVX512:
            if (best == Isa::AVX512) return Isa::AVX512;
            return best == Isa::AVX2 ? Isa::AVX2 : Isa::Scalar;
    }
    return Isa::Scalar;
}

const detail::KernelTable& table() noexcept {
    const Isa isa = active_isa();
    if (isa == Isa::AVX2 || isa == Isa::AVX512) {
        if (const auto* t = detail::avx2_kernels()) {
            return *t;
        }
    }
    return kScalarTable;
}

// --- dtype visitation ---------------------------------------------------------

template <typename F>
void visit_real_dtype(DType dt, const char* what, F&& f) {
    switch (dt) {
        case DType::Bool:    f(std::type_identity<bool>{}); return;
        case DType::Int8:    f(std::type_identity<std::int8_t>{}); return;
        case DType::Int16:   f(std::type_identity<std::int16_t>{}); return;
        case DType::Int32:   f(std::type_identity<std::int32_t>{}); return;
        case DType::Int64:   f(std::type_identity<std::int64_t>{}); return;
        case DType::UInt8:   f(std::type_identity<std::uint8_t>{}); return;
        case DType::UInt16:  f(std::type_identity<std::uint16_t>{}); return;
        case DType::UInt32:  f(std::type_identity<std::uint32_t>{}); return;
        case DType::UInt64:  f(std::type_identity<std::uint64_t>{}); return;
        case DType::Float32: f(std::type_identity<float>{}); return;
        case DType::Float64: f(std::type_identity<double>{}); return;
        case DType::Float16:
        case DType::BFloat16:
            break;
    }
    throw InvalidArgumentError(std::string(what) + ": unsupported dtype " +
                               std::string(to_string(dt)));
}

void require_host(const NDArray& x, const char* what) {
    if (!x.device().is_host()) {
        throw InvalidArgumentError(std::string(what) + ": host arrays only (got " +
                                   x.device().to_string() + ")");
    }
}

template <typename T>
T* typed(NDArray& x) {
    return static_cast<T*>(x.data());
}

std::size_t require_period(const NDArray& x, std::size_t cols, const char* what) {
    if (cols == 0) {
        throw InvalidArgumentError(std::string(what) + ": empty parameter vector");
    }
    if (x.numel() % cols != 0) {
        throw ShapeError(std::string(what) + ": numel " + std::to_string(x.numel()) +
                         " is not a multiple of feature count " + std::to_string(cols));
    }
    return x.numel() / cols;
}

} // namespace

// --- public API ---------------------------------------------------------------

const char* isa_name(Isa isa) noexcept {
    switch (isa) {
        case Isa::Scalar: return "scalar";
        case Isa::AVX2:   return "avx2";
        case Isa::AVX512: return "avx512";
        case Isa::NEON:   return "neon";
    }
    return "unknown";
}

Isa detected_isa() noexcept {
    static const Isa cached = [] {
        const auto& f = cpu_features();
        if (f.avx2 && detail::avx2_kernels() != nullptr) {
            return cpu_has_avx512() ? Isa::AVX512 : Isa::AVX2;
        }
        if (f.neon) {
            return Isa::NEON;
        }
        return Isa::Scalar;
    }();
    return cached;
}

Isa active_isa() noexcept {
    const int o = g_override.load(std::memory_order_relaxed);
    if (o == kNoOverride) {
        return detected_isa();
    }
    return static_cast<Isa>(o);
}

void set_isa_override(Isa isa) noexcept {
    g_override.store(static_cast<int>(clamp_isa(isa)), std::memory_order_relaxed);
}

void clear_isa_override() noexcept {
    g_override.store(kNoOverride, std::memory_order_relaxed);
}

namespace detail {
const KernelTable& scalar_kernels() noexcept { return kScalarTable; }
const KernelTable& active_kernels() noexcept { return table(); }
} // namespace detail

void scalar_op_f32(float* x, std::size_t n, BinaryOp op, double a) noexcept {
    table().scalar_op_f32(x, n, op, a);
}
void scalar_op_f64(double* x, std::size_t n, BinaryOp op, double a) noexcept {
    table().scalar_op_f64(x, n, op, a);
}
void clip_f32(float* x, std::size_t n, double lo, double hi) noexcept {
    table().clip_f32(x, n, lo, hi);
}
void clip_f64(double* x, std::size_t n, double lo, double hi) noexcept {
    table().clip_f64(x, n, lo, hi);
}
void log1p_f32(float* x, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = static_cast<float>(std::log1p(static_cast<double>(x[i])));
    }
}
void log1p_f64(double* x, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = std::log1p(x[i]);
    }
}
void standardize_f32(float* x, std::size_t rows, std::size_t cols, const double* mean,
                     const double* scale) noexcept {
    table().standardize_f32(x, rows, cols, mean, scale);
}
void standardize_f64(double* x, std::size_t rows, std::size_t cols, const double* mean,
                     const double* scale) noexcept {
    table().standardize_f64(x, rows, cols, mean, scale);
}
void minmax_f32(float* x, std::size_t rows, std::size_t cols, const double* sub,
                const double* div, double mul, double add) noexcept {
    table().minmax_f32(x, rows, cols, sub, div, mul, add);
}
void minmax_f64(double* x, std::size_t rows, std::size_t cols, const double* sub,
                const double* div, double mul, double add) noexcept {
    table().minmax_f64(x, rows, cols, sub, div, mul, add);
}
void round_to_f32(double* x, std::size_t n) noexcept {
    table().round_to_f32(x, n);
}

void convert(const void* src, DType src_dtype, void* dst, DType dst_dtype, std::size_t n) {
    if (n == 0) {
        return;
    }
    if (src_dtype == dst_dtype) {
        std::memcpy(dst, src, n * size_of(src_dtype));
        return;
    }
    visit_real_dtype(src_dtype, "simd::convert", [&](auto src_tag) {
        using S = typename decltype(src_tag)::type;
        visit_real_dtype(dst_dtype, "simd::convert", [&](auto dst_tag) {
            using D = typename decltype(dst_tag)::type;
            const S* s = static_cast<const S*>(src);
            D* d = static_cast<D*>(dst);
            for (std::size_t i = 0; i < n; ++i) {
                if constexpr (std::is_same_v<D, bool>) {
                    d[i] = static_cast<double>(s[i]) != 0.0;
                } else {
                    d[i] = static_cast<D>(static_cast<double>(s[i]));
                }
            }
        });
    });
}

void apply_scalar_op(NDArray& x, BinaryOp op, double a) {
    require_host(x, "simd::apply_scalar_op");
    if (x.numel() == 0) return;
    visit_real_dtype(x.dtype(), "simd::apply_scalar_op", [&](auto tag) {
        using T = typename decltype(tag)::type;
        if constexpr (std::is_same_v<T, float>) {
            scalar_op_f32(typed<float>(x), x.numel(), op, a);
        } else if constexpr (std::is_same_v<T, double>) {
            scalar_op_f64(typed<double>(x), x.numel(), op, a);
        } else {
            T* p = typed<T>(x);
            for (std::size_t i = 0; i < x.numel(); ++i) {
                p[i] = apply_binary(p[i], op, a);
            }
        }
    });
}

void clip_inplace(NDArray& x, double lo, double hi) {
    require_host(x, "simd::clip_inplace");
    if (x.numel() == 0) return;
    visit_real_dtype(x.dtype(), "simd::clip_inplace", [&](auto tag) {
        using T = typename decltype(tag)::type;
        if constexpr (std::is_same_v<T, float>) {
            clip_f32(typed<float>(x), x.numel(), lo, hi);
        } else if constexpr (std::is_same_v<T, double>) {
            clip_f64(typed<double>(x), x.numel(), lo, hi);
        } else {
            clip_ref(typed<T>(x), x.numel(), lo, hi);
        }
    });
}

void log1p_inplace(NDArray& x) {
    require_host(x, "simd::log1p_inplace");
    if (x.numel() == 0) return;
    visit_real_dtype(x.dtype(), "simd::log1p_inplace", [&](auto tag) {
        using T = typename decltype(tag)::type;
        if constexpr (std::is_same_v<T, float>) {
            log1p_f32(typed<float>(x), x.numel());
        } else if constexpr (std::is_same_v<T, double>) {
            log1p_f64(typed<double>(x), x.numel());
        } else {
            T* p = typed<T>(x);
            for (std::size_t i = 0; i < x.numel(); ++i) {
                p[i] = static_cast<T>(std::log1p(static_cast<double>(p[i])));
            }
        }
    });
}

void standardize_inplace(NDArray& x, const std::vector<double>& mean,
                         const std::vector<double>& scale) {
    require_host(x, "simd::standardize_inplace");
    if (mean.size() != scale.size()) {
        throw InvalidArgumentError("simd::standardize_inplace: mean/scale size mismatch");
    }
    const std::size_t cols = mean.size();
    const std::size_t rows = require_period(x, cols, "simd::standardize_inplace");
    if (rows == 0) return;
    visit_real_dtype(x.dtype(), "simd::standardize_inplace", [&](auto tag) {
        using T = typename decltype(tag)::type;
        if constexpr (std::is_same_v<T, float>) {
            standardize_f32(typed<float>(x), rows, cols, mean.data(), scale.data());
        } else if constexpr (std::is_same_v<T, double>) {
            standardize_f64(typed<double>(x), rows, cols, mean.data(), scale.data());
        } else {
            standardize_ref(typed<T>(x), rows, cols, mean.data(), scale.data());
        }
    });
}

void minmax_inplace(NDArray& x, const std::vector<double>& data_min,
                    const std::vector<double>& data_max, double feature_min,
                    double feature_max) {
    require_host(x, "simd::minmax_inplace");
    if (data_min.size() != data_max.size()) {
        throw InvalidArgumentError("simd::minmax_inplace: data_min/data_max size mismatch");
    }
    const std::size_t cols = data_min.size();
    const std::size_t rows = require_period(x, cols, "simd::minmax_inplace");
    if (rows == 0) return;
    std::vector<double> range(cols);
    for (std::size_t j = 0; j < cols; ++j) {
        const double d = data_max[j] - data_min[j];
        range[j] = d == 0.0 ? 1.0 : d;
    }
    const double span = feature_max - feature_min;
    visit_real_dtype(x.dtype(), "simd::minmax_inplace", [&](auto tag) {
        using T = typename decltype(tag)::type;
        if constexpr (std::is_same_v<T, float>) {
            minmax_f32(typed<float>(x), rows, cols, data_min.data(), range.data(), span,
                       feature_min);
        } else if constexpr (std::is_same_v<T, double>) {
            minmax_f64(typed<double>(x), rows, cols, data_min.data(), range.data(), span,
                       feature_min);
        } else {
            minmax_ref(typed<T>(x), rows, cols, data_min.data(), range.data(), span,
                       feature_min);
        }
    });
}

NDArray cast(const NDArray& x, DType to) {
    require_host(x, "simd::cast");
    NDArray out(x.shape(), to);
    convert(x.data(), x.dtype(), out.data(), to, x.numel());
    return out;
}

} // namespace simd
} // namespace nexusdata
