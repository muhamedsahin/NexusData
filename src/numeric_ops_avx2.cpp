// Compiled with -mavx2 (/arch:AVX2 on MSVC) but deliberately without FMA:
// fused multiply-add would change rounding and break bit-exactness with the
// scalar reference kernels.
#include "nexusdata/simd/numeric_ops.hpp"

#include <algorithm>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace nexusdata {
namespace simd {
namespace detail {

#if defined(__AVX2__)

namespace {

struct F32x8 {
    __m256d lo;
    __m256d hi;
};

inline F32x8 load_f32x8(const float* p) {
    const __m256 v = _mm256_loadu_ps(p);
    return {_mm256_cvtps_pd(_mm256_castps256_ps128(v)),
            _mm256_cvtps_pd(_mm256_extractf128_ps(v, 1))};
}

inline void store_f32x8(float* p, const F32x8& v) {
    _mm256_storeu_ps(p, _mm256_set_m128(_mm256_cvtpd_ps(v.hi), _mm256_cvtpd_ps(v.lo)));
}

template <BinaryOp OP>
inline __m256d op_pd(__m256d v, __m256d a) {
    if constexpr (OP == BinaryOp::Add) return _mm256_add_pd(v, a);
    else if constexpr (OP == BinaryOp::Sub) return _mm256_sub_pd(v, a);
    else if constexpr (OP == BinaryOp::Mul) return _mm256_mul_pd(v, a);
    else return _mm256_div_pd(v, a);
}

template <BinaryOp OP>
inline double op_sd(double v, double a) {
    if constexpr (OP == BinaryOp::Add) return v + a;
    else if constexpr (OP == BinaryOp::Sub) return v - a;
    else if constexpr (OP == BinaryOp::Mul) return v * a;
    else return v / a;
}

template <BinaryOp OP>
void scalar_op_f32_t(float* x, std::size_t n, double a) noexcept {
    const __m256d va = _mm256_set1_pd(a);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        F32x8 v = load_f32x8(x + i);
        v.lo = op_pd<OP>(v.lo, va);
        v.hi = op_pd<OP>(v.hi, va);
        store_f32x8(x + i, v);
    }
    for (; i < n; ++i) {
        x[i] = static_cast<float>(op_sd<OP>(static_cast<double>(x[i]), a));
    }
}

template <BinaryOp OP>
void scalar_op_f64_t(double* x, std::size_t n, double a) noexcept {
    const __m256d va = _mm256_set1_pd(a);
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        _mm256_storeu_pd(x + i, op_pd<OP>(_mm256_loadu_pd(x + i), va));
    }
    for (; i < n; ++i) {
        x[i] = op_sd<OP>(x[i], a);
    }
}

void scalar_op_f32(float* x, std::size_t n, BinaryOp op, double a) noexcept {
    switch (op) {
        case BinaryOp::Add: scalar_op_f32_t<BinaryOp::Add>(x, n, a); break;
        case BinaryOp::Sub: scalar_op_f32_t<BinaryOp::Sub>(x, n, a); break;
        case BinaryOp::Mul: scalar_op_f32_t<BinaryOp::Mul>(x, n, a); break;
        case BinaryOp::Div: scalar_op_f32_t<BinaryOp::Div>(x, n, a); break;
    }
}

void scalar_op_f64(double* x, std::size_t n, BinaryOp op, double a) noexcept {
    switch (op) {
        case BinaryOp::Add: scalar_op_f64_t<BinaryOp::Add>(x, n, a); break;
        case BinaryOp::Sub: scalar_op_f64_t<BinaryOp::Sub>(x, n, a); break;
        case BinaryOp::Mul: scalar_op_f64_t<BinaryOp::Mul>(x, n, a); break;
        case BinaryOp::Div: scalar_op_f64_t<BinaryOp::Div>(x, n, a); break;
    }
}

// Clamping in float with float-rounded bounds equals clamping in double then
// rounding (round-to-nearest is monotone). Operand order keeps NaN inputs.
void clip_f32(float* x, std::size_t n, double lo, double hi) noexcept {
    const __m256 vlo = _mm256_set1_ps(static_cast<float>(lo));
    const __m256 vhi = _mm256_set1_ps(static_cast<float>(hi));
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 v = _mm256_loadu_ps(x + i);
        _mm256_storeu_ps(x + i, _mm256_min_ps(vhi, _mm256_max_ps(vlo, v)));
    }
    for (; i < n; ++i) {
        x[i] = static_cast<float>(std::clamp(static_cast<double>(x[i]), lo, hi));
    }
}

void clip_f64(double* x, std::size_t n, double lo, double hi) noexcept {
    const __m256d vlo = _mm256_set1_pd(lo);
    const __m256d vhi = _mm256_set1_pd(hi);
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const __m256d v = _mm256_loadu_pd(x + i);
        _mm256_storeu_pd(x + i, _mm256_min_pd(vhi, _mm256_max_pd(vlo, v)));
    }
    for (; i < n; ++i) {
        x[i] = std::clamp(x[i], lo, hi);
    }
}

/// Run @p row_fn over a row-major [rows, cols] buffer. Narrow rows are grouped
/// into wider "super rows" with repeated parameter vectors so the vector loop
/// still does useful work when cols < 8.
template <typename T, typename RowFn>
void for_each_row_block(T* x, std::size_t rows, std::size_t cols, const double* p0,
                        const double* p1, RowFn&& row_fn) {
    constexpr std::size_t kMinWidth = 32;
    if (cols >= kMinWidth || rows < 4) {
        for (std::size_t r = 0; r < rows; ++r) {
            row_fn(x + r * cols, cols, p0, p1);
        }
        return;
    }
    // cols < kMinWidth  =>  width < 2 * kMinWidth.
    const std::size_t rep = (kMinWidth + cols - 1) / cols;
    const std::size_t width = rep * cols;
    double e0[2 * kMinWidth];
    double e1[2 * kMinWidth];
    for (std::size_t k = 0; k < rep; ++k) {
        std::copy(p0, p0 + cols, e0 + k * cols);
        std::copy(p1, p1 + cols, e1 + k * cols);
    }
    const std::size_t blocks = rows / rep;
    for (std::size_t b = 0; b < blocks; ++b) {
        row_fn(x + b * width, width, e0, e1);
    }
    for (std::size_t r = blocks * rep; r < rows; ++r) {
        row_fn(x + r * cols, cols, p0, p1);
    }
}

void standardize_row_f32(float* x, std::size_t n, const double* mean, const double* scale) {
    std::size_t j = 0;
    for (; j + 8 <= n; j += 8) {
        F32x8 v = load_f32x8(x + j);
        v.lo = _mm256_div_pd(_mm256_sub_pd(v.lo, _mm256_loadu_pd(mean + j)),
                             _mm256_loadu_pd(scale + j));
        v.hi = _mm256_div_pd(_mm256_sub_pd(v.hi, _mm256_loadu_pd(mean + j + 4)),
                             _mm256_loadu_pd(scale + j + 4));
        store_f32x8(x + j, v);
    }
    for (; j < n; ++j) {
        x[j] = static_cast<float>((static_cast<double>(x[j]) - mean[j]) / scale[j]);
    }
}

void standardize_row_f64(double* x, std::size_t n, const double* mean, const double* scale) {
    std::size_t j = 0;
    for (; j + 4 <= n; j += 4) {
        const __m256d v = _mm256_loadu_pd(x + j);
        _mm256_storeu_pd(x + j, _mm256_div_pd(_mm256_sub_pd(v, _mm256_loadu_pd(mean + j)),
                                              _mm256_loadu_pd(scale + j)));
    }
    for (; j < n; ++j) {
        x[j] = (x[j] - mean[j]) / scale[j];
    }
}

void standardize_f32(float* x, std::size_t rows, std::size_t cols, const double* mean,
                     const double* scale) noexcept {
    for_each_row_block(x, rows, cols, mean, scale, standardize_row_f32);
}

void standardize_f64(double* x, std::size_t rows, std::size_t cols, const double* mean,
                     const double* scale) noexcept {
    for_each_row_block(x, rows, cols, mean, scale, standardize_row_f64);
}

void minmax_f32(float* x, std::size_t rows, std::size_t cols, const double* sub,
                const double* div, double mul, double add) noexcept {
    const __m256d vm = _mm256_set1_pd(mul);
    const __m256d va = _mm256_set1_pd(add);
    for_each_row_block(x, rows, cols, sub, div,
                       [&](float* row, std::size_t n, const double* s, const double* d) {
        std::size_t j = 0;
        for (; j + 8 <= n; j += 8) {
            F32x8 v = load_f32x8(row + j);
            v.lo = _mm256_add_pd(va, _mm256_mul_pd(vm, _mm256_div_pd(
                _mm256_sub_pd(v.lo, _mm256_loadu_pd(s + j)), _mm256_loadu_pd(d + j))));
            v.hi = _mm256_add_pd(va, _mm256_mul_pd(vm, _mm256_div_pd(
                _mm256_sub_pd(v.hi, _mm256_loadu_pd(s + j + 4)), _mm256_loadu_pd(d + j + 4))));
            store_f32x8(row + j, v);
        }
        for (; j < n; ++j) {
            row[j] = static_cast<float>(add + mul * ((static_cast<double>(row[j]) - s[j]) / d[j]));
        }
    });
}

void minmax_f64(double* x, std::size_t rows, std::size_t cols, const double* sub,
                const double* div, double mul, double add) noexcept {
    const __m256d vm = _mm256_set1_pd(mul);
    const __m256d va = _mm256_set1_pd(add);
    for_each_row_block(x, rows, cols, sub, div,
                       [&](double* row, std::size_t n, const double* s, const double* d) {
        std::size_t j = 0;
        for (; j + 4 <= n; j += 4) {
            const __m256d v = _mm256_loadu_pd(row + j);
            _mm256_storeu_pd(row + j, _mm256_add_pd(va, _mm256_mul_pd(vm, _mm256_div_pd(
                _mm256_sub_pd(v, _mm256_loadu_pd(s + j)), _mm256_loadu_pd(d + j)))));
        }
        for (; j < n; ++j) {
            row[j] = add + mul * ((row[j] - s[j]) / d[j]);
        }
    });
}

void round_to_f32(double* x, std::size_t n) noexcept {
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        _mm256_storeu_pd(x + i, _mm256_cvtps_pd(_mm256_cvtpd_ps(_mm256_loadu_pd(x + i))));
    }
    for (; i < n; ++i) {
        x[i] = static_cast<double>(static_cast<float>(x[i]));
    }
}

void normalize_planes_f32(float* p, std::size_t planes, std::size_t plane_size,
                          const float* mean, const float* stddev) noexcept {
    for (std::size_t ch = 0; ch < planes; ++ch) {
        float* q = p + ch * plane_size;
        const float m = mean[ch];
        const float s = stddev[ch];
        const __m256 vm = _mm256_set1_ps(m);
        const __m256 vs = _mm256_set1_ps(s);
        std::size_t i = 0;
        for (; i + 8 <= plane_size; i += 8) {
            _mm256_storeu_ps(q + i, _mm256_div_ps(_mm256_sub_ps(_mm256_loadu_ps(q + i), vm), vs));
        }
        for (; i < plane_size; ++i) {
            q[i] = (q[i] - m) / s;
        }
    }
}

const KernelTable kAvx2Table{
    &scalar_op_f32,   &scalar_op_f64,   &clip_f32,   &clip_f64,     &standardize_f32,
    &standardize_f64, &minmax_f32,      &minmax_f64, &round_to_f32, &normalize_planes_f32,
};

} // namespace

const KernelTable* avx2_kernels() noexcept { return &kAvx2Table; }

#else

const KernelTable* avx2_kernels() noexcept { return nullptr; }

#endif

} // namespace detail
} // namespace simd
} // namespace nexusdata
