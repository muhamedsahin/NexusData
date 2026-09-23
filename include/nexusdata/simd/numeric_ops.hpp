#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "nexusdata/core/dtype.hpp"
#include "nexusdata/core/ndarray.hpp"

namespace nexusdata {
namespace simd {

/// Instruction set used by the runtime dispatcher.
enum class Isa : std::uint8_t {
    Scalar = 0,
    AVX2,
    /// Detected, currently served by the AVX2 kernels.
    AVX512,
    NEON,
};

[[nodiscard]] const char* isa_name(Isa isa) noexcept;

/// Best ISA supported by this CPU and this build.
[[nodiscard]] Isa detected_isa() noexcept;

/// ISA the kernels dispatch to right now (detected unless overridden).
[[nodiscard]] Isa active_isa() noexcept;

/// Force dispatch to @p isa (clamped to what the CPU supports). For tests/benchmarks.
void set_isa_override(Isa isa) noexcept;
void clear_isa_override() noexcept;

enum class BinaryOp : std::uint8_t { Add, Sub, Mul, Div };

// --- Raw contiguous kernels (in-place, host memory) ---------------------------
//
// Precision contract: every kernel computes in double and rounds once to the
// storage type, exactly like `set_double(a, i, f(as_double(a, i)))`. No FMA
// contraction is used, so results are bit-identical to the scalar reference.

/// x[i] = x[i] (op) a
void scalar_op_f32(float* x, std::size_t n, BinaryOp op, double a) noexcept;
void scalar_op_f64(double* x, std::size_t n, BinaryOp op, double a) noexcept;

/// x[i] = clamp(x[i], lo, hi); NaN inputs are preserved.
void clip_f32(float* x, std::size_t n, double lo, double hi) noexcept;
void clip_f64(double* x, std::size_t n, double lo, double hi) noexcept;

/// x[i] = log1p(x[i]) (scalar libm per element; no approximation).
void log1p_f32(float* x, std::size_t n) noexcept;
void log1p_f64(double* x, std::size_t n) noexcept;

/// Row-major [rows, cols]: x[r, j] = (x[r, j] - mean[j]) / scale[j]
void standardize_f32(float* x, std::size_t rows, std::size_t cols,
                     const double* mean, const double* scale) noexcept;
void standardize_f64(double* x, std::size_t rows, std::size_t cols,
                     const double* mean, const double* scale) noexcept;

/// Row-major [rows, cols]: x[r, j] = add + mul * ((x[r, j] - sub[j]) / div[j])
/// (MinMax scaling with add = feature_min, mul = span, sub = data_min, div = range).
void minmax_f32(float* x, std::size_t rows, std::size_t cols, const double* sub,
                const double* div, double mul, double add) noexcept;
void minmax_f64(double* x, std::size_t rows, std::size_t cols, const double* sub,
                const double* div, double mul, double add) noexcept;

/// x[i] = double(float(x[i])) — rounds a double buffer to float32 precision.
void round_to_f32(double* x, std::size_t n) noexcept;

/// Convert @p n elements between dtypes (static_cast semantics; Bool reads as 0/1
/// and writes as v != 0). Float16/BFloat16 are unsupported and throw.
void convert(const void* src, DType src_dtype, void* dst, DType dst_dtype, std::size_t n);

// --- NDArray-level operations (any real dtype, host, in-place) ---------------

/// Throws InvalidArgumentError for non-host arrays or Float16/BFloat16.
void apply_scalar_op(NDArray& x, BinaryOp op, double a);
void clip_inplace(NDArray& x, double lo, double hi);
void log1p_inplace(NDArray& x);

/// Per-column standardize with period mean.size(); numel must be a multiple of it.
void standardize_inplace(NDArray& x, const std::vector<double>& mean,
                         const std::vector<double>& scale);

/// Per-column MinMax (same formula as MinMaxTransform / MinMaxScaler: a zero
/// data range is treated as 1).
void minmax_inplace(NDArray& x, const std::vector<double>& data_min,
                    const std::vector<double>& data_max, double feature_min,
                    double feature_max);

/// New contiguous host array with dtype @p to (identity copy when equal).
[[nodiscard]] NDArray cast(const NDArray& x, DType to);

namespace detail {

/// Per-ISA kernel table (filled by the scalar / AVX2 / NEON translation units).
struct KernelTable {
    void (*scalar_op_f32)(float*, std::size_t, BinaryOp, double) noexcept;
    void (*scalar_op_f64)(double*, std::size_t, BinaryOp, double) noexcept;
    void (*clip_f32)(float*, std::size_t, double, double) noexcept;
    void (*clip_f64)(double*, std::size_t, double, double) noexcept;
    void (*standardize_f32)(float*, std::size_t, std::size_t, const double*,
                            const double*) noexcept;
    void (*standardize_f64)(double*, std::size_t, std::size_t, const double*,
                            const double*) noexcept;
    void (*minmax_f32)(float*, std::size_t, std::size_t, const double*, const double*,
                       double, double) noexcept;
    void (*minmax_f64)(double*, std::size_t, std::size_t, const double*, const double*,
                       double, double) noexcept;
    void (*round_to_f32)(double*, std::size_t) noexcept;
    void (*normalize_planes_f32)(float*, std::size_t, std::size_t, const float*,
                                 const float*) noexcept;
};

/// Kernel table selected by active_isa().
[[nodiscard]] const KernelTable& active_kernels() noexcept;

[[nodiscard]] const KernelTable& scalar_kernels() noexcept;
/// nullptr when the build has no AVX2 translation unit.
[[nodiscard]] const KernelTable* avx2_kernels() noexcept;

} // namespace detail

} // namespace simd
} // namespace nexusdata
