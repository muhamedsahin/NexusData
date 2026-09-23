#pragma once

// Per-element arithmetic shared by the CPU kernels (src/image_ops.cpp) and the
// CUDA kernels (cuda/cuda_kernels.cu). Keeping one definition is what makes the
// GPU path bit-identical to the CPU path: both evaluate the same IEEE float
// expressions in the same order (nvcc is invoked with --fmad=false and IEEE
// division; host code is compiled without FP contraction).

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define NEXUSDATA_HD __host__ __device__ __forceinline__
#else
#define NEXUSDATA_HD inline
#endif

namespace nexusdata {
namespace kmath {

/// std::clamp semantics (NaN passes through).
NEXUSDATA_HD float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (hi < v ? hi : v);
}
NEXUSDATA_HD double clampd(double v, double lo, double hi) {
    return v < lo ? lo : (hi < v ? hi : v);
}

NEXUSDATA_HD int mini(int a, int b) { return a < b ? a : b; }

/// std::lround for non-negative finite floats below 2^23 (f - trunc(f) is exact there).
NEXUSDATA_HD std::uint8_t round_u8(float f) {
    int r = static_cast<int>(f);
    if (f - static_cast<float>(r) >= 0.5f) {
        ++r;
    }
    return static_cast<std::uint8_t>(r);
}

/// One axis of a bilinear resize (same expressions as imgdetail::sample_bilinear).
struct BilinearTap {
    int i0;
    int i1;
    float w; ///< weight of i1
};

NEXUSDATA_HD BilinearTap bilinear_tap(int k, int src_n, int dst_n) {
    float s = (static_cast<float>(k) + 0.5f) * static_cast<float>(src_n) /
                  static_cast<float>(dst_n) -
              0.5f;
    s = clampf(s, 0.0f, static_cast<float>(src_n - 1));
    const int s0 = static_cast<int>(s); // s >= 0: truncation == floor
    BilinearTap t;
    t.i0 = s0;
    t.i1 = mini(s0 + 1, src_n - 1);
    t.w = s - static_cast<float>(s0);
    return t;
}

/// Nearest source index: floor(k * src_n / dst_n), computed exactly in integers.
/// Equals the v1 expression `size_t(k * double(src_n) / dst_n)` because the
/// exact quotient is at least 1/dst_n away from the next integer.
NEXUSDATA_HD int nearest_tap(int k, int src_n, int dst_n) {
    const long long v = static_cast<long long>(k) * src_n / dst_n;
    return v < src_n - 1 ? static_cast<int>(v) : src_n - 1;
}

/// Interpolate one channel from its four neighbours (v1 Resize arithmetic).
NEXUSDATA_HD std::uint8_t bilinear_u8(float v00, float v01, float v10, float v11, float wx,
                                      float wy) {
    const float v0 = v00 * (1 - wx) + v01 * wx;
    const float v1 = v10 * (1 - wx) + v11 * wx;
    return round_u8(v0 * (1 - wy) + v1 * wy);
}

/// ToTensor (+ NormalizeImage) of one uint8 value.
NEXUSDATA_HD float u8_to_f32(std::uint8_t u) { return static_cast<float>(u) / 255.0f; }
NEXUSDATA_HD float u8_to_f32_norm(std::uint8_t u, float mean, float stddev) {
    return (static_cast<float>(u) / 255.0f - mean) / stddev;
}

} // namespace kmath
} // namespace nexusdata
