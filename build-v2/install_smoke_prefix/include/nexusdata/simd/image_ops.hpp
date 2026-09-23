#pragma once

#include <cstddef>
#include <cstdint>

namespace nexusdata {
namespace simd {

/// Resampling filter for resize_hwc_u8 / resize_to_chw_f32.
enum class ResampleMode : std::uint8_t { Nearest, Bilinear };

/// Precision contract: all image kernels reproduce the float arithmetic of the
/// reference transforms in pipeline/image.hpp bit-for-bit (same expressions,
/// no FMA); speed comes from hoisting per-row/per-column coordinate math,
/// per-channel lookup tables and fixed-channel specializations.

/// HWC uint8 -> CHW float32: dst[ch][y][x] = (src / 255 - mean[ch]) / stddev[ch].
/// Pass mean == stddev == nullptr for plain src / 255 (ToTensor).
void hwc_u8_to_chw_f32(const std::uint8_t* src, float* dst, std::size_t h, std::size_t w,
                       std::size_t c, const float* mean = nullptr,
                       const float* stddev = nullptr) noexcept;

/// In-place per-plane normalize of a CHW float32 buffer: p = (p - mean[ch]) / stddev[ch].
void normalize_planes_f32(float* p, std::size_t planes, std::size_t plane_size,
                          const float* mean, const float* stddev) noexcept;

/// Out-of-place horizontal / vertical flip of an HWC uint8 image (src != dst).
void hflip_hwc_u8(const std::uint8_t* src, std::uint8_t* dst, std::size_t h, std::size_t w,
                  std::size_t c) noexcept;
void vflip_hwc_u8(const std::uint8_t* src, std::uint8_t* dst, std::size_t h, std::size_t w,
                  std::size_t c) noexcept;

/// Resize HWC uint8 [sh, sw, c] -> [dh, dw, c]. Throws std::bad_alloc only.
void resize_hwc_u8(const std::uint8_t* src, std::size_t sh, std::size_t sw, std::uint8_t* dst,
                   std::size_t dh, std::size_t dw, std::size_t c, ResampleMode mode);

/// Fused resize + ToTensor (+ optional normalize): HWC uint8 [sh, sw, c] ->
/// CHW float32 [c, dh, dw] in one pass without an intermediate uint8 image.
/// Equivalent to resize_hwc_u8 followed by hwc_u8_to_chw_f32.
void resize_to_chw_f32(const std::uint8_t* src, std::size_t sh, std::size_t sw, std::size_t c,
                       float* dst, std::size_t dh, std::size_t dw, ResampleMode mode,
                       const float* mean = nullptr, const float* stddev = nullptr);

} // namespace simd
} // namespace nexusdata
