#include "nexusdata/simd/image_ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "nexusdata/backend/detail/kernel_math.hpp"
#include "nexusdata/backend/kernels.hpp"
#include "nexusdata/simd/numeric_ops.hpp"

namespace nexusdata {
namespace simd {

namespace {

using kmath::round_u8;

constexpr std::size_t kMaxLutChannels = 16;

/// lut[ch * 256 + u] = (u / 255 - mean[ch]) / stddev[ch]  (or u / 255 without mean).
void build_lut(float* lut, std::size_t c, const float* mean, const float* stddev) noexcept {
    for (std::size_t ch = 0; ch < c; ++ch) {
        for (int u = 0; u < 256; ++u) {
            float v = static_cast<float>(u) / 255.0f;
            if (mean != nullptr) {
                v = (v - mean[ch]) / stddev[ch];
            }
            lut[ch * 256 + static_cast<std::size_t>(u)] = v;
        }
    }
}

inline float to_float(std::uint8_t u, std::size_t ch, const float* mean,
                      const float* stddev) noexcept {
    float v = static_cast<float>(u) / 255.0f;
    if (mean != nullptr) {
        v = (v - mean[ch]) / stddev[ch];
    }
    return v;
}

template <std::size_t C>
void hwc_to_chw_lut(const std::uint8_t* src, float* dst, std::size_t hw, const float* lut) {
    float* planes[C];
    for (std::size_t ch = 0; ch < C; ++ch) {
        planes[ch] = dst + ch * hw;
    }
    for (std::size_t i = 0; i < hw; ++i) {
        for (std::size_t ch = 0; ch < C; ++ch) {
            planes[ch][i] = lut[ch * 256 + src[i * C + ch]];
        }
    }
}

template <std::size_t C>
void hflip_fixed(const std::uint8_t* src, std::uint8_t* dst, std::size_t h, std::size_t w) {
    for (std::size_t y = 0; y < h; ++y) {
        const std::uint8_t* s = src + y * w * C;
        std::uint8_t* d = dst + y * w * C;
        for (std::size_t x = 0; x < w; ++x) {
            std::memcpy(d + x * C, s + (w - 1 - x) * C, C);
        }
    }
}

/// Per-axis bilinear taps, computed with exactly the expressions used by
/// imgdetail::sample_bilinear / Resize::apply.
struct Taps {
    std::vector<std::size_t> i0;
    std::vector<std::size_t> i1;
    std::vector<float> w; // weight of i1
};

Taps bilinear_taps(std::size_t src_n, std::size_t dst_n) {
    Taps t;
    t.i0.resize(dst_n);
    t.i1.resize(dst_n);
    t.w.resize(dst_n);
    for (std::size_t k = 0; k < dst_n; ++k) {
        const kmath::BilinearTap tap = kmath::bilinear_tap(
            static_cast<int>(k), static_cast<int>(src_n), static_cast<int>(dst_n));
        t.i0[k] = static_cast<std::size_t>(tap.i0);
        t.i1[k] = static_cast<std::size_t>(tap.i1);
        t.w[k] = tap.w;
    }
    return t;
}

std::vector<std::size_t> nearest_taps(std::size_t src_n, std::size_t dst_n) {
    std::vector<std::size_t> idx(dst_n);
    for (std::size_t k = 0; k < dst_n; ++k) {
        idx[k] = static_cast<std::size_t>(kmath::nearest_tap(
            static_cast<int>(k), static_cast<int>(src_n), static_cast<int>(dst_n)));
    }
    return idx;
}

/// Visit every output pixel of a resize; @p emit(y, x, ch, value_u8).
template <typename Emit>
void resize_visit(const std::uint8_t* src, std::size_t sh, std::size_t sw, std::size_t dh,
                  std::size_t dw, std::size_t c, ResampleMode mode, Emit&& emit) {
    if (mode == ResampleMode::Nearest) {
        const auto ys = nearest_taps(sh, dh);
        const auto xs = nearest_taps(sw, dw);
        for (std::size_t y = 0; y < dh; ++y) {
            const std::uint8_t* row = src + ys[y] * sw * c;
            for (std::size_t x = 0; x < dw; ++x) {
                const std::uint8_t* p = row + xs[x] * c;
                for (std::size_t ch = 0; ch < c; ++ch) {
                    emit(y, x, ch, p[ch]);
                }
            }
        }
        return;
    }
    const Taps ty = bilinear_taps(sh, dh);
    const Taps tx = bilinear_taps(sw, dw);
    for (std::size_t y = 0; y < dh; ++y) {
        const std::uint8_t* r0 = src + ty.i0[y] * sw * c;
        const std::uint8_t* r1 = src + ty.i1[y] * sw * c;
        const float wy = ty.w[y];
        for (std::size_t x = 0; x < dw; ++x) {
            const std::size_t a = tx.i0[x] * c;
            const std::size_t b = tx.i1[x] * c;
            const float wx = tx.w[x];
            for (std::size_t ch = 0; ch < c; ++ch) {
                emit(y, x, ch, kmath::bilinear_u8(r0[a + ch], r0[b + ch], r1[a + ch],
                                                  r1[b + ch], wx, wy));
            }
        }
    }
}

} // namespace

void hwc_u8_to_chw_f32(const std::uint8_t* src, float* dst, std::size_t h, std::size_t w,
                       std::size_t c, const float* mean, const float* stddev) noexcept {
    const std::size_t hw = h * w;
    if (hw == 0 || c == 0) {
        return;
    }
    if (c <= kMaxLutChannels) {
        float lut[kMaxLutChannels * 256];
        build_lut(lut, c, mean, stddev);
        switch (c) {
            case 1: hwc_to_chw_lut<1>(src, dst, hw, lut); return;
            case 3: hwc_to_chw_lut<3>(src, dst, hw, lut); return;
            case 4: hwc_to_chw_lut<4>(src, dst, hw, lut); return;
            default:
                for (std::size_t i = 0; i < hw; ++i) {
                    for (std::size_t ch = 0; ch < c; ++ch) {
                        dst[ch * hw + i] = lut[ch * 256 + src[i * c + ch]];
                    }
                }
                return;
        }
    }
    for (std::size_t i = 0; i < hw; ++i) {
        for (std::size_t ch = 0; ch < c; ++ch) {
            dst[ch * hw + i] = to_float(src[i * c + ch], ch, mean, stddev);
        }
    }
}

void normalize_planes_f32(float* p, std::size_t planes, std::size_t plane_size,
                          const float* mean, const float* stddev) noexcept {
    detail::active_kernels().normalize_planes_f32(p, planes, plane_size, mean, stddev);
}

void hflip_hwc_u8(const std::uint8_t* src, std::uint8_t* dst, std::size_t h, std::size_t w,
                  std::size_t c) noexcept {
    switch (c) {
        case 1: hflip_fixed<1>(src, dst, h, w); return;
        case 3: hflip_fixed<3>(src, dst, h, w); return;
        case 4: hflip_fixed<4>(src, dst, h, w); return;
        default:
            for (std::size_t y = 0; y < h; ++y) {
                for (std::size_t x = 0; x < w; ++x) {
                    std::memcpy(dst + (y * w + x) * c, src + (y * w + (w - 1 - x)) * c, c);
                }
            }
    }
}

void vflip_hwc_u8(const std::uint8_t* src, std::uint8_t* dst, std::size_t h, std::size_t w,
                  std::size_t c) noexcept {
    const std::size_t row = w * c;
    for (std::size_t y = 0; y < h; ++y) {
        std::memcpy(dst + y * row, src + (h - 1 - y) * row, row);
    }
}

void resize_hwc_u8(const std::uint8_t* src, std::size_t sh, std::size_t sw, std::uint8_t* dst,
                   std::size_t dh, std::size_t dw, std::size_t c, ResampleMode mode) {
    resize_visit(src, sh, sw, dh, dw, c, mode,
                 [&](std::size_t y, std::size_t x, std::size_t ch, std::uint8_t v) {
                     dst[(y * dw + x) * c + ch] = v;
                 });
}

void resize_to_chw_f32(const std::uint8_t* src, std::size_t sh, std::size_t sw, std::size_t c,
                       float* dst, std::size_t dh, std::size_t dw, ResampleMode mode,
                       const float* mean, const float* stddev) {
    const std::size_t plane = dh * dw;
    if (c <= kMaxLutChannels) {
        float lut[kMaxLutChannels * 256];
        build_lut(lut, c, mean, stddev);
        resize_visit(src, sh, sw, dh, dw, c, mode,
                     [&](std::size_t y, std::size_t x, std::size_t ch, std::uint8_t v) {
                         dst[ch * plane + y * dw + x] = lut[ch * 256 + v];
                     });
        return;
    }
    resize_visit(src, sh, sw, dh, dw, c, mode,
                 [&](std::size_t y, std::size_t x, std::size_t ch, std::uint8_t v) {
                     dst[ch * plane + y * dw + x] = to_float(v, ch, mean, stddev);
                 });
}

} // namespace simd

namespace kernels {

namespace {

/// Resized value of channel @p ch at resized coordinates (ry, rx) inside a crop.
template <ResampleKind R>
struct Sampler {
    const std::uint8_t* base; // crop origin
    std::size_t stride;       // source row stride (bytes)
    std::size_t c;
    const kmath::BilinearTap* ty;
    const kmath::BilinearTap* tx;
    const int* ny;
    const int* nx;

    std::uint8_t operator()(int ry, int rx, std::size_t ch) const {
        if constexpr (R == ResampleKind::None) {
            return base[static_cast<std::size_t>(ry) * stride + static_cast<std::size_t>(rx) * c + ch];
        } else if constexpr (R == ResampleKind::Nearest) {
            return base[static_cast<std::size_t>(ny[ry]) * stride +
                        static_cast<std::size_t>(nx[rx]) * c + ch];
        } else {
            const kmath::BilinearTap& a = ty[ry];
            const kmath::BilinearTap& b = tx[rx];
            const std::uint8_t* r0 = base + static_cast<std::size_t>(a.i0) * stride;
            const std::uint8_t* r1 = base + static_cast<std::size_t>(a.i1) * stride;
            const std::size_t x0 = static_cast<std::size_t>(b.i0) * c + ch;
            const std::size_t x1 = static_cast<std::size_t>(b.i1) * c + ch;
            return kmath::bilinear_u8(r0[x0], r0[x1], r1[x0], r1[x1], b.w, a.w);
        }
    }
};

template <ResampleKind R>
void augment_samples(const ImageAugmentParams& p, int begin, int end,
                     const kmath::BilinearTap* ty, const kmath::BilinearTap* tx, const int* ny,
                     const int* nx, const float* lut) {
    const std::size_t c = static_cast<std::size_t>(p.c);
    const std::size_t dh = static_cast<std::size_t>(p.dh);
    const std::size_t dw = static_cast<std::size_t>(p.dw);
    const std::size_t src_img = static_cast<std::size_t>(p.sh) * static_cast<std::size_t>(p.sw) * c;
    const std::size_t stride = static_cast<std::size_t>(p.sw) * c;
    for (int s = begin; s < end; ++s) {
        AugmentSample a = p.samples != nullptr ? p.samples[s] : AugmentSample{};
        a.hflip = static_cast<std::uint8_t>((a.hflip != 0) != (p.hflip_all != 0));
        a.vflip = static_cast<std::uint8_t>((a.vflip != 0) != (p.vflip_all != 0));
        Sampler<R> at{p.src + static_cast<std::size_t>(s) * src_img +
                          static_cast<std::size_t>(a.crop_y) * stride +
                          static_cast<std::size_t>(a.crop_x) * c,
                      stride, c, ty, tx, ny, nx};
        for (std::size_t y = 0; y < dh; ++y) {
            const int ry = static_cast<int>(a.vflip ? dh - 1 - y : y);
            for (std::size_t x = 0; x < dw; ++x) {
                const int rx = static_cast<int>(a.hflip ? dw - 1 - x : x);
                if (p.layout == ImageLayout::HwcU8) {
                    auto* d = static_cast<std::uint8_t*>(p.dst) +
                              ((static_cast<std::size_t>(s) * dh + y) * dw + x) * c;
                    for (std::size_t ch = 0; ch < c; ++ch) d[ch] = at(ry, rx, ch);
                } else {
                    auto* d = static_cast<float*>(p.dst) +
                              static_cast<std::size_t>(s) * c * dh * dw + y * dw + x;
                    for (std::size_t ch = 0; ch < c; ++ch) {
                        d[ch * dh * dw] = lut[ch * 256 + at(ry, rx, ch)];
                    }
                }
            }
        }
    }
}

} // namespace

void run_image_augment_host(const ImageAugmentParams& p, int begin, int end) {
    if (begin >= end || p.dh <= 0 || p.dw <= 0 || p.c <= 0) {
        return;
    }
    std::vector<kmath::BilinearTap> ty, tx;
    std::vector<int> ny, nx;
    if (p.resize == ResampleKind::Bilinear) {
        ty.resize(static_cast<std::size_t>(p.dh));
        tx.resize(static_cast<std::size_t>(p.dw));
        for (int k = 0; k < p.dh; ++k) ty[static_cast<std::size_t>(k)] = kmath::bilinear_tap(k, p.crop_h, p.dh);
        for (int k = 0; k < p.dw; ++k) tx[static_cast<std::size_t>(k)] = kmath::bilinear_tap(k, p.crop_w, p.dw);
    } else if (p.resize == ResampleKind::Nearest) {
        ny.resize(static_cast<std::size_t>(p.dh));
        nx.resize(static_cast<std::size_t>(p.dw));
        for (int k = 0; k < p.dh; ++k) ny[static_cast<std::size_t>(k)] = kmath::nearest_tap(k, p.crop_h, p.dh);
        for (int k = 0; k < p.dw; ++k) nx[static_cast<std::size_t>(k)] = kmath::nearest_tap(k, p.crop_w, p.dw);
    }
    std::vector<float> lut;
    if (p.layout == ImageLayout::ChwF32) {
        lut.resize(static_cast<std::size_t>(p.c) * 256);
        for (int ch = 0; ch < p.c; ++ch) {
            for (int u = 0; u < 256; ++u) {
                const auto v = static_cast<std::uint8_t>(u);
                lut[static_cast<std::size_t>(ch) * 256 + static_cast<std::size_t>(u)] =
                    p.normalize ? kmath::u8_to_f32_norm(v, p.mean[ch], p.stddev[ch])
                                : kmath::u8_to_f32(v);
            }
        }
    }
    switch (p.resize) {
        case ResampleKind::None:
            augment_samples<ResampleKind::None>(p, begin, end, nullptr, nullptr, nullptr, nullptr,
                                                lut.data());
            return;
        case ResampleKind::Nearest:
            augment_samples<ResampleKind::Nearest>(p, begin, end, nullptr, nullptr, ny.data(),
                                                   nx.data(), lut.data());
            return;
        case ResampleKind::Bilinear:
            augment_samples<ResampleKind::Bilinear>(p, begin, end, ty.data(), tx.data(), nullptr,
                                                    nullptr, lut.data());
            return;
    }
}

} // namespace kernels
} // namespace nexusdata
