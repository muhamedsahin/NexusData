#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "nexusdata/backend/cpu_dispatch.hpp"
#include "nexusdata/core/array_utils.hpp"
#include "nexusdata/simd/image_ops.hpp"
#include "nexusdata/simd/numeric_ops.hpp"

using namespace nexusdata;

namespace {

// --- v1 reference implementations (verbatim semantics of the scalar loops) ---

NDArray ref_map(const NDArray& x, double (*f)(double, const void*), const void* ctx) {
    NDArray out = x.clone();
    for (std::size_t i = 0; i < out.numel(); ++i) {
        detail::set_double(out, i, f(detail::as_double(out, i), ctx));
    }
    return out;
}

struct ClipCtx { double lo, hi; };
double ref_clip(double v, const void* c) {
    const auto* p = static_cast<const ClipCtx*>(c);
    return std::clamp(v, p->lo, p->hi);
}
double ref_log1p(double v, const void*) { return std::log1p(v); }

NDArray ref_standardize(const NDArray& x, const std::vector<double>& mean,
                        const std::vector<double>& scale) {
    NDArray out = x.clone();
    const std::size_t f = mean.size();
    for (std::size_t i = 0; i < out.numel(); ++i) {
        const std::size_t j = i % f;
        detail::set_double(out, i, (detail::as_double(out, i) - mean[j]) / scale[j]);
    }
    return out;
}

NDArray ref_minmax(const NDArray& x, const std::vector<double>& dmin,
                   const std::vector<double>& dmax, double fmin, double fmax) {
    NDArray out = x.clone();
    const std::size_t f = dmin.size();
    const double span = fmax - fmin;
    for (std::size_t i = 0; i < out.numel(); ++i) {
        const std::size_t j = i % f;
        double denom = dmax[j] - dmin[j];
        if (denom == 0.0) denom = 1.0;
        detail::set_double(out, i, fmin + span * ((detail::as_double(out, i) - dmin[j]) / denom));
    }
    return out;
}

/// Bitwise equality, treating any two NaNs of a floating dtype as equal.
bool same(const NDArray& a, const NDArray& b) {
    if (a.dtype() != b.dtype() || a.numel() != b.numel()) return false;
    if (a.dtype() == DType::Float32) {
        const float* p = a.data<float>();
        const float* q = b.data<float>();
        for (std::size_t i = 0; i < a.numel(); ++i) {
            if (std::isnan(p[i]) && std::isnan(q[i])) continue;
            if (std::memcmp(&p[i], &q[i], sizeof(float)) != 0) return false;
        }
        return true;
    }
    if (a.dtype() == DType::Float64) {
        const double* p = a.data<double>();
        const double* q = b.data<double>();
        for (std::size_t i = 0; i < a.numel(); ++i) {
            if (std::isnan(p[i]) && std::isnan(q[i])) continue;
            if (std::memcmp(&p[i], &q[i], sizeof(double)) != 0) return false;
        }
        return true;
    }
    return a.nbytes() == 0 || std::memcmp(a.data(), b.data(), a.nbytes()) == 0;
}

NDArray random_array(DType dt, std::size_t n, std::uint32_t seed, bool specials) {
    NDArray a(Shape{n}, dt);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-100.0, 100.0);
    for (std::size_t i = 0; i < n; ++i) {
        double v = u(rng);
        if (dt != DType::Float32 && dt != DType::Float64) {
            v = std::round(v); // integral dtypes: keep conversions defined
            if (dt == DType::UInt8 || dt == DType::UInt16 || dt == DType::UInt32 ||
                dt == DType::UInt64) {
                v = std::abs(v);
            }
        }
        detail::set_double(a, i, v);
    }
    if (specials && (dt == DType::Float32 || dt == DType::Float64) && n >= 6) {
        detail::set_double(a, 0, std::numeric_limits<double>::quiet_NaN());
        detail::set_double(a, 1, -0.0);
        detail::set_double(a, 2, std::numeric_limits<double>::infinity());
        detail::set_double(a, 3, -std::numeric_limits<double>::infinity());
        detail::set_double(a, 4, 1e-30);
        detail::set_double(a, 5, 3.4e38);
    }
    return a;
}

std::vector<simd::Isa> isas_to_test() {
    std::vector<simd::Isa> v{simd::Isa::Scalar};
    if (simd::detected_isa() != simd::Isa::Scalar) v.push_back(simd::detected_isa());
    return v;
}

struct IsaGuard {
    explicit IsaGuard(simd::Isa isa) { simd::set_isa_override(isa); }
    ~IsaGuard() { simd::clear_isa_override(); }
};

const std::vector<std::size_t> kSizes{0, 1, 3, 7, 8, 9, 15, 16, 17, 31, 64, 1000, 4099};

} // namespace

TEST_CASE("cpu feature flags are consistent") {
    const auto& f = cpu_features();
    if (f.avx2) CHECK(f.avx);
    if (f.fma) CHECK(f.avx);
    if (f.avx512bw) CHECK(f.avx512f);
    MESSAGE("isa=" << simd::isa_name(simd::detected_isa()) << " avx2=" << f.avx2
                   << " fma=" << f.fma << " avx512f=" << f.avx512f);
}

TEST_CASE("isa override clamps to the CPU") {
    simd::set_isa_override(simd::Isa::AVX512);
    const simd::Isa got = simd::active_isa();
    if (simd::detected_isa() == simd::Isa::AVX512) {
        CHECK(got == simd::Isa::AVX512);
    } else if (simd::detected_isa() == simd::Isa::AVX2) {
        CHECK(got == simd::Isa::AVX2);
    } else {
        CHECK(got == simd::Isa::Scalar);
    }
    simd::clear_isa_override();
    CHECK(simd::active_isa() == simd::detected_isa());
}

TEST_CASE("clip / log1p / scalar ops are bit-exact vs v1 loops") {
    const std::vector<DType> dtypes{DType::Float32, DType::Float64, DType::Int32,
                                    DType::Int64,   DType::UInt8,   DType::Int16};
    for (simd::Isa isa : isas_to_test()) {
        IsaGuard g(isa);
        for (DType dt : dtypes) {
            for (std::size_t n : kSizes) {
                CAPTURE(simd::isa_name(isa));
                CAPTURE(to_string(dt));
                CAPTURE(n);
                const NDArray x = random_array(dt, n, static_cast<std::uint32_t>(n * 7 + 1), true);

                ClipCtx c{-12.5, 40.25};
                if (dt == DType::UInt8) c = {3.0, 60.0};
                NDArray a = x.clone();
                simd::clip_inplace(a, c.lo, c.hi);
                CHECK(same(a, ref_map(x, ref_clip, &c)));

                if (dt == DType::Float32 || dt == DType::Float64) {
                    NDArray l = x.clone();
                    simd::log1p_inplace(l);
                    CHECK(same(l, ref_map(x, ref_log1p, nullptr)));

                    for (auto op : {simd::BinaryOp::Add, simd::BinaryOp::Sub,
                                    simd::BinaryOp::Mul, simd::BinaryOp::Div}) {
                        NDArray s = x.clone();
                        simd::apply_scalar_op(s, op, 0.37);
                        NDArray r = x.clone();
                        for (std::size_t i = 0; i < r.numel(); ++i) {
                            const double v = detail::as_double(r, i);
                            double o = v;
                            switch (op) {
                                case simd::BinaryOp::Add: o = v + 0.37; break;
                                case simd::BinaryOp::Sub: o = v - 0.37; break;
                                case simd::BinaryOp::Mul: o = v * 0.37; break;
                                case simd::BinaryOp::Div: o = v / 0.37; break;
                            }
                            detail::set_double(r, i, o);
                        }
                        CHECK(same(s, r));
                    }
                }
            }
        }
    }
}

TEST_CASE("standardize / minmax are bit-exact vs v1 loops for all widths") {
    for (simd::Isa isa : isas_to_test()) {
        IsaGuard g(isa);
        for (DType dt : {DType::Float32, DType::Float64, DType::Int32}) {
            for (std::size_t cols : {1u, 2u, 3u, 5u, 8u, 9u, 17u, 31u, 32u, 40u}) {
                for (std::size_t rows : {1u, 3u, 4u, 11u, 64u}) {
                    CAPTURE(simd::isa_name(isa));
                    CAPTURE(to_string(dt));
                    CAPTURE(cols);
                    CAPTURE(rows);
                    NDArray x = random_array(dt, rows * cols, static_cast<std::uint32_t>(cols * 131 + rows),
                                             dt != DType::Int32);
                    std::vector<double> mean(cols), scale(cols), dmin(cols), dmax(cols);
                    for (std::size_t j = 0; j < cols; ++j) {
                        mean[j] = 0.5 * static_cast<double>(j) - 1.0;
                        scale[j] = 1.0 + 0.25 * static_cast<double>(j);
                        dmin[j] = -static_cast<double>(j);
                        dmax[j] = (j % 4 == 0) ? dmin[j] : static_cast<double>(j) * 3.0; // zero ranges too
                    }
                    NDArray s = x.clone();
                    simd::standardize_inplace(s, mean, scale);
                    CHECK(same(s, ref_standardize(x, mean, scale)));

                    NDArray m = x.clone();
                    simd::minmax_inplace(m, dmin, dmax, -1.0, 2.0);
                    CHECK(same(m, ref_minmax(x, dmin, dmax, -1.0, 2.0)));
                }
            }
        }
    }
}

TEST_CASE("period / dtype validation") {
    NDArray x(Shape{10}, DType::Float32);
    CHECK_THROWS_AS(simd::standardize_inplace(x, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}), ShapeError);
    CHECK_THROWS_AS(simd::standardize_inplace(x, {0.0}, {1.0, 1.0}), InvalidArgumentError);
    NDArray h(Shape{4}, DType::Float16);
    CHECK_THROWS_AS(simd::clip_inplace(h, 0.0, 1.0), InvalidArgumentError);
}

TEST_CASE("cast matches as_double/set_double") {
    const NDArray x = random_array(DType::Float32, 257, 5, false);
    for (DType to : {DType::Float64, DType::Int32, DType::Int64, DType::Float32, DType::Bool}) {
        NDArray a = simd::cast(x, to);
        NDArray r(x.shape(), to);
        for (std::size_t i = 0; i < x.numel(); ++i) {
            detail::set_double(r, i, detail::as_double(x, i));
        }
        CAPTURE(to_string(to));
        CHECK(same(a, r));
    }
}

// --- image kernels vs v1 reference transforms -----------------------------------

namespace {

std::vector<std::uint8_t> random_image(std::size_t h, std::size_t w, std::size_t c,
                                       std::uint32_t seed) {
    std::vector<std::uint8_t> img(h * w * c);
    std::mt19937 rng(seed);
    for (auto& v : img) v = static_cast<std::uint8_t>(rng() & 0xFF);
    return img;
}

std::vector<std::uint8_t> ref_resize(const std::vector<std::uint8_t>& src, std::size_t sh,
                                     std::size_t sw, std::size_t c, int height, int width,
                                     bool nearest) {
    std::vector<std::uint8_t> dst(static_cast<std::size_t>(height) * width * c);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (nearest) {
                const std::size_t sy = std::min(
                    sh - 1, static_cast<std::size_t>(y * static_cast<double>(sh) / height));
                const std::size_t sx = std::min(
                    sw - 1, static_cast<std::size_t>(x * static_cast<double>(sw) / width));
                std::memcpy(&dst[(static_cast<std::size_t>(y) * width + x) * c],
                            &src[(sy * sw + sx) * c], c);
                continue;
            }
            float fy = (y + 0.5f) * static_cast<float>(sh) / static_cast<float>(height) - 0.5f;
            float fx = (x + 0.5f) * static_cast<float>(sw) / static_cast<float>(width) - 0.5f;
            fy = std::clamp(fy, 0.0f, static_cast<float>(sh - 1));
            fx = std::clamp(fx, 0.0f, static_cast<float>(sw - 1));
            const int y0 = static_cast<int>(std::floor(fy));
            const int x0 = static_cast<int>(std::floor(fx));
            const int y1 = std::min(y0 + 1, static_cast<int>(sh - 1));
            const int x1 = std::min(x0 + 1, static_cast<int>(sw - 1));
            const float wy = fy - static_cast<float>(y0);
            const float wx = fx - static_cast<float>(x0);
            for (std::size_t ch = 0; ch < c; ++ch) {
                const float v00 = src[(static_cast<std::size_t>(y0) * sw + x0) * c + ch];
                const float v01 = src[(static_cast<std::size_t>(y0) * sw + x1) * c + ch];
                const float v10 = src[(static_cast<std::size_t>(y1) * sw + x0) * c + ch];
                const float v11 = src[(static_cast<std::size_t>(y1) * sw + x1) * c + ch];
                const float v0 = v00 * (1 - wx) + v01 * wx;
                const float v1 = v10 * (1 - wx) + v11 * wx;
                dst[(static_cast<std::size_t>(y) * width + x) * c + ch] =
                    static_cast<std::uint8_t>(std::lround(v0 * (1 - wy) + v1 * wy));
            }
        }
    }
    return dst;
}

std::vector<float> ref_to_chw(const std::vector<std::uint8_t>& src, std::size_t h, std::size_t w,
                              std::size_t c, const float* mean, const float* stdv) {
    std::vector<float> out(c * h * w);
    for (std::size_t y = 0; y < h; ++y)
        for (std::size_t x = 0; x < w; ++x)
            for (std::size_t ch = 0; ch < c; ++ch)
                out[ch * h * w + y * w + x] = static_cast<float>(src[(y * w + x) * c + ch]) / 255.0f;
    if (mean) {
        for (std::size_t ch = 0; ch < c; ++ch)
            for (std::size_t i = 0; i < h * w; ++i)
                out[ch * h * w + i] = (out[ch * h * w + i] - mean[ch]) / stdv[ch];
    }
    return out;
}

bool same_floats(const std::vector<float>& a, const float* b) {
    return std::memcmp(a.data(), b, a.size() * sizeof(float)) == 0;
}

} // namespace

TEST_CASE("resize kernels are bit-exact vs v1 Resize") {
    struct Case { std::size_t sh, sw, c; int dh, dw; };
    const std::vector<Case> cases{{17, 23, 3, 8, 9}, {8, 8, 1, 31, 17}, {5, 64, 4, 5, 64},
                                  {33, 7, 5, 12, 40}, {1, 1, 3, 4, 4}, {64, 48, 3, 224, 224}};
    for (const auto& k : cases) {
        const auto img = random_image(k.sh, k.sw, k.c, static_cast<std::uint32_t>(k.sh * 31 + k.c));
        for (bool nearest : {false, true}) {
            CAPTURE(k.sh); CAPTURE(k.sw); CAPTURE(k.c); CAPTURE(nearest);
            const auto ref = ref_resize(img, k.sh, k.sw, k.c, k.dh, k.dw, nearest);
            std::vector<std::uint8_t> got(ref.size());
            simd::resize_hwc_u8(img.data(), k.sh, k.sw, got.data(), static_cast<std::size_t>(k.dh),
                                static_cast<std::size_t>(k.dw), k.c,
                                nearest ? simd::ResampleMode::Nearest : simd::ResampleMode::Bilinear);
            CHECK(got == ref);

            // Fused resize -> CHW normalize equals resize then ToTensor + Normalize.
            std::vector<float> mean(k.c), stdv(k.c);
            for (std::size_t ch = 0; ch < k.c; ++ch) {
                mean[ch] = 0.4f + 0.05f * static_cast<float>(ch);
                stdv[ch] = 0.2f + 0.01f * static_cast<float>(ch);
            }
            const auto ref_f = ref_to_chw(ref, static_cast<std::size_t>(k.dh),
                                          static_cast<std::size_t>(k.dw), k.c, mean.data(), stdv.data());
            std::vector<float> got_f(ref_f.size());
            simd::resize_to_chw_f32(img.data(), k.sh, k.sw, k.c, got_f.data(),
                                    static_cast<std::size_t>(k.dh), static_cast<std::size_t>(k.dw),
                                    nearest ? simd::ResampleMode::Nearest : simd::ResampleMode::Bilinear,
                                    mean.data(), stdv.data());
            CHECK(same_floats(ref_f, got_f.data()));
        }
    }
}

TEST_CASE("to_chw / normalize / flips are bit-exact vs v1") {
    for (std::size_t c : {1u, 3u, 4u, 5u, 20u}) {
        const std::size_t h = 13, w = 29;
        const auto img = random_image(h, w, c, static_cast<std::uint32_t>(c));
        std::vector<float> mean(c), stdv(c);
        for (std::size_t ch = 0; ch < c; ++ch) {
            mean[ch] = 0.485f - 0.01f * static_cast<float>(ch);
            stdv[ch] = 0.229f + 0.003f * static_cast<float>(ch);
        }
        CAPTURE(c);
        std::vector<float> got(c * h * w);
        simd::hwc_u8_to_chw_f32(img.data(), got.data(), h, w, c);
        CHECK(same_floats(ref_to_chw(img, h, w, c, nullptr, nullptr), got.data()));
        simd::hwc_u8_to_chw_f32(img.data(), got.data(), h, w, c, mean.data(), stdv.data());
        const auto ref_n = ref_to_chw(img, h, w, c, mean.data(), stdv.data());
        CHECK(same_floats(ref_n, got.data()));

        for (simd::Isa isa : isas_to_test()) {
            IsaGuard g(isa);
            auto planes = ref_to_chw(img, h, w, c, nullptr, nullptr);
            simd::normalize_planes_f32(planes.data(), c, h * w, mean.data(), stdv.data());
            CHECK(std::memcmp(planes.data(), ref_n.data(), planes.size() * 4) == 0);
        }

        std::vector<std::uint8_t> hf(img.size()), vf(img.size());
        simd::hflip_hwc_u8(img.data(), hf.data(), h, w, c);
        simd::vflip_hwc_u8(img.data(), vf.data(), h, w, c);
        for (std::size_t y = 0; y < h; ++y)
            for (std::size_t x = 0; x < w; ++x)
                for (std::size_t ch = 0; ch < c; ++ch) {
                    REQUIRE(hf[(y * w + x) * c + ch] == img[(y * w + (w - 1 - x)) * c + ch]);
                    REQUIRE(vf[(y * w + x) * c + ch] == img[((h - 1 - y) * w + x) * c + ch]);
                }
    }
}
