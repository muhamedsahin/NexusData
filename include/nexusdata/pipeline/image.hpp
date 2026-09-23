#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>

#include "nexusdata/core/array_utils.hpp"
#include "nexusdata/core/error.hpp"
#include "nexusdata/core/ndarray.hpp"
#include "nexusdata/core/random.hpp"
#include "nexusdata/image/image.hpp"
#include "nexusdata/pipeline/transform.hpp"
#include "nexusdata/simd/image_ops.hpp"
#include "nexusdata/simd/numeric_ops.hpp"

namespace nexusdata {

namespace imgdetail {

inline void require_hwc_u8(const NDArray& a, std::size_t& h, std::size_t& w, std::size_t& c) {
    if (a.dtype() != DType::UInt8 || a.shape().size() != 3) {
        throw ShapeError("image transform expects HWC UInt8, got dtype/shape mismatch");
    }
    h = a.shape()[0];
    w = a.shape()[1];
    c = a.shape()[2];
}

inline NDArray make_hwc(std::size_t h, std::size_t w, std::size_t c) {
    return NDArray(Shape{h, w, c}, DType::UInt8);
}

inline void sample_bilinear(const NDArray& src,
                            std::size_t sh, std::size_t sw, std::size_t c,
                            float y, float x, std::uint8_t* out) {
    y = std::clamp(y, 0.0f, static_cast<float>(sh - 1));
    x = std::clamp(x, 0.0f, static_cast<float>(sw - 1));
    const int y0 = static_cast<int>(std::floor(y));
    const int x0 = static_cast<int>(std::floor(x));
    const int y1 = std::min(y0 + 1, static_cast<int>(sh - 1));
    const int x1 = std::min(x0 + 1, static_cast<int>(sw - 1));
    const float wy = y - static_cast<float>(y0);
    const float wx = x - static_cast<float>(x0);
    const auto* p = src.data<std::uint8_t>();
    for (std::size_t ch = 0; ch < c; ++ch) {
        const float v00 = p[(static_cast<std::size_t>(y0) * sw + x0) * c + ch];
        const float v01 = p[(static_cast<std::size_t>(y0) * sw + x1) * c + ch];
        const float v10 = p[(static_cast<std::size_t>(y1) * sw + x0) * c + ch];
        const float v11 = p[(static_cast<std::size_t>(y1) * sw + x1) * c + ch];
        const float v0 = v00 * (1 - wx) + v01 * wx;
        const float v1 = v10 * (1 - wx) + v11 * wx;
        out[ch] = static_cast<std::uint8_t>(std::lround(v0 * (1 - wy) + v1 * wy));
    }
}

} // namespace imgdetail

enum class ResizeInterp { Nearest, Bilinear };

class Resize : public Transform {
public:
    Resize(int height, int width, ResizeInterp interp = ResizeInterp::Bilinear)
        : height_(height), width_(width), interp_(interp) {
        if (height_ <= 0 || width_ <= 0) {
            throw InvalidArgumentError("Resize: invalid size");
        }
    }

    [[nodiscard]] Sample apply(const Sample& sample) const override {
        std::size_t sh = 0, sw = 0, c = 0;
        imgdetail::require_hwc_u8(sample.input, sh, sw, c);
        NDArray out = imgdetail::make_hwc(static_cast<std::size_t>(height_),
                                          static_cast<std::size_t>(width_), c);
        simd::resize_hwc_u8(sample.input.data<std::uint8_t>(), sh, sw, out.data<std::uint8_t>(),
                            static_cast<std::size_t>(height_), static_cast<std::size_t>(width_),
                            c,
                            interp_ == ResizeInterp::Nearest ? simd::ResampleMode::Nearest
                                                             : simd::ResampleMode::Bilinear);
        return sample.with_input(std::move(out));
    }

    [[nodiscard]] bool is_fusable() const override { return true; }

    [[nodiscard]] int height() const noexcept { return height_; }
    [[nodiscard]] int width() const noexcept { return width_; }
    [[nodiscard]] ResizeInterp interp() const noexcept { return interp_; }

private:
    int height_ = 0;
    int width_ = 0;
    ResizeInterp interp_ = ResizeInterp::Bilinear;
};

class CenterCrop : public Transform {
public:
    CenterCrop(int height, int width) : height_(height), width_(width) {
        if (height_ <= 0 || width_ <= 0) {
            throw InvalidArgumentError("CenterCrop: invalid size");
        }
    }

    [[nodiscard]] Sample apply(const Sample& sample) const override {
        std::size_t sh = 0, sw = 0, c = 0;
        imgdetail::require_hwc_u8(sample.input, sh, sw, c);
        if (static_cast<int>(sh) < height_ || static_cast<int>(sw) < width_) {
            throw ShapeError("CenterCrop: image smaller than crop size");
        }
        const std::size_t y0 = (sh - static_cast<std::size_t>(height_)) / 2;
        const std::size_t x0 = (sw - static_cast<std::size_t>(width_)) / 2;
        NDArray out = imgdetail::make_hwc(static_cast<std::size_t>(height_),
                                          static_cast<std::size_t>(width_), c);
        const auto* src = sample.input.data<std::uint8_t>();
        auto* dst = out.data<std::uint8_t>();
        for (int y = 0; y < height_; ++y) {
            std::memcpy(dst + static_cast<std::size_t>(y) * width_ * c,
                        src + ((y0 + y) * sw + x0) * c,
                        static_cast<std::size_t>(width_) * c);
        }
        return sample.with_input(std::move(out));
    }

private:
    int height_ = 0;
    int width_ = 0;
};

class RandomCrop : public Transform {
public:
    RandomCrop(int height, int width, std::uint64_t seed = 0)
        : height_(height), width_(width), rng_(seed) {}

    [[nodiscard]] Sample apply(const Sample& sample) const override {
        std::size_t sh = 0, sw = 0, c = 0;
        imgdetail::require_hwc_u8(sample.input, sh, sw, c);
        if (static_cast<int>(sh) < height_ || static_cast<int>(sw) < width_) {
            throw ShapeError("RandomCrop: image smaller than crop size");
        }
        auto& rng = const_cast<PCG32&>(rng_);
        const std::uint32_t max_y = static_cast<std::uint32_t>(sh - height_);
        const std::uint32_t max_x = static_cast<std::uint32_t>(sw - width_);
        const std::size_t y0 = max_y == 0 ? 0 : rng.next_bounded(max_y + 1);
        const std::size_t x0 = max_x == 0 ? 0 : rng.next_bounded(max_x + 1);
        NDArray out = imgdetail::make_hwc(static_cast<std::size_t>(height_),
                                          static_cast<std::size_t>(width_), c);
        const auto* src = sample.input.data<std::uint8_t>();
        auto* dst = out.data<std::uint8_t>();
        for (int y = 0; y < height_; ++y) {
            std::memcpy(dst + static_cast<std::size_t>(y) * width_ * c,
                        src + ((y0 + y) * sw + x0) * c,
                        static_cast<std::size_t>(width_) * c);
        }
        return sample.with_input(std::move(out));
    }

    [[nodiscard]] bool is_random() const override { return true; }
    void set_seed(std::uint64_t seed) override { rng_ = PCG32(seed); }

private:
    int height_ = 0;
    int width_ = 0;
    PCG32 rng_;
};

class RandomHorizontalFlip : public Transform {
public:
    explicit RandomHorizontalFlip(float p = 0.5f, std::uint64_t seed = 0)
        : p_(p), rng_(seed) {}

    [[nodiscard]] Sample apply(const Sample& sample) const override {
        auto& rng = const_cast<PCG32&>(rng_);
        if (rng.next_float() >= p_) {
            return sample;
        }
        std::size_t h = 0, w = 0, c = 0;
        imgdetail::require_hwc_u8(sample.input, h, w, c);
        NDArray out = imgdetail::make_hwc(h, w, c);
        simd::hflip_hwc_u8(sample.input.data<std::uint8_t>(), out.data<std::uint8_t>(), h, w, c);
        return sample.with_input(std::move(out));
    }

    [[nodiscard]] bool is_random() const override { return true; }
    void set_seed(std::uint64_t seed) override { rng_ = PCG32(seed); }

private:
    float p_ = 0.5f;
    PCG32 rng_;
};

class RandomVerticalFlip : public Transform {
public:
    explicit RandomVerticalFlip(float p = 0.5f, std::uint64_t seed = 0)
        : p_(p), rng_(seed) {}

    [[nodiscard]] Sample apply(const Sample& sample) const override {
        auto& rng = const_cast<PCG32&>(rng_);
        if (rng.next_float() >= p_) {
            return sample;
        }
        std::size_t h = 0, w = 0, c = 0;
        imgdetail::require_hwc_u8(sample.input, h, w, c);
        NDArray out = imgdetail::make_hwc(h, w, c);
        simd::vflip_hwc_u8(sample.input.data<std::uint8_t>(), out.data<std::uint8_t>(), h, w, c);
        return sample.with_input(std::move(out));
    }

    [[nodiscard]] bool is_random() const override { return true; }
    void set_seed(std::uint64_t seed) override { rng_ = PCG32(seed); }

private:
    float p_ = 0.5f;
    PCG32 rng_;
};

class Grayscale : public Transform {
public:
    explicit Grayscale(int num_output_channels = 1) : out_c_(num_output_channels) {
        if (out_c_ != 1 && out_c_ != 3) {
            throw InvalidArgumentError("Grayscale: num_output_channels must be 1 or 3");
        }
    }

    [[nodiscard]] Sample apply(const Sample& sample) const override {
        std::size_t h = 0, w = 0, c = 0;
        imgdetail::require_hwc_u8(sample.input, h, w, c);
        NDArray out = imgdetail::make_hwc(h, w, static_cast<std::size_t>(out_c_));
        const auto* src = sample.input.data<std::uint8_t>();
        auto* dst = out.data<std::uint8_t>();
        for (std::size_t i = 0; i < h * w; ++i) {
            float g = 0;
            if (c == 1) {
                g = src[i];
            } else {
                g = 0.2989f * src[i * c + 0] + 0.5870f * src[i * c + 1] +
                    0.1140f * src[i * c + std::min<std::size_t>(2, c - 1)];
            }
            const auto v = static_cast<std::uint8_t>(std::clamp(std::lround(g), 0L, 255L));
            for (int ch = 0; ch < out_c_; ++ch) {
                dst[i * out_c_ + ch] = v;
            }
        }
        return sample.with_input(std::move(out));
    }

private:
    int out_c_ = 1;
};

/// Per-channel normalize on float CHW or HWC float tensor: (x - mean) / std.
class NormalizeImage : public Transform {
public:
    NormalizeImage(std::vector<float> mean, std::vector<float> stddev)
        : mean_(std::move(mean)), std_(std::move(stddev)) {
        if (mean_.size() != std_.size() || mean_.empty()) {
            throw InvalidArgumentError("NormalizeImage: mean/std size mismatch");
        }
    }

    [[nodiscard]] Sample apply(const Sample& sample) const override {
        Sample s = sample.with_input(NDArray{});
        const NDArray& in = sample.input;
        if (in.dtype() == DType::UInt8 && in.shape().size() == 3) {
            // HWC uint8: ToTensor + normalize in one pass (same float arithmetic).
            const std::size_t h = in.shape()[0];
            const std::size_t w = in.shape()[1];
            const std::size_t c = in.shape()[2];
            if (c != mean_.size()) {
                throw ShapeError("NormalizeImage: channel mismatch");
            }
            s.input = NDArray(Shape{c, h, w}, DType::Float32);
            simd::hwc_u8_to_chw_f32(in.data<std::uint8_t>(), s.input.data<float>(), h, w, c,
                                    mean_.data(), std_.data());
            return s;
        }
        NDArray x = in;
        if (x.dtype() == DType::UInt8) {
            x = image_to_tensor(x); // CHW float
        }
        if (x.dtype() != DType::Float32 || x.shape().size() != 3) {
            throw ShapeError("NormalizeImage: expected CHW float32 or HWC uint8");
        }
        // Assume CHW
        const std::size_t c = x.shape()[0];
        const std::size_t hw = x.shape()[1] * x.shape()[2];
        if (c != mean_.size()) {
            throw ShapeError("NormalizeImage: channel mismatch");
        }
        s.input = x.clone();
        simd::normalize_planes_f32(s.input.data<float>(), c, hw, mean_.data(), std_.data());
        return s;
    }

    [[nodiscard]] const std::vector<float>& mean() const noexcept { return mean_; }
    [[nodiscard]] const std::vector<float>& stddev() const noexcept { return std_; }

private:
    std::vector<float> mean_;
    std::vector<float> std_;
};

class ToTensor : public Transform {
public:
    [[nodiscard]] Sample apply(const Sample& sample) const override {
        return sample.with_input(image_to_tensor(sample.input));
    }

    [[nodiscard]] bool is_fusable() const override { return true; }
};

class Pad : public Transform {
public:
    Pad(int padding, std::uint8_t fill = 0) : pad_(padding), fill_(fill) {
        if (pad_ < 0) {
            throw InvalidArgumentError("Pad: negative padding");
        }
    }

    [[nodiscard]] Sample apply(const Sample& sample) const override {
        std::size_t h = 0, w = 0, c = 0;
        imgdetail::require_hwc_u8(sample.input, h, w, c);
        const std::size_t nh = h + 2 * static_cast<std::size_t>(pad_);
        const std::size_t nw = w + 2 * static_cast<std::size_t>(pad_);
        NDArray out = imgdetail::make_hwc(nh, nw, c);
        std::memset(out.data(), fill_, out.nbytes());
        const auto* src = sample.input.data<std::uint8_t>();
        auto* dst = out.data<std::uint8_t>();
        for (std::size_t y = 0; y < h; ++y) {
            std::memcpy(dst + ((y + pad_) * nw + pad_) * c, src + y * w * c, w * c);
        }
        return sample.with_input(std::move(out));
    }

private:
    int pad_ = 0;
    std::uint8_t fill_ = 0;
};

} // namespace nexusdata
