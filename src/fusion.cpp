#include "nexusdata/pipeline/fusion.hpp"

#include <algorithm>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

#include "nexusdata/backend/thread_pool.hpp"
#include "nexusdata/core/allocator.hpp"
#include "nexusdata/core/error.hpp"
#include "nexusdata/pipeline/image.hpp"

namespace nexusdata {

// --- FuseStep factories -------------------------------------------------------------

FuseStepPtr FuseStep::clip(double lo, double hi, std::string name) {
    auto s = std::make_shared<FuseStep>();
    s->kind = Kind::Clip;
    s->a = lo;
    s->b = hi;
    s->name = std::move(name);
    return s;
}

FuseStepPtr FuseStep::log1p(std::string name) {
    auto s = std::make_shared<FuseStep>();
    s->kind = Kind::Log1p;
    s->name = std::move(name);
    return s;
}

FuseStepPtr FuseStep::standardize(std::vector<double> mean, std::vector<double> scale,
                                  std::string name) {
    if (mean.empty() || mean.size() != scale.size()) {
        throw InvalidArgumentError(name + ": mean/scale size mismatch");
    }
    auto s = std::make_shared<FuseStep>();
    s->kind = Kind::Standardize;
    s->period = mean.size();
    s->p0 = std::move(mean);
    s->p1 = std::move(scale);
    s->name = std::move(name);
    return s;
}

FuseStepPtr FuseStep::minmax(const std::vector<double>& data_min,
                             const std::vector<double>& data_max, double feature_min,
                             double feature_max, std::string name) {
    if (data_min.empty() || data_min.size() != data_max.size()) {
        throw InvalidArgumentError(name + ": size mismatch");
    }
    auto s = std::make_shared<FuseStep>();
    s->kind = Kind::MinMax;
    s->period = data_min.size();
    s->p0 = data_min;
    s->p1.resize(data_min.size());
    for (std::size_t j = 0; j < data_min.size(); ++j) {
        const double d = data_max[j] - data_min[j];
        s->p1[j] = d == 0.0 ? 1.0 : d;
    }
    s->a = feature_max - feature_min;
    s->b = feature_min;
    s->name = std::move(name);
    return s;
}

FuseStepPtr FuseStep::scalar(simd::BinaryOp op, double a, std::string name) {
    auto s = std::make_shared<FuseStep>();
    s->kind = Kind::ScalarOp;
    s->op = op;
    s->a = a;
    s->name = std::move(name);
    return s;
}

FuseStepPtr FuseStep::make_custom(std::string name,
                                  std::function<void(double*, std::size_t, std::size_t)> fn,
                                  std::size_t period) {
    if (!fn) {
        throw InvalidArgumentError("FuseStep::make_custom: empty kernel for " + name);
    }
    auto s = std::make_shared<FuseStep>();
    s->kind = Kind::Custom;
    s->custom = std::move(fn);
    s->period = period;
    s->name = std::move(name);
    return s;
}

// --- execution --------------------------------------------------------------------

namespace {

constexpr std::size_t kTile = 512;

std::size_t tile_for(std::size_t period) {
    if (period == 0 || period > kTile) {
        return kTile;
    }
    return (kTile / period) * period;
}

void periodic_kernel(const FuseStep& s, double* v, std::size_t rows, std::size_t cols,
                     std::size_t c0) {
    if (s.kind == FuseStep::Kind::Standardize) {
        simd::standardize_f64(v, rows, cols, s.p0.data() + c0, s.p1.data() + c0);
    } else {
        simd::minmax_f64(v, rows, cols, s.p0.data() + c0, s.p1.data() + c0, s.a, s.b);
    }
}

void apply_step(const FuseStep& s, double* v, std::size_t n, std::size_t col0) {
    switch (s.kind) {
        case FuseStep::Kind::Clip:
            simd::clip_f64(v, n, s.a, s.b);
            return;
        case FuseStep::Kind::Log1p:
            simd::log1p_f64(v, n);
            return;
        case FuseStep::Kind::ScalarOp:
            simd::scalar_op_f64(v, n, s.op, s.a);
            return;
        case FuseStep::Kind::Custom:
            s.custom(v, n, col0);
            return;
        case FuseStep::Kind::Standardize:
        case FuseStep::Kind::MinMax: {
            const std::size_t p = s.period;
            if (col0 == 0 && n % p == 0) {
                periodic_kernel(s, v, n / p, p, 0);
                return;
            }
            std::size_t pos = 0;
            std::size_t c = col0;
            while (pos < n) {
                const std::size_t len = std::min(n - pos, p - c);
                periodic_kernel(s, v + pos, 1, len, c);
                pos += len;
                c = 0;
            }
            return;
        }
    }
}

template <typename T>
void load_tile(const T* src, double* t, std::size_t n) {
    for (std::size_t k = 0; k < n; ++k) {
        t[k] = static_cast<double>(src[k]);
    }
}

template <typename T>
void store_tile(const double* t, T* dst, std::size_t n) {
    for (std::size_t k = 0; k < n; ++k) {
        if constexpr (std::is_same_v<T, bool>) {
            dst[k] = t[k] != 0.0;
        } else {
            dst[k] = static_cast<T>(t[k]);
        }
    }
}

/// Emulate writing to / re-reading from the storage dtype between steps.
template <typename T>
void round_tile(double* t, std::size_t n) {
    if constexpr (std::is_same_v<T, double>) {
        (void)t;
        (void)n;
    } else if constexpr (std::is_same_v<T, float>) {
        simd::round_to_f32(t, n);
    } else if constexpr (std::is_same_v<T, bool>) {
        for (std::size_t k = 0; k < n; ++k) {
            t[k] = t[k] != 0.0 ? 1.0 : 0.0;
        }
    } else {
        for (std::size_t k = 0; k < n; ++k) {
            t[k] = static_cast<double>(static_cast<T>(t[k]));
        }
    }
}

template <typename T>
void run_range(const FuseStepPtr* steps, std::size_t count, const T* src, T* dst, std::size_t begin,
               std::size_t end, std::size_t row_len, std::size_t tile) {
    double buf[kTile];
    for (std::size_t off = begin; off < end; off += tile) {
        const std::size_t m = std::min(tile, end - off);
        load_tile(src + off, buf, m);
        const std::size_t col0 = row_len == 0 ? 0 : off % row_len;
        for (std::size_t i = 0; i < count; ++i) {
            apply_step(*steps[i], buf, m, col0);
            if (i + 1 < count) {
                round_tile<T>(buf, m);
            }
        }
        store_tile(buf, dst + off, m);
    }
}

template <typename F>
void visit_dtype(DType dt, F&& f) {
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
    throw InvalidArgumentError("FusedTransform: unsupported dtype " + std::string(to_string(dt)));
}

NDArray alloc_uninit(const Shape& shape, DType dtype) {
    const std::size_t bytes = numel(shape) * size_of(dtype);
    if (bytes == 0) {
        return NDArray(shape, dtype);
    }
    return NDArray::from_shared(make_aligned_buffer(bytes), shape, dtype, Device::host());
}

NDArray run_steps(const FuseStepPtr* steps, std::size_t count, const NDArray& x, std::size_t row_len,
                  bool parallel) {
    if (!x.device().is_host()) {
        throw InvalidArgumentError("FusedTransform: host arrays only (got " +
                                   x.device().to_string() + ")");
    }
    NDArray out = alloc_uninit(x.shape(), x.dtype());
    const std::size_t n = x.numel();
    if (n == 0) {
        return out;
    }
    std::size_t period = 0;
    for (std::size_t i = 0; i < count; ++i) {
        if (steps[i]->period != 0) {
            period = steps[i]->period;
            break;
        }
    }
    const std::size_t tile = tile_for(period);
    visit_dtype(x.dtype(), [&](auto tag) {
        using T = typename decltype(tag)::type;
        const T* src = static_cast<const T*>(x.data());
        T* dst = static_cast<T*>(out.data());
        if (!parallel || n < FusedTransform::kParallelMinElements) {
            run_range(steps, count, src, dst, 0, n, row_len, tile);
            return;
        }
        ThreadPool& pool = default_thread_pool();
        const std::size_t parts = std::max<std::size_t>(1, pool.size() * 2);
        std::size_t chunk = (n + parts - 1) / parts;
        chunk = ((chunk + tile - 1) / tile) * tile;
        const std::size_t tasks = (n + chunk - 1) / chunk;
        pool.parallel_for(tasks, [&](std::size_t t) {
            const std::size_t b = t * chunk;
            run_range(steps, count, src, dst, b, std::min(n, b + chunk), row_len, tile);
        });
    });
    return out;
}

void check_steps_row(const FuseStepPtr* steps, std::size_t count, std::size_t row_len) {
    for (std::size_t i = 0; i < count; ++i) {
        const auto& s = steps[i];
        if (s->period != 0 && row_len != s->period) {
            throw ShapeError(s->name + ": feature count mismatch (expected " +
                             std::to_string(s->period) + ", got " + std::to_string(row_len) +
                             ")");
        }
    }
}

} // namespace

FusedTransform::FusedTransform(std::vector<FuseStepPtr> steps) : steps_(std::move(steps)) {
    if (steps_.empty()) {
        throw InvalidArgumentError("FusedTransform: empty step list");
    }
    for (const auto& s : steps_) {
        if (!s) {
            throw InvalidArgumentError("FusedTransform: null step");
        }
        if (period_ == 0 && s->period != 0) {
            period_ = s->period;
        }
    }
}

void FusedTransform::check_row(std::size_t row_len) const {
    check_steps_row(steps_.data(), steps_.size(), row_len);
}

NDArray FusedTransform::run(const NDArray& x, std::size_t row_len) const {
    return run_steps(steps_.data(), steps_.size(), x, row_len, false);
}

Sample FusedTransform::apply(const Sample& sample) const {
    check_row(sample.input.numel());
    return sample.with_input(
        run_steps(steps_.data(), steps_.size(), sample.input, sample.input.numel(), false));
}

Batch FusedTransform::apply_batch(const Batch& batch) const {
    Batch out = batch;
    const NDArray& x = batch.inputs;
    if (x.numel() == 0) {
        return out;
    }
    if (x.shape().empty() || x.shape()[0] == 0) {
        throw ShapeError("FusedTransform::apply_batch: inputs must be [B, ...]");
    }
    const std::size_t row_len = x.numel() / x.shape()[0];
    check_row(row_len);
    out.inputs = run_steps(steps_.data(), steps_.size(), x, row_len, true);
    return out;
}

Sample apply_fuse_step(const FuseStepPtr& step, const Sample& sample) {
    check_steps_row(&step, 1, sample.input.numel());
    return sample.with_input(run_steps(&step, 1, sample.input, sample.input.numel(), false));
}

Batch Transform::apply_batch(const Batch& batch) const {
    auto step = fuse_step();
    if (!step) {
        throw InvalidArgumentError(
            "Transform::apply_batch: transform does not support batch application "
            "(override apply_batch() or fuse_step())");
    }
    return FusedTransform({std::move(step)}).apply_batch(batch);
}

// --- fused image chain -------------------------------------------------------------

FusedImageTransform::FusedImageTransform(Spec spec, std::vector<TransformConstPtr> originals)
    : spec_(std::move(spec)), originals_(std::move(originals)) {
    if (spec_.mean.size() != spec_.stddev.size()) {
        throw InvalidArgumentError("FusedImageTransform: mean/std size mismatch");
    }
    if (spec_.resize && (spec_.height == 0 || spec_.width == 0)) {
        throw InvalidArgumentError("FusedImageTransform: invalid resize target");
    }
}

Sample FusedImageTransform::apply(const Sample& sample) const {
    const NDArray& x = sample.input;
    const bool fast = x.device().is_host() && x.dtype() == DType::UInt8 &&
                      x.shape().size() == 3 && x.shape()[0] > 0 && x.shape()[1] > 0;
    if (!fast) {
        Sample out = sample;
        for (const auto& t : originals_) {
            out = t->apply(out);
        }
        return out;
    }
    const std::size_t h = x.shape()[0];
    const std::size_t w = x.shape()[1];
    const std::size_t c = x.shape()[2];
    const bool norm = !spec_.mean.empty();
    if (norm && c != spec_.mean.size()) {
        throw ShapeError("NormalizeImage: channel mismatch");
    }
    const std::size_t dh = spec_.resize ? spec_.height : h;
    const std::size_t dw = spec_.resize ? spec_.width : w;
    NDArray out = alloc_uninit(Shape{c, dh, dw}, DType::Float32);
    const auto* src = static_cast<const std::uint8_t*>(x.data());
    auto* dst = static_cast<float*>(out.data());
    const float* mean = norm ? spec_.mean.data() : nullptr;
    const float* stdv = norm ? spec_.stddev.data() : nullptr;
    if (spec_.resize) {
        simd::resize_to_chw_f32(src, h, w, c, dst, dh, dw, spec_.mode, mean, stdv);
    } else {
        simd::hwc_u8_to_chw_f32(src, dst, h, w, c, mean, stdv);
    }
    return sample.with_input(std::move(out));
}

// --- plan compiler -------------------------------------------------------------------

namespace {

std::optional<std::pair<TransformConstPtr, std::size_t>> match_image_chain(
    const std::vector<TransformConstPtr>& ts, std::size_t i) {
    FusedImageTransform::Spec spec;
    std::vector<TransformConstPtr> originals;
    std::size_t j = i;
    if (j < ts.size()) {
        if (const auto* r = dynamic_cast<const Resize*>(ts[j].get())) {
            spec.resize = true;
            spec.height = static_cast<std::size_t>(r->height());
            spec.width = static_cast<std::size_t>(r->width());
            spec.mode = r->interp() == ResizeInterp::Nearest ? simd::ResampleMode::Nearest
                                                             : simd::ResampleMode::Bilinear;
            originals.push_back(ts[j]);
            ++j;
        }
    }
    bool to_float = false;
    if (j < ts.size() && dynamic_cast<const ToTensor*>(ts[j].get()) != nullptr) {
        originals.push_back(ts[j]);
        ++j;
        to_float = true;
    }
    if (j < ts.size()) {
        if (const auto* nm = dynamic_cast<const NormalizeImage*>(ts[j].get())) {
            spec.mean = nm->mean();
            spec.stddev = nm->stddev();
            originals.push_back(ts[j]);
            ++j;
            to_float = true;
        }
    }
    if (!to_float || originals.size() < 2) {
        return std::nullopt;
    }
    return std::make_pair(
        TransformConstPtr(std::make_shared<FusedImageTransform>(std::move(spec),
                                                                std::move(originals))),
        j);
}

} // namespace

namespace detail {

std::vector<TransformConstPtr> compile_transform_plan(
    const std::vector<TransformConstPtr>& transforms) {
    std::vector<TransformConstPtr> plan;
    plan.reserve(transforms.size());
    std::size_t i = 0;
    while (i < transforms.size()) {
        const TransformConstPtr& t = transforms[i];
        if (!t) {
            ++i;
            continue;
        }
        if (auto img = match_image_chain(transforms, i)) {
            plan.push_back(std::move(img->first));
            i = img->second;
            continue;
        }
        std::vector<FuseStepPtr> steps;
        std::size_t j = i;
        while (j < transforms.size()) {
            const TransformConstPtr& u = transforms[j];
            if (!u) {
                ++j;
                continue;
            }
            if (u->is_random()) {
                break;
            }
            auto step = u->fuse_step();
            if (!step) {
                break;
            }
            steps.push_back(std::move(step));
            ++j;
        }
        if (steps.size() >= 2) {
            plan.push_back(std::make_shared<FusedTransform>(std::move(steps)));
            i = j;
            continue;
        }
        plan.push_back(t);
        ++i;
    }
    return plan;
}

} // namespace detail

} // namespace nexusdata
