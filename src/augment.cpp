#include "nexusdata/pipeline/augment.hpp"

#include <algorithm>
#include <string>

#include "nexusdata/backend/gpu_executor.hpp"
#include "nexusdata/backend/philox.hpp"
#include "nexusdata/backend/thread_pool.hpp"
#include "nexusdata/core/allocator.hpp"
#include "nexusdata/core/error.hpp"

namespace nexusdata {

namespace {

constexpr std::uint32_t kAugmentStream = 0x41554721u; // decorrelates from other Philox users

std::uint32_t bounded(std::uint32_t u, std::uint32_t range) {
    return static_cast<std::uint32_t>((static_cast<std::uint64_t>(u) * range) >> 32);
}

float unit(std::uint32_t u) { return static_cast<float>(u >> 8) * (1.0f / 16777216.0f); }

NDArray alloc_uninit(const Shape& shape, DType dtype) {
    const std::size_t bytes = numel(shape) * size_of(dtype);
    if (bytes == 0) {
        return NDArray(shape, dtype);
    }
    return NDArray::from_shared(make_aligned_buffer(bytes), shape, dtype, Device::host());
}

void require_nhwc_u8(const NDArray& x, const char* who) {
    if (x.dtype() != DType::UInt8 || x.shape().size() != 4) {
        throw ShapeError(std::string(who) + ": expected [N, H, W, C] uint8, got " +
                         shape_to_string(x.shape()) + " " + std::string(to_string(x.dtype())));
    }
}

} // namespace

FusedAugment::FusedAugment(AugmentSpec spec, std::uint64_t seed)
    : spec_(std::move(spec)), seed_(seed) {
    if (spec_.mean.size() != spec_.stddev.size()) {
        throw InvalidArgumentError("FusedAugment: mean/std size mismatch");
    }
    if (!spec_.mean.empty() && !spec_.to_tensor) {
        throw InvalidArgumentError("FusedAugment: normalization requires to_tensor");
    }
    if (spec_.mean.size() > static_cast<std::size_t>(kernels::kMaxChannels)) {
        throw InvalidArgumentError("FusedAugment: at most 16 channels");
    }
    if (spec_.crop != AugmentSpec::Crop::None && (spec_.crop_height == 0 || spec_.crop_width == 0)) {
        throw InvalidArgumentError("FusedAugment: crop size must be positive");
    }
    if ((spec_.out_height == 0) != (spec_.out_width == 0)) {
        throw InvalidArgumentError("FusedAugment: set both out_height and out_width (or neither)");
    }
    if (spec_.hflip_p < 0.0f || spec_.hflip_p > 1.0f || spec_.vflip_p < 0.0f || spec_.vflip_p > 1.0f) {
        throw InvalidArgumentError("FusedAugment: flip probabilities must be in [0, 1]");
    }
}

bool FusedAugment::is_random() const {
    auto random_p = [](float p) { return p > 0.0f && p < 1.0f; };
    return spec_.crop == AugmentSpec::Crop::Random || random_p(spec_.hflip_p) ||
           random_p(spec_.vflip_p);
}

void FusedAugment::set_seed(std::uint64_t seed) {
    seed_ = seed;
    counter_.store(0);
}

kernels::ImageAugmentParams FusedAugment::plan(int n, int sh, int sw, int c,
                                               std::vector<kernels::AugmentSample>& samples) const {
    if (c <= 0 || c > kernels::kMaxChannels) {
        throw ShapeError("FusedAugment: channel count must be in [1, 16]");
    }
    if (!spec_.mean.empty() && static_cast<std::size_t>(c) != spec_.mean.size()) {
        throw ShapeError("NormalizeImage: channel mismatch");
    }
    kernels::ImageAugmentParams p;
    p.n = n;
    p.sh = sh;
    p.sw = sw;
    p.c = c;
    p.crop_h = sh;
    p.crop_w = sw;
    if (spec_.crop != AugmentSpec::Crop::None) {
        p.crop_h = static_cast<int>(spec_.crop_height);
        p.crop_w = static_cast<int>(spec_.crop_width);
        if (p.crop_h > sh || p.crop_w > sw) {
            throw ShapeError("FusedAugment: image smaller than crop size");
        }
    }
    p.dh = spec_.out_height != 0 ? static_cast<int>(spec_.out_height) : p.crop_h;
    p.dw = spec_.out_width != 0 ? static_cast<int>(spec_.out_width) : p.crop_w;
    if (p.dh == p.crop_h && p.dw == p.crop_w) {
        p.resize = kernels::ResampleKind::None; // same-size resize is the identity
    } else {
        p.resize = spec_.interp == simd::ResampleMode::Nearest ? kernels::ResampleKind::Nearest
                                                               : kernels::ResampleKind::Bilinear;
    }
    p.layout = spec_.to_tensor ? kernels::ImageLayout::ChwF32 : kernels::ImageLayout::HwcU8;
    p.normalize = !spec_.mean.empty();
    for (std::size_t i = 0; i < spec_.mean.size(); ++i) {
        p.mean[i] = spec_.mean[i];
        p.stddev[i] = spec_.stddev[i];
    }

    samples.assign(static_cast<std::size_t>(n), kernels::AugmentSample{});
    const std::uint32_t max_y = static_cast<std::uint32_t>(sh - p.crop_h);
    const std::uint32_t max_x = static_cast<std::uint32_t>(sw - p.crop_w);
    const std::uint64_t base = counter_.fetch_add(static_cast<std::uint64_t>(n));
    for (int s = 0; s < n; ++s) {
        kernels::AugmentSample& a = samples[static_cast<std::size_t>(s)];
        const std::uint64_t idx = base + static_cast<std::uint64_t>(s);
        Philox4x32 rng(seed_);
        rng.set_counter(static_cast<std::uint32_t>(idx), static_cast<std::uint32_t>(idx >> 32),
                        kAugmentStream, 0u);
        std::uint32_t o[4];
        rng.next(o);
        if (spec_.crop == AugmentSpec::Crop::Center) {
            a.crop_y = static_cast<std::int32_t>(max_y / 2);
            a.crop_x = static_cast<std::int32_t>(max_x / 2);
        } else if (spec_.crop == AugmentSpec::Crop::Random) {
            a.crop_y = static_cast<std::int32_t>(bounded(o[0], max_y + 1));
            a.crop_x = static_cast<std::int32_t>(bounded(o[1], max_x + 1));
        }
        a.hflip = unit(o[2]) < spec_.hflip_p ? 1 : 0;
        a.vflip = unit(o[3]) < spec_.vflip_p ? 1 : 0;
    }
    return p;
}

Sample FusedAugment::apply(const Sample& sample) const {
    const NDArray& x = sample.input;
    if (!x.device().is_host() || x.dtype() != DType::UInt8 || x.shape().size() != 3) {
        throw ShapeError("FusedAugment: expected host [H, W, C] uint8 sample");
    }
    std::vector<kernels::AugmentSample> geo;
    kernels::ImageAugmentParams p =
        plan(1, static_cast<int>(x.shape()[0]), static_cast<int>(x.shape()[1]),
             static_cast<int>(x.shape()[2]), geo);
    const auto c = static_cast<std::size_t>(p.c);
    const auto dh = static_cast<std::size_t>(p.dh);
    const auto dw = static_cast<std::size_t>(p.dw);
    NDArray out = spec_.to_tensor ? alloc_uninit(Shape{c, dh, dw}, DType::Float32)
                                  : alloc_uninit(Shape{dh, dw, c}, DType::UInt8);
    p.src = static_cast<const std::uint8_t*>(x.data());
    p.dst = out.data();
    p.samples = geo.data();
    kernels::run_image_augment_host(p, 0, 1);
    return sample.with_input(std::move(out));
}

namespace {

Shape batch_out_shape(const kernels::ImageAugmentParams& p) {
    const auto n = static_cast<std::size_t>(p.n);
    const auto c = static_cast<std::size_t>(p.c);
    const auto dh = static_cast<std::size_t>(p.dh);
    const auto dw = static_cast<std::size_t>(p.dw);
    return p.layout == kernels::ImageLayout::ChwF32 ? Shape{n, c, dh, dw} : Shape{n, dh, dw, c};
}

} // namespace

Batch FusedAugment::apply_batch(const Batch& batch) const {
    const NDArray x = batch.inputs.device().is_host() ? batch.inputs : batch.inputs.cpu();
    require_nhwc_u8(x, "FusedAugment::apply_batch");
    std::vector<kernels::AugmentSample> geo;
    kernels::ImageAugmentParams p =
        plan(static_cast<int>(x.shape()[0]), static_cast<int>(x.shape()[1]),
             static_cast<int>(x.shape()[2]), static_cast<int>(x.shape()[3]), geo);
    const Shape shape = batch_out_shape(p);
    NDArray out = alloc_uninit(shape, spec_.to_tensor ? DType::Float32 : DType::UInt8);
    p.src = static_cast<const std::uint8_t*>(x.data());
    p.dst = out.data();
    p.samples = geo.data();
    const std::size_t per_sample = static_cast<std::size_t>(p.dh) * static_cast<std::size_t>(p.dw) *
                                   static_cast<std::size_t>(p.c);
    if (p.n <= 1 || per_sample * static_cast<std::size_t>(p.n) < (std::size_t{1} << 16)) {
        kernels::run_image_augment_host(p, 0, p.n);
    } else {
        ThreadPool& pool = default_thread_pool();
        const std::size_t tasks = std::min<std::size_t>(static_cast<std::size_t>(p.n),
                                                        std::max<std::size_t>(1, pool.size() * 2));
        const std::size_t per = (static_cast<std::size_t>(p.n) + tasks - 1) / tasks;
        pool.parallel_for(tasks, [&](std::size_t t) {
            const std::size_t b = t * per;
            const std::size_t e = std::min<std::size_t>(static_cast<std::size_t>(p.n), b + per);
            if (b < e) {
                kernels::run_image_augment_host(p, static_cast<int>(b), static_cast<int>(e));
            }
        });
    }
    Batch r = batch;
    r.inputs = std::move(out);
    return r;
}

Batch FusedAugment::apply_batch_gpu(const Batch& batch, GpuBatchContext& ctx) const {
    const NDArray& x = batch.inputs;
    require_nhwc_u8(x, "FusedAugment::apply_batch_gpu");
    std::vector<kernels::AugmentSample> geo;
    kernels::ImageAugmentParams base =
        plan(static_cast<int>(x.shape()[0]), static_cast<int>(x.shape()[1]),
             static_cast<int>(x.shape()[2]), static_cast<int>(x.shape()[3]), geo);
    const auto* d_geo = static_cast<const kernels::AugmentSample*>(ctx.executor->upload_aux(
        *ctx.lane, geo.data(), geo.size() * sizeof(kernels::AugmentSample)));

    GpuRowJob job;
    job.input = &x;
    job.output_shape = batch_out_shape(base);
    job.output_dtype = spec_.to_tensor ? DType::Float32 : DType::UInt8;
    job.kernel = [&](std::size_t r0, std::size_t rows, const void* d_in, void* d_out,
                     CudaStreamHandle stream) {
        kernels::ImageAugmentParams p = base;
        p.src = static_cast<const std::uint8_t*>(d_in);
        p.dst = d_out;
        p.n = static_cast<int>(rows);
        p.samples = d_geo + r0;
        kernels::launch_image_augment_cuda(p, stream);
    };
    Batch r = batch;
    r.inputs = ctx.executor->run_rows(*ctx.lane, job, ctx.output_on_device);
    return r;
}

} // namespace nexusdata
