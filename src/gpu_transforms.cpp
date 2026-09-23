// Batch-level CPU / CUDA execution of the built-in transforms (the per-sample
// paths live in fusion.cpp / augment.cpp).

#include <algorithm>
#include <cstring>
#include <string>

#include "nexusdata/backend/gpu_executor.hpp"
#include "nexusdata/backend/kernels.hpp"
#include "nexusdata/backend/thread_pool.hpp"
#include "nexusdata/core/allocator.hpp"
#include "nexusdata/core/error.hpp"
#include "nexusdata/pipeline/fusion.hpp"

namespace nexusdata {

namespace {

NDArray host_view(const NDArray& x) { return x.device().is_host() ? x : x.cpu(); }

NDArray alloc_uninit(const Shape& shape, DType dtype) {
    const std::size_t bytes = numel(shape) * size_of(dtype);
    if (bytes == 0) {
        return NDArray(shape, dtype);
    }
    return NDArray::from_shared(make_aligned_buffer(bytes), shape, dtype, Device::host());
}

/// Steps whose CUDA kernel reproduces the CPU bits exactly.
bool gpu_kind(const FuseStep& s) {
    return s.kind != FuseStep::Kind::Custom && s.kind != FuseStep::Kind::Log1p;
}

kernels::TabularStepKind tabular_kind(const FuseStep& s) {
    switch (s.kind) {
        case FuseStep::Kind::Clip: return kernels::TabularStepKind::Clip;
        case FuseStep::Kind::Log1p: return kernels::TabularStepKind::Log1p;
        case FuseStep::Kind::Standardize: return kernels::TabularStepKind::Standardize;
        case FuseStep::Kind::MinMax: return kernels::TabularStepKind::MinMax;
        case FuseStep::Kind::ScalarOp:
            switch (s.op) {
                case simd::BinaryOp::Add: return kernels::TabularStepKind::ScalarAdd;
                case simd::BinaryOp::Sub: return kernels::TabularStepKind::ScalarSub;
                case simd::BinaryOp::Mul: return kernels::TabularStepKind::ScalarMul;
                case simd::BinaryOp::Div: return kernels::TabularStepKind::ScalarDiv;
            }
            break;
        case FuseStep::Kind::Custom:
            break;
    }
    throw InvalidArgumentError("FusedTransform: step '" + s.name + "' has no CUDA kernel");
}

} // namespace

// --- defaults ---------------------------------------------------------------------------

bool Transform::supports_gpu_batch() const {
    const auto step = fuse_step();
    return step != nullptr && gpu_kind(*step);
}

Batch Transform::apply_batch_gpu(const Batch& batch, GpuBatchContext& ctx) const {
    auto step = fuse_step();
    if (!step || !gpu_kind(*step)) {
        throw InvalidArgumentError(
            "Transform::apply_batch_gpu: transform has no GPU implementation "
            "(check supports_gpu_batch())");
    }
    return FusedTransform({std::move(step)}).apply_batch_gpu(batch, ctx);
}

Batch Compose::apply_batch_gpu(const Batch& batch, GpuBatchContext& ctx) const {
    if (!supports_gpu_batch()) {
        throw InvalidArgumentError("Compose::apply_batch_gpu: a step has no GPU implementation");
    }
    std::vector<const Transform*> steps;
    for (const auto& t : plan_) {
        if (t) steps.push_back(t.get());
    }
    Batch out = batch;
    GpuBatchContext inner = ctx;
    for (std::size_t i = 0; i < steps.size(); ++i) {
        inner.output_on_device = i + 1 < steps.size() ? true : ctx.output_on_device;
        out = steps[i]->apply_batch_gpu(out, inner);
    }
    return out;
}

namespace detail {

Batch apply_rowwise(const Transform& t, const Batch& batch) {
    const NDArray x = host_view(batch.inputs);
    if (x.shape().empty()) {
        throw ShapeError("apply_rowwise: inputs must be [B, ...]");
    }
    const std::size_t rows = x.shape()[0];
    const Shape row_shape(x.shape().begin() + 1, x.shape().end());
    const std::size_t row_bytes = rows == 0 ? 0 : x.nbytes() / rows;
    std::vector<NDArray> outs(rows);
    for (std::size_t i = 0; i < rows; ++i) {
        NDArray row(row_shape, x.dtype());
        if (row_bytes > 0) {
            std::memcpy(row.data(), static_cast<const std::uint8_t*>(x.data()) + i * row_bytes,
                        row_bytes);
        }
        Sample s;
        s.input = std::move(row);
        outs[i] = t.apply(s).input;
    }
    Batch r = batch;
    if (rows == 0) {
        return r;
    }
    Shape out_shape{rows};
    for (std::size_t d : outs[0].shape()) out_shape.push_back(d);
    NDArray stacked = NDArray::stack(outs);
    stacked.reshape(out_shape);
    r.inputs = std::move(stacked);
    return r;
}

Batch apply_batch_on_host(const Transform& t, const Batch& batch, const GpuBatchContext& ctx) {
    Batch in = batch;
    in.inputs = host_view(batch.inputs);
    Batch out = t.apply_batch(in);
    if (ctx.output_on_device && ctx.executor != nullptr) {
        out.inputs = out.inputs.cuda(ctx.executor->device());
    }
    return out;
}

} // namespace detail

// --- FusedTransform ---------------------------------------------------------------------

bool FusedTransform::supports_gpu_batch() const {
    if (steps_.size() > static_cast<std::size_t>(kernels::kMaxTabularSteps)) {
        return false;
    }
    return std::all_of(steps_.begin(), steps_.end(), [](const FuseStepPtr& s) { return gpu_kind(*s); });
}

Batch FusedTransform::apply_batch_gpu(const Batch& batch, GpuBatchContext& ctx) const {
    const NDArray& x = batch.inputs;
    if (x.numel() == 0) {
        return batch;
    }
    if (x.shape().empty() || x.shape()[0] == 0) {
        throw ShapeError("FusedTransform::apply_batch_gpu: inputs must be [B, ...]");
    }
    const std::size_t row_len = x.numel() / x.shape()[0];
    check_row(row_len);
    const bool dtype_ok = x.dtype() == DType::Float32 || x.dtype() == DType::Float64;
    if (!dtype_ok || !supports_gpu_batch()) {
        return detail::apply_batch_on_host(*this, batch, ctx);
    }

    kernels::TabularParams base;
    base.dtype = x.dtype() == DType::Float32 ? kernels::TabularDType::F32 : kernels::TabularDType::F64;
    base.row_len = row_len;
    base.num_steps = static_cast<int>(steps_.size());
    for (std::size_t i = 0; i < steps_.size(); ++i) {
        const FuseStep& s = *steps_[i];
        kernels::TabularStep& k = base.steps[i];
        k.kind = tabular_kind(s);
        k.a = s.a;
        k.b = s.b;
        if (s.kind == FuseStep::Kind::Standardize || s.kind == FuseStep::Kind::MinMax) {
            const std::size_t bytes = s.p0.size() * sizeof(double);
            k.p0 = static_cast<const double*>(ctx.executor->upload_aux(*ctx.lane, s.p0.data(), bytes));
            k.p1 = static_cast<const double*>(ctx.executor->upload_aux(*ctx.lane, s.p1.data(), bytes));
        }
    }

    GpuRowJob job;
    job.input = &x;
    job.output_shape = x.shape();
    job.output_dtype = x.dtype();
    job.kernel = [&](std::size_t, std::size_t rows, const void* d_in, void* d_out,
                     CudaStreamHandle stream) {
        kernels::TabularParams p = base;
        p.src = d_in;
        p.dst = d_out;
        p.n = rows * row_len;
        kernels::launch_tabular_cuda(p, stream);
    };
    Batch out = batch;
    out.inputs = ctx.executor->run_rows(*ctx.lane, job, ctx.output_on_device);
    return out;
}

// --- FusedImageTransform ----------------------------------------------------------------

bool FusedImageTransform::batch_fast_path(const NDArray& x) const {
    if (x.dtype() != DType::UInt8 || x.shape().size() != 4 || x.shape()[1] == 0 ||
        x.shape()[2] == 0) {
        return false;
    }
    const std::size_t c = x.shape()[3];
    if (c == 0 || c > static_cast<std::size_t>(kernels::kMaxChannels)) {
        return false;
    }
    if (!spec_.mean.empty() && c != spec_.mean.size()) {
        throw ShapeError("NormalizeImage: channel mismatch");
    }
    return true;
}

Batch FusedImageTransform::apply_batch(const Batch& batch) const {
    const NDArray x = host_view(batch.inputs);
    if (!batch_fast_path(x)) {
        return detail::apply_rowwise(*this, batch);
    }
    const std::size_t n = x.shape()[0], h = x.shape()[1], w = x.shape()[2], c = x.shape()[3];
    const std::size_t dh = spec_.resize ? spec_.height : h;
    const std::size_t dw = spec_.resize ? spec_.width : w;
    NDArray out = alloc_uninit(Shape{n, c, dh, dw}, DType::Float32);
    const auto* src = static_cast<const std::uint8_t*>(x.data());
    auto* dst = static_cast<float*>(out.data());
    const bool norm = !spec_.mean.empty();
    const float* mean = norm ? spec_.mean.data() : nullptr;
    const float* stdv = norm ? spec_.stddev.data() : nullptr;
    auto run = [&](std::size_t i) {
        const std::uint8_t* s = src + i * h * w * c;
        float* d = dst + i * c * dh * dw;
        if (spec_.resize) {
            simd::resize_to_chw_f32(s, h, w, c, d, dh, dw, spec_.mode, mean, stdv);
        } else {
            simd::hwc_u8_to_chw_f32(s, d, h, w, c, mean, stdv);
        }
    };
    if (n <= 1 || n * c * dh * dw < (std::size_t{1} << 16)) {
        for (std::size_t i = 0; i < n; ++i) run(i);
    } else {
        default_thread_pool().parallel_for(n, run);
    }
    Batch r = batch;
    r.inputs = std::move(out);
    return r;
}

Batch FusedImageTransform::apply_batch_gpu(const Batch& batch, GpuBatchContext& ctx) const {
    const NDArray& x = batch.inputs;
    if (!batch_fast_path(x)) {
        return detail::apply_batch_on_host(*this, batch, ctx);
    }
    const int h = static_cast<int>(x.shape()[1]);
    const int w = static_cast<int>(x.shape()[2]);
    const int c = static_cast<int>(x.shape()[3]);
    const int dh = spec_.resize ? static_cast<int>(spec_.height) : h;
    const int dw = spec_.resize ? static_cast<int>(spec_.width) : w;
    const bool norm = !spec_.mean.empty();

    kernels::ImageAugmentParams base;
    base.sh = h;
    base.sw = w;
    base.c = c;
    base.crop_h = h;
    base.crop_w = w;
    base.dh = dh;
    base.dw = dw;
    base.layout = kernels::ImageLayout::ChwF32;
    base.normalize = norm;
    for (std::size_t i = 0; i < spec_.mean.size(); ++i) {
        base.mean[i] = spec_.mean[i];
        base.stddev[i] = spec_.stddev[i];
    }
    const bool plain = !spec_.resize || (dh == h && dw == w);
    if (!plain) {
        base.resize = spec_.mode == simd::ResampleMode::Nearest ? kernels::ResampleKind::Nearest
                                                                : kernels::ResampleKind::Bilinear;
    }

    GpuRowJob job;
    job.input = &x;
    job.output_shape = Shape{x.shape()[0], static_cast<std::size_t>(c), static_cast<std::size_t>(dh),
                             static_cast<std::size_t>(dw)};
    job.output_dtype = DType::Float32;
    job.kernel = [&](std::size_t, std::size_t rows, const void* d_in, void* d_out,
                     CudaStreamHandle stream) {
        if (plain) {
            // No resampling: the shared-memory transpose kernel.
            cuda_api::launch_hwc_u8_to_chw_f32(static_cast<const std::uint8_t*>(d_in),
                                               static_cast<float*>(d_out), static_cast<int>(rows),
                                               h, w, c, norm ? spec_.mean.data() : nullptr,
                                               norm ? spec_.stddev.data() : nullptr, stream);
            return;
        }
        kernels::ImageAugmentParams p = base;
        p.src = static_cast<const std::uint8_t*>(d_in);
        p.dst = d_out;
        p.n = static_cast<int>(rows);
        kernels::launch_image_augment_cuda(p, stream);
    };
    Batch r = batch;
    r.inputs = ctx.executor->run_rows(*ctx.lane, job, ctx.output_on_device);
    return r;
}

} // namespace nexusdata
