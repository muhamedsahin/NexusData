#include "nexusdata/backend/gpu_executor.hpp"

#include <algorithm>
#include <cstring>
#include <string>

#include "nexusdata/backend/device.hpp"
#include "nexusdata/core/error.hpp"
#include "nexusdata/simd/image_ops.hpp"

namespace nexusdata {

struct GpuExecutor::DevicePool {
    struct Entry {
        void* ptr = nullptr;
        CudaEventHandle fence = nullptr;
    };

    int device = 0;
    std::size_t max_cached = 0;
    std::mutex mu;
    std::map<std::size_t, std::vector<Entry>> free;
    std::size_t cached = 0;
    std::size_t hits = 0;
    std::size_t misses = 0;
    bool closed = false;

    static void destroy(const Entry& e) noexcept {
        cuda_api::event_destroy(e.fence);
        cuda_api::device_free(e.ptr);
    }

    void give_back(void* ptr, std::size_t cls, CudaEventHandle fence) noexcept {
        bool fenced = false;
        try {
            cuda_api::set_device(device);
            // Legacy default stream: completes after all prior work on blocking streams,
            // i.e. after whatever the consumer enqueued against this buffer.
            cuda_api::event_record(fence, nullptr);
            fenced = true;
        } catch (...) {
        }
        {
            std::lock_guard lock(mu);
            if (fenced && !closed && cached + cls <= max_cached) {
                free[cls].push_back(Entry{ptr, fence});
                cached += cls;
                return;
            }
        }
        destroy(Entry{ptr, fence}); // cudaFree synchronizes: safe without a fence
    }

    void trim() noexcept {
        std::map<std::size_t, std::vector<Entry>> drop;
        {
            std::lock_guard lock(mu);
            drop.swap(free);
            cached = 0;
        }
        for (auto& [cls, list] : drop) {
            for (const Entry& e : list) destroy(e);
        }
    }
};

bool GpuExecutor::usable(int device) noexcept {
    try {
        return device >= 0 && cuda_api::available() && device < cuda_api::device_count();
    } catch (...) {
        return false;
    }
}

GpuExecutor& GpuExecutor::instance(int device) {
    static std::mutex mu;
    static std::map<int, GpuExecutor*> executors;
    if (!usable(device)) {
        throw InvalidArgumentError("GpuExecutor: CUDA device " + std::to_string(device) +
                                   " is not available");
    }
    std::lock_guard lock(mu);
    GpuExecutor*& ex = executors[device];
    if (ex == nullptr) {
        GpuExecutorOptions opt;
        opt.device = device;
        ex = new GpuExecutor(opt);
    }
    return *ex;
}

GpuExecutor::GpuExecutor(GpuExecutorOptions options) : opt_(options) {
    if (!usable(opt_.device)) {
        throw InvalidArgumentError("GpuExecutor: CUDA device unavailable");
    }
    opt_.max_lanes = std::max<std::size_t>(1, opt_.max_lanes);
    dev_pool_ = std::make_shared<DevicePool>();
    dev_pool_->device = opt_.device;
    dev_pool_->max_cached = opt_.max_cached_device_bytes;
    pinned_pool_ = BufferPool::pinned(opt_.max_cached_pinned_bytes);
}

GpuExecutor::~GpuExecutor() {
    std::lock_guard lock(mu_);
    for (auto& lane : lanes_) {
        try {
            cuda_api::stream_synchronize(lane->h2d);
            cuda_api::stream_synchronize(lane->compute);
            cuda_api::stream_synchronize(lane->d2h);
        } catch (...) {
        }
        lane->scratch.clear();
        for (auto e : lane->uploaded) cuda_api::event_destroy(e);
        for (auto e : lane->computed) cuda_api::event_destroy(e);
        cuda_api::stream_destroy(lane->h2d);
        cuda_api::stream_destroy(lane->compute);
        cuda_api::stream_destroy(lane->d2h);
    }
    {
        std::lock_guard pool_lock(dev_pool_->mu);
        dev_pool_->closed = true; // outstanding buffers are freed on release
    }
    dev_pool_->trim();
}

GpuExecutor::Lease& GpuExecutor::Lease::operator=(Lease&& o) noexcept {
    if (this != &o) {
        if (ex_ != nullptr) ex_->release(lane_);
        ex_ = o.ex_;
        lane_ = o.lane_;
        o.ex_ = nullptr;
        o.lane_ = nullptr;
    }
    return *this;
}

GpuExecutor::Lease::~Lease() {
    if (ex_ != nullptr) ex_->release(lane_);
}

GpuExecutor::Lease GpuExecutor::acquire() {
    std::unique_lock lock(mu_);
    waiting_.fetch_add(1);
    cv_.wait(lock, [&] { return !free_lanes_.empty() || lanes_.size() < opt_.max_lanes; });
    waiting_.fetch_sub(1);
    GpuLane* lane = nullptr;
    if (!free_lanes_.empty()) {
        lane = free_lanes_.back();
        free_lanes_.pop_back();
    } else {
        cuda_api::set_device(opt_.device);
        auto fresh = std::make_unique<GpuLane>();
        fresh->h2d = cuda_api::stream_create();
        fresh->compute = cuda_api::stream_create();
        fresh->d2h = cuda_api::stream_create();
        for (auto& e : fresh->uploaded) e = cuda_api::event_create(false);
        for (auto& e : fresh->computed) e = cuda_api::event_create(false);
        lane = fresh.get();
        lanes_.push_back(std::move(fresh));
    }
    busy_.fetch_add(1);
    return Lease(this, lane);
}

void GpuExecutor::release(GpuLane* lane) noexcept {
    {
        std::lock_guard lock(mu_);
        free_lanes_.push_back(lane);
        busy_.fetch_sub(1);
    }
    cv_.notify_one();
}

AlignedBuffer GpuExecutor::device_buffer(std::size_t bytes, CudaStreamHandle stream) {
    const std::size_t cls = BufferPool::size_class(std::max<std::size_t>(bytes, 1));
    DevicePool::Entry e;
    {
        std::lock_guard lock(dev_pool_->mu);
        auto it = dev_pool_->free.find(cls);
        if (it != dev_pool_->free.end() && !it->second.empty()) {
            e = it->second.back();
            it->second.pop_back();
            dev_pool_->cached -= cls;
            ++dev_pool_->hits;
        } else {
            ++dev_pool_->misses;
        }
    }
    cuda_api::set_device(opt_.device);
    if (e.ptr != nullptr) {
        cuda_api::stream_wait_event(stream, e.fence);
    } else {
        try {
            e.ptr = cuda_api::device_alloc(cls);
        } catch (const Error&) {
            dev_pool_->trim(); // idle cached blocks may be what is exhausting VRAM
            e.ptr = cuda_api::device_alloc(cls);
        }
        e.fence = cuda_api::event_create(false);
    }
    std::shared_ptr<DevicePool> pool = dev_pool_;
    CudaEventHandle fence = e.fence;
    return AlignedBuffer(e.ptr, [pool, cls, fence](void* p) { pool->give_back(p, cls, fence); });
}

NDArray GpuExecutor::device_array(Shape shape, DType dtype, CudaStreamHandle stream) {
    const std::size_t bytes = numel(shape) * size_of(dtype);
    return NDArray::from_shared(device_buffer(bytes, stream), std::move(shape), dtype,
                                Device::cuda(opt_.device));
}

NDArray GpuExecutor::pinned_array(Shape shape, DType dtype) {
    const std::size_t bytes = numel(shape) * size_of(dtype);
    return NDArray::from_shared(pinned_pool_->acquire(bytes), std::move(shape), dtype,
                                Device::host());
}

const void* GpuExecutor::upload_aux(GpuLane& lane, const void* host, std::size_t bytes) {
    AlignedBuffer buf = device_buffer(bytes, lane.compute);
    if (bytes > 0) {
        // Pageable source: the call returns once the bytes are staged, so @p host may die after it.
        cuda_api::memcpy_h2d(buf.get(), host, bytes, lane.compute);
    }
    const void* p = buf.get();
    lane.scratch.push_back(std::move(buf));
    return p;
}

NDArray GpuExecutor::run_rows(GpuLane& lane, const GpuRowJob& job, bool output_on_device) {
    if (job.input == nullptr || !job.kernel) {
        throw InvalidArgumentError("GpuExecutor::run_rows: missing input or kernel");
    }
    const NDArray& in = *job.input;
    if (in.shape().empty() || job.output_shape.empty() || job.output_shape[0] != in.shape()[0]) {
        throw ShapeError("GpuExecutor::run_rows: input and output need the same leading dimension");
    }
    const bool host_in = in.device().is_host();
    if (!host_in && in.device().index != opt_.device) {
        throw InvalidArgumentError("GpuExecutor::run_rows: input lives on another CUDA device");
    }
    cuda_api::set_device(opt_.device);

    const std::size_t rows_total = in.shape()[0];
    NDArray out_dev = device_array(job.output_shape, job.output_dtype, lane.compute);
    NDArray out_host;
    if (!output_on_device) {
        out_host = pinned_array(job.output_shape, job.output_dtype);
    }
    if (rows_total == 0 || out_dev.nbytes() == 0) {
        return output_on_device ? out_dev : out_host;
    }
    const std::size_t in_row = in.nbytes() / rows_total;
    const std::size_t out_row = out_dev.nbytes() / rows_total;

    AlignedBuffer d_in_buf;
    AlignedBuffer stage;
    const auto* h_in = static_cast<const std::uint8_t*>(in.data());
    const std::uint8_t* d_in = static_cast<const std::uint8_t*>(in.data());
    if (host_in && in.nbytes() > 0) {
        d_in_buf = device_buffer(in.nbytes(), lane.h2d);
        d_in = static_cast<const std::uint8_t*>(d_in_buf.get());
        if (!cuda_api::is_host_pinned(in.data())) {
            stage = pinned_pool_->acquire(in.nbytes());
        }
    }
    auto* d_out = static_cast<std::uint8_t*>(out_dev.data());
    auto* h_out = output_on_device ? nullptr : static_cast<std::uint8_t*>(out_host.data());

    std::size_t chunks = std::clamp<std::size_t>(in.nbytes() / std::max<std::size_t>(opt_.min_chunk_bytes, 1),
                                                 1, GpuLane::kMaxChunks);
    chunks = std::min(chunks, rows_total);
    const std::size_t rows_per = (rows_total + chunks - 1) / chunks;
    chunks = (rows_total + rows_per - 1) / rows_per;

    try {
        for (std::size_t i = 0; i < chunks; ++i) {
            const std::size_t r0 = i * rows_per;
            const std::size_t rows = std::min(rows_per, rows_total - r0);
            const std::size_t in_off = r0 * in_row;
            const std::size_t out_off = r0 * out_row;
            if (host_in && in_row > 0) {
                const void* src = h_in + in_off;
                if (stage) {
                    // Host copy of chunk i overlaps the DMA of chunk i-1.
                    auto* s = static_cast<std::uint8_t*>(stage.get()) + in_off;
                    std::memcpy(s, src, rows * in_row);
                    src = s;
                }
                cuda_api::memcpy_h2d(const_cast<std::uint8_t*>(d_in) + in_off, src, rows * in_row,
                                     lane.h2d);
                cuda_api::event_record(lane.uploaded[i], lane.h2d);
                cuda_api::stream_wait_event(lane.compute, lane.uploaded[i]);
            }
            job.kernel(r0, rows, d_in + in_off, d_out + out_off, lane.compute);
            if (!output_on_device) {
                cuda_api::event_record(lane.computed[i], lane.compute);
                cuda_api::stream_wait_event(lane.d2h, lane.computed[i]);
                cuda_api::memcpy_d2h(h_out + out_off, d_out + out_off, rows * out_row, lane.d2h);
            }
        }
        cuda_api::stream_synchronize(lane.h2d);
        cuda_api::stream_synchronize(lane.compute);
        cuda_api::stream_synchronize(lane.d2h);
    } catch (...) {
        try {
            cuda_api::stream_synchronize(lane.h2d);
            cuda_api::stream_synchronize(lane.compute);
            cuda_api::stream_synchronize(lane.d2h);
        } catch (...) {
        }
        lane.scratch.clear();
        throw;
    }
    lane.scratch.clear();
    return output_on_device ? out_dev : out_host;
}

GpuExecutor::Stats GpuExecutor::stats() const {
    Stats s;
    {
        std::lock_guard lock(dev_pool_->mu);
        s.device_hits = dev_pool_->hits;
        s.device_misses = dev_pool_->misses;
        s.cached_device_bytes = dev_pool_->cached;
    }
    std::lock_guard lock(mu_);
    s.lanes_created = lanes_.size();
    return s;
}

// --- legacy high-level API ---------------------------------------------------------

NDArray batch_to_tensor_normalize(const NDArray& hwc_u8_batch, const std::vector<float>& mean,
                                  const std::vector<float>& stddev, Device device,
                                  CudaStreamHandle stream) {
    const Device resolved =
        resolve_device(device, GpuOpKind::NormalizeCollate, hwc_u8_batch.nbytes());
    if (hwc_u8_batch.dtype() != DType::UInt8) {
        throw ShapeError("batch_to_tensor_normalize: expected UInt8 HWC");
    }
    const Shape& s = hwc_u8_batch.shape();
    if (s.size() != 3 && s.size() != 4) {
        throw ShapeError("batch_to_tensor_normalize: expected [N,H,W,C] or [H,W,C]");
    }
    const bool single = s.size() == 3;
    const std::size_t n = single ? 1 : s[0];
    const std::size_t h = s[s.size() - 3];
    const std::size_t w = s[s.size() - 2];
    const std::size_t c = s[s.size() - 1];
    if (mean.size() != c || stddev.size() != c) {
        throw ShapeError("batch_to_tensor_normalize: mean/std channel mismatch");
    }
    const Shape out_shape = single ? Shape{c, h, w} : Shape{n, c, h, w};

    if (!resolved.is_cuda()) {
        const NDArray src = hwc_u8_batch.device().is_host() ? hwc_u8_batch : hwc_u8_batch.cpu();
        NDArray out(out_shape, DType::Float32);
        const auto* in = src.data<std::uint8_t>();
        auto* dst = out.data<float>();
        for (std::size_t i = 0; i < n; ++i) {
            simd::hwc_u8_to_chw_f32(in + i * h * w * c, dst + i * c * h * w, h, w, c, mean.data(),
                                    stddev.data());
        }
        return out;
    }

    GpuExecutor& ex = GpuExecutor::instance(resolved.index);
    const int ni = static_cast<int>(n), hi = static_cast<int>(h), wi = static_cast<int>(w),
              ci = static_cast<int>(c);
    if (stream != nullptr) {
        // Caller-owned stream: plain sequential H2D -> kernel on that stream.
        cuda_api::set_device(ex.device());
        NDArray out = ex.device_array(out_shape, DType::Float32, stream);
        NDArray d_in = hwc_u8_batch;
        if (hwc_u8_batch.device().is_host()) {
            d_in = ex.device_array(hwc_u8_batch.shape(), DType::UInt8, stream);
            cuda_api::memcpy_h2d(d_in.data(), hwc_u8_batch.data(), hwc_u8_batch.nbytes(), stream);
        }
        cuda_api::launch_hwc_u8_to_chw_f32(static_cast<const std::uint8_t*>(d_in.data()),
                                           static_cast<float*>(out.data()), ni, hi, wi, ci,
                                           mean.data(), stddev.data(), stream);
        cuda_api::stream_synchronize(stream);
        return out;
    }

    NDArray batched = hwc_u8_batch;
    if (single) {
        batched = NDArray::from_shared(hwc_u8_batch.shared_storage(), Shape{1, h, w, c},
                                       DType::UInt8, hwc_u8_batch.device());
    }
    GpuRowJob job;
    job.input = &batched;
    job.output_shape = Shape{n, c, h, w};
    job.output_dtype = DType::Float32;
    job.kernel = [&](std::size_t, std::size_t rows, const void* d_in, void* d_out,
                     CudaStreamHandle st) {
        cuda_api::launch_hwc_u8_to_chw_f32(static_cast<const std::uint8_t*>(d_in),
                                           static_cast<float*>(d_out), static_cast<int>(rows), hi,
                                           wi, ci, mean.data(), stddev.data(), st);
    };
    auto lease = ex.acquire();
    NDArray out = ex.run_rows(lease.lane(), job, /*output_on_device=*/true);
    if (single) {
        out.reshape(out_shape);
    }
    return out;
}

} // namespace nexusdata
