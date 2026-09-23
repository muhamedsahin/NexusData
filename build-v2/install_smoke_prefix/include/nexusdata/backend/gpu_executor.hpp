#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "nexusdata/backend/cuda_api.hpp"
#include "nexusdata/core/ndarray.hpp"
#include "nexusdata/core/object_pool.hpp"

namespace nexusdata {

/// One set of streams for an in-flight GPU batch: uploads, kernels and
/// downloads run on separate streams ordered by per-chunk events, so the H2D
/// copy of chunk i+1 overlaps the kernel of chunk i and the D2H copy of chunk i-1.
struct GpuLane {
    static constexpr std::size_t kMaxChunks = 8;
    CudaStreamHandle h2d = nullptr;
    CudaStreamHandle compute = nullptr;
    CudaStreamHandle d2h = nullptr;
    std::array<CudaEventHandle, kMaxChunks> uploaded{};
    std::array<CudaEventHandle, kMaxChunks> computed{};
    /// Device buffers that must live until the lane's current batch completes.
    std::vector<AlignedBuffer> scratch;
};

struct GpuExecutorOptions {
    int device = 0;
    /// Concurrent batches (lanes); acquire() blocks beyond this.
    std::size_t max_lanes = 4;
    /// Batches are split into chunks of at least this many input bytes (up to kMaxChunks).
    std::size_t min_chunk_bytes = std::size_t{1} << 20;
    std::size_t max_cached_device_bytes = std::size_t{1} << 30;
    std::size_t max_cached_pinned_bytes = std::size_t{512} << 20;
};

class GpuExecutor;

/// Execution context handed to Transform::apply_batch_gpu().
struct GpuBatchContext {
    GpuExecutor* executor = nullptr;
    GpuLane* lane = nullptr;
    /// Leave the result in device memory (true) or download it into pinned host memory.
    bool output_on_device = false;
};

/// Row-parallel batch job: rows (samples) are independent, so the batch can be
/// cut into chunks along axis 0 and pipelined through the lane's streams.
struct GpuRowJob {
    /// [B, ...] input on host (pageable or pinned) or on this executor's device.
    const NDArray* input = nullptr;
    Shape output_shape;
    DType output_dtype = DType::Float32;
    /// Enqueue the kernel for rows [row0, row0 + rows) on @p stream. d_in / d_out
    /// point at the first row of the chunk.
    std::function<void(std::size_t row0, std::size_t rows, const void* d_in, void* d_out,
                       CudaStreamHandle stream)>
        kernel;
};

/// Process-wide GPU work scheduler for data-loading kernels (one per device).
///
/// Device memory is recycled by size class. A buffer released by the host is
/// fenced with an event on the legacy default stream and reused only after that
/// event, so arrays handed to users stay safe as long as their consumers' work
/// is ordered before the release (e.g. PyTorch's default stream).
///
/// Thread-safety: all methods are thread-safe; a lane is used by one thread at a time.
class GpuExecutor {
public:
    /// True when the build has CUDA and @p device exists.
    [[nodiscard]] static bool usable(int device = 0) noexcept;

    /// Lazily created executor for @p device (throws InvalidArgumentError when !usable()).
    /// Intentionally never destroyed: CUDA may be torn down before static destructors run.
    [[nodiscard]] static GpuExecutor& instance(int device = 0);

    explicit GpuExecutor(GpuExecutorOptions options);
    ~GpuExecutor();
    GpuExecutor(const GpuExecutor&) = delete;
    GpuExecutor& operator=(const GpuExecutor&) = delete;

    class Lease {
    public:
        Lease() = default;
        Lease(GpuExecutor* ex, GpuLane* lane) : ex_(ex), lane_(lane) {}
        Lease(Lease&& o) noexcept : ex_(o.ex_), lane_(o.lane_) { o.ex_ = nullptr; o.lane_ = nullptr; }
        Lease& operator=(Lease&& o) noexcept;
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        ~Lease();
        [[nodiscard]] GpuLane& lane() const { return *lane_; }

    private:
        GpuExecutor* ex_ = nullptr;
        GpuLane* lane_ = nullptr;
    };

    /// Borrow a lane (blocks while max_lanes batches are in flight).
    [[nodiscard]] Lease acquire();

    /// Lanes currently leased.
    [[nodiscard]] std::size_t busy_lanes() const noexcept { return busy_.load(); }
    /// Batches a new acquire() would wait behind: the GPU queue depth fed to
    /// CostModel::decide(). Zero while a lane is free, because recorded GPU timings
    /// already include contention between concurrently running lanes.
    [[nodiscard]] std::size_t queue_depth() const noexcept {
        const std::size_t pending = busy_.load() + waiting_.load();
        return pending >= opt_.max_lanes ? pending - opt_.max_lanes + 1 : 0;
    }
    [[nodiscard]] int device() const noexcept { return opt_.device; }
    [[nodiscard]] const GpuExecutorOptions& options() const noexcept { return opt_; }

    /// Pooled device array (uninitialized). The first use on @p stream is ordered
    /// after the buffer's release fence.
    [[nodiscard]] NDArray device_array(Shape shape, DType dtype, CudaStreamHandle stream);
    /// Pooled page-locked host array (uninitialized).
    [[nodiscard]] NDArray pinned_array(Shape shape, DType dtype);

    /// Copy @p bytes of host memory to a device buffer kept alive for the lane's batch.
    [[nodiscard]] const void* upload_aux(GpuLane& lane, const void* host, std::size_t bytes);

    /// Run @p job through the lane's H2D -> compute -> D2H pipeline and wait for it.
    [[nodiscard]] NDArray run_rows(GpuLane& lane, const GpuRowJob& job, bool output_on_device);

    struct Stats {
        std::size_t device_hits = 0;
        std::size_t device_misses = 0;
        std::size_t cached_device_bytes = 0;
        std::size_t lanes_created = 0;
    };
    [[nodiscard]] Stats stats() const;

private:
    struct DevicePool;
    void release(GpuLane* lane) noexcept;
    [[nodiscard]] AlignedBuffer device_buffer(std::size_t bytes, CudaStreamHandle stream);

    GpuExecutorOptions opt_;
    std::shared_ptr<DevicePool> dev_pool_;
    std::unique_ptr<BufferPool> pinned_pool_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::vector<std::unique_ptr<GpuLane>> lanes_;
    std::vector<GpuLane*> free_lanes_;
    std::atomic<std::size_t> busy_{0};
    std::atomic<std::size_t> waiting_{0};
};

} // namespace nexusdata
