#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "nexusdata/backend/bounded_queue.hpp"
#include "nexusdata/backend/cost_model.hpp"
#include "nexusdata/backend/device.hpp"
#include "nexusdata/backend/thread_pool.hpp"
#include "nexusdata/core/object_pool.hpp"
#include "nexusdata/dataset/dataset.hpp"
#include "nexusdata/loading/batch.hpp"
#include "nexusdata/pipeline/stage_graph.hpp"
#include "nexusdata/pipeline/transform.hpp"
#include "nexusdata/sampling/random.hpp"
#include "nexusdata/sampling/sequential.hpp"

namespace nexusdata {

struct DataLoaderOptions {
    std::size_t batch_size = 1;
    bool shuffle = false;
    bool drop_last = false;
    std::uint64_t seed = 0;
    /// Background workers for sample fetch + collate. 0 = synchronous (main thread).
    int num_workers = 0;
    /// Queue capacity ≈ num_workers * prefetch_factor (minimum 2 when workers > 0).
    int prefetch_factor = 2;
    /// Stage collated batches in page-locked host memory before H2D (when CUDA).
    bool pin_memory = false;
    bool persistent_workers = false;
    int timeout_ms = 0;
    /// Where to place batch outputs. Auto may pick CUDA for large image batches.
    Device device = Device::host();

    // --- v2.0 (all defaults reproduce v1 behaviour) --------------------------

    /// Deliver batches in sampler order (reorder buffer). false = hand out each
    /// batch as soon as it is ready (higher throughput; contents per batch are unchanged).
    bool in_order = true;
    /// Optional batch-level transform run in its own pipeline stage after collate.
    /// Must report supports_batch() (e.g. a Compose of tabular transforms).
    TransformConstPtr batch_transform;
    /// Threads for the batch_transform stage (when num_workers > 0).
    int transform_workers = 1;
    /// Op class under which batch_transform timings are recorded in cost_model
    /// (default: batch_transform->batch_op_kind()).
    std::optional<GpuOpKind> batch_transform_op;
    /// Where batch_transform runs when it supports_gpu_batch(): Auto = per-batch
    /// CostModel decision (cost_model, else CostModel::global()), CUDA = always
    /// on the GPU when present, Host = always on the CPU. A GPU batch stays in
    /// device memory unless `device` is Host; any GPU failure reruns it on the CPU.
    Device transform_device = Device::auto_select();
    /// Learned device placement for Device::auto_select(); null = static AutoDevicePolicy.
    std::shared_ptr<CostModel> cost_model;
    /// With pin_memory: recycle pinned staging buffers through a BufferPool.
    bool reuse_pinned_buffers = true;
};

using CollateFn = std::function<Batch(const std::vector<Sample>&)>;

/// DataLoader with optional multi-worker prefetch.
///
/// v2.0: with num_workers > 0 batches flow through a StageGraph:
///   sampler -> load (fetch + collate, num_workers threads)
///           -> [batch_transform, transform_workers threads]
///           -> [transfer: pin + H2D, 1 thread] -> iterator
/// Each stage has its own bounded queue; a reorder buffer keeps sampler order
/// when in_order is true.
///
/// Determinism: batch i always contains the same sample indices for a given seed,
/// independent of num_workers (workers only parallelize materialization).
///
/// Thread-safety: not safe to iterate from multiple threads; destructor joins workers.
class DataLoader {
public:
    DataLoader(DatasetConstPtr dataset,
               DataLoaderOptions options = {},
               CollateFn collate = collate_stack);

    template <typename DatasetT>
    DataLoader(DatasetT dataset,
               DataLoaderOptions options = {},
               CollateFn collate = collate_stack)
        : DataLoader(DatasetConstPtr(std::make_shared<DatasetT>(std::move(dataset))),
                     std::move(options),
                     std::move(collate)) {}

    ~DataLoader();

    DataLoader(const DataLoader&) = delete;
    DataLoader& operator=(const DataLoader&) = delete;

    [[nodiscard]] std::size_t size() const noexcept { return num_batches_; }
    [[nodiscard]] const DataLoaderOptions& options() const noexcept { return options_; }

    class Iterator {
    public:
        using value_type = Batch;

        Iterator() = default;
        Iterator(DataLoader* loader, std::size_t batch_index);

        Batch operator*() const;
        Iterator& operator++();
        [[nodiscard]] bool operator==(const Iterator& other) const noexcept;
        [[nodiscard]] bool operator!=(const Iterator& other) const noexcept {
            return !(*this == other);
        }

    private:
        DataLoader* loader_ = nullptr;
        std::size_t batch_index_ = 0;
        mutable std::optional<Batch> cached_;
    };

    [[nodiscard]] Iterator begin();
    [[nodiscard]] Iterator end();

    void reset_epoch(std::uint64_t seed);

    /// v2.0: per-stage counters of the running prefetch pipeline (empty when synchronous).
    [[nodiscard]] std::vector<StageStats> pipeline_stats() const;

private:
    friend class Iterator;

    [[nodiscard]] Batch materialize_batch(std::size_t batch_index) const;
    [[nodiscard]] Batch load_batch(std::size_t batch_index) const;
    [[nodiscard]] Batch transform_batch(Batch b) const;
    [[nodiscard]] Batch finalize_batch(Batch b) const;
    [[nodiscard]] bool needs_transfer_stage() const noexcept;
    void start_prefetch();
    void stop_prefetch();
    Batch next_prefetched(std::size_t expected_index);

    DatasetConstPtr dataset_;
    DataLoaderOptions options_;
    CollateFn collate_;
    std::vector<std::size_t> order_;
    std::size_t num_batches_ = 0;

    std::unique_ptr<StageGraph<Batch>> graph_;
    std::unique_ptr<BufferPool> pinned_pool_;
    std::size_t next_consume_ = 0;
};

} // namespace nexusdata
