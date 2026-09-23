#include "nexusdata/loading/dataloader.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <string>
#include <utility>

#include "nexusdata/backend/cuda_api.hpp"
#include "nexusdata/backend/gpu_executor.hpp"
#include "nexusdata/backend/pinned.hpp"
#include "nexusdata/core/error.hpp"
#include "nexusdata/core/metrics.hpp"
#include "nexusdata/core/random.hpp"

namespace nexusdata {

namespace {

NDArray stage_pinned(const NDArray& src, BufferPool* pool) {
    if (!src.device().is_host() || src.nbytes() == 0 || cuda_api::is_host_pinned(src.data())) {
        return src;
    }
    NDArray pin = pool != nullptr ? pool->make_array(src.shape(), src.dtype(), /*zero=*/false)
                                  : NDArray::pinned(src.shape(), src.dtype());
    std::memcpy(pin.data(), src.data(), src.nbytes());
    return pin;
}

} // namespace

DataLoader::DataLoader(DatasetConstPtr dataset,
                       DataLoaderOptions options,
                       CollateFn collate)
    : dataset_(std::move(dataset)),
      options_(options),
      collate_(std::move(collate)) {
    if (!dataset_) {
        throw InvalidArgumentError("DataLoader: null dataset");
    }
    if (options_.batch_size == 0) {
        throw InvalidArgumentError("DataLoader: batch_size must be > 0");
    }
    if (options_.num_workers < 0) {
        throw InvalidArgumentError("DataLoader: num_workers must be >= 0");
    }
    if (options_.prefetch_factor < 1) {
        throw InvalidArgumentError("DataLoader: prefetch_factor must be >= 1");
    }
    if (options_.transform_workers < 1) {
        throw InvalidArgumentError("DataLoader: transform_workers must be >= 1");
    }
    if (options_.batch_transform && !options_.batch_transform->supports_batch()) {
        throw InvalidArgumentError(
            "DataLoader: batch_transform does not support batch application "
            "(supports_batch() == false); wrap per-sample transforms in MapDataset instead");
    }
    if (!collate_) {
        collate_ = collate_stack;
    }
    if (options_.pin_memory && options_.reuse_pinned_buffers) {
        pinned_pool_ = BufferPool::pinned();
    }
    reset_epoch(options_.seed);
}

DataLoader::~DataLoader() {
    stop_prefetch();
}

void DataLoader::reset_epoch(std::uint64_t seed) {
    stop_prefetch();
    options_.seed = seed;
    const std::size_t n = dataset_->size();
    if (options_.shuffle) {
        PCG32 rng(seed);
        order_ = shuffled_indices(n, rng);
    } else {
        order_.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            order_[i] = i;
        }
    }

    if (n == 0) {
        num_batches_ = 0;
    } else if (options_.drop_last) {
        num_batches_ = n / options_.batch_size;
    } else {
        num_batches_ = (n + options_.batch_size - 1) / options_.batch_size;
    }
    next_consume_ = 0;
}

Batch DataLoader::load_batch(std::size_t batch_index) const {
    if (batch_index >= num_batches_) {
        throw IndexError("DataLoader: batch index out of range");
    }
    const std::size_t start = batch_index * options_.batch_size;
    std::size_t end = start + options_.batch_size;
    if (end > order_.size()) {
        end = order_.size();
    }
    std::vector<Sample> samples;
    samples.reserve(end - start);
    for (std::size_t i = start; i < end; ++i) {
        samples.push_back(dataset_->get(order_[i]));
    }
    return collate_(samples);
}

Batch DataLoader::transform_batch(Batch b) const {
    if (!options_.batch_transform) {
        return b;
    }
    const Transform& t = *options_.batch_transform;
    const std::size_t nbytes = b.inputs.nbytes();
    const GpuOpKind op = options_.batch_transform_op.value_or(t.batch_op_kind());
    const Device want = options_.transform_device;
    const int dev = want.is_cuda() && want.index >= 0 ? want.index : 0;

    const bool gpu_capable = !want.is_host() && t.supports_gpu_batch() && GpuExecutor::usable(dev);
    CostModel* model = options_.cost_model.get();
    if (gpu_capable && model == nullptr) {
        model = &CostModel::global();
    }

    if (gpu_capable) {
        GpuExecutor& ex = GpuExecutor::instance(dev);
        const bool use_gpu = want.is_cuda() || model->decide(op, nbytes, ex.queue_depth()).use_gpu;
        if (use_gpu) {
            auto lease = ex.acquire();
            GpuBatchContext ctx;
            ctx.executor = &ex;
            ctx.lane = &lease.lane();
            // Auto placement keeps GPU results on the device instead of paying a D2H + H2D.
            ctx.output_on_device = !options_.device.is_host();
            ScopedCostTimer timer(*model, op, ExecPath::Gpu, nbytes);
            try {
                return t.apply_batch_gpu(b, ctx);
            } catch (const std::exception&) {
                // GPU failure (e.g. out of memory): drop the sample and redo the batch on the CPU.
                timer.cancel();
            }
        }
    }
    if (model == nullptr) {
        return t.apply_batch(b);
    }
    ScopedCostTimer timer(*model, op, ExecPath::Cpu, nbytes);
    Batch out = t.apply_batch(b);
    if (gpu_capable && !options_.device.is_host()) {
        // The GPU path is timed up to device-resident outputs; charge the CPU path the
        // same destination, otherwise the model never sees the upload it causes.
        if (options_.pin_memory) {
            out.inputs = stage_pinned(out.inputs, pinned_pool_.get());
        }
        out.inputs = out.inputs.cuda(dev);
    }
    return out;
}

Batch DataLoader::finalize_batch(Batch b) const {
    if (options_.pin_memory) {
        b.inputs = stage_pinned(b.inputs, pinned_pool_.get());
        b.labels = stage_pinned(b.labels, pinned_pool_.get());
        for (auto& [key, t] : b.extras) {
            t = stage_pinned(t, pinned_pool_.get());
        }
    }

    Device target;
    if (b.inputs.device().is_cuda()) {
        // Produced on the GPU by the transform stage: labels follow the inputs.
        target = b.inputs.device();
        if (options_.device.is_host()) {
            target = Device::host();
            b.inputs = b.inputs.cpu();
        }
    } else if (options_.cost_model) {
        const std::size_t depth = graph_ ? graph_->ready_items() : 0;
        target = resolve_device(options_.device, GpuOpKind::NormalizeCollate, b.inputs.nbytes(),
                                *options_.cost_model, depth);
    } else {
        target = resolve_device(options_.device, GpuOpKind::NormalizeCollate, b.inputs.nbytes());
    }
    if (target.is_cuda() && cuda_api::available()) {
        b.inputs = b.inputs.cuda(target.index);
        b.labels = b.labels.cuda(target.index);
        // Ragged extras stay on the host: one small transfer per sample is rarely worth it.
        for (auto& [key, t] : b.extras) {
            t = t.cuda(target.index);
        }
    }
    return b;
}

bool DataLoader::needs_transfer_stage() const noexcept {
    return options_.pin_memory || !options_.device.is_host();
}

Batch DataLoader::materialize_batch(std::size_t batch_index) const {
    return finalize_batch(transform_batch(load_batch(batch_index)));
}

void DataLoader::start_prefetch() {
    if (options_.num_workers <= 0 || num_batches_ == 0) {
        return;
    }
    if (graph_) {
        return;
    }
    const auto workers = static_cast<std::size_t>(options_.num_workers);
    const std::size_t cap = static_cast<std::size_t>(
        std::max(2, options_.num_workers * options_.prefetch_factor));

    StageGraphOptions gopt;
    gopt.in_order = options_.in_order;

    auto g = StageGraph<std::size_t>::from_range("sampler", num_batches_, workers, gopt)
                 .then("load", StageOptions{workers, cap},
                       [this](std::size_t i) { return load_batch(i); });
    if (options_.batch_transform) {
        g = std::move(g).then(
            "transform",
            StageOptions{static_cast<std::size_t>(options_.transform_workers), 2},
            [this](Batch b) { return transform_batch(std::move(b)); });
    }
    if (needs_transfer_stage()) {
        g = std::move(g).then("transfer", StageOptions{1, 2},
                              [this](Batch b) { return finalize_batch(std::move(b)); });
    }
    graph_ = std::make_unique<StageGraph<Batch>>(std::move(g));
    next_consume_ = 0;
    graph_->start();
}

void DataLoader::stop_prefetch() {
    if (graph_) {
        graph_->stop();
        graph_.reset();
    }
}

Batch DataLoader::next_prefetched(std::size_t expected_index) {
    if (!graph_) {
        return materialize_batch(expected_index);
    }
    auto item = graph_->next();
    if (!item) {
        throw Error("DataLoader: prefetch pipeline ended before batch " +
                    std::to_string(expected_index));
    }
    ++next_consume_;
    return std::move(*item);
}

std::vector<StageStats> DataLoader::pipeline_stats() const {
    return graph_ ? graph_->stats() : std::vector<StageStats>{};
}

DataLoader::Iterator DataLoader::begin() {
    stop_prefetch();
    next_consume_ = 0;
    start_prefetch();
    return Iterator(this, 0);
}

DataLoader::Iterator DataLoader::end() {
    return Iterator(this, num_batches_);
}

DataLoader::Iterator::Iterator(DataLoader* loader, std::size_t batch_index)
    : loader_(loader), batch_index_(batch_index) {}

Batch DataLoader::Iterator::operator*() const {
    if (!loader_) {
        throw InvalidArgumentError("DataLoader::Iterator: singular iterator");
    }
    if (!cached_) {
        const auto load = [&] {
            if (loader_->options_.num_workers > 0) return loader_->next_prefetched(batch_index_);
            return loader_->materialize_batch(batch_index_);
        };
        if (metrics::enabled()) {
            const auto t0 = std::chrono::steady_clock::now();
            cached_ = load();
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            metrics::add("dataloader.batch_seconds", seconds);
            metrics::add("dataloader.batches", 1);
            if (loader_->options_.device.is_host()) {
                metrics::add("dataloader.cpu_batches", 1);
            } else {
                metrics::add("dataloader.gpu_batches", 1);
            }
            if (loader_->graph_) {
                for (const StageStats& st : loader_->pipeline_stats()) {
                    metrics::add("dataloader.queue_size." + st.name, static_cast<double>(st.queue_size));
                }
            }
        } else {
            cached_ = load();
        }
    }
    return *cached_;
}

DataLoader::Iterator& DataLoader::Iterator::operator++() {
    if (loader_ && !cached_ && loader_->graph_) {
        // Skipping without dereferencing still consumes one pipeline slot.
        (void)**this;
    }
    cached_.reset();
    ++batch_index_;
    if (loader_ && batch_index_ >= loader_->num_batches_) {
        loader_->stop_prefetch();
    }
    return *this;
}

bool DataLoader::Iterator::operator==(const Iterator& other) const noexcept {
    return loader_ == other.loader_ && batch_index_ == other.batch_index_;
}

} // namespace nexusdata
