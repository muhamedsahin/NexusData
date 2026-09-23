#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "nexusdata/core/registry.hpp"
#include "nexusdata/dataset/compose.hpp"
#include "nexusdata/dataset/csv/csv_dataset.hpp"
#include "nexusdata/dataset/map.hpp"
#include "nexusdata/loading/dataloader.hpp"
#include "nexusdata/pipeline/transform.hpp"

namespace nexusdata {

/// Loader options collected after `batches()`. `prefetch` can follow `batches`,
/// matching the chain `from_csv(...).batches(n).prefetch(k).build()`.
/// `batch()` still builds the loader immediately.
class BatchPlan {
public:
    BatchPlan(DatasetPtr dataset, DataLoaderOptions options)
        : dataset_(std::move(dataset)), options_(options) {
        if (!dataset_) throw InvalidArgumentError("BatchPlan: null dataset");
    }

    [[nodiscard]] BatchPlan prefetch(int factor) const {
        if (factor < 1) throw InvalidArgumentError("BatchPlan::prefetch: factor >= 1");
        BatchPlan next = *this;
        next.options_.prefetch_factor = factor;
        return next;
    }

    [[nodiscard]] BatchPlan workers(int n) const {
        if (n < 0) throw InvalidArgumentError("BatchPlan::workers: n >= 0");
        BatchPlan next = *this;
        next.options_.num_workers = n;
        return next;
    }

    [[nodiscard]] std::unique_ptr<DataLoader> build() const {
        return std::make_unique<DataLoader>(DatasetConstPtr(dataset_), options_);
    }

private:
    DatasetPtr dataset_;
    DataLoaderOptions options_;
};

/// Thin chain over Dataset / MapDataset / FilterDataset / DataLoader.
/// Core classes are unchanged. `batch` returns a loader; DataLoader is not movable.
class DatasetChain {
public:
    explicit DatasetChain(DatasetPtr dataset) : dataset_(std::move(dataset)) {
        if (!dataset_) throw InvalidArgumentError("DatasetChain: null dataset");
    }

    [[nodiscard]] DatasetChain map(TransformConstPtr transform) const {
        if (!transform) throw InvalidArgumentError("DatasetChain::map: null transform");
        DatasetChain next = *this;
        next.steps_.push_back(std::move(transform));
        return next;
    }

    /// `map(ClipTransform{0, 1})`. Shared pointers use the overload above.
    template <class T, typename std::enable_if<!std::is_convertible<T, TransformConstPtr>::value, int>::type = 0>
    [[nodiscard]] DatasetChain map(T transform) const {
        return map(std::static_pointer_cast<const Transform>(std::make_shared<T>(std::move(transform))));
    }

    [[nodiscard]] DatasetChain filter(FilterDataset::Pred pred) const {
        if (!pred) throw InvalidArgumentError("DatasetChain::filter: null predicate");
        DatasetChain next(std::make_shared<FilterDataset>(realize(), std::move(pred)));
        next.shuffle_ = shuffle_;
        next.seed_ = seed_;
        next.prefetch_ = prefetch_;
        return next;
    }

    [[nodiscard]] DatasetChain shuffle(std::uint64_t seed) const {
        DatasetChain next = *this;
        next.shuffle_ = true;
        next.seed_ = seed;
        return next;
    }

    [[nodiscard]] DatasetChain prefetch(int factor) const {
        if (factor < 1) throw InvalidArgumentError("DatasetChain::prefetch: factor >= 1");
        DatasetChain next = *this;
        next.prefetch_ = factor;
        return next;
    }

    [[nodiscard]] DatasetPtr dataset() const { return realize(); }

    [[nodiscard]] std::unique_ptr<DataLoader> batch(std::size_t batch_size, bool drop_last = false) const {
        return batches(batch_size, drop_last).build();
    }

    /// Same options as `batch`, but prefetch and workers can still be set.
    [[nodiscard]] BatchPlan batches(std::size_t batch_size, bool drop_last = false) const {
        if (batch_size == 0) throw InvalidArgumentError("DatasetChain::batches: batch_size >= 1");
        DataLoaderOptions opt;
        opt.batch_size = batch_size;
        opt.drop_last = drop_last;
        opt.shuffle = shuffle_;
        opt.seed = seed_;
        opt.prefetch_factor = prefetch_;
        return BatchPlan(realize(), opt);
    }

private:
    [[nodiscard]] DatasetPtr realize() const {
        if (steps_.empty()) return dataset_;
        TransformConstPtr t = steps_.size() == 1
                                  ? steps_[0]
                                  : std::static_pointer_cast<const Transform>(
                                        std::make_shared<Compose>(steps_));
        return std::make_shared<MapDataset>(DatasetConstPtr(dataset_), std::move(t));
    }

    DatasetPtr dataset_;
    std::vector<TransformConstPtr> steps_;
    bool shuffle_ = false;
    std::uint64_t seed_ = 0;
    int prefetch_ = 2;
};

[[nodiscard]] inline DatasetChain from_csv(const std::string& path, CSVOptions options = {}) {
    return DatasetChain(std::make_shared<CSVDataset>(path, std::move(options)));
}

/// Any scheme registered with DataSourceRegistry, including plugins.
[[nodiscard]] inline DatasetChain from_uri(const std::string& uri, const std::string& config_json = "{}") {
    return DatasetChain(open(uri, config_json));
}

} // namespace nexusdata
