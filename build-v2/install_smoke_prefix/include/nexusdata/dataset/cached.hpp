#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "nexusdata/cache/lru_cache.hpp"
#include "nexusdata/core/error.hpp"
#include "nexusdata/dataset/dataset.hpp"

namespace nexusdata {

/// Wraps a dataset with an LRU sample cache (key = prefix + index).
class CachedDataset : public Dataset {
public:
    CachedDataset(DatasetConstPtr parent,
                  std::shared_ptr<LruSampleCache> cache,
                  std::string key_prefix = "ds")
        : parent_(std::move(parent)),
          cache_(std::move(cache)),
          key_prefix_(std::move(key_prefix)) {
        if (!parent_) {
            throw InvalidArgumentError("CachedDataset: null parent");
        }
        if (!cache_) {
            throw InvalidArgumentError("CachedDataset: null cache");
        }
    }

    [[nodiscard]] std::size_t size() const override { return parent_->size(); }

    [[nodiscard]] Sample get(std::size_t index) const override {
        const std::string key = key_prefix_ + ":" + std::to_string(index);
        if (auto hit = cache_->get(key)) {
            return *hit;
        }
        Sample s = parent_->get(index);
        cache_->put(key, s);
        return s;
    }

private:
    DatasetConstPtr parent_;
    std::shared_ptr<LruSampleCache> cache_;
    std::string key_prefix_;
};

} // namespace nexusdata
