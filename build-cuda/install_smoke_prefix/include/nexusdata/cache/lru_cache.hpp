#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

#include "nexusdata/core/error.hpp"
#include "nexusdata/core/ndarray.hpp"
#include "nexusdata/dataset/sample.hpp"

namespace nexusdata {

/// Thread-safe LRU cache for Samples keyed by (dataset_id, index) or opaque string.
/// Evicts by approximate byte budget (sum of input+label nbytes).
class LruSampleCache {
public:
    explicit LruSampleCache(std::size_t max_bytes) : max_bytes_(max_bytes) {
        if (max_bytes_ == 0) {
            throw InvalidArgumentError("LruSampleCache: max_bytes must be > 0");
        }
    }

    [[nodiscard]] std::optional<Sample> get(const std::string& key) {
        std::lock_guard lock(mu_);
        auto it = map_.find(key);
        if (it == map_.end()) {
            return std::nullopt;
        }
        // Move to front
        lru_.splice(lru_.begin(), lru_, it->second);
        return it->second->second;
    }

    void put(const std::string& key, Sample sample) {
        const std::size_t bytes = sample.input.nbytes() + sample.label.nbytes();
        std::lock_guard lock(mu_);
        auto it = map_.find(key);
        if (it != map_.end()) {
            used_bytes_ -= it->second->second.input.nbytes() + it->second->second.label.nbytes();
            lru_.erase(it->second);
            map_.erase(it);
        }
        lru_.emplace_front(key, std::move(sample));
        map_[key] = lru_.begin();
        used_bytes_ += bytes;
        evict();
    }

    [[nodiscard]] std::size_t used_bytes() const {
        std::lock_guard lock(mu_);
        return used_bytes_;
    }

    [[nodiscard]] std::size_t size() const {
        std::lock_guard lock(mu_);
        return map_.size();
    }

    void clear() {
        std::lock_guard lock(mu_);
        lru_.clear();
        map_.clear();
        used_bytes_ = 0;
    }

private:
    void evict() {
        while (used_bytes_ > max_bytes_ && !lru_.empty()) {
            auto& back = lru_.back();
            used_bytes_ -= back.second.input.nbytes() + back.second.label.nbytes();
            map_.erase(back.first);
            lru_.pop_back();
        }
    }

    using List = std::list<std::pair<std::string, Sample>>;
    std::size_t max_bytes_ = 0;
    std::size_t used_bytes_ = 0;
    List lru_;
    std::unordered_map<std::string, typename List::iterator> map_;
    mutable std::mutex mu_;
};

/// FNV-1a 64-bit for cache keys / pipeline invalidation.
[[nodiscard]] inline std::uint64_t fnv1a64(const void* data, std::size_t n,
                                           std::uint64_t seed = 14695981039346656037ULL) {
    auto h = seed;
    const auto* p = static_cast<const std::uint8_t*>(data);
    for (std::size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

[[nodiscard]] inline std::uint64_t fnv1a64_str(const std::string& s) {
    return fnv1a64(s.data(), s.size());
}

} // namespace nexusdata
