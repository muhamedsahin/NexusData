#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

#include "nexusdata/core/error.hpp"
#include "nexusdata/core/random.hpp"
#include "nexusdata/sampling/sampler.hpp"

namespace nexusdata {

/// Sample with replacement according to weights ( systematically: alias method simplified via CDF).
class WeightedRandomSampler : public Sampler {
public:
    WeightedRandomSampler(std::vector<double> weights, std::size_t num_samples, std::uint64_t seed,
                          bool replacement = true)
        : weights_(std::move(weights)), n_(num_samples), seed_(seed), replacement_(replacement) {
        if (weights_.empty()) {
            throw InvalidArgumentError("WeightedRandomSampler: empty weights");
        }
        if (!replacement_ && n_ > weights_.size()) {
            throw InvalidArgumentError("WeightedRandomSampler: without replacement n > population");
        }
        double sum = 0.0;
        for (double w : weights_) {
            if (w < 0.0) {
                throw InvalidArgumentError("WeightedRandomSampler: negative weight");
            }
            sum += w;
        }
        if (sum <= 0.0) {
            throw InvalidArgumentError("WeightedRandomSampler: weight sum must be > 0");
        }
        cdf_.resize(weights_.size());
        double acc = 0.0;
        for (std::size_t i = 0; i < weights_.size(); ++i) {
            acc += weights_[i] / sum;
            cdf_[i] = acc;
        }
        cdf_.back() = 1.0;
    }

    [[nodiscard]] std::size_t size() const override { return n_; }

    [[nodiscard]] std::vector<std::size_t> indices() const override {
        PCG32 rng(seed_);
        std::vector<std::size_t> out;
        out.reserve(n_);
        if (replacement_) {
            for (std::size_t i = 0; i < n_; ++i) {
                out.push_back(sample_one(rng));
            }
        } else {
            auto order = shuffled_indices(weights_.size(), rng);
            // Weighted without replacement: sequential elimination is expensive;
            // v0.6 uses weighted shuffle approximation via key = rng / weight.
            std::vector<std::pair<double, std::size_t>> keys;
            keys.reserve(weights_.size());
            for (std::size_t i = 0; i < weights_.size(); ++i) {
                const double u = std::max(1e-12, static_cast<double>(rng.next_float()));
                keys.push_back({std::log(u) / weights_[i], i});
            }
            std::sort(keys.begin(), keys.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
            for (std::size_t i = 0; i < n_; ++i) {
                out.push_back(keys[i].second);
            }
        }
        return out;
    }

private:
    [[nodiscard]] std::size_t sample_one(PCG32& rng) const {
        const double u = rng.next_double();
        auto it = std::lower_bound(cdf_.begin(), cdf_.end(), u);
        return static_cast<std::size_t>(it - cdf_.begin());
    }

    std::vector<double> weights_;
    std::vector<double> cdf_;
    std::size_t n_ = 0;
    std::uint64_t seed_ = 0;
    bool replacement_ = true;
};

/// Group indices into batches of batch_size (optionally drop last).
class BatchSampler : public Sampler {
public:
    BatchSampler(std::shared_ptr<const Sampler> base, std::size_t batch_size, bool drop_last)
        : base_(std::move(base)), batch_size_(batch_size), drop_last_(drop_last) {
        if (!base_) {
            throw InvalidArgumentError("BatchSampler: null base");
        }
        if (batch_size_ == 0) {
            throw InvalidArgumentError("BatchSampler: batch_size > 0");
        }
    }

    /// Number of *batches*.
    [[nodiscard]] std::size_t size() const override {
        const std::size_t n = base_->size();
        if (drop_last_) {
            return n / batch_size_;
        }
        return (n + batch_size_ - 1) / batch_size_;
    }

    /// Flattened indices in batch order (not batch boundaries).
    [[nodiscard]] std::vector<std::size_t> indices() const override {
        auto idx = base_->indices();
        if (drop_last_) {
            const std::size_t keep = (idx.size() / batch_size_) * batch_size_;
            idx.resize(keep);
        }
        return idx;
    }

    [[nodiscard]] std::vector<std::vector<std::size_t>> batches() const {
        auto idx = indices();
        std::vector<std::vector<std::size_t>> out;
        for (std::size_t i = 0; i < idx.size(); i += batch_size_) {
            const std::size_t end = std::min(i + batch_size_, idx.size());
            out.emplace_back(idx.begin() + static_cast<std::ptrdiff_t>(i),
                             idx.begin() + static_cast<std::ptrdiff_t>(end));
        }
        return out;
    }

private:
    std::shared_ptr<const Sampler> base_;
    std::size_t batch_size_ = 1;
    bool drop_last_ = false;
};

/// Bucket by sequence length: indices sorted by length, then batched.
class BucketSampler : public Sampler {
public:
    BucketSampler(std::vector<std::size_t> lengths, std::size_t batch_size, std::uint64_t seed,
                  bool drop_last = false)
        : lengths_(std::move(lengths)), batch_size_(batch_size), seed_(seed), drop_last_(drop_last) {
        if (batch_size_ == 0) {
            throw InvalidArgumentError("BucketSampler: batch_size > 0");
        }
    }

    [[nodiscard]] std::size_t size() const override {
        const std::size_t n = lengths_.size();
        if (drop_last_) {
            return (n / batch_size_) * batch_size_;
        }
        return n;
    }

    [[nodiscard]] std::vector<std::size_t> indices() const override {
        std::vector<std::size_t> order(lengths_.size());
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
            return lengths_[a] < lengths_[b];
        });
        // Shuffle within buckets of similar length (window = batch_size * 4)
        PCG32 rng(seed_);
        const std::size_t window = std::max<std::size_t>(batch_size_ * 4, 1);
        for (std::size_t i = 0; i < order.size(); i += window) {
            const std::size_t end = std::min(i + window, order.size());
            for (std::size_t j = end - 1; j > i; --j) {
                const std::uint32_t r = rng.next_bounded(static_cast<std::uint32_t>(j - i + 1));
                std::swap(order[j], order[i + r]);
            }
        }
        if (drop_last_) {
            order.resize((order.size() / batch_size_) * batch_size_);
        }
        return order;
    }

private:
    std::vector<std::size_t> lengths_;
    std::size_t batch_size_ = 1;
    std::uint64_t seed_ = 0;
    bool drop_last_ = false;
};

/// Shard indices across distributed ranks: index % world_size == rank.
class DistributedSampler : public Sampler {
public:
    DistributedSampler(std::size_t n, int rank, int world_size, bool shuffle, std::uint64_t seed,
                       int epoch = 0)
        : n_(n), rank_(rank), world_size_(world_size), shuffle_(shuffle), seed_(seed), epoch_(epoch) {
        if (world_size_ <= 0 || rank_ < 0 || rank_ >= world_size_) {
            throw InvalidArgumentError("DistributedSampler: invalid rank/world_size");
        }
    }

    void set_epoch(int epoch) { epoch_ = epoch; }

    [[nodiscard]] std::size_t size() const override {
        return (n_ + static_cast<std::size_t>(world_size_) - 1) / static_cast<std::size_t>(world_size_);
    }

    [[nodiscard]] std::vector<std::size_t> indices() const override {
        std::vector<std::size_t> all(n_);
        std::iota(all.begin(), all.end(), 0);
        if (shuffle_) {
            PCG32 rng(seed_ + static_cast<std::uint64_t>(epoch_) * 0x9E3779B97F4A7C15ULL);
            shuffle(all, rng);
        }
        // Pad to multiple of world_size for even shards
        while (all.size() % static_cast<std::size_t>(world_size_) != 0) {
            all.push_back(all[all.size() % n_]);
        }
        std::vector<std::size_t> out;
        out.reserve(all.size() / static_cast<std::size_t>(world_size_));
        for (std::size_t i = static_cast<std::size_t>(rank_); i < all.size();
             i += static_cast<std::size_t>(world_size_)) {
            out.push_back(all[i]);
        }
        return out;
    }

private:
    std::size_t n_ = 0;
    int rank_ = 0;
    int world_size_ = 1;
    bool shuffle_ = false;
    std::uint64_t seed_ = 0;
    int epoch_ = 0;
};

} // namespace nexusdata
