#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "nexusdata/core/random.hpp"
#include "nexusdata/sampling/sampler.hpp"

namespace nexusdata {

/// Shuffled indices via PCG32 Fisher-Yates. Same seed => same order.
class RandomSampler : public Sampler {
public:
    RandomSampler(std::size_t n, std::uint64_t seed, bool replacement = false)
        : n_(n), seed_(seed), replacement_(replacement) {}

    [[nodiscard]] std::size_t size() const override { return n_; }

    [[nodiscard]] std::vector<std::size_t> indices() const override {
        PCG32 rng(seed_);
        if (!replacement_) {
            return shuffled_indices(n_, rng);
        }
        std::vector<std::size_t> out(n_);
        for (std::size_t i = 0; i < n_; ++i) {
            out[i] = rng.next_bounded(static_cast<std::uint32_t>(n_));
        }
        return out;
    }

    /// Reseed for a new epoch: seed_base + epoch (caller convention).
    void set_seed(std::uint64_t seed) { seed_ = seed; }

private:
    std::size_t n_ = 0;
    std::uint64_t seed_ = 0;
    bool replacement_ = false;
};

} // namespace nexusdata
