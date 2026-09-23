#pragma once

#include <cstddef>
#include <vector>

#include "nexusdata/sampling/sampler.hpp"

namespace nexusdata {

/// Emits 0, 1, 2, ..., n-1.
class SequentialSampler : public Sampler {
public:
    explicit SequentialSampler(std::size_t n) : n_(n) {}

    [[nodiscard]] std::size_t size() const override { return n_; }

    [[nodiscard]] std::vector<std::size_t> indices() const override {
        std::vector<std::size_t> out(n_);
        for (std::size_t i = 0; i < n_; ++i) {
            out[i] = i;
        }
        return out;
    }

private:
    std::size_t n_ = 0;
};

} // namespace nexusdata
