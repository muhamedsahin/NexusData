#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nexusdata {

/// Produces a sequence of sample indices for one epoch.
/// Thread-safety: not thread-safe; DataLoader owns one sampler per epoch.
class Sampler {
public:
    virtual ~Sampler() = default;
    [[nodiscard]] virtual std::size_t size() const = 0;
    [[nodiscard]] virtual std::vector<std::size_t> indices() const = 0;
};

} // namespace nexusdata
