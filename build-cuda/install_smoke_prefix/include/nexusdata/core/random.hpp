#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "nexusdata/core/error.hpp"

namespace nexusdata {

/// Permuted Congruential Generator (32-bit output).
/// Deterministic across platforms for the same (seed, stream).
/// Thread-safety: not thread-safe; each thread needs its own instance.
class PCG32 {
public:
    /// Construct from seed. Stream id defaults to 54 (odd stream via encoding).
    explicit PCG32(std::uint64_t seed, std::uint64_t stream = 54ULL);

    /// Next uniform 32-bit value in [0, 2^32).
    [[nodiscard]] std::uint32_t next_u32();

    /// Unbiased integer in [0, bound). Requires bound > 0.
    [[nodiscard]] std::uint32_t next_bounded(std::uint32_t bound);

    /// Uniform float in [0, 1).
    [[nodiscard]] float next_float();

    /// Uniform double in [0, 1).
    [[nodiscard]] double next_double();

    [[nodiscard]] std::uint64_t state() const noexcept { return state_; }
    [[nodiscard]] std::uint64_t stream() const noexcept { return inc_; }

private:
    std::uint64_t state_ = 0;
    std::uint64_t inc_ = 0;
};

/// In-place Fisher-Yates shuffle using PCG32 (not std::shuffle).
template <typename T>
void shuffle(std::vector<T>& values, PCG32& rng) {
    if (values.size() <= 1) {
        return;
    }
    for (std::size_t i = values.size() - 1; i > 0; --i) {
        const std::uint32_t j = rng.next_bounded(static_cast<std::uint32_t>(i + 1));
        using std::swap;
        swap(values[i], values[j]);
    }
}

/// Build indices [0, n) and shuffle them.
[[nodiscard]] std::vector<std::size_t> shuffled_indices(std::size_t n, PCG32& rng);

} // namespace nexusdata
