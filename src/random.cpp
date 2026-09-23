#include "nexusdata/core/random.hpp"

namespace nexusdata {

namespace {

constexpr std::uint64_t kPCGMultiplier = 6364136223846793005ULL;

} // namespace

PCG32::PCG32(std::uint64_t seed, std::uint64_t stream) {
    // Stream must be odd for full period (PCG paper).
    inc_ = (stream << 1u) | 1u;
    state_ = 0;
    (void)next_u32();
    state_ += seed;
    (void)next_u32();
}

std::uint32_t PCG32::next_u32() {
    const std::uint64_t oldstate = state_;
    state_ = oldstate * kPCGMultiplier + inc_;
    const std::uint32_t xorshifted =
        static_cast<std::uint32_t>(((oldstate >> 18u) ^ oldstate) >> 27u);
    const std::uint32_t rot = static_cast<std::uint32_t>(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31u));
}

std::uint32_t PCG32::next_bounded(std::uint32_t bound) {
    if (bound == 0) {
        throw InvalidArgumentError("PCG32::next_bounded: bound must be > 0");
    }
    // Lemire's nearly-divisionless method.
    std::uint64_t r = next_u32();
    std::uint64_t m = r * static_cast<std::uint64_t>(bound);
    std::uint32_t l = static_cast<std::uint32_t>(m);
    if (l < bound) {
        std::uint32_t t = static_cast<std::uint32_t>(-bound) % bound;
        while (l < t) {
            r = next_u32();
            m = r * static_cast<std::uint64_t>(bound);
            l = static_cast<std::uint32_t>(m);
        }
    }
    return static_cast<std::uint32_t>(m >> 32);
}

float PCG32::next_float() {
    // 24 bits of mantissa precision.
    return static_cast<float>(next_u32() >> 8) * (1.0f / 16777216.0f);
}

double PCG32::next_double() {
    // 53 bits: combine two draws.
    const std::uint64_t a = next_u32() >> 5;
    const std::uint64_t b = next_u32() >> 6;
    return (a * 67108864.0 + static_cast<double>(b)) * (1.0 / 9007199254740992.0);
}

std::vector<std::size_t> shuffled_indices(std::size_t n, PCG32& rng) {
    std::vector<std::size_t> indices(n);
    for (std::size_t i = 0; i < n; ++i) {
        indices[i] = i;
    }
    shuffle(indices, rng);
    return indices;
}

} // namespace nexusdata
