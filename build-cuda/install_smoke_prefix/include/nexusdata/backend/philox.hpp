#pragma once

#include <cstddef>
#include <cstdint>

namespace nexusdata {

/// Philox 4x32-10 counter-based RNG (Salmon et al.).
/// Same (key, counter) ⇒ same output on CPU and GPU.
/// Thread-safety: instances are independent; not shared without sync.
class Philox4x32 {
public:
    Philox4x32() = default;

    /// key = seed parts; counter typically (sample_index, epoch, op_id, 0)
    explicit Philox4x32(std::uint64_t seed) {
        key_[0] = static_cast<std::uint32_t>(seed);
        key_[1] = static_cast<std::uint32_t>(seed >> 32);
    }

    void set_key(std::uint32_t k0, std::uint32_t k1) {
        key_[0] = k0;
        key_[1] = k1;
    }

    void set_counter(std::uint32_t c0, std::uint32_t c1, std::uint32_t c2, std::uint32_t c3) {
        ctr_[0] = c0;
        ctr_[1] = c1;
        ctr_[2] = c2;
        ctr_[3] = c3;
    }

    /// Advance counter and return 4 uint32 outputs.
    void next(std::uint32_t out[4]) {
        std::uint32_t c[4] = {ctr_[0], ctr_[1], ctr_[2], ctr_[3]};
        std::uint32_t k[2] = {key_[0], key_[1]};
        round10(c, k);
        out[0] = c[0];
        out[1] = c[1];
        out[2] = c[2];
        out[3] = c[3];
        // increment counter
        if (++ctr_[0] == 0)
            if (++ctr_[1] == 0)
                if (++ctr_[2] == 0) ++ctr_[3];
    }

    [[nodiscard]] std::uint32_t next_u32() {
        std::uint32_t o[4];
        next(o);
        return o[0];
    }

    [[nodiscard]] float next_float() {
        return static_cast<float>(next_u32() >> 8) * (1.0f / 16777216.0f);
    }

    /// Convenience: seed from (seed, sample_index, epoch).
    static Philox4x32 from_sample(std::uint64_t seed, std::uint64_t sample_index,
                                  std::uint32_t epoch) {
        Philox4x32 p(seed);
        p.set_counter(static_cast<std::uint32_t>(sample_index),
                      static_cast<std::uint32_t>(sample_index >> 32), epoch, 0u);
        return p;
    }

private:
    static constexpr std::uint32_t kPhiloxM0 = 0xD2511F53u;
    static constexpr std::uint32_t kPhiloxM1 = 0xCD9E8D57u;
    static constexpr std::uint32_t kPhiloxW0 = 0x9E3779B9u;
    static constexpr std::uint32_t kPhiloxW1 = 0xBB67AE85u;

    static void mulhilo(std::uint32_t a, std::uint32_t b, std::uint32_t& hi, std::uint32_t& lo) {
        const std::uint64_t p = static_cast<std::uint64_t>(a) * static_cast<std::uint64_t>(b);
        hi = static_cast<std::uint32_t>(p >> 32);
        lo = static_cast<std::uint32_t>(p);
    }

    static void round10(std::uint32_t c[4], std::uint32_t k[2]) {
        for (int i = 0; i < 10; ++i) {
            std::uint32_t hi0 = 0, lo0 = 0, hi1 = 0, lo1 = 0;
            mulhilo(kPhiloxM0, c[0], hi0, lo0);
            mulhilo(kPhiloxM1, c[2], hi1, lo1);
            c[0] = hi1 ^ c[1] ^ k[0];
            c[1] = lo1;
            c[2] = hi0 ^ c[3] ^ k[1];
            c[3] = lo0;
            k[0] += kPhiloxW0;
            k[1] += kPhiloxW1;
        }
    }

    std::uint32_t key_[2] = {0, 0};
    std::uint32_t ctr_[4] = {0, 0, 0, 0};
};

} // namespace nexusdata
