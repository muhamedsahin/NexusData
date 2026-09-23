#pragma once

#include <cstddef>
#include <cstdint>

namespace nexusdata {

/// Runtime CPU feature flags. x86 vector flags are only set when the OS also
/// saves the corresponding register state (XGETBV), so they are safe to dispatch on.
struct CpuFeatures {
    bool sse2 = false;
    bool avx2 = false;
    bool avx512f = false;
    bool neon = false;
    /// v2.0 additions.
    bool avx = false;
    bool fma = false;
    bool avx512bw = false;
    /// ARM Scalable Vector Extension (Linux/aarch64 via HWCAP; false elsewhere).
    bool sve = false;
};

/// Detect CPU features once (thread-safe).
[[nodiscard]] const CpuFeatures& cpu_features();

[[nodiscard]] inline bool cpu_has_avx2() { return cpu_features().avx2; }
[[nodiscard]] inline bool cpu_has_avx512() {
    return cpu_features().avx512f && cpu_features().avx512bw;
}

/// Cache-line size used for padding (false-sharing avoidance).
inline constexpr std::size_t kCacheLineSize = 64;

template <typename T>
struct alignas(kCacheLineSize) CacheLinePadded {
    T value{};
};

} // namespace nexusdata
