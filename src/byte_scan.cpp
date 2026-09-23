#include "nexusdata/simd/byte_scan.hpp"

#include <cstring>

#include "nexusdata/backend/cpu_dispatch.hpp"

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace nexusdata {

namespace {

unsigned ctz32(std::uint32_t x) {
#if defined(_MSC_VER)
    unsigned long idx = 0;
    _BitScanForward(&idx, x);
    return static_cast<unsigned>(idx);
#else
    return static_cast<unsigned>(__builtin_ctz(x));
#endif
}

unsigned ctz64(std::uint64_t x) {
#if defined(_MSC_VER)
    unsigned long idx = 0;
    _BitScanForward64(&idx, x);
    return static_cast<unsigned>(idx);
#else
    return static_cast<unsigned>(__builtin_ctzll(x));
#endif
}

/// SWAR byte search: process 8 bytes per iteration (v0.8 scalar hotspot).
std::vector<std::size_t> find_bytes_swar(const std::uint8_t* data, std::size_t len,
                                         std::uint8_t needle) {
    std::vector<std::size_t> out;
    out.reserve(len / 64 + 1);
    const std::uint64_t n64 = 0x0101010101010101ULL * static_cast<std::uint64_t>(needle);
    std::size_t i = 0;
    for (; i + 8 <= len; i += 8) {
        std::uint64_t chunk = 0;
        std::memcpy(&chunk, data + i, 8);
        const std::uint64_t x = chunk ^ n64;
        std::uint64_t y = (x - 0x0101010101010101ULL) & ~x & 0x8080808080808080ULL;
        while (y != 0) {
            const unsigned bit = ctz64(y) / 8;
            out.push_back(i + bit);
            y &= y - 1;
        }
    }
    for (; i < len; ++i) {
        if (data[i] == needle) {
            out.push_back(i);
        }
    }
    return out;
}

} // namespace

std::vector<std::size_t> find_bytes_scalar(const std::uint8_t* data, std::size_t len,
                                           std::uint8_t needle) {
    return find_bytes_swar(data, len, needle);
}

#if defined(__AVX2__)
namespace {

std::vector<std::size_t> find_bytes_avx2(const std::uint8_t* data, std::size_t len,
                                         std::uint8_t needle) {
    std::vector<std::size_t> out;
    out.reserve(len / 64 + 1);
    const __m256i vneedle = _mm256_set1_epi8(static_cast<char>(needle));
    std::size_t i = 0;
    for (; i + 32 <= len; i += 32) {
        const __m256i chunk =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i));
        const __m256i eq = _mm256_cmpeq_epi8(chunk, vneedle);
        std::uint32_t mask = static_cast<std::uint32_t>(_mm256_movemask_epi8(eq));
        while (mask) {
            const unsigned bit = ctz32(mask);
            out.push_back(i + bit);
            mask &= mask - 1;
        }
    }
    for (; i < len; ++i) {
        if (data[i] == needle) {
            out.push_back(i);
        }
    }
    return out;
}

} // namespace
#endif

std::vector<std::size_t> find_bytes(const std::uint8_t* data, std::size_t len,
                                    std::uint8_t needle) {
#if defined(__AVX2__)
    if (cpu_has_avx2()) {
        return find_bytes_avx2(data, len, needle);
    }
#endif
    return find_bytes_scalar(data, len, needle);
}

std::vector<std::size_t> find_csv_record_starts(const std::uint8_t* data, std::size_t len) {
    std::vector<std::size_t> starts;
    starts.reserve(len / 32 + 1);
    starts.push_back(0);
    if (len == 0) {
        return starts;
    }

    // Fast path (v0.8): no '"' → single newline SIMD/SWAR scan.
    if (std::memchr(data, '"', len) == nullptr) {
        const auto nls = find_bytes(data, len, static_cast<std::uint8_t>('\n'));
        for (std::size_t o : nls) {
            if (o + 1 < len) {
                starts.push_back(o + 1);
            }
        }
        return starts;
    }

    bool in_quotes = false;
    for (std::size_t i = 0; i < len; ++i) {
        const std::uint8_t c = data[i];
        if (c == '"') {
            if (in_quotes && i + 1 < len && data[i + 1] == '"') {
                ++i;
            } else {
                in_quotes = !in_quotes;
            }
        } else if ((c == '\n') && !in_quotes) {
            if (i + 1 < len) {
                starts.push_back(i + 1);
            }
        }
    }
    return starts;
}

} // namespace nexusdata
