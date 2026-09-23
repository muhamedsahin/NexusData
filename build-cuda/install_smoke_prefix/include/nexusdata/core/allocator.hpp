#pragma once

#include <cstddef>
#include <memory>

namespace nexusdata {

/// Default alignment for NDArray buffers (cache-line friendly; SIMD-ready).
inline constexpr std::size_t kDefaultAlignment = 64;

/// Allocate @p bytes aligned to @p alignment (power of two, >= sizeof(void*)).
/// Throws std::bad_alloc on failure.
[[nodiscard]] void* aligned_alloc_bytes(std::size_t bytes, std::size_t alignment = kDefaultAlignment);

/// Free memory returned by aligned_alloc_bytes.
void aligned_free_bytes(void* ptr) noexcept;

/// Deleter for aligned buffers used with std::shared_ptr.
struct AlignedDeleter {
    void operator()(void* ptr) const noexcept { aligned_free_bytes(ptr); }
};

using AlignedBuffer = std::shared_ptr<void>;

[[nodiscard]] AlignedBuffer make_aligned_buffer(std::size_t bytes,
                                                std::size_t alignment = kDefaultAlignment);

} // namespace nexusdata
