#include "nexusdata/core/allocator.hpp"

#include <cstdlib>
#include <new>

#if defined(_WIN32)
#include <malloc.h>
#endif

namespace nexusdata {

void* aligned_alloc_bytes(std::size_t bytes, std::size_t alignment) {
    if (bytes == 0) {
        return nullptr;
    }
    if (alignment < sizeof(void*) || (alignment & (alignment - 1)) != 0) {
        alignment = kDefaultAlignment;
    }

#if defined(_WIN32)
    void* ptr = _aligned_malloc(bytes, alignment);
    if (!ptr) {
        throw std::bad_alloc();
    }
    return ptr;
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, bytes) != 0 || ptr == nullptr) {
        throw std::bad_alloc();
    }
    return ptr;
#endif
}

void aligned_free_bytes(void* ptr) noexcept {
    if (!ptr) {
        return;
    }
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

AlignedBuffer make_aligned_buffer(std::size_t bytes, std::size_t alignment) {
    if (bytes == 0) {
        return AlignedBuffer{};
    }
    void* ptr = aligned_alloc_bytes(bytes, alignment);
    return AlignedBuffer(ptr, AlignedDeleter{});
}

} // namespace nexusdata
