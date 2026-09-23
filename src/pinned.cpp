#include "nexusdata/backend/pinned.hpp"

#include "nexusdata/backend/cuda_api.hpp"
#include "nexusdata/core/allocator.hpp"

namespace nexusdata {

void* pinned_alloc(std::size_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }
    if (cuda_api::available()) {
        return cuda_api::host_alloc_pinned(bytes);
    }
    return aligned_alloc_bytes(bytes, kDefaultAlignment);
}

void pinned_free(void* ptr) noexcept {
    if (!ptr) {
        return;
    }
    if (cuda_api::available()) {
        cuda_api::host_free_pinned(ptr);
        return;
    }
    aligned_free_bytes(ptr);
}

} // namespace nexusdata
