#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "nexusdata/backend/device.hpp"
#include "nexusdata/core/error.hpp"

namespace nexusdata {

/// Page-locked (pinned) host memory when CUDA is enabled; else 64-byte aligned host RAM.
[[nodiscard]] void* pinned_alloc(std::size_t bytes);
void pinned_free(void* ptr) noexcept;

struct PinnedDeleter {
    void operator()(void* p) const noexcept { pinned_free(p); }
};

using PinnedBuffer = std::shared_ptr<void>;

[[nodiscard]] inline PinnedBuffer make_pinned_buffer(std::size_t bytes) {
    if (bytes == 0) {
        return {};
    }
    return PinnedBuffer(pinned_alloc(bytes), PinnedDeleter{});
}

/// Simple pool of pinned slabs for H2D staging (double/triple buffering).
class PinnedPool {
public:
    PinnedPool(std::size_t slab_bytes, std::size_t num_slabs)
        : slab_bytes_(slab_bytes) {
        if (num_slabs == 0 || slab_bytes == 0) {
            throw InvalidArgumentError("PinnedPool: invalid size");
        }
        slabs_.reserve(num_slabs);
        for (std::size_t i = 0; i < num_slabs; ++i) {
            slabs_.push_back(make_pinned_buffer(slab_bytes));
        }
    }

    [[nodiscard]] std::size_t num_slabs() const noexcept { return slabs_.size(); }
    [[nodiscard]] std::size_t slab_bytes() const noexcept { return slab_bytes_; }
    [[nodiscard]] void* slab(std::size_t i) const {
        if (i >= slabs_.size()) {
            throw IndexError("PinnedPool: slab index out of range");
        }
        return slabs_[i].get();
    }

private:
    std::size_t slab_bytes_ = 0;
    std::vector<PinnedBuffer> slabs_;
};

} // namespace nexusdata
