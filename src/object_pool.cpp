#include "nexusdata/core/object_pool.hpp"

#include <cstring>
#include <map>

#include "nexusdata/backend/pinned.hpp"
#include "nexusdata/core/error.hpp"
#include "nexusdata/core/ndarray.hpp"

namespace nexusdata {

struct BufferPool::State {
    std::mutex mu;
    std::map<std::size_t, std::vector<void*>> free;
    std::size_t cached_bytes = 0;
    std::size_t hits = 0;
    std::size_t misses = 0;
    std::size_t outstanding = 0;
    bool closed = false;
    BufferPoolOptions opt;

    void* upstream_alloc(std::size_t bytes) const {
        if (opt.alloc_fn != nullptr) {
            void* p = opt.alloc_fn(bytes);
            if (p == nullptr) {
                throw std::bad_alloc();
            }
            return p;
        }
        return aligned_alloc_bytes(bytes, opt.alignment);
    }

    void upstream_free(void* p) const noexcept {
        if (opt.free_fn != nullptr) {
            opt.free_fn(p);
        } else {
            aligned_free_bytes(p);
        }
    }

    void release(void* p, std::size_t cls) noexcept {
        {
            std::lock_guard lock(mu);
            --outstanding;
            if (!closed && cached_bytes + cls <= opt.max_cached_bytes) {
                try {
                    free[cls].push_back(p);
                    cached_bytes += cls;
                    return;
                } catch (...) {
                    // fall through to free
                }
            }
        }
        upstream_free(p);
    }

    void free_all_locked() noexcept {
        for (auto& [cls, list] : free) {
            (void)cls;
            for (void* p : list) {
                upstream_free(p);
            }
        }
        free.clear();
        cached_bytes = 0;
    }
};

BufferPool::BufferPool(BufferPoolOptions options) : state_(std::make_shared<State>()) {
    if ((options.alloc_fn == nullptr) != (options.free_fn == nullptr)) {
        throw InvalidArgumentError("BufferPool: alloc_fn and free_fn must be set together");
    }
    if (options.alignment == 0 || (options.alignment & (options.alignment - 1)) != 0) {
        throw InvalidArgumentError("BufferPool: alignment must be a power of two");
    }
    state_->opt = options;
}

BufferPool::~BufferPool() {
    std::lock_guard lock(state_->mu);
    state_->closed = true;
    state_->free_all_locked();
}

std::unique_ptr<BufferPool> BufferPool::pinned(std::size_t max_cached_bytes) {
    BufferPoolOptions o;
    o.max_cached_bytes = max_cached_bytes;
    o.alloc_fn = &pinned_alloc;
    o.free_fn = &pinned_free;
    return std::make_unique<BufferPool>(o);
}

std::size_t BufferPool::size_class(std::size_t bytes) noexcept {
    if (bytes <= 64) {
        return 64;
    }
    std::size_t base = 64;
    while (base * 2 < bytes) {
        base *= 2;
    }
    // bytes in (base, 2 * base]
    const std::size_t step = base / 4;
    return base + ((bytes - base + step - 1) / step) * step;
}

AlignedBuffer BufferPool::acquire(std::size_t bytes) {
    if (bytes == 0) {
        return {};
    }
    const std::size_t cls = size_class(bytes);
    void* p = nullptr;
    {
        std::lock_guard lock(state_->mu);
        auto it = state_->free.find(cls);
        if (it != state_->free.end() && !it->second.empty()) {
            p = it->second.back();
            it->second.pop_back();
            state_->cached_bytes -= cls;
            ++state_->hits;
        } else {
            ++state_->misses;
        }
        ++state_->outstanding;
    }
    if (p == nullptr) {
        try {
            p = state_->upstream_alloc(cls);
        } catch (...) {
            std::lock_guard lock(state_->mu);
            --state_->outstanding;
            throw;
        }
    }
    std::shared_ptr<State> st = state_;
    return AlignedBuffer(p, [st, cls](void* q) { st->release(q, cls); });
}

NDArray BufferPool::make_array(Shape shape, DType dtype, bool zero) {
    const std::size_t bytes = ::nexusdata::numel(shape) * size_of(dtype);
    AlignedBuffer buf = acquire(bytes);
    if (zero && bytes > 0) {
        std::memset(buf.get(), 0, bytes);
    }
    return NDArray::from_shared(std::move(buf), std::move(shape), dtype, Device::host());
}

void BufferPool::trim() {
    std::lock_guard lock(state_->mu);
    state_->free_all_locked();
}

BufferPool::Stats BufferPool::stats() const {
    std::lock_guard lock(state_->mu);
    return Stats{state_->hits, state_->misses, state_->cached_bytes, state_->outstanding};
}

} // namespace nexusdata
