#pragma once

#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "nexusdata/core/allocator.hpp"
#include "nexusdata/core/dtype.hpp"
#include "nexusdata/core/shape.hpp"

namespace nexusdata {

class NDArray;

/// Free-list pool of reusable objects. Handles return their object to the pool
/// on destruction (after the optional reset hook); if the pool is gone or full
/// the object is simply deleted.
///
/// Thread-safety: acquire() and handle destruction are thread-safe.
template <typename T>
class ObjectPool {
    struct State {
        std::mutex mu;
        std::vector<std::unique_ptr<T>> free;
        std::size_t max_cached = 0;
        std::function<void(T&)> reset;
        std::atomic<std::size_t> created{0};
        std::atomic<std::size_t> reused{0};
    };

public:
    struct Returner {
        std::weak_ptr<State> state;
        void operator()(T* p) const noexcept {
            std::unique_ptr<T> obj(p);
            auto st = state.lock();
            if (!st) {
                return;
            }
            try {
                if (st->reset) {
                    st->reset(*obj);
                }
                std::lock_guard lock(st->mu);
                if (st->free.size() < st->max_cached) {
                    st->free.push_back(std::move(obj));
                }
            } catch (...) {
                // Object is dropped (deleted by unique_ptr).
            }
        }
    };

    using Handle = std::unique_ptr<T, Returner>;
    using Factory = std::function<std::unique_ptr<T>()>;

    explicit ObjectPool(Factory factory = [] { return std::make_unique<T>(); },
                        std::function<void(T&)> reset = {}, std::size_t max_cached = 256)
        : factory_(std::move(factory)), state_(std::make_shared<State>()) {
        state_->max_cached = max_cached;
        state_->reset = std::move(reset);
    }

    ObjectPool(const ObjectPool&) = delete;
    ObjectPool& operator=(const ObjectPool&) = delete;

    [[nodiscard]] Handle acquire() {
        {
            std::lock_guard lock(state_->mu);
            if (!state_->free.empty()) {
                T* p = state_->free.back().release();
                state_->free.pop_back();
                state_->reused.fetch_add(1, std::memory_order_relaxed);
                return Handle(p, Returner{state_});
            }
        }
        state_->created.fetch_add(1, std::memory_order_relaxed);
        return Handle(factory_().release(), Returner{state_});
    }

    [[nodiscard]] std::size_t cached() const {
        std::lock_guard lock(state_->mu);
        return state_->free.size();
    }
    [[nodiscard]] std::size_t created() const noexcept { return state_->created.load(); }
    [[nodiscard]] std::size_t reused() const noexcept { return state_->reused.load(); }

private:
    Factory factory_;
    std::shared_ptr<State> state_;
};

struct BufferPoolOptions {
    /// Upper bound on idle bytes kept for reuse; larger returns are freed.
    std::size_t max_cached_bytes = std::size_t{256} << 20;
    std::size_t alignment = kDefaultAlignment;
    /// Custom upstream allocator (e.g. pinned_alloc/pinned_free). Both or neither.
    void* (*alloc_fn)(std::size_t) = nullptr;
    void (*free_fn)(void*) = nullptr;
};

/// Size-class recycling pool for tensor storage. acquire() rounds the request up
/// to a size class (quarter power-of-two steps, <= 25% slack) and returns a
/// buffer whose deleter hands the memory back to the pool.
///
/// Thread-safety: all methods are thread-safe; buffers may be released from any thread,
/// including after the pool is destroyed (they are then freed directly).
class BufferPool {
public:
    explicit BufferPool(BufferPoolOptions options = {});
    ~BufferPool();

    BufferPool(const BufferPool&) = delete;
    BufferPool& operator=(const BufferPool&) = delete;

    /// Pool backed by page-locked memory when CUDA is available (plain aligned RAM otherwise).
    [[nodiscard]] static std::unique_ptr<BufferPool> pinned(std::size_t max_cached_bytes =
                                                                std::size_t{256} << 20);

    [[nodiscard]] AlignedBuffer acquire(std::size_t bytes);

    /// Host NDArray backed by a pooled buffer (zero-filled unless @p zero is false).
    [[nodiscard]] NDArray make_array(Shape shape, DType dtype, bool zero = true);

    /// Free every idle buffer.
    void trim();

    struct Stats {
        std::size_t hits = 0;
        std::size_t misses = 0;
        std::size_t cached_bytes = 0;
        std::size_t outstanding = 0;
    };
    [[nodiscard]] Stats stats() const;

    [[nodiscard]] static std::size_t size_class(std::size_t bytes) noexcept;

    struct State;

private:
    std::shared_ptr<State> state_;
};

} // namespace nexusdata
