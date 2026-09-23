#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include "nexusdata/backend/cpu_dispatch.hpp"
#include "nexusdata/core/error.hpp"

namespace nexusdata {

/// Bounded blocking queue (MPMC) for prefetch backpressure.
/// Thread-safety: push/pop are thread-safe.
template <typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(std::size_t capacity) : cap_(capacity), buf_(capacity) {
        if (capacity == 0) {
            throw InvalidArgumentError("BoundedQueue: capacity must be > 0");
        }
    }

    void close() {
        {
            std::lock_guard lock(mu_);
            closed_ = true;
        }
        not_empty_.notify_all();
        not_full_.notify_all();
    }

    [[nodiscard]] bool closed() const {
        std::lock_guard lock(mu_);
        return closed_;
    }

    /// Blocks until space available or closed. Returns false if closed.
    bool push(T item) {
        std::unique_lock lock(mu_);
        not_full_.wait(lock, [&] { return closed_ || size_ < cap_; });
        if (closed_) {
            return false;
        }
        buf_[tail_] = std::move(item);
        tail_ = (tail_ + 1) % cap_;
        ++size_;
        not_empty_.notify_one();
        return true;
    }

    /// Blocks until item available or closed+empty. Returns nullopt if closed and empty.
    std::optional<T> pop() {
        std::unique_lock lock(mu_);
        not_empty_.wait(lock, [&] { return closed_ || size_ > 0; });
        if (size_ == 0) {
            return std::nullopt;
        }
        T item = std::move(buf_[head_]);
        head_ = (head_ + 1) % cap_;
        --size_;
        not_full_.notify_one();
        return item;
    }

    [[nodiscard]] std::size_t size() const {
        std::lock_guard lock(mu_);
        return size_;
    }

private:
    const std::size_t cap_;
    std::vector<T> buf_;
    std::size_t head_ = 0;
    std::size_t tail_ = 0;
    std::size_t size_ = 0;
    bool closed_ = false;
    mutable std::mutex mu_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
};

} // namespace nexusdata
