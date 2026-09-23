#pragma once

#include <atomic>
#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "nexusdata/backend/cpu_dispatch.hpp"
#include "nexusdata/core/error.hpp"

namespace nexusdata {

/// Fixed-size worker pool with a mutex + condition_variable task queue.
/// (Work-stealing can replace this later; v0.4 prioritizes correctness + TSan cleanliness.)
///
/// Thread-safety: submit() is thread-safe. Destroying the pool joins all workers.
class ThreadPool {
public:
    explicit ThreadPool(std::size_t num_threads = 0) {
        if (num_threads == 0) {
            num_threads = std::max<std::size_t>(1, std::thread::hardware_concurrency());
        }
        workers_.reserve(num_threads);
        for (std::size_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard lock(mu_);
            stopping_ = true;
        }
        cv_.notify_all();
        for (auto& w : workers_) {
            if (w.joinable()) {
                w.join();
            }
        }
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    [[nodiscard]] std::size_t size() const noexcept { return workers_.size(); }

    template <typename F>
    auto submit(F&& f) -> std::future<std::invoke_result_t<F>> {
        using R = std::invoke_result_t<F>;
        auto task = std::make_shared<std::packaged_task<R()>>(std::forward<F>(f));
        std::future<R> fut = task->get_future();
        {
            std::lock_guard lock(mu_);
            if (stopping_) {
                throw InvalidArgumentError("ThreadPool: submit on stopped pool");
            }
            tasks_.emplace([task]() { (*task)(); });
        }
        cv_.notify_one();
        return fut;
    }

    /// Run @p n tasks over indices [0, n), blocking until all complete.
    template <typename F>
    void parallel_for(std::size_t n, F&& f) {
        if (n == 0) {
            return;
        }
        if (workers_.empty() || n == 1) {
            for (std::size_t i = 0; i < n; ++i) {
                f(i);
            }
            return;
        }
        std::vector<std::future<void>> futs;
        futs.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            futs.push_back(submit([&f, i] { f(i); }));
        }
        for (auto& fut : futs) {
            fut.get();
        }
    }

private:
    void worker_loop() {
        for (;;) {
            std::function<void()> task;
            {
                std::unique_lock lock(mu_);
                cv_.wait(lock, [&] { return stopping_ || !tasks_.empty(); });
                if (stopping_ && tasks_.empty()) {
                    return;
                }
                task = std::move(tasks_.front());
                tasks_.pop();
            }
            task();
        }
    }

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool stopping_ = false;
};

/// Process-wide default pool (lazy).
[[nodiscard]] ThreadPool& default_thread_pool();

} // namespace nexusdata
