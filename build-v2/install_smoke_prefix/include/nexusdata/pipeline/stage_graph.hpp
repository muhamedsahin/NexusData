#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "nexusdata/backend/bounded_queue.hpp"
#include "nexusdata/core/error.hpp"

namespace nexusdata {

/// Per-stage parallelism and output buffering.
struct StageOptions {
    std::size_t num_threads = 1;
    /// Capacity of the queue this stage writes into (backpressure bound).
    std::size_t queue_capacity = 4;
};

struct StageGraphOptions {
    /// Deliver items in source order (reorder buffer) or as soon as they finish.
    bool in_order = true;
    /// Max items between source and consumer (credit window). 0 = sum over
    /// stages of (queue_capacity + num_threads). Bounds memory and the reorder buffer.
    std::size_t max_in_flight = 0;
};

/// Snapshot of one stage's counters.
struct StageStats {
    std::string name;
    std::size_t num_threads = 0;
    std::size_t queue_capacity = 0;
    /// Items currently waiting in this stage's output queue.
    std::size_t queue_size = 0;
    std::uint64_t items = 0;
    /// Summed wall time spent inside the stage function (all threads).
    double busy_seconds = 0.0;
};

namespace detail {

template <typename T>
struct StageEnvelope {
    std::size_t seq = 0;
    std::optional<T> value;
    std::exception_ptr error;
};

struct StageCounters {
    std::string name;
    std::size_t num_threads = 0;
    std::size_t queue_capacity = 0;
    std::atomic<std::uint64_t> items{0};
    std::atomic<std::uint64_t> busy_ns{0};
    std::function<std::size_t()> queue_size;
};

/// Shared runtime of a StageGraph: threads, shutdown and the credit window.
class StageGraphCore {
public:
    ~StageGraphCore() { stop(); }

    void add_stage(std::shared_ptr<StageCounters> c, std::function<void()> launch,
                   std::function<void()> close) {
        counters_.push_back(std::move(c));
        launchers_.push_back(std::move(launch));
        closers_.push_back(std::move(close));
    }

    void spawn(std::function<void()> fn) { threads_.emplace_back(std::move(fn)); }

    void start(const StageGraphOptions& opt) {
        if (started_) {
            return;
        }
        started_ = true;
        std::size_t window = opt.max_in_flight;
        if (window == 0) {
            for (const auto& c : counters_) {
                window += c->queue_capacity + c->num_threads;
            }
        }
        window_ = std::max<std::size_t>(window, 1);
        for (auto& l : launchers_) {
            l();
        }
    }

    void stop() {
        {
            std::lock_guard lock(mu_);
            if (stopped_) {
                return;
            }
            stopped_ = true;
            stopping_ = true;
        }
        credit_cv_.notify_all();
        for (auto& c : closers_) {
            c();
        }
        for (auto& t : threads_) {
            if (t.joinable()) {
                t.join();
            }
        }
        threads_.clear();
    }

    [[nodiscard]] bool stopping() const {
        std::lock_guard lock(mu_);
        return stopping_;
    }

    /// Source side: wait until @p seq fits in the window. False when stopping.
    bool acquire_credit(std::size_t seq) {
        std::unique_lock lock(mu_);
        credit_cv_.wait(lock, [&] { return stopping_ || seq < consumed_ + window_; });
        return !stopping_;
    }

    /// Consumer side: one item left the graph.
    void release_credit() {
        {
            std::lock_guard lock(mu_);
            ++consumed_;
        }
        credit_cv_.notify_all();
    }

    [[nodiscard]] std::vector<StageStats> stats() const {
        std::vector<StageStats> out;
        out.reserve(counters_.size());
        for (const auto& c : counters_) {
            StageStats s;
            s.name = c->name;
            s.num_threads = c->num_threads;
            s.queue_capacity = c->queue_capacity;
            s.queue_size = c->queue_size ? c->queue_size() : 0;
            s.items = c->items.load(std::memory_order_relaxed);
            s.busy_seconds =
                static_cast<double>(c->busy_ns.load(std::memory_order_relaxed)) * 1e-9;
            out.push_back(std::move(s));
        }
        return out;
    }

    [[nodiscard]] std::size_t window() const noexcept { return window_; }

private:
    std::vector<std::shared_ptr<StageCounters>> counters_;
    std::vector<std::function<void()>> launchers_;
    std::vector<std::function<void()>> closers_;
    std::vector<std::thread> threads_;
    mutable std::mutex mu_;
    std::condition_variable credit_cv_;
    std::size_t consumed_ = 0;
    std::size_t window_ = 1;
    bool started_ = false;
    bool stopping_ = false;
    bool stopped_ = false;
};

} // namespace detail

/// Multi-stage asynchronous pipeline:
///
///   source -> stage 1 (N1 threads) -> queue -> stage 2 (N2 threads) -> ... -> next()
///
/// Every stage has its own thread count and bounded output queue, so a slow
/// stage applies backpressure to the ones before it. Items carry their source
/// sequence number; with in_order = true a reorder buffer at the sink restores
/// source order even though workers finish out of order. Exceptions thrown by a
/// stage are delivered by next() at the failing item's position.
///
/// Threads start on the first next() (or start()) and are joined by stop() /
/// the destructor, which also discards undelivered items.
///
/// Thread-safety: next() must be called from one consumer thread at a time.
template <typename T>
class StageGraph {
public:
    using Envelope = detail::StageEnvelope<T>;
    using Queue = BoundedQueue<Envelope>;

    /// Sequential source: @p gen returns nullopt at end of stream.
    [[nodiscard]] static StageGraph from_generator(std::string name,
                                                   std::function<std::optional<T>()> gen,
                                                   std::size_t queue_capacity = 4,
                                                   StageGraphOptions options = {}) {
        StageGraph g;
        g.core_ = std::make_shared<detail::StageGraphCore>();
        g.options_ = std::make_shared<StageGraphOptions>(options);
        auto out = std::make_shared<Queue>(queue_capacity);
        auto counters = std::make_shared<detail::StageCounters>();
        counters->name = std::move(name);
        counters->num_threads = 1;
        counters->queue_capacity = queue_capacity;
        counters->queue_size = [out] { return out->size(); };
        auto fn = std::make_shared<std::function<std::optional<T>()>>(std::move(gen));
        detail::StageGraphCore* core = g.core_.get();
        g.core_->add_stage(
            counters,
            [core, out, fn, counters] {
                core->spawn([core, out, fn, counters] {
                    for (std::size_t seq = 0;; ++seq) {
                        if (!core->acquire_credit(seq)) {
                            break;
                        }
                        Envelope env;
                        env.seq = seq;
                        const auto t0 = std::chrono::steady_clock::now();
                        bool done = false;
                        try {
                            env.value = (*fn)();
                            done = !env.value.has_value();
                        } catch (...) {
                            env.error = std::current_exception();
                        }
                        counters->busy_ns.fetch_add(
                            static_cast<std::uint64_t>(
                                std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now() - t0)
                                    .count()),
                            std::memory_order_relaxed);
                        if (done) {
                            break;
                        }
                        counters->items.fetch_add(1, std::memory_order_relaxed);
                        const bool failed = env.error != nullptr;
                        if (!out->push(std::move(env)) || failed) {
                            break;
                        }
                    }
                    out->close();
                });
            },
            [out] { out->close(); });
        g.tail_ = out;
        return g;
    }

    /// Source over indices [0, n).
    [[nodiscard]] static StageGraph from_range(std::string name, std::size_t n,
                                               std::size_t queue_capacity = 4,
                                               StageGraphOptions options = {}) {
        static_assert(std::is_convertible_v<std::size_t, T>, "from_range needs T = size_t");
        auto next = std::make_shared<std::size_t>(0);
        return from_generator(
            std::move(name),
            [next, n]() -> std::optional<T> {
                if (*next >= n) {
                    return std::nullopt;
                }
                return static_cast<T>((*next)++);
            },
            queue_capacity, options);
    }

    /// Append a stage running @p fn(T&&) -> U on opt.num_threads threads.
    template <typename F>
    [[nodiscard]] auto then(std::string name, StageOptions opt, F&& fn) && {
        using U = std::decay_t<std::invoke_result_t<F&, T&&>>;
        if (!core_) {
            throw InvalidArgumentError("StageGraph::then on a moved-from graph");
        }
        if (opt.num_threads == 0 || opt.queue_capacity == 0) {
            throw InvalidArgumentError("StageGraph::then(" + name +
                                       "): num_threads and queue_capacity must be > 0");
        }
        using OutQueue = BoundedQueue<detail::StageEnvelope<U>>;
        auto in = tail_;
        auto out = std::make_shared<OutQueue>(opt.queue_capacity);
        auto counters = std::make_shared<detail::StageCounters>();
        counters->name = std::move(name);
        counters->num_threads = opt.num_threads;
        counters->queue_capacity = opt.queue_capacity;
        counters->queue_size = [out] { return out->size(); };
        auto f = std::make_shared<std::decay_t<F>>(std::forward<F>(fn));
        auto remaining = std::make_shared<std::atomic<std::size_t>>(opt.num_threads);
        detail::StageGraphCore* core = core_.get();
        const std::size_t threads = opt.num_threads;
        core_->add_stage(
            counters,
            [core, in, out, f, counters, remaining, threads] {
                for (std::size_t t = 0; t < threads; ++t) {
                    core->spawn([in, out, f, counters, remaining] {
                        for (;;) {
                            auto item = in->pop();
                            if (!item) {
                                break;
                            }
                            detail::StageEnvelope<U> env;
                            env.seq = item->seq;
                            env.error = item->error;
                            if (!env.error) {
                                const auto t0 = std::chrono::steady_clock::now();
                                try {
                                    env.value.emplace((*f)(std::move(*item->value)));
                                } catch (...) {
                                    env.error = std::current_exception();
                                }
                                counters->busy_ns.fetch_add(
                                    static_cast<std::uint64_t>(
                                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                                            std::chrono::steady_clock::now() - t0)
                                            .count()),
                                    std::memory_order_relaxed);
                                counters->items.fetch_add(1, std::memory_order_relaxed);
                            }
                            if (!out->push(std::move(env))) {
                                break;
                            }
                        }
                        if (remaining->fetch_sub(1) == 1) {
                            out->close();
                        }
                    });
                }
            },
            [out] { out->close(); });

        StageGraph<U> g;
        g.core_ = std::move(core_);
        g.options_ = std::move(options_);
        g.tail_ = out;
        tail_.reset();
        return g;
    }

    /// Launch all stage threads (idempotent; next() calls it implicitly).
    void start() {
        if (!core_) {
            throw InvalidArgumentError("StageGraph::start on a moved-from graph");
        }
        core_->start(*options_);
    }

    /// Next item, or nullopt at end of stream. Rethrows stage exceptions.
    [[nodiscard]] std::optional<T> next() {
        start();
        if (options_->in_order) {
            for (;;) {
                auto it = reorder_.find(expected_);
                if (it != reorder_.end()) {
                    Envelope env = std::move(it->second);
                    reorder_.erase(it);
                    return deliver(std::move(env));
                }
                auto item = tail_->pop();
                if (!item) {
                    return std::nullopt;
                }
                reorder_.emplace(item->seq, std::move(*item));
            }
        }
        auto item = tail_->pop();
        if (!item) {
            return std::nullopt;
        }
        return deliver(std::move(*item));
    }

    /// Stop all threads and drop pending items. Safe to call repeatedly.
    void stop() {
        if (core_) {
            core_->stop();
        }
        reorder_.clear();
    }

    [[nodiscard]] std::vector<StageStats> stats() const {
        return core_ ? core_->stats() : std::vector<StageStats>{};
    }

    /// Items currently parked in the reorder buffer.
    [[nodiscard]] std::size_t reorder_buffer_size() const noexcept { return reorder_.size(); }
    [[nodiscard]] std::size_t in_flight_window() const noexcept {
        return core_ ? core_->window() : 0;
    }
    /// Items waiting in the final queue (for adaptive prefetch / cost decisions).
    [[nodiscard]] std::size_t ready_items() const { return tail_ ? tail_->size() : 0; }

    StageGraph() = default;
    StageGraph(StageGraph&&) noexcept = default;
    StageGraph& operator=(StageGraph&& other) noexcept {
        if (this != &other) {
            stop();
            core_ = std::move(other.core_);
            options_ = std::move(other.options_);
            tail_ = std::move(other.tail_);
            reorder_ = std::move(other.reorder_);
            expected_ = other.expected_;
        }
        return *this;
    }
    StageGraph(const StageGraph&) = delete;
    StageGraph& operator=(const StageGraph&) = delete;
    ~StageGraph() { stop(); }

private:
    template <typename>
    friend class StageGraph;

    std::optional<T> deliver(Envelope env) {
        ++expected_;
        core_->release_credit();
        if (env.error) {
            std::rethrow_exception(env.error);
        }
        return std::move(env.value);
    }

    std::shared_ptr<detail::StageGraphCore> core_;
    std::shared_ptr<StageGraphOptions> options_;
    std::shared_ptr<Queue> tail_;
    std::map<std::size_t, Envelope> reorder_;
    std::size_t expected_ = 0;
};

} // namespace nexusdata
