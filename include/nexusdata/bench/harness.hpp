#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>

namespace nexusdata {
namespace bench {

/// Steady-clock stopwatch for microbenchmarks.
class Timer {
public:
    void start() { t0_ = std::chrono::steady_clock::now(); }
    [[nodiscard]] double stop_seconds() const {
        const auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(t1 - t0_).count();
    }

private:
    std::chrono::steady_clock::time_point t0_{};
};

/// True when @p seconds is at most baseline * (1 + slack_ratio).
[[nodiscard]] inline bool within_budget(double seconds, double baseline, double slack_ratio = 0.5) {
    return seconds <= baseline * (1.0 + slack_ratio);
}

/// Run @p fn @p iters times after one warmup; return best seconds.
template <typename Fn>
double time_best(Fn&& fn, int iters = 5) {
    fn(); // warmup
    double best = 1e300;
    for (int i = 0; i < iters; ++i) {
        Timer t;
        t.start();
        fn();
        best = std::min(best, t.stop_seconds());
    }
    return best;
}

inline void print_result(const std::string& name, double seconds, std::size_t bytes = 0,
                         std::size_t ops = 0) {
    std::cout << std::left << std::setw(28) << name << "  " << std::fixed << std::setprecision(6)
              << seconds << " s";
    if (bytes > 0 && seconds > 0.0) {
        const double gib = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
        std::cout << "  " << std::setprecision(2) << (gib / seconds) << " GiB/s";
    }
    if (ops > 0 && seconds > 0.0) {
        std::cout << "  " << std::setprecision(1) << (ops / seconds) << " ops/s";
    }
    std::cout << "\n";
}

} // namespace bench
} // namespace nexusdata
