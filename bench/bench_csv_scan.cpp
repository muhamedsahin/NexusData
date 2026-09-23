#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "nexusdata/backend/cpu_dispatch.hpp"
#include "nexusdata/simd/byte_scan.hpp"

using namespace nexusdata;

int main() {
    constexpr std::size_t n = 64 * 1024 * 1024; // 64 MiB
    std::vector<std::uint8_t> buf(n);
    for (std::size_t i = 0; i < n; ++i) {
        buf[i] = static_cast<std::uint8_t>(i % 251);
        if (i % 64 == 0) {
            buf[i] = '\n';
        }
    }

    const auto& feat = cpu_features();
    std::cout << "avx2=" << feat.avx2 << " sse2=" << feat.sse2 << "\n";

    auto bench = [&](const char* name, auto&& fn) {
        // warmup
        fn();
        const auto t0 = std::chrono::steady_clock::now();
        auto hits = fn();
        const auto t1 = std::chrono::steady_clock::now();
        const double sec = std::chrono::duration<double>(t1 - t0).count();
        const double gbs = (n / (1024.0 * 1024.0 * 1024.0)) / sec;
        std::cout << name << ": hits=" << hits.size() << " time=" << sec
                  << "s throughput=" << gbs << " GiB/s\n";
    };

    bench("scalar", [&] {
        return find_bytes_scalar(buf.data(), buf.size(), '\n');
    });
    bench("dispatch", [&] { return find_bytes(buf.data(), buf.size(), '\n'); });
    return 0;
}
