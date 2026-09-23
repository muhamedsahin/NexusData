/// NexusData v0.8 microbenchmark suite (Release recommended).
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "nexusdata/backend/cpu_dispatch.hpp"
#include "nexusdata/backend/device.hpp"
#include "nexusdata/bench/harness.hpp"
#include "nexusdata/dataset/csv/csv_dataset.hpp"
#include "nexusdata/dataset/in_memory.hpp"
#include "nexusdata/loading/batch.hpp"
#include "nexusdata/loading/dataloader.hpp"
#include "nexusdata/simd/byte_scan.hpp"

using namespace nexusdata;

namespace {

std::string write_csv(const std::string& path, std::size_t rows, std::size_t cols) {
    std::ofstream out(path);
    for (std::size_t c = 0; c < cols; ++c) {
        if (c) out << ',';
        out << "f" << c;
    }
    out << ",label\n";
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            if (c) out << ',';
            out << (r % 1000) + c * 0.01;
        }
        out << ',' << (r % 2) << '\n';
    }
    return path;
}

InMemoryDataset make_toy(std::size_t n, std::size_t f) {
    NDArray features({n, f}, DType::Float32);
    NDArray labels({n}, DType::Int64);
    for (std::size_t i = 0; i < n; ++i) {
        labels.data<std::int64_t>()[i] = static_cast<std::int64_t>(i);
        for (std::size_t j = 0; j < f; ++j) {
            features.data<float>()[i * f + j] = static_cast<float>(i);
        }
    }
    return InMemoryDataset(std::move(features), std::move(labels));
}

} // namespace

int main() {
    const auto& feat = cpu_features();
    std::cout << "NexusData v1.0 bench  avx2=" << feat.avx2 << " sse2=" << feat.sse2 << "\n";

    // --- byte_scan ---
    {
        constexpr std::size_t n = 32 * 1024 * 1024;
        std::vector<std::uint8_t> buf(n);
        for (std::size_t i = 0; i < n; ++i) {
            buf[i] = static_cast<std::uint8_t>(i % 251);
            if (i % 64 == 0) buf[i] = '\n';
        }
        const double ts = bench::time_best(
            [&] { (void)find_bytes_scalar(buf.data(), buf.size(), '\n'); }, 3);
        bench::print_result("byte_scan_scalar", ts, n);
        const double td = bench::time_best(
            [&] { (void)find_bytes(buf.data(), buf.size(), '\n'); }, 3);
        bench::print_result("byte_scan_dispatch", td, n);
        const double tc = bench::time_best(
            [&] { (void)find_csv_record_starts(buf.data(), buf.size()); }, 3);
        bench::print_result("csv_record_starts", tc, n);
    }

    // --- calibration ---
    {
        auto rep = calibrate_auto_device_policy();
        std::cout << "calibrate: host_memcpy=" << rep.host_memcpy_gib_s << " GiB/s"
                  << " cuda=" << rep.cuda_available
                  << " image_thr=" << rep.policy.image_nbytes_threshold
                  << " tabular_thr=" << rep.policy.tabular_nbytes_threshold << "\n";
    }

    // --- CSV load ---
    {
        const auto path = write_csv("bench_v08.csv", 20000, 16);
        CSVOptions opt;
        opt.label_column = "label";
        const double t = bench::time_best(
            [&] {
                CSVDataset ds(path, opt);
                (void)ds.size();
            },
            3);
        bench::print_result("csv_load_20k_x16", t, 0, 1);
        std::remove(path.c_str());
    }

    // --- collate ---
    {
        std::vector<Sample> batch;
        batch.reserve(64);
        for (int i = 0; i < 64; ++i) {
            Sample s;
            s.input = NDArray(Shape{128}, DType::Float32);
            s.label = NDArray(Shape{1}, DType::Int64);
            batch.push_back(std::move(s));
        }
        const std::size_t bytes = 64 * 128 * 4;
        const double t = bench::time_best(
            [&] { (void)collate_stack(batch); }, 50);
        bench::print_result("collate_stack_B64_F128", t, bytes, 50);
    }

    // --- DataLoader ---
    {
        auto ds = make_toy(4096, 32);
        DataLoaderOptions opt;
        opt.batch_size = 64;
        opt.shuffle = true;
        opt.seed = 1;
        opt.num_workers = 0;
        const double t0 = bench::time_best(
            [&] {
                DataLoader loader(ds, opt);
                for (const Batch& b : loader) {
                    (void)b.inputs.numel();
                }
            },
            3);
        bench::print_result("dataloader_sync_4k", t0, 4096 * 32 * 4, 1);

        opt.num_workers = 2;
        opt.prefetch_factor = 2;
        const double t1 = bench::time_best(
            [&] {
                DataLoader loader(ds, opt);
                for (const Batch& b : loader) {
                    (void)b.inputs.numel();
                }
            },
            3);
        bench::print_result("dataloader_workers2_4k", t1, 4096 * 32 * 4, 1);
    }

    return 0;
}
