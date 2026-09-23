#include <doctest/doctest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "nexusdata/backend/cpu_dispatch.hpp"
#include "nexusdata/backend/thread_pool.hpp"
#include "nexusdata/cache/disk_cache.hpp"
#include "nexusdata/cache/lru_cache.hpp"
#include "nexusdata/dataset/cached.hpp"
#include "nexusdata/dataset/in_memory.hpp"
#include "nexusdata/dataset/jsonl.hpp"
#include "nexusdata/loading/dataloader.hpp"
#include "nexusdata/simd/byte_scan.hpp"

using namespace nexusdata;

static InMemoryDataset make_toy(std::size_t n, std::size_t f) {
    NDArray features({n, f}, DType::Float32);
    NDArray labels({n}, DType::Int64);
    for (std::size_t i = 0; i < n; ++i) {
        labels.data<int64_t>()[i] = static_cast<int64_t>(i);
        for (std::size_t j = 0; j < f; ++j) {
            features.data<float>()[i * f + j] = static_cast<float>(i);
        }
    }
    return InMemoryDataset(std::move(features), std::move(labels));
}

static std::vector<std::int64_t> collect_labels(DataLoader& loader) {
    std::vector<std::int64_t> out;
    for (const Batch& b : loader) {
        const auto* p = b.labels.data<int64_t>();
        for (std::size_t i = 0; i < b.labels.numel(); ++i) {
            out.push_back(p[i]);
        }
    }
    return out;
}

TEST_CASE("ThreadPool parallel_for") {
    ThreadPool pool(4);
    std::vector<int> v(100, 0);
    pool.parallel_for(v.size(), [&](std::size_t i) { v[i] = static_cast<int>(i); });
    for (int i = 0; i < 100; ++i) {
        CHECK(v[i] == i);
    }
}

TEST_CASE("find_bytes scalar matches dispatched") {
    std::string s = "a,b\nc,d\ne";
    auto a = find_bytes_scalar(reinterpret_cast<const std::uint8_t*>(s.data()), s.size(), '\n');
    auto b = find_bytes(reinterpret_cast<const std::uint8_t*>(s.data()), s.size(), '\n');
    CHECK(a == b);
    CHECK(a.size() == 2);
}

TEST_CASE("DataLoader num_workers same order as sync") {
    auto ds = make_toy(40, 3);
    DataLoaderOptions aopt;
    aopt.batch_size = 4;
    aopt.shuffle = true;
    aopt.seed = 123;
    aopt.num_workers = 0;

    DataLoaderOptions bopt = aopt;
    bopt.num_workers = 2;
    bopt.prefetch_factor = 2;

    DataLoader a(ds, aopt);
    DataLoader b(ds, bopt);
    CHECK(collect_labels(a) == collect_labels(b));
}

TEST_CASE("LruSampleCache evicts by bytes") {
    auto cache = std::make_shared<LruSampleCache>(64); // tiny
    Sample s;
    s.input = NDArray(Shape{8}, DType::Float32); // 32 bytes
    s.label = NDArray(Shape{1}, DType::Float32); // 4
    cache->put("0", s);
    cache->put("1", s);
    // second insert may evict first depending on budget
    CHECK(cache->size() >= 1);
    CHECK(cache->used_bytes() <= 64);
}

TEST_CASE("DiskCache roundtrip") {
    DiskCache dc("test_disk_cache");
    NDArray a(Shape{2, 2}, DType::Float32);
    a.data<float>()[0] = 1;
    a.data<float>()[1] = 2;
    a.data<float>()[2] = 3;
    a.data<float>()[3] = 4;
    const auto key = DiskCache::make_key("content", "pipe");
    dc.put(key, a);
    {
        NDArray b = dc.get(key);
        CHECK(b.data<float>()[3] == doctest::Approx(4.0f));
    }
    std::filesystem::remove_all("test_disk_cache");
}

TEST_CASE("CachedDataset hits") {
    auto ds = std::make_shared<InMemoryDataset>(make_toy(5, 2));
    auto cache = std::make_shared<LruSampleCache>(1 << 20);
    CachedDataset cached(ds, cache, "t");
    Sample a = cached.get(1);
    Sample b = cached.get(1);
    CHECK(a.input.data<float>()[0] == b.input.data<float>()[0]);
    CHECK(cache->size() >= 1);
}

TEST_CASE("JSONLDataset") {
    {
        std::ofstream out("test.jsonl");
        out << "[1,2,0]\n[3,4,1]\n[5,6,0]\n";
    }
    {
        JSONLDataset ds("test.jsonl");
        CHECK(ds.size() == 3);
        CHECK(ds.num_features() == 2);
        CHECK(ds.get(1).input.data<float>()[0] == doctest::Approx(3.0f));
        CHECK(ds.get(1).label.data<float>()[0] == doctest::Approx(1.0f));
    }
    std::filesystem::remove("test.jsonl");
}

TEST_CASE("cpu_features accessible") {
    const auto& f = cpu_features();
    (void)f.avx2;
    CHECK(true);
}
