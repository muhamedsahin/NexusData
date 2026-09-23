#include <doctest/doctest.h>

#include <set>
#include <vector>

#include "nexusdata/dataset/in_memory.hpp"
#include "nexusdata/loading/dataloader.hpp"

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

TEST_CASE("same seed = same batch order") {
    auto ds = make_toy(20, 4);
    DataLoaderOptions opt;
    opt.batch_size = 4;
    opt.shuffle = true;
    opt.seed = 42;

    DataLoader a(ds, opt);
    DataLoader b(ds, opt);
    auto la = collect_labels(a);
    auto lb = collect_labels(b);
    CHECK(la == lb);
    CHECK(la.size() == 20);
}

TEST_CASE("drop_last discards incomplete batch") {
    auto ds = make_toy(10, 2);
    DataLoaderOptions opt;
    opt.batch_size = 3;
    opt.shuffle = false;
    opt.drop_last = true;

    DataLoader loader(ds, opt);
    CHECK(loader.size() == 3); // 9 samples, 1 dropped
    std::size_t seen = 0;
    for (const Batch& b : loader) {
        CHECK(b.inputs.shape()[0] == 3);
        seen += b.inputs.shape()[0];
    }
    CHECK(seen == 9);
}

TEST_CASE("drop_last=false keeps remainder") {
    auto ds = make_toy(10, 2);
    DataLoaderOptions opt;
    opt.batch_size = 3;
    opt.shuffle = false;
    opt.drop_last = false;

    DataLoader loader(ds, opt);
    CHECK(loader.size() == 4);
    std::vector<std::size_t> sizes;
    for (const Batch& b : loader) {
        sizes.push_back(b.inputs.shape()[0]);
    }
    CHECK(sizes == std::vector<std::size_t>{3, 3, 3, 1});
}

TEST_CASE("sequential order without shuffle") {
    auto ds = make_toy(6, 1);
    DataLoaderOptions opt;
    opt.batch_size = 2;
    opt.shuffle = false;
    DataLoader loader(ds, opt);
    auto labels = collect_labels(loader);
    CHECK(labels == std::vector<std::int64_t>{0, 1, 2, 3, 4, 5});
}

TEST_CASE("batch shapes") {
    auto ds = make_toy(8, 5);
    DataLoaderOptions opt;
    opt.batch_size = 4;
    DataLoader loader(ds, opt);
    for (const Batch& b : loader) {
        CHECK(b.inputs.shape() == Shape{4, 5});
        CHECK(b.labels.shape() == Shape{4});
    }
}
