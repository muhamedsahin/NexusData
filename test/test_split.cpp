#include <doctest/doctest.h>

#include <set>
#include <vector>

#include "nexusdata/dataset/in_memory.hpp"
#include "nexusdata/dataset/split.hpp"
#include "nexusdata/dataset/stats.hpp"

using namespace nexusdata;

static InMemoryDataset make_binary(std::size_t n0, std::size_t n1) {
    const std::size_t n = n0 + n1;
    NDArray features({n, 1}, DType::Float32);
    NDArray labels({n}, DType::Int64);
    for (std::size_t i = 0; i < n0; ++i) {
        features.data<float>()[i] = static_cast<float>(i);
        labels.data<int64_t>()[i] = 0;
    }
    for (std::size_t i = 0; i < n1; ++i) {
        features.data<float>()[n0 + i] = static_cast<float>(100 + i);
        labels.data<int64_t>()[n0 + i] = 1;
    }
    return InMemoryDataset(std::move(features), std::move(labels));
}

TEST_CASE("train and val do not intersect") {
    auto ds = make_binary(50, 50);
    auto [train, val] = random_split(ds, {0.8, 0.2}, 42);
    CHECK(train.size() == 80);
    CHECK(val.size() == 20);
    std::set<std::size_t> a, b;
    for (std::size_t i = 0; i < train.size(); ++i) {
        a.insert(static_cast<std::size_t>(train.get(i).label.data<int64_t>()[0] * 0 +
                                          train.indices()[i]));
    }
    for (std::size_t i = 0; i < val.size(); ++i) {
        b.insert(val.indices()[i]);
    }
    for (auto idx : train.indices()) {
        CHECK(b.count(idx) == 0);
    }
}

TEST_CASE("stratified_split preserves class ratios roughly") {
    auto ds = make_binary(80, 20);
    NDArray labels = ds.labels().clone();
    auto [train, val] = stratified_split(ds, labels, 0.75, 42);
    std::size_t tr0 = 0, tr1 = 0, va0 = 0, va1 = 0;
    for (std::size_t i = 0; i < train.size(); ++i) {
        (train.get(i).label.data<int64_t>()[0] == 0 ? tr0 : tr1)++;
    }
    for (std::size_t i = 0; i < val.size(); ++i) {
        (val.get(i).label.data<int64_t>()[0] == 0 ? va0 : va1)++;
    }
    CHECK(tr0 + va0 == 80);
    CHECK(tr1 + va1 == 20);
    // ~75% of each class in train
    CHECK(tr0 >= 55);
    CHECK(tr1 >= 10);
}

TEST_CASE("k_fold covers each index once in test") {
    auto ds = make_binary(10, 10);
    auto folds = k_fold(ds, 5, true, 1);
    CHECK(folds.size() == 5);
    std::set<std::size_t> seen;
    for (const auto& fold : folds) {
        for (std::size_t idx : fold.test.indices()) {
            CHECK(seen.count(idx) == 0);
            seen.insert(idx);
        }
    }
    CHECK(seen.size() == 20);
}

TEST_CASE("stratified_k_fold") {
    auto ds = make_binary(20, 20);
    auto folds = stratified_k_fold(ds, ds.labels(), 4, true, 3);
    CHECK(folds.size() == 4);
    for (const auto& fold : folds) {
        std::size_t c0 = 0, c1 = 0;
        for (std::size_t i = 0; i < fold.test.size(); ++i) {
            (fold.test.get(i).label.data<int64_t>()[0] == 0 ? c0 : c1)++;
        }
        CHECK(c0 == 5);
        CHECK(c1 == 5);
    }
}

TEST_CASE("group_k_fold no group leakage") {
    NDArray features({6, 1}, DType::Float32);
    NDArray labels({6}, DType::Int64);
    NDArray groups({6}, DType::Int64);
    for (int i = 0; i < 6; ++i) {
        features.data<float>()[i] = static_cast<float>(i);
        labels.data<int64_t>()[i] = i % 2;
        groups.data<int64_t>()[i] = i / 2; // groups 0,0,1,1,2,2
    }
    InMemoryDataset ds(std::move(features), std::move(labels));
    auto folds = group_k_fold(ds, groups, 3);
    for (const auto& fold : folds) {
        std::set<std::int64_t> train_g, test_g;
        for (std::size_t idx : fold.train.indices()) {
            train_g.insert(groups.data<int64_t>()[idx]);
        }
        for (std::size_t idx : fold.test.indices()) {
            test_g.insert(groups.data<int64_t>()[idx]);
        }
        for (auto g : test_g) {
            CHECK(train_g.count(g) == 0);
        }
    }
}

TEST_CASE("time_series_split expanding train") {
    auto ds = make_binary(5, 5); // n=10
    auto folds = time_series_split(ds, 3, 0);
    CHECK(folds.size() == 3);
    // train sizes should be non-decreasing
    CHECK(folds[0].train.size() <= folds[1].train.size());
    CHECK(folds[1].train.size() <= folds[2].train.size());
}

TEST_CASE("train_val_test_split") {
    auto ds = make_binary(40, 10);
    auto [tr, va, te] = train_val_test_split(ds, 0.6, 0.2, 0.2, 9);
    CHECK(tr.size() + va.size() + te.size() == 50);
}

TEST_CASE("compute_stats and histogram") {
    auto ds = make_binary(10, 10);
    auto st = compute_stats(ds.features(), &ds.labels());
    CHECK(st.num_samples == 20);
    CHECK(st.class_counts.size() == 2);
    CHECK(st.features.size() == 1);
    auto hist = histogram(ds.labels(), 2);
    CHECK(hist.size() == 2);
    CHECK(hist[0] + hist[1] == 20);
}
