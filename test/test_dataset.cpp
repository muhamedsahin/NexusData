#include <doctest/doctest.h>

#include "nexusdata/dataset/in_memory.hpp"
#include "nexusdata/dataset/subset.hpp"

using namespace nexusdata;

static InMemoryDataset make_toy(std::size_t n, std::size_t f) {
    NDArray features({n, f}, DType::Float32);
    NDArray labels({n}, DType::Float32);
    for (std::size_t i = 0; i < n; ++i) {
        labels.data<float>()[i] = static_cast<float>(i);
        for (std::size_t j = 0; j < f; ++j) {
            features.data<float>()[i * f + j] = static_cast<float>(i * 10 + j);
        }
    }
    return InMemoryDataset(std::move(features), std::move(labels));
}

TEST_CASE("InMemoryDataset get") {
    auto ds = make_toy(5, 3);
    CHECK(ds.size() == 5);
    Sample s = ds.get(2);
    CHECK(s.input.shape() == Shape{3});
    CHECK(s.input.data<float>()[0] == 20.0f);
    CHECK(s.input.data<float>()[2] == 22.0f);
    CHECK(s.label.data<float>()[0] == 2.0f);
    CHECK_THROWS_AS(ds.get(5), IndexError);
}

TEST_CASE("Subset view") {
    auto ds = std::make_shared<InMemoryDataset>(make_toy(10, 2));
    Subset sub(ds, {9, 0, 3});
    CHECK(sub.size() == 3);
    CHECK(sub.get(0).label.data<float>()[0] == 9.0f);
    CHECK(sub.get(1).label.data<float>()[0] == 0.0f);
    CHECK(sub.get(2).label.data<float>()[0] == 3.0f);
}
