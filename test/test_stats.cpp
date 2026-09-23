#include <doctest/doctest.h>

#include "nexusdata/dataset/in_memory.hpp"
#include "nexusdata/dataset/stats.hpp"

using namespace nexusdata;

TEST_CASE("compute_dataset_stats") {
    NDArray features({4, 2}, DType::Float32);
    NDArray labels({4}, DType::Float32);
    float* p = features.data<float>();
    p[0] = 1; p[1] = 2;
    p[2] = 3; p[3] = 4;
    p[4] = 5; p[5] = 6;
    p[6] = 7; p[7] = 8;
    for (int i = 0; i < 4; ++i) {
        labels.data<float>()[i] = static_cast<float>(i % 2);
    }
    InMemoryDataset ds(std::move(features), std::move(labels));
    auto st = compute_dataset_stats(ds);
    CHECK(st.num_samples == 4);
    CHECK(st.num_features == 2);
    CHECK(st.class_counts[0.0] == 2);
    CHECK(st.class_counts[1.0] == 2);
    CHECK(st.features[0].mean == doctest::Approx(4.0));
}
