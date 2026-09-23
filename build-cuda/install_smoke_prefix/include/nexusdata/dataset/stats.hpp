#pragma once

#include <cmath>
#include <cstddef>
#include <map>
#include <string>
#include <vector>

#include "nexusdata/core/array_utils.hpp"
#include "nexusdata/core/ndarray.hpp"
#include "nexusdata/dataset/dataset.hpp"

namespace nexusdata {

struct ColumnStats {
    double mean = 0.0;
    double stddev = 0.0;
    double min_v = 0.0;
    double max_v = 0.0;
    std::size_t missing = 0;
    std::size_t count = 0;
};

struct DatasetStats {
    std::size_t num_samples = 0;
    std::size_t num_features = 0;
    std::map<double, std::size_t> class_counts; // from labels
    std::vector<ColumnStats> features;
};

/// Compute basic statistics over a feature matrix [N,F] and optional labels [N].
[[nodiscard]] DatasetStats compute_stats(const NDArray& features,
                                         const NDArray* labels = nullptr);

/// Convenience: iterate map-style dataset (copies each sample — OK for v0.2 sizes).
[[nodiscard]] DatasetStats compute_dataset_stats(const Dataset& dataset);

/// Simple histogram for a 1-D array into @p bins equal-width bins over [min,max].
[[nodiscard]] std::vector<std::size_t> histogram(const NDArray& values,
                                                 std::size_t bins,
                                                 double* out_min = nullptr,
                                                 double* out_max = nullptr);

} // namespace nexusdata
