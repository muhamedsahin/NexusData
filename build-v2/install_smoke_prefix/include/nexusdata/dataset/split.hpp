#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "nexusdata/core/error.hpp"
#include "nexusdata/core/ndarray.hpp"
#include "nexusdata/dataset/dataset.hpp"
#include "nexusdata/dataset/subset.hpp"

namespace nexusdata {

/// One train/test fold.
struct Fold {
    Subset train;
    Subset test;
};

[[nodiscard]] std::vector<Subset> random_split(DatasetConstPtr dataset,
                                               const std::vector<double>& fractions,
                                               std::uint64_t seed);

[[nodiscard]] std::pair<Subset, Subset> random_split(DatasetConstPtr dataset,
                                                     double train_fraction,
                                                     double val_fraction,
                                                     std::uint64_t seed);

template <typename DatasetT>
[[nodiscard]] std::pair<Subset, Subset> random_split(DatasetT dataset,
                                                     std::initializer_list<double> fractions,
                                                     std::uint64_t seed) {
    static_assert(std::is_base_of_v<Dataset, DatasetT>,
                  "random_split requires a Dataset-derived type");
    if (fractions.size() != 2) {
        throw InvalidArgumentError("random_split(initializer_list): need exactly 2 fractions");
    }
    auto it = fractions.begin();
    const double a = *it++;
    const double b = *it;
    auto ptr = std::make_shared<DatasetT>(std::move(dataset));
    return random_split(DatasetConstPtr(ptr), a, b, seed);
}

/// Stratified two-way split preserving class proportions in @p labels ([N] numeric).
[[nodiscard]] std::pair<Subset, Subset> stratified_split(DatasetConstPtr dataset,
                                                         const NDArray& labels,
                                                         double train_fraction,
                                                         std::uint64_t seed);

template <typename DatasetT>
[[nodiscard]] std::pair<Subset, Subset> stratified_split(DatasetT dataset,
                                                         const NDArray& labels,
                                                         double train_fraction,
                                                         std::uint64_t seed) {
    auto ptr = std::make_shared<DatasetT>(std::move(dataset));
    return stratified_split(DatasetConstPtr(ptr), labels, train_fraction, seed);
}

/// K-Fold: returns K folds; each sample appears in exactly one test fold.
[[nodiscard]] std::vector<Fold> k_fold(DatasetConstPtr dataset,
                                       int n_splits,
                                       bool shuffle,
                                       std::uint64_t seed);

template <typename DatasetT>
[[nodiscard]] std::vector<Fold> k_fold(DatasetT dataset,
                                       int n_splits,
                                       bool shuffle,
                                       std::uint64_t seed) {
    auto ptr = std::make_shared<DatasetT>(std::move(dataset));
    return k_fold(DatasetConstPtr(ptr), n_splits, shuffle, seed);
}

/// Stratified K-Fold using @p labels.
[[nodiscard]] std::vector<Fold> stratified_k_fold(DatasetConstPtr dataset,
                                                  const NDArray& labels,
                                                  int n_splits,
                                                  bool shuffle,
                                                  std::uint64_t seed);

template <typename DatasetT>
[[nodiscard]] std::vector<Fold> stratified_k_fold(DatasetT dataset,
                                                  const NDArray& labels,
                                                  int n_splits,
                                                  bool shuffle,
                                                  std::uint64_t seed) {
    auto ptr = std::make_shared<DatasetT>(std::move(dataset));
    return stratified_k_fold(DatasetConstPtr(ptr), labels, n_splits, shuffle, seed);
}

[[nodiscard]] std::vector<Fold> group_k_fold(DatasetConstPtr dataset,
                                             const NDArray& groups,
                                             int n_splits);

template <typename DatasetT>
[[nodiscard]] std::vector<Fold> group_k_fold(DatasetT dataset,
                                             const NDArray& groups,
                                             int n_splits) {
    auto ptr = std::make_shared<DatasetT>(std::move(dataset));
    return group_k_fold(DatasetConstPtr(ptr), groups, n_splits);
}

/// Time-series split: expanding train window, contiguous test blocks.
[[nodiscard]] std::vector<Fold> time_series_split(DatasetConstPtr dataset,
                                                  int n_splits,
                                                  int gap = 0);

template <typename DatasetT>
[[nodiscard]] std::vector<Fold> time_series_split(DatasetT dataset,
                                                  int n_splits,
                                                  int gap = 0) {
    auto ptr = std::make_shared<DatasetT>(std::move(dataset));
    return time_series_split(DatasetConstPtr(ptr), n_splits, gap);
}

/// Three-way split (fractions must sum to 1).
[[nodiscard]] std::tuple<Subset, Subset, Subset>
train_val_test_split(DatasetConstPtr dataset,
                     double train_fraction,
                     double val_fraction,
                     double test_fraction,
                     std::uint64_t seed);

template <typename DatasetT>
[[nodiscard]] std::tuple<Subset, Subset, Subset>
train_val_test_split(DatasetT dataset,
                     double train_fraction,
                     double val_fraction,
                     double test_fraction,
                     std::uint64_t seed) {
    auto ptr = std::make_shared<DatasetT>(std::move(dataset));
    return train_val_test_split(DatasetConstPtr(ptr), train_fraction, val_fraction,
                                test_fraction, seed);
}

} // namespace nexusdata
