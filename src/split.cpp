#include "nexusdata/dataset/split.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <tuple>

#include "nexusdata/core/array_utils.hpp"
#include "nexusdata/core/random.hpp"

namespace nexusdata {

namespace {

std::vector<std::size_t> split_counts(std::size_t n, const std::vector<double>& fractions) {
    if (fractions.empty()) {
        throw InvalidArgumentError("random_split: empty fractions");
    }
    double sum = 0.0;
    for (double f : fractions) {
        if (f < 0.0) {
            throw InvalidArgumentError("random_split: negative fraction");
        }
        sum += f;
    }
    if (std::abs(sum - 1.0) > 1e-6) {
        throw InvalidArgumentError("random_split: fractions must sum to 1.0 (got " +
                                   std::to_string(sum) + ")");
    }

    std::vector<std::size_t> counts(fractions.size(), 0);
    std::size_t assigned = 0;
    for (std::size_t i = 0; i + 1 < fractions.size(); ++i) {
        counts[i] = static_cast<std::size_t>(std::llround(fractions[i] * static_cast<double>(n)));
        assigned += counts[i];
    }
    if (assigned > n) {
        std::size_t overflow = assigned - n;
        for (std::size_t i = 0; i + 1 < counts.size() && overflow > 0; ++i) {
            const std::size_t take = std::min(counts[i], overflow);
            counts[i] -= take;
            overflow -= take;
            assigned -= take;
        }
    }
    counts.back() = n - assigned;
    return counts;
}

std::vector<double> extract_labels_1d(const NDArray& labels, std::size_t expected_n) {
    std::size_t n = 0, f = 0;
    detail::require_matrix(labels, n, f);
    if (n != expected_n) {
        throw ShapeError("labels length does not match dataset size");
    }
    std::vector<double> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = detail::as_double(labels, i * f);
    }
    return out;
}

std::map<double, std::vector<std::size_t>> group_by_label(const std::vector<double>& y) {
    std::map<double, std::vector<std::size_t>> by_class;
    for (std::size_t i = 0; i < y.size(); ++i) {
        by_class[y[i]].push_back(i);
    }
    return by_class;
}

} // namespace

std::vector<Subset> random_split(DatasetConstPtr dataset,
                                 const std::vector<double>& fractions,
                                 std::uint64_t seed) {
    if (!dataset) {
        throw InvalidArgumentError("random_split: null dataset");
    }
    const std::size_t n = dataset->size();
    const auto counts = split_counts(n, fractions);

    PCG32 rng(seed);
    auto order = shuffled_indices(n, rng);

    std::vector<Subset> parts;
    parts.reserve(counts.size());
    std::size_t offset = 0;
    for (std::size_t c : counts) {
        std::vector<std::size_t> idx(order.begin() + static_cast<std::ptrdiff_t>(offset),
                                     order.begin() + static_cast<std::ptrdiff_t>(offset + c));
        parts.emplace_back(dataset, std::move(idx));
        offset += c;
    }
    return parts;
}

std::pair<Subset, Subset> random_split(DatasetConstPtr dataset,
                                       double train_fraction,
                                       double val_fraction,
                                       std::uint64_t seed) {
    auto parts = random_split(dataset, std::vector<double>{train_fraction, val_fraction}, seed);
    return {std::move(parts[0]), std::move(parts[1])};
}

std::pair<Subset, Subset> stratified_split(DatasetConstPtr dataset,
                                           const NDArray& labels,
                                           double train_fraction,
                                           std::uint64_t seed) {
    if (!dataset) {
        throw InvalidArgumentError("stratified_split: null dataset");
    }
    if (train_fraction <= 0.0 || train_fraction >= 1.0) {
        throw InvalidArgumentError("stratified_split: train_fraction must be in (0,1)");
    }
    const auto y = extract_labels_1d(labels, dataset->size());
    auto by_class = group_by_label(y);

    PCG32 rng(seed);
    std::vector<std::size_t> train_idx;
    std::vector<std::size_t> val_idx;
    train_idx.reserve(dataset->size());
    val_idx.reserve(dataset->size());

    for (auto& [cls, idxs] : by_class) {
        (void)cls;
        shuffle(idxs, rng);
        const std::size_t n_train =
            static_cast<std::size_t>(std::llround(train_fraction * static_cast<double>(idxs.size())));
        const std::size_t n_tr = std::min(n_train, idxs.size());
        // Ensure both sides get at least one if class has >=2 samples when possible.
        std::size_t cut = n_tr;
        if (idxs.size() >= 2) {
            cut = std::clamp(n_tr, std::size_t{1}, idxs.size() - 1);
        }
        train_idx.insert(train_idx.end(), idxs.begin(), idxs.begin() + static_cast<std::ptrdiff_t>(cut));
        val_idx.insert(val_idx.end(), idxs.begin() + static_cast<std::ptrdiff_t>(cut), idxs.end());
    }

    shuffle(train_idx, rng);
    shuffle(val_idx, rng);
    return {Subset(dataset, std::move(train_idx)), Subset(dataset, std::move(val_idx))};
}

std::vector<Fold> k_fold(DatasetConstPtr dataset, int n_splits, bool shuffle, std::uint64_t seed) {
    if (!dataset) {
        throw InvalidArgumentError("k_fold: null dataset");
    }
    if (n_splits < 2) {
        throw InvalidArgumentError("k_fold: n_splits must be >= 2");
    }
    const std::size_t n = dataset->size();
    if (static_cast<std::size_t>(n_splits) > n) {
        throw InvalidArgumentError("k_fold: n_splits cannot exceed dataset size");
    }

    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    if (shuffle) {
        PCG32 rng(seed);
        nexusdata::shuffle(order, rng);
    }

    std::vector<Fold> folds;
    folds.reserve(static_cast<std::size_t>(n_splits));
    for (int fold = 0; fold < n_splits; ++fold) {
        std::vector<std::size_t> test_idx;
        std::vector<std::size_t> train_idx;
        for (std::size_t i = 0; i < n; ++i) {
            if (static_cast<int>(i % static_cast<std::size_t>(n_splits)) == fold) {
                test_idx.push_back(order[i]);
            } else {
                train_idx.push_back(order[i]);
            }
        }
        folds.push_back(Fold{Subset(dataset, std::move(train_idx)),
                             Subset(dataset, std::move(test_idx))});
    }
    return folds;
}

std::vector<Fold> stratified_k_fold(DatasetConstPtr dataset,
                                    const NDArray& labels,
                                    int n_splits,
                                    bool shuffle,
                                    std::uint64_t seed) {
    if (!dataset) {
        throw InvalidArgumentError("stratified_k_fold: null dataset");
    }
    if (n_splits < 2) {
        throw InvalidArgumentError("stratified_k_fold: n_splits must be >= 2");
    }
    const auto y = extract_labels_1d(labels, dataset->size());
    auto by_class = group_by_label(y);

    PCG32 rng(seed);
    // Assign each sample a fold id.
    std::vector<int> fold_of(dataset->size(), -1);
    for (auto& [cls, idxs] : by_class) {
        (void)cls;
        if (shuffle) {
            nexusdata::shuffle(idxs, rng);
        }
        if (static_cast<int>(idxs.size()) < n_splits) {
            throw InvalidArgumentError(
                "stratified_k_fold: class has fewer samples than n_splits");
        }
        for (std::size_t i = 0; i < idxs.size(); ++i) {
            fold_of[idxs[i]] = static_cast<int>(i % static_cast<std::size_t>(n_splits));
        }
    }

    std::vector<Fold> folds;
    folds.reserve(static_cast<std::size_t>(n_splits));
    for (int fold = 0; fold < n_splits; ++fold) {
        std::vector<std::size_t> test_idx;
        std::vector<std::size_t> train_idx;
        for (std::size_t i = 0; i < fold_of.size(); ++i) {
            if (fold_of[i] == fold) {
                test_idx.push_back(i);
            } else {
                train_idx.push_back(i);
            }
        }
        folds.push_back(Fold{Subset(dataset, std::move(train_idx)),
                             Subset(dataset, std::move(test_idx))});
    }
    return folds;
}

std::vector<Fold> group_k_fold(DatasetConstPtr dataset, const NDArray& groups, int n_splits) {
    if (!dataset) {
        throw InvalidArgumentError("group_k_fold: null dataset");
    }
    if (n_splits < 2) {
        throw InvalidArgumentError("group_k_fold: n_splits must be >= 2");
    }
    const auto g = extract_labels_1d(groups, dataset->size());
    std::map<double, std::vector<std::size_t>> by_group;
    for (std::size_t i = 0; i < g.size(); ++i) {
        by_group[g[i]].push_back(i);
    }
    if (static_cast<int>(by_group.size()) < n_splits) {
        throw InvalidArgumentError("group_k_fold: fewer unique groups than n_splits");
    }

    std::vector<double> group_ids;
    group_ids.reserve(by_group.size());
    for (const auto& [gid, _] : by_group) {
        group_ids.push_back(gid);
    }

    std::vector<int> group_fold(group_ids.size());
    for (std::size_t i = 0; i < group_ids.size(); ++i) {
        group_fold[i] = static_cast<int>(i % static_cast<std::size_t>(n_splits));
    }

    std::map<double, int> gid_to_fold;
    for (std::size_t i = 0; i < group_ids.size(); ++i) {
        gid_to_fold[group_ids[i]] = group_fold[i];
    }

    std::vector<Fold> folds;
    folds.reserve(static_cast<std::size_t>(n_splits));
    for (int fold = 0; fold < n_splits; ++fold) {
        std::vector<std::size_t> test_idx;
        std::vector<std::size_t> train_idx;
        for (std::size_t i = 0; i < g.size(); ++i) {
            if (gid_to_fold[g[i]] == fold) {
                test_idx.push_back(i);
            } else {
                train_idx.push_back(i);
            }
        }
        folds.push_back(Fold{Subset(dataset, std::move(train_idx)),
                             Subset(dataset, std::move(test_idx))});
    }
    return folds;
}

std::vector<Fold> time_series_split(DatasetConstPtr dataset, int n_splits, int gap) {
    if (!dataset) {
        throw InvalidArgumentError("time_series_split: null dataset");
    }
    if (n_splits < 2) {
        throw InvalidArgumentError("time_series_split: n_splits must be >= 2");
    }
    if (gap < 0) {
        throw InvalidArgumentError("time_series_split: gap must be >= 0");
    }
    const std::size_t n = dataset->size();
    // Similar to sklearn: n_splits+1 folds of roughly equal size along time.
    const int n_folds = n_splits + 1;
    if (static_cast<std::size_t>(n_folds) > n) {
        throw InvalidArgumentError("time_series_split: not enough samples");
    }

    std::vector<std::size_t> fold_sizes(static_cast<std::size_t>(n_folds),
                                        n / static_cast<std::size_t>(n_folds));
    for (std::size_t i = 0; i < n % static_cast<std::size_t>(n_folds); ++i) {
        fold_sizes[i] += 1;
    }

    std::vector<std::size_t> boundaries(static_cast<std::size_t>(n_folds) + 1, 0);
    for (int i = 0; i < n_folds; ++i) {
        boundaries[static_cast<std::size_t>(i) + 1] =
            boundaries[static_cast<std::size_t>(i)] + fold_sizes[static_cast<std::size_t>(i)];
    }

    std::vector<Fold> folds;
    for (int i = 1; i <= n_splits; ++i) {
        const std::size_t train_end = boundaries[static_cast<std::size_t>(i)];
        std::size_t test_start = train_end + static_cast<std::size_t>(gap);
        const std::size_t test_end = boundaries[static_cast<std::size_t>(i) + 1];
        if (test_start > test_end) {
            test_start = test_end;
        }
        std::vector<std::size_t> train_idx(train_end);
        std::iota(train_idx.begin(), train_idx.end(), 0);
        std::vector<std::size_t> test_idx;
        for (std::size_t t = test_start; t < test_end; ++t) {
            test_idx.push_back(t);
        }
        folds.push_back(Fold{Subset(dataset, std::move(train_idx)),
                             Subset(dataset, std::move(test_idx))});
    }
    return folds;
}

std::tuple<Subset, Subset, Subset>
train_val_test_split(DatasetConstPtr dataset,
                     double train_fraction,
                     double val_fraction,
                     double test_fraction,
                     std::uint64_t seed) {
    auto parts = random_split(
        dataset, std::vector<double>{train_fraction, val_fraction, test_fraction}, seed);
    return {std::move(parts[0]), std::move(parts[1]), std::move(parts[2])};
}

} // namespace nexusdata
