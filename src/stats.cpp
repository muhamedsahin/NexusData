#include "nexusdata/dataset/stats.hpp"

#include <algorithm>
#include <limits>

namespace nexusdata {

DatasetStats compute_stats(const NDArray& features, const NDArray* labels) {
    std::size_t n = 0, f = 0;
    detail::require_matrix(features, n, f);

    DatasetStats stats;
    stats.num_samples = n;
    stats.num_features = f;
    stats.features.resize(f);

    for (std::size_t j = 0; j < f; ++j) {
        detail::Welford acc;
        double min_v = std::numeric_limits<double>::infinity();
        double max_v = -std::numeric_limits<double>::infinity();
        std::size_t missing = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const double v = detail::as_double(features, i * f + j);
            if (detail::is_nan_value(v)) {
                ++missing;
                continue;
            }
            acc.update(v);
            min_v = std::min(min_v, v);
            max_v = std::max(max_v, v);
        }
        ColumnStats cs;
        cs.mean = acc.mean;
        cs.stddev = acc.stddev(false);
        cs.min_v = std::isfinite(min_v) ? min_v : 0.0;
        cs.max_v = std::isfinite(max_v) ? max_v : 0.0;
        cs.missing = missing;
        cs.count = acc.n;
        stats.features[j] = cs;
    }

    if (labels) {
        std::size_t ln = 0, lf = 0;
        detail::require_matrix(*labels, ln, lf);
        if (ln != n) {
            throw ShapeError("compute_stats: labels length mismatch");
        }
        for (std::size_t i = 0; i < n; ++i) {
            const double y = detail::as_double(*labels, i * lf);
            ++stats.class_counts[y];
        }
    }
    return stats;
}

DatasetStats compute_dataset_stats(const Dataset& dataset) {
    const std::size_t n = dataset.size();
    if (n == 0) {
        return {};
    }
    Sample first = dataset.get(0);
    const std::size_t f = first.input.numel();
    NDArray features(Shape{n, f}, DType::Float64);
    NDArray labels(Shape{n}, DType::Float64);
    for (std::size_t i = 0; i < n; ++i) {
        Sample s = dataset.get(i);
        if (s.input.numel() != f) {
            throw ShapeError("compute_dataset_stats: inconsistent feature size");
        }
        for (std::size_t j = 0; j < f; ++j) {
            detail::set_double(features, i * f + j, detail::as_double(s.input, j));
        }
        detail::set_double(labels, i, detail::as_double(s.label, 0));
    }
    return compute_stats(features, &labels);
}

std::vector<std::size_t> histogram(const NDArray& values,
                                   std::size_t bins,
                                   double* out_min,
                                   double* out_max) {
    if (bins == 0) {
        throw InvalidArgumentError("histogram: bins must be > 0");
    }
    if (values.numel() == 0) {
        return std::vector<std::size_t>(bins, 0);
    }
    double min_v = std::numeric_limits<double>::infinity();
    double max_v = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < values.numel(); ++i) {
        const double v = detail::as_double(values, i);
        if (detail::is_nan_value(v)) {
            continue;
        }
        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);
    }
    if (!std::isfinite(min_v)) {
        min_v = 0.0;
        max_v = 1.0;
    }
    if (min_v == max_v) {
        max_v = min_v + 1.0;
    }
    if (out_min) {
        *out_min = min_v;
    }
    if (out_max) {
        *out_max = max_v;
    }

    std::vector<std::size_t> hist(bins, 0);
    const double width = (max_v - min_v) / static_cast<double>(bins);
    for (std::size_t i = 0; i < values.numel(); ++i) {
        const double v = detail::as_double(values, i);
        if (detail::is_nan_value(v)) {
            continue;
        }
        std::size_t b = static_cast<std::size_t>((v - min_v) / width);
        if (b >= bins) {
            b = bins - 1;
        }
        ++hist[b];
    }
    return hist;
}

} // namespace nexusdata
