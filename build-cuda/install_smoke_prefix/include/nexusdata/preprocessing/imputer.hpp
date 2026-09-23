#pragma once

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <vector>

#include "nexusdata/core/array_utils.hpp"
#include "nexusdata/preprocessing/scaler.hpp"
#include "nexusdata/preprocessing/transformer.hpp"

namespace nexusdata {

enum class ImputeStrategy {
    Mean,
    Median,
    Mode,
    Constant,
};

/// Replace NaN values column-wise.
class Imputer : public Transformer {
public:
    explicit Imputer(ImputeStrategy strategy = ImputeStrategy::Mean, double fill_value = 0.0)
        : strategy_(strategy), fill_value_(fill_value) {}

    void fit(const NDArray& X) override {
        std::size_t n = 0, f = 0;
        detail::require_matrix(X, n, f);
        statistics_.assign(f, fill_value_);

        for (std::size_t j = 0; j < f; ++j) {
            std::vector<double> vals;
            vals.reserve(n);
            for (std::size_t i = 0; i < n; ++i) {
                const double v = detail::as_double(X, i * f + j);
                if (!detail::is_nan_value(v)) {
                    vals.push_back(v);
                }
            }
            if (vals.empty()) {
                statistics_[j] = fill_value_;
                continue;
            }
            switch (strategy_) {
                case ImputeStrategy::Mean: {
                    double sum = 0.0;
                    for (double v : vals) {
                        sum += v;
                    }
                    statistics_[j] = sum / static_cast<double>(vals.size());
                    break;
                }
                case ImputeStrategy::Median: {
                    std::sort(vals.begin(), vals.end());
                    const std::size_t m = vals.size() / 2;
                    statistics_[j] = (vals.size() % 2 == 0)
                                         ? 0.5 * (vals[m - 1] + vals[m])
                                         : vals[m];
                    break;
                }
                case ImputeStrategy::Mode: {
                    std::map<double, std::size_t> counts;
                    for (double v : vals) {
                        ++counts[v];
                    }
                    std::size_t best = 0;
                    double mode = vals[0];
                    for (const auto& [v, c] : counts) {
                        if (c > best) {
                            best = c;
                            mode = v;
                        }
                    }
                    statistics_[j] = mode;
                    break;
                }
                case ImputeStrategy::Constant:
                    statistics_[j] = fill_value_;
                    break;
            }
        }
        n_features_ = f;
        fitted_ = true;
    }

    [[nodiscard]] NDArray transform(const NDArray& X) const override {
        ensure_fitted();
        return detail::transform_matrix(X, n_features_, [&](std::size_t, std::size_t j, double v) {
            return detail::is_nan_value(v) ? statistics_[j] : v;
        });
    }

    [[nodiscard]] bool is_fitted() const override { return fitted_; }
    [[nodiscard]] std::string name() const override { return "Imputer"; }
    [[nodiscard]] const std::vector<double>& statistics() const { return statistics_; }

    void save(std::ostream& out) const override {
        ensure_fitted();
        out << "NEXUSDATA_PREPROCESSOR 1\nTYPE Imputer\n";
        out << "STRATEGY " << static_cast<int>(strategy_) << "\n";
        out << "FILL " << fill_value_ << "\n";
        detail::write_vec(out, "STATS", statistics_);
    }

    void load(std::istream& in) override {
        std::string tag, tk, tv, k;
        int ver = 0;
        in >> tag >> ver >> tk >> tv;
        if (tv != "Imputer") {
            throw IOError("Imputer::load: type mismatch");
        }
        int strat = 0;
        in >> k >> strat >> k >> fill_value_;
        strategy_ = static_cast<ImputeStrategy>(strat);
        statistics_ = detail::read_vec(in, "STATS");
        n_features_ = statistics_.size();
        fitted_ = true;
    }

private:
    void ensure_fitted() const {
        if (!fitted_) {
            throw InvalidArgumentError("Imputer: not fitted");
        }
    }

    ImputeStrategy strategy_ = ImputeStrategy::Mean;
    double fill_value_ = 0.0;
    bool fitted_ = false;
    std::size_t n_features_ = 0;
    std::vector<double> statistics_;
};

/// Threshold binarization: x > threshold -> 1 else 0. fit() is no-op.
class Binarizer : public Transformer {
public:
    explicit Binarizer(double threshold = 0.0) : threshold_(threshold) {}

    void fit(const NDArray& X) override {
        std::size_t n = 0, f = 0;
        detail::require_matrix(X, n, f);
        n_features_ = f;
        fitted_ = true;
    }

    [[nodiscard]] NDArray transform(const NDArray& X) const override {
        ensure_fitted();
        return detail::transform_matrix(X, n_features_, [&](std::size_t, std::size_t, double v) {
            return v > threshold_ ? 1.0 : 0.0;
        });
    }

    [[nodiscard]] bool is_fitted() const override { return fitted_; }
    [[nodiscard]] std::string name() const override { return "Binarizer"; }

    void save(std::ostream& out) const override {
        ensure_fitted();
        out << "NEXUSDATA_PREPROCESSOR 1\nTYPE Binarizer\n";
        out << "THRESHOLD " << threshold_ << "\n";
        out << "N_FEATURES " << n_features_ << "\n";
    }

    void load(std::istream& in) override {
        std::string tag, tk, tv, k;
        int ver = 0;
        in >> tag >> ver >> tk >> tv;
        if (tv != "Binarizer") {
            throw IOError("Binarizer::load: type mismatch");
        }
        in >> k >> threshold_ >> k >> n_features_;
        fitted_ = true;
    }

private:
    void ensure_fitted() const {
        if (!fitted_) {
            throw InvalidArgumentError("Binarizer: not fitted");
        }
    }

    double threshold_ = 0.0;
    bool fitted_ = false;
    std::size_t n_features_ = 0;
};

enum class NormType { L1, L2, Max };

/// Row-wise normalizer.
class Normalizer : public Transformer {
public:
    explicit Normalizer(NormType norm = NormType::L2) : norm_(norm) {}

    void fit(const NDArray& X) override {
        std::size_t n = 0, f = 0;
        detail::require_matrix(X, n, f);
        n_features_ = f;
        fitted_ = true;
    }

    [[nodiscard]] NDArray transform(const NDArray& X) const override {
        ensure_fitted();
        std::size_t n = 0, f = 0;
        detail::require_matrix(X, n, f);
        if (f != n_features_) {
            throw ShapeError("Normalizer: feature mismatch");
        }
        NDArray out(Shape{n, f}, DType::Float64);
        for (std::size_t i = 0; i < n; ++i) {
            double norm = 0.0;
            for (std::size_t j = 0; j < f; ++j) {
                const double v = detail::as_double(X, i * f + j);
                switch (norm_) {
                    case NormType::L1:  norm += std::abs(v); break;
                    case NormType::L2:  norm += v * v; break;
                    case NormType::Max: norm = std::max(norm, std::abs(v)); break;
                }
            }
            if (norm_ == NormType::L2) {
                norm = std::sqrt(norm);
            }
            if (norm == 0.0) {
                norm = 1.0;
            }
            for (std::size_t j = 0; j < f; ++j) {
                detail::set_double(out, i * f + j, detail::as_double(X, i * f + j) / norm);
            }
        }
        return out;
    }

    [[nodiscard]] bool is_fitted() const override { return fitted_; }
    [[nodiscard]] std::string name() const override { return "Normalizer"; }

    void save(std::ostream& out) const override {
        ensure_fitted();
        out << "NEXUSDATA_PREPROCESSOR 1\nTYPE Normalizer\n";
        out << "NORM " << static_cast<int>(norm_) << "\n";
        out << "N_FEATURES " << n_features_ << "\n";
    }

    void load(std::istream& in) override {
        std::string tag, tk, tv, k;
        int ver = 0, nrm = 0;
        in >> tag >> ver >> tk >> tv;
        if (tv != "Normalizer") {
            throw IOError("Normalizer::load: type mismatch");
        }
        in >> k >> nrm >> k >> n_features_;
        norm_ = static_cast<NormType>(nrm);
        fitted_ = true;
    }

private:
    void ensure_fitted() const {
        if (!fitted_) {
            throw InvalidArgumentError("Normalizer: not fitted");
        }
    }

    NormType norm_ = NormType::L2;
    bool fitted_ = false;
    std::size_t n_features_ = 0;
};

} // namespace nexusdata
