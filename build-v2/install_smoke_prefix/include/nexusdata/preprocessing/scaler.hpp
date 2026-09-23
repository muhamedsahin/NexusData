#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "nexusdata/core/array_utils.hpp"
#include "nexusdata/core/error.hpp"
#include "nexusdata/preprocessing/transformer.hpp"
#include "nexusdata/simd/numeric_ops.hpp"

namespace nexusdata {

namespace detail {

inline void write_vec(std::ostream& out, const std::string& key, const std::vector<double>& v) {
    out << key << " " << v.size();
    for (double x : v) {
        out << " " << std::setprecision(17) << x;
    }
    out << "\n";
}

inline std::vector<double> read_vec(std::istream& in, const std::string& expect_key) {
    std::string key;
    std::size_t n = 0;
    in >> key >> n;
    if (key != expect_key) {
        throw IOError("expected key " + expect_key + ", got " + key);
    }
    std::vector<double> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        in >> v[i];
    }
    return v;
}

inline NDArray transform_matrix(
    const NDArray& X,
    std::size_t expected_f,
    const std::function<double(std::size_t /*row*/, std::size_t /*col*/, double)>& fn) {
    std::size_t n = 0, f = 0;
    require_matrix(X, n, f);
    if (f != expected_f) {
        throw ShapeError("transform: expected " + std::to_string(expected_f) +
                         " features, got " + std::to_string(f));
    }
    NDArray out = simd::cast(X, DType::Float64);
    out.reshape(Shape{n, f});
    double* p = out.data<double>();
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < f; ++j) {
            const std::size_t idx = i * f + j;
            p[idx] = fn(i, j, p[idx]);
        }
    }
    return out;
}

/// X (any real dtype, [N] or [N, F]) -> Float64 [N, F] copy, validated against @p expected_f.
inline NDArray to_f64_matrix(const NDArray& X, std::size_t expected_f, std::size_t& n) {
    std::size_t f = 0;
    require_matrix(X, n, f);
    if (f != expected_f) {
        throw ShapeError("transform: expected " + std::to_string(expected_f) +
                         " features, got " + std::to_string(f));
    }
    NDArray out = simd::cast(X, DType::Float64);
    out.reshape(Shape{n, f});
    return out;
}

} // namespace detail

/// z = (x - mean) / std  (population std by default; zero-variance -> scale 1)
class StandardScaler : public Transformer {
public:
    explicit StandardScaler(bool with_mean = true, bool with_std = true)
        : with_mean_(with_mean), with_std_(with_std) {}

    void fit(const NDArray& X) override {
        std::size_t n = 0, f = 0;
        detail::require_matrix(X, n, f);
        mean_.assign(f, 0.0);
        scale_.assign(f, 1.0);
        std::vector<detail::Welford> acc(f);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < f; ++j) {
                acc[j].update(detail::as_double(X, i * f + j));
            }
        }
        for (std::size_t j = 0; j < f; ++j) {
            mean_[j] = with_mean_ ? acc[j].mean : 0.0;
            double s = with_std_ ? acc[j].stddev(/*sample=*/false) : 1.0;
            if (s == 0.0 || std::isnan(s)) {
                s = 1.0;
            }
            scale_[j] = s;
        }
        fitted_ = true;
        n_features_ = f;
    }

    [[nodiscard]] NDArray transform(const NDArray& X) const override {
        ensure_fitted();
        std::size_t n = 0;
        NDArray out = detail::to_f64_matrix(X, n_features_, n);
        simd::standardize_f64(out.data<double>(), n, n_features_, mean_.data(), scale_.data());
        return out;
    }

    [[nodiscard]] bool is_fitted() const override { return fitted_; }
    [[nodiscard]] std::string name() const override { return "StandardScaler"; }

    [[nodiscard]] const std::vector<double>& mean() const { return mean_; }
    [[nodiscard]] const std::vector<double>& scale() const { return scale_; }

    void save(std::ostream& out) const override {
        ensure_fitted();
        out << "NEXUSDATA_PREPROCESSOR 1\n";
        out << "TYPE StandardScaler\n";
        out << "WITH_MEAN " << (with_mean_ ? 1 : 0) << "\n";
        out << "WITH_STD " << (with_std_ ? 1 : 0) << "\n";
        detail::write_vec(out, "MEAN", mean_);
        detail::write_vec(out, "SCALE", scale_);
    }

    void load(std::istream& in) override {
        std::string tag;
        int ver = 0;
        in >> tag >> ver;
        if (tag != "NEXUSDATA_PREPROCESSOR" || ver != 1) {
            throw IOError("StandardScaler::load: bad header");
        }
        std::string type_key, type_val;
        in >> type_key >> type_val;
        if (type_val != "StandardScaler") {
            throw IOError("StandardScaler::load: type mismatch");
        }
        std::string k;
        int wm = 0, ws = 0;
        in >> k >> wm >> k >> ws;
        with_mean_ = wm != 0;
        with_std_ = ws != 0;
        mean_ = detail::read_vec(in, "MEAN");
        scale_ = detail::read_vec(in, "SCALE");
        n_features_ = mean_.size();
        fitted_ = true;
    }

private:
    void ensure_fitted() const {
        if (!fitted_) {
            throw InvalidArgumentError("StandardScaler: not fitted");
        }
    }

    bool with_mean_ = true;
    bool with_std_ = true;
    bool fitted_ = false;
    std::size_t n_features_ = 0;
    std::vector<double> mean_;
    std::vector<double> scale_;
};

/// Scale features to [feature_range_min, feature_range_max].
class MinMaxScaler : public Transformer {
public:
    MinMaxScaler(double feature_min = 0.0, double feature_max = 1.0)
        : feature_min_(feature_min), feature_max_(feature_max) {
        if (feature_min_ >= feature_max_) {
            throw InvalidArgumentError("MinMaxScaler: invalid feature range");
        }
    }

    void fit(const NDArray& X) override {
        std::size_t n = 0, f = 0;
        detail::require_matrix(X, n, f);
        data_min_.assign(f, std::numeric_limits<double>::infinity());
        data_max_.assign(f, -std::numeric_limits<double>::infinity());
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < f; ++j) {
                const double v = detail::as_double(X, i * f + j);
                if (detail::is_nan_value(v)) {
                    continue;
                }
                data_min_[j] = std::min(data_min_[j], v);
                data_max_[j] = std::max(data_max_[j], v);
            }
        }
        for (std::size_t j = 0; j < f; ++j) {
            if (!std::isfinite(data_min_[j])) {
                data_min_[j] = 0.0;
                data_max_[j] = 1.0;
            }
        }
        n_features_ = f;
        fitted_ = true;
    }

    [[nodiscard]] NDArray transform(const NDArray& X) const override {
        ensure_fitted();
        std::size_t n = 0;
        NDArray out = detail::to_f64_matrix(X, n_features_, n);
        simd::minmax_inplace(out, data_min_, data_max_, feature_min_, feature_max_);
        return out;
    }

    [[nodiscard]] bool is_fitted() const override { return fitted_; }
    [[nodiscard]] std::string name() const override { return "MinMaxScaler"; }
    [[nodiscard]] const std::vector<double>& data_min() const { return data_min_; }
    [[nodiscard]] const std::vector<double>& data_max() const { return data_max_; }

    void save(std::ostream& out) const override {
        ensure_fitted();
        out << "NEXUSDATA_PREPROCESSOR 1\nTYPE MinMaxScaler\n";
        out << "FEATURE_MIN " << feature_min_ << "\nFEATURE_MAX " << feature_max_ << "\n";
        detail::write_vec(out, "DATA_MIN", data_min_);
        detail::write_vec(out, "DATA_MAX", data_max_);
    }

    void load(std::istream& in) override {
        std::string tag, type_key, type_val, k;
        int ver = 0;
        in >> tag >> ver >> type_key >> type_val;
        if (type_val != "MinMaxScaler") {
            throw IOError("MinMaxScaler::load: type mismatch");
        }
        in >> k >> feature_min_ >> k >> feature_max_;
        data_min_ = detail::read_vec(in, "DATA_MIN");
        data_max_ = detail::read_vec(in, "DATA_MAX");
        n_features_ = data_min_.size();
        fitted_ = true;
    }

private:
    void ensure_fitted() const {
        if (!fitted_) {
            throw InvalidArgumentError("MinMaxScaler: not fitted");
        }
    }

    double feature_min_ = 0.0;
    double feature_max_ = 1.0;
    bool fitted_ = false;
    std::size_t n_features_ = 0;
    std::vector<double> data_min_;
    std::vector<double> data_max_;
};

/// RobustScaler: (x - median) / IQR
class RobustScaler : public Transformer {
public:
    void fit(const NDArray& X) override {
        std::size_t n = 0, f = 0;
        detail::require_matrix(X, n, f);
        center_.resize(f);
        scale_.resize(f);
        for (std::size_t j = 0; j < f; ++j) {
            std::vector<double> col;
            col.reserve(n);
            for (std::size_t i = 0; i < n; ++i) {
                const double v = detail::as_double(X, i * f + j);
                if (!detail::is_nan_value(v)) {
                    col.push_back(v);
                }
            }
            if (col.empty()) {
                center_[j] = 0.0;
                scale_[j] = 1.0;
                continue;
            }
            std::sort(col.begin(), col.end());
            center_[j] = quantile_sorted(col, 0.5);
            const double q1 = quantile_sorted(col, 0.25);
            const double q3 = quantile_sorted(col, 0.75);
            double iqr = q3 - q1;
            if (iqr == 0.0) {
                iqr = 1.0;
            }
            scale_[j] = iqr;
        }
        n_features_ = f;
        fitted_ = true;
    }

    [[nodiscard]] NDArray transform(const NDArray& X) const override {
        ensure_fitted();
        return detail::transform_matrix(X, n_features_, [&](std::size_t, std::size_t j, double v) {
            return (v - center_[j]) / scale_[j];
        });
    }

    [[nodiscard]] bool is_fitted() const override { return fitted_; }
    [[nodiscard]] std::string name() const override { return "RobustScaler"; }

    void save(std::ostream& out) const override {
        ensure_fitted();
        out << "NEXUSDATA_PREPROCESSOR 1\nTYPE RobustScaler\n";
        detail::write_vec(out, "CENTER", center_);
        detail::write_vec(out, "SCALE", scale_);
    }

    void load(std::istream& in) override {
        std::string tag, tk, tv;
        int ver = 0;
        in >> tag >> ver >> tk >> tv;
        if (tv != "RobustScaler") {
            throw IOError("RobustScaler::load: type mismatch");
        }
        center_ = detail::read_vec(in, "CENTER");
        scale_ = detail::read_vec(in, "SCALE");
        n_features_ = center_.size();
        fitted_ = true;
    }

private:
    static double quantile_sorted(const std::vector<double>& sorted, double q) {
        if (sorted.empty()) {
            return 0.0;
        }
        const double pos = q * static_cast<double>(sorted.size() - 1);
        const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
        const std::size_t hi = static_cast<std::size_t>(std::ceil(pos));
        if (lo == hi) {
            return sorted[lo];
        }
        const double w = pos - static_cast<double>(lo);
        return sorted[lo] * (1.0 - w) + sorted[hi] * w;
    }

    void ensure_fitted() const {
        if (!fitted_) {
            throw InvalidArgumentError("RobustScaler: not fitted");
        }
    }

    bool fitted_ = false;
    std::size_t n_features_ = 0;
    std::vector<double> center_;
    std::vector<double> scale_;
};

/// MaxAbsScaler: x / max(|x|)
class MaxAbsScaler : public Transformer {
public:
    void fit(const NDArray& X) override {
        std::size_t n = 0, f = 0;
        detail::require_matrix(X, n, f);
        scale_.assign(f, 1.0);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < f; ++j) {
                const double v = std::abs(detail::as_double(X, i * f + j));
                if (!detail::is_nan_value(v)) {
                    scale_[j] = std::max(scale_[j], v);
                }
            }
        }
        for (double& s : scale_) {
            if (s == 0.0) {
                s = 1.0;
            }
        }
        n_features_ = f;
        fitted_ = true;
    }

    [[nodiscard]] NDArray transform(const NDArray& X) const override {
        ensure_fitted();
        return detail::transform_matrix(X, n_features_, [&](std::size_t, std::size_t j, double v) {
            return v / scale_[j];
        });
    }

    [[nodiscard]] bool is_fitted() const override { return fitted_; }
    [[nodiscard]] std::string name() const override { return "MaxAbsScaler"; }

    void save(std::ostream& out) const override {
        ensure_fitted();
        out << "NEXUSDATA_PREPROCESSOR 1\nTYPE MaxAbsScaler\n";
        detail::write_vec(out, "SCALE", scale_);
    }

    void load(std::istream& in) override {
        std::string tag, tk, tv;
        int ver = 0;
        in >> tag >> ver >> tk >> tv;
        if (tv != "MaxAbsScaler") {
            throw IOError("MaxAbsScaler::load: type mismatch");
        }
        scale_ = detail::read_vec(in, "SCALE");
        n_features_ = scale_.size();
        fitted_ = true;
    }

private:
    void ensure_fitted() const {
        if (!fitted_) {
            throw InvalidArgumentError("MaxAbsScaler: not fitted");
        }
    }

    bool fitted_ = false;
    std::size_t n_features_ = 0;
    std::vector<double> scale_;
};

} // namespace nexusdata
