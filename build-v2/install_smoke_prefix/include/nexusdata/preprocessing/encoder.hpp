#pragma once

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "nexusdata/core/array_utils.hpp"
#include "nexusdata/preprocessing/scaler.hpp"
#include "nexusdata/preprocessing/transformer.hpp"

namespace nexusdata {

/// Encode 1-D labels to 0..K-1.
class LabelEncoder : public Transformer {
public:
    void fit(const NDArray& y) override {
        std::size_t n = 0, f = 0;
        detail::require_matrix(y, n, f);
        if (f != 1 && y.shape().size() != 1) {
            // allow [N] or [N,1]
        }
        if (y.shape().size() == 1) {
            n = y.shape()[0];
            f = 1;
        }
        std::set<double> uniq;
        for (std::size_t i = 0; i < n; ++i) {
            uniq.insert(detail::as_double(y, i * f));
        }
        classes_.assign(uniq.begin(), uniq.end());
        class_to_index_.clear();
        for (std::size_t i = 0; i < classes_.size(); ++i) {
            class_to_index_[classes_[i]] = static_cast<std::int64_t>(i);
        }
        fitted_ = true;
    }

    [[nodiscard]] NDArray transform(const NDArray& y) const override {
        ensure_fitted();
        std::size_t n = y.shape()[0];
        NDArray out(Shape{n}, DType::Int64);
        const std::size_t stride = (y.shape().size() == 2) ? y.shape()[1] : 1;
        for (std::size_t i = 0; i < n; ++i) {
            const double v = detail::as_double(y, i * stride);
            auto it = class_to_index_.find(v);
            if (it == class_to_index_.end()) {
                throw InvalidArgumentError("LabelEncoder: unseen label " + std::to_string(v));
            }
            out.data<std::int64_t>()[i] = it->second;
        }
        return out;
    }

    [[nodiscard]] bool is_fitted() const override { return fitted_; }
    [[nodiscard]] std::string name() const override { return "LabelEncoder"; }
    [[nodiscard]] const std::vector<double>& classes() const { return classes_; }

    void save(std::ostream& out) const override {
        ensure_fitted();
        out << "NEXUSDATA_PREPROCESSOR 1\nTYPE LabelEncoder\n";
        detail::write_vec(out, "CLASSES", classes_);
    }

    void load(std::istream& in) override {
        std::string tag, tk, tv;
        int ver = 0;
        in >> tag >> ver >> tk >> tv;
        if (tv != "LabelEncoder") {
            throw IOError("LabelEncoder::load: type mismatch");
        }
        classes_ = detail::read_vec(in, "CLASSES");
        class_to_index_.clear();
        for (std::size_t i = 0; i < classes_.size(); ++i) {
            class_to_index_[classes_[i]] = static_cast<std::int64_t>(i);
        }
        fitted_ = true;
    }

private:
    void ensure_fitted() const {
        if (!fitted_) {
            throw InvalidArgumentError("LabelEncoder: not fitted");
        }
    }

    bool fitted_ = false;
    std::vector<double> classes_;
    std::map<double, std::int64_t> class_to_index_;
};

/// One-hot encode integer/categorical columns. Fits unique values per column.
class OneHotEncoder : public Transformer {
public:
    void fit(const NDArray& X) override {
        std::size_t n = 0, f = 0;
        detail::require_matrix(X, n, f);
        categories_.assign(f, {});
        for (std::size_t j = 0; j < f; ++j) {
            std::set<double> uniq;
            for (std::size_t i = 0; i < n; ++i) {
                uniq.insert(detail::as_double(X, i * f + j));
            }
            categories_[j].assign(uniq.begin(), uniq.end());
        }
        n_features_in_ = f;
        n_features_out_ = 0;
        for (const auto& c : categories_) {
            n_features_out_ += c.size();
        }
        fitted_ = true;
    }

    [[nodiscard]] NDArray transform(const NDArray& X) const override {
        ensure_fitted();
        std::size_t n = 0, f = 0;
        detail::require_matrix(X, n, f);
        if (f != n_features_in_) {
            throw ShapeError("OneHotEncoder: feature count mismatch");
        }
        NDArray out(Shape{n, n_features_out_}, DType::Float64);
        for (std::size_t i = 0; i < n; ++i) {
            std::size_t offset = 0;
            for (std::size_t j = 0; j < f; ++j) {
                const double v = detail::as_double(X, i * f + j);
                bool found = false;
                for (std::size_t k = 0; k < categories_[j].size(); ++k) {
                    if (categories_[j][k] == v) {
                        detail::set_double(out, i * n_features_out_ + offset + k, 1.0);
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    throw InvalidArgumentError("OneHotEncoder: unseen category");
                }
                offset += categories_[j].size();
            }
        }
        return out;
    }

    [[nodiscard]] bool is_fitted() const override { return fitted_; }
    [[nodiscard]] std::string name() const override { return "OneHotEncoder"; }
    [[nodiscard]] std::size_t n_features_out() const { return n_features_out_; }

    void save(std::ostream& out) const override {
        ensure_fitted();
        out << "NEXUSDATA_PREPROCESSOR 1\nTYPE OneHotEncoder\n";
        out << "N_FEATURES_IN " << n_features_in_ << "\n";
        out << "N_CATS " << categories_.size() << "\n";
        for (std::size_t j = 0; j < categories_.size(); ++j) {
            detail::write_vec(out, "CAT", categories_[j]);
        }
    }

    void load(std::istream& in) override {
        std::string tag, tk, tv, k;
        int ver = 0;
        in >> tag >> ver >> tk >> tv;
        if (tv != "OneHotEncoder") {
            throw IOError("OneHotEncoder::load: type mismatch");
        }
        std::size_t n_cats = 0;
        in >> k >> n_features_in_ >> k >> n_cats;
        categories_.resize(n_cats);
        n_features_out_ = 0;
        for (std::size_t j = 0; j < n_cats; ++j) {
            categories_[j] = detail::read_vec(in, "CAT");
            n_features_out_ += categories_[j].size();
        }
        fitted_ = true;
    }

private:
    void ensure_fitted() const {
        if (!fitted_) {
            throw InvalidArgumentError("OneHotEncoder: not fitted");
        }
    }

    bool fitted_ = false;
    std::size_t n_features_in_ = 0;
    std::size_t n_features_out_ = 0;
    std::vector<std::vector<double>> categories_;
};

/// Map categories to ordinal integers 0..K-1 per column (same as LabelEncoder per col).
class OrdinalEncoder : public Transformer {
public:
    void fit(const NDArray& X) override {
        std::size_t n = 0, f = 0;
        detail::require_matrix(X, n, f);
        categories_.assign(f, {});
        for (std::size_t j = 0; j < f; ++j) {
            std::set<double> uniq;
            for (std::size_t i = 0; i < n; ++i) {
                uniq.insert(detail::as_double(X, i * f + j));
            }
            categories_[j].assign(uniq.begin(), uniq.end());
        }
        n_features_ = f;
        fitted_ = true;
    }

    [[nodiscard]] NDArray transform(const NDArray& X) const override {
        ensure_fitted();
        std::size_t n = 0, f = 0;
        detail::require_matrix(X, n, f);
        if (f != n_features_) {
            throw ShapeError("OrdinalEncoder: feature mismatch");
        }
        NDArray out(Shape{n, f}, DType::Float64);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < f; ++j) {
                const double v = detail::as_double(X, i * f + j);
                auto it = std::find(categories_[j].begin(), categories_[j].end(), v);
                if (it == categories_[j].end()) {
                    throw InvalidArgumentError("OrdinalEncoder: unseen category");
                }
                const auto code = static_cast<double>(it - categories_[j].begin());
                detail::set_double(out, i * f + j, code);
            }
        }
        return out;
    }

    [[nodiscard]] bool is_fitted() const override { return fitted_; }
    [[nodiscard]] std::string name() const override { return "OrdinalEncoder"; }

    void save(std::ostream& out) const override {
        ensure_fitted();
        out << "NEXUSDATA_PREPROCESSOR 1\nTYPE OrdinalEncoder\n";
        out << "N_CATS " << categories_.size() << "\n";
        for (const auto& c : categories_) {
            detail::write_vec(out, "CAT", c);
        }
    }

    void load(std::istream& in) override {
        std::string tag, tk, tv, k;
        int ver = 0;
        in >> tag >> ver >> tk >> tv;
        if (tv != "OrdinalEncoder") {
            throw IOError("OrdinalEncoder::load: type mismatch");
        }
        std::size_t n_cats = 0;
        in >> k >> n_cats;
        categories_.resize(n_cats);
        for (std::size_t j = 0; j < n_cats; ++j) {
            categories_[j] = detail::read_vec(in, "CAT");
        }
        n_features_ = n_cats;
        fitted_ = true;
    }

private:
    void ensure_fitted() const {
        if (!fitted_) {
            throw InvalidArgumentError("OrdinalEncoder: not fitted");
        }
    }

    bool fitted_ = false;
    std::size_t n_features_ = 0;
    std::vector<std::vector<double>> categories_;
};

} // namespace nexusdata
