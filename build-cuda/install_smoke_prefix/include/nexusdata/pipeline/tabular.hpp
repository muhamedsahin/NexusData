#pragma once

#include <algorithm>
#include <cmath>
#include <memory>

#include "nexusdata/core/array_utils.hpp"
#include "nexusdata/core/ndarray.hpp"
#include "nexusdata/pipeline/fusion.hpp"
#include "nexusdata/pipeline/transform.hpp"

namespace nexusdata {

// v2.0: every tabular transform is a single FuseStep executed by the SIMD
// fusion engine (one allocation + one pass, bit-identical to the v1 scalar
// as_double/set_double loops). Inside a Compose, neighbouring tabular steps are
// merged into one FusedTransform.

/// Element-wise clip of sample.input to [min_v, max_v].
class ClipTransform : public Transform {
public:
    ClipTransform(double min_v, double max_v) : min_v_(min_v), max_v_(max_v) {
        if (min_v_ > max_v_) {
            throw InvalidArgumentError("ClipTransform: min > max");
        }
        step_ = FuseStep::clip(min_v_, max_v_);
    }

    [[nodiscard]] Sample apply(const Sample& sample) const override {
        return apply_fuse_step(step_, sample);
    }

    [[nodiscard]] bool is_fusable() const override { return true; }
    [[nodiscard]] FuseStepPtr fuse_step() const override { return step_; }

private:
    double min_v_ = 0.0;
    double max_v_ = 1.0;
    FuseStepPtr step_;
};

/// log1p on sample.input (requires non-negative values for real results).
class Log1pTransform : public Transform {
public:
    Log1pTransform() : step_(FuseStep::log1p()) {}

    [[nodiscard]] Sample apply(const Sample& sample) const override {
        return apply_fuse_step(step_, sample);
    }

    [[nodiscard]] bool is_fusable() const override { return true; }
    [[nodiscard]] FuseStepPtr fuse_step() const override { return step_; }

private:
    FuseStepPtr step_;
};

/// (x - mean) / scale per feature. mean/scale length must equal feature count.
class StandardizeTransform : public Transform {
public:
    StandardizeTransform(std::vector<double> mean, std::vector<double> scale)
        : mean_(std::move(mean)), scale_(std::move(scale)) {
        if (mean_.size() != scale_.size() || mean_.empty()) {
            throw InvalidArgumentError("StandardizeTransform: mean/scale size mismatch");
        }
        for (double& s : scale_) {
            if (s == 0.0 || std::isnan(s)) {
                s = 1.0;
            }
        }
        step_ = FuseStep::standardize(mean_, scale_);
    }

    /// Throws ShapeError unless sample.input has exactly mean.size() elements.
    [[nodiscard]] Sample apply(const Sample& sample) const override {
        return apply_fuse_step(step_, sample);
    }

    [[nodiscard]] bool is_fusable() const override { return true; }
    [[nodiscard]] FuseStepPtr fuse_step() const override { return step_; }

private:
    std::vector<double> mean_;
    std::vector<double> scale_;
    FuseStepPtr step_;
};

/// Scale to [feature_min, feature_max] using fitted data_min/data_max.
class MinMaxTransform : public Transform {
public:
    MinMaxTransform(std::vector<double> data_min,
                    std::vector<double> data_max,
                    double feature_min = 0.0,
                    double feature_max = 1.0)
        : data_min_(std::move(data_min)),
          data_max_(std::move(data_max)),
          feature_min_(feature_min),
          feature_max_(feature_max) {
        if (data_min_.size() != data_max_.size() || data_min_.empty()) {
            throw InvalidArgumentError("MinMaxTransform: size mismatch");
        }
        step_ = FuseStep::minmax(data_min_, data_max_, feature_min_, feature_max_);
    }

    [[nodiscard]] Sample apply(const Sample& sample) const override {
        return apply_fuse_step(step_, sample);
    }

    [[nodiscard]] bool is_fusable() const override { return true; }
    [[nodiscard]] FuseStepPtr fuse_step() const override { return step_; }

private:
    std::vector<double> data_min_;
    std::vector<double> data_max_;
    double feature_min_ = 0.0;
    double feature_max_ = 1.0;
    FuseStepPtr step_;
};

} // namespace nexusdata
