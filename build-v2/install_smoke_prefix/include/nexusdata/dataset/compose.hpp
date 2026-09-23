#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "nexusdata/core/error.hpp"
#include "nexusdata/dataset/dataset.hpp"
#include "nexusdata/dataset/subset.hpp"

namespace nexusdata {

/// Keep indices where predicate(sample) is true (materializes index list at construction).
class FilterDataset : public Dataset {
public:
    using Pred = std::function<bool(const Sample&)>;

    FilterDataset(DatasetConstPtr parent, Pred pred) : parent_(std::move(parent)) {
        if (!parent_) {
            throw InvalidArgumentError("FilterDataset: null parent");
        }
        for (std::size_t i = 0; i < parent_->size(); ++i) {
            if (pred(parent_->get(i))) {
                indices_.push_back(i);
            }
        }
    }

    [[nodiscard]] std::size_t size() const override { return indices_.size(); }
    [[nodiscard]] Sample get(std::size_t index) const override {
        if (index >= indices_.size()) {
            throw IndexError("FilterDataset::get out of range");
        }
        return parent_->get(indices_[index]);
    }

private:
    DatasetConstPtr parent_;
    std::vector<std::size_t> indices_;
};

/// Zip two datasets of equal length: input=a's input, label=b's label (v0.6 convention).
class ZipDataset : public Dataset {
public:
    ZipDataset(DatasetConstPtr a, DatasetConstPtr b) : a_(std::move(a)), b_(std::move(b)) {
        if (!a_ || !b_) {
            throw InvalidArgumentError("ZipDataset: null");
        }
        if (a_->size() != b_->size()) {
            throw ShapeError("ZipDataset: size mismatch");
        }
    }
    [[nodiscard]] std::size_t size() const override { return a_->size(); }
    [[nodiscard]] Sample get(std::size_t index) const override {
        Sample x = a_->get(index);
        Sample y = b_->get(index);
        Sample out = std::move(x);
        out.label = std::move(y.label);
        // Fields of `a` win on key collisions.
        for (auto& [k, v] : y.extra_tensors) {
            if (!out.extra_tensors.contains(k)) out.extra_tensors.set(k, std::move(v));
        }
        for (auto& [k, v] : y.metadata) {
            if (!out.metadata.contains(k)) out.metadata.set(k, std::move(v));
        }
        return out;
    }

private:
    DatasetConstPtr a_;
    DatasetConstPtr b_;
};

/// Concatenate map-style datasets along sample axis.
class ConcatDataset : public Dataset {
public:
    explicit ConcatDataset(std::vector<DatasetConstPtr> parts) : parts_(std::move(parts)) {
        if (parts_.empty()) {
            throw InvalidArgumentError("ConcatDataset: empty");
        }
        offsets_.push_back(0);
        for (const auto& p : parts_) {
            if (!p) {
                throw InvalidArgumentError("ConcatDataset: null part");
            }
            offsets_.push_back(offsets_.back() + p->size());
        }
    }
    [[nodiscard]] std::size_t size() const override { return offsets_.back(); }
    [[nodiscard]] Sample get(std::size_t index) const override {
        if (index >= size()) {
            throw IndexError("ConcatDataset::get out of range");
        }
        std::size_t pi = 0;
        while (pi + 1 < offsets_.size() && index >= offsets_[pi + 1]) {
            ++pi;
        }
        return parts_[pi]->get(index - offsets_[pi]);
    }

private:
    std::vector<DatasetConstPtr> parts_;
    std::vector<std::size_t> offsets_;
};

/// Take first n samples.
[[nodiscard]] inline Subset take(DatasetConstPtr ds, std::size_t n) {
    if (!ds) {
        throw InvalidArgumentError("take: null");
    }
    n = std::min(n, ds->size());
    std::vector<std::size_t> idx(n);
    for (std::size_t i = 0; i < n; ++i) {
        idx[i] = i;
    }
    return Subset(std::move(ds), std::move(idx));
}

/// Skip first n samples.
[[nodiscard]] inline Subset skip(DatasetConstPtr ds, std::size_t n) {
    if (!ds) {
        throw InvalidArgumentError("skip: null");
    }
    if (n >= ds->size()) {
        return Subset(std::move(ds), {});
    }
    std::vector<std::size_t> idx;
    idx.reserve(ds->size() - n);
    for (std::size_t i = n; i < ds->size(); ++i) {
        idx.push_back(i);
    }
    return Subset(std::move(ds), std::move(idx));
}

/// Repeat each index `times` times (virtual length = n * times).
class RepeatDataset : public Dataset {
public:
    RepeatDataset(DatasetConstPtr parent, std::size_t times)
        : parent_(std::move(parent)), times_(times) {
        if (!parent_) {
            throw InvalidArgumentError("RepeatDataset: null");
        }
        if (times_ == 0) {
            throw InvalidArgumentError("RepeatDataset: times > 0");
        }
    }
    [[nodiscard]] std::size_t size() const override { return parent_->size() * times_; }
    [[nodiscard]] Sample get(std::size_t index) const override {
        if (index >= size()) {
            throw IndexError("RepeatDataset::get out of range");
        }
        return parent_->get(index % parent_->size());
    }

private:
    DatasetConstPtr parent_;
    std::size_t times_ = 1;
};

} // namespace nexusdata
