#include "nexusdata/dataset/in_memory.hpp"

#include <cstring>
#include <string>

namespace nexusdata {

InMemoryDataset::InMemoryDataset(NDArray features, NDArray labels)
    : features_(std::move(features)), labels_(std::move(labels)) {
    if (features_.shape().empty() || labels_.shape().empty()) {
        throw InvalidArgumentError("InMemoryDataset: empty features or labels");
    }
    n_ = features_.shape()[0];
    if (labels_.shape()[0] != n_) {
        throw ShapeError("InMemoryDataset: features and labels length mismatch");
    }
    if (features_.shape().size() == 1) {
        n_features_ = 1;
        features_.reshape(Shape{n_, 1});
    } else if (features_.shape().size() == 2) {
        n_features_ = features_.shape()[1];
    } else {
        throw ShapeError("InMemoryDataset: features must be rank 1 or 2");
    }
    if (labels_.shape().size() == 2 && labels_.shape()[1] == 1) {
        labels_.reshape(Shape{n_});
    } else if (labels_.shape().size() != 1) {
        throw ShapeError("InMemoryDataset: labels must be rank 1 (or [N,1])");
    }
}

std::size_t InMemoryDataset::size() const {
    return n_;
}

Sample InMemoryDataset::get(std::size_t index) const {
    if (index >= n_) {
        throw IndexError(format_index_error("InMemoryDataset::get", index, n_));
    }

    Sample s;
    s.input = NDArray(Shape{n_features_}, features_.dtype());
    s.label = NDArray(Shape{1}, labels_.dtype());

    const std::size_t feat_bytes = n_features_ * size_of(features_.dtype());
    const auto* src_f =
        static_cast<const std::uint8_t*>(features_.data()) + index * feat_bytes;
    std::memcpy(s.input.data(), src_f, feat_bytes);

    const std::size_t lab_bytes = size_of(labels_.dtype());
    const auto* src_l =
        static_cast<const std::uint8_t*>(labels_.data()) + index * lab_bytes;
    std::memcpy(s.label.data(), src_l, lab_bytes);
    return s;
}

} // namespace nexusdata
