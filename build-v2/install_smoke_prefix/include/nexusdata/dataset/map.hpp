#pragma once

#include <memory>

#include "nexusdata/dataset/dataset.hpp"
#include "nexusdata/pipeline/transform.hpp"

namespace nexusdata {

/// Applies a Transform lazily on each get(i). Parent kept via shared_ptr.
class MapDataset : public Dataset {
public:
    MapDataset(DatasetConstPtr parent, TransformConstPtr transform)
        : parent_(std::move(parent)), transform_(std::move(transform)) {
        if (!parent_) {
            throw InvalidArgumentError("MapDataset: null parent");
        }
        if (!transform_) {
            throw InvalidArgumentError("MapDataset: null transform");
        }
    }

    template <typename DatasetT>
    MapDataset(DatasetT parent, TransformConstPtr transform)
        : MapDataset(DatasetConstPtr(std::make_shared<DatasetT>(std::move(parent))),
                     std::move(transform)) {}

    [[nodiscard]] std::size_t size() const override { return parent_->size(); }

    [[nodiscard]] Sample get(std::size_t index) const override {
        return transform_->apply(parent_->get(index));
    }

    [[nodiscard]] const DatasetConstPtr& parent() const noexcept { return parent_; }

private:
    DatasetConstPtr parent_;
    TransformConstPtr transform_;
};

[[nodiscard]] inline MapDataset map(DatasetConstPtr ds, TransformConstPtr t) {
    return MapDataset(std::move(ds), std::move(t));
}

} // namespace nexusdata
