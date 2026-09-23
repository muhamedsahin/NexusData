#include "nexusdata/dataset/subset.hpp"

#include <string>

namespace nexusdata {

Subset::Subset(DatasetConstPtr parent, std::vector<std::size_t> indices)
    : parent_(std::move(parent)), indices_(std::move(indices)) {
    if (!parent_) {
        throw InvalidArgumentError("Subset: parent dataset is null");
    }
    const std::size_t n = parent_->size();
    for (std::size_t idx : indices_) {
        if (idx >= n) {
            throw IndexError("Subset: index " + std::to_string(idx) +
                             " out of parent range [0, " + std::to_string(n) + ")");
        }
    }
}

std::size_t Subset::size() const {
    return indices_.size();
}

Sample Subset::get(std::size_t index) const {
    if (index >= indices_.size()) {
        throw IndexError("Subset::get: index out of range");
    }
    return parent_->get(indices_[index]);
}

} // namespace nexusdata
