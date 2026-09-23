#pragma once

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "nexusdata/core/error.hpp"
#include "nexusdata/dataset/dataset.hpp"

namespace nexusdata {

/// View over a parent dataset selecting a subset of indices.
/// Owns a shared_ptr to the parent — safe against dangling.
/// Thread-safety: concurrent get() safe if parent get() is.
class Subset : public Dataset {
public:
    Subset() = default;

    Subset(DatasetConstPtr parent, std::vector<std::size_t> indices);

    [[nodiscard]] std::size_t size() const override;
    [[nodiscard]] Sample get(std::size_t index) const override;

    [[nodiscard]] const DatasetConstPtr& parent() const noexcept { return parent_; }
    [[nodiscard]] const std::vector<std::size_t>& indices() const noexcept {
        return indices_;
    }

private:
    DatasetConstPtr parent_;
    std::vector<std::size_t> indices_;
};

} // namespace nexusdata
