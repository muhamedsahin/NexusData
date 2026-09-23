#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "nexusdata/core/dtype.hpp"
#include "nexusdata/core/error.hpp"
#include "nexusdata/core/ndarray.hpp"
#include "nexusdata/dataset/dataset.hpp"

namespace nexusdata {

/// Dataset backed by stacked feature matrix [N, F] and labels [N].
/// Copy is shallow (NDArray shared storage).
class InMemoryDataset : public Dataset {
public:
    InMemoryDataset() = default;

    /// @param features shape [N, F] (or [N] for F=1)
    /// @param labels   shape [N] or [N, 1]
    InMemoryDataset(NDArray features, NDArray labels);

    [[nodiscard]] std::size_t size() const override;
    [[nodiscard]] Sample get(std::size_t index) const override;

    [[nodiscard]] const NDArray& features() const noexcept { return features_; }
    [[nodiscard]] const NDArray& labels() const noexcept { return labels_; }

private:
    NDArray features_;
    NDArray labels_;
    std::size_t n_ = 0;
    std::size_t n_features_ = 0;
};

} // namespace nexusdata
