#pragma once

#include <cstddef>
#include <memory>

#include "nexusdata/dataset/sample.hpp"

namespace nexusdata {

/// Map-style dataset: random-access by index.
/// Lifetime: prefer holding datasets via shared_ptr when composing
/// (Subset, DataLoader, random_split) so views outlive safely.
/// Thread-safety: get() must be safe for concurrent reads if the
/// concrete implementation documents it (InMemory/CSV are read-only after load).
class Dataset {
public:
    virtual ~Dataset() = default;

    [[nodiscard]] virtual std::size_t size() const = 0;
    [[nodiscard]] virtual Sample get(std::size_t index) const = 0;

    [[nodiscard]] bool empty() const { return size() == 0; }
};

using DatasetPtr = std::shared_ptr<Dataset>;
using DatasetConstPtr = std::shared_ptr<const Dataset>;

} // namespace nexusdata
