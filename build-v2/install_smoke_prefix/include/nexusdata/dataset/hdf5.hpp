#pragma once

#include <cstddef>
#include <string>

#include "nexusdata/core/error.hpp"
#include "nexusdata/dataset/dataset.hpp"

namespace nexusdata {

struct Hdf5Options {
    std::string dataset_path = "/data";
    std::string label_path; // empty = no labels / use index
};

/// Optional HDF5 dataset reader.
/// Default build: constructor throws; enable with -DNEXUSDATA_WITH_HDF5=ON (future).
class Hdf5Dataset : public Dataset {
public:
    Hdf5Dataset(const std::string& path, Hdf5Options opt = {});

    [[nodiscard]] std::size_t size() const override;
    [[nodiscard]] Sample get(std::size_t index) const override;

private:
    std::size_t n_ = 0;
};

[[nodiscard]] bool hdf5_support_enabled();

} // namespace nexusdata
