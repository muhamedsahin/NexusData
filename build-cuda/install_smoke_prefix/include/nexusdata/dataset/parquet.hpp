#pragma once

#include <cstddef>
#include <string>

#include "nexusdata/core/error.hpp"
#include "nexusdata/dataset/dataset.hpp"

namespace nexusdata {

struct ParquetOptions {
    std::string columns; // comma-separated; empty = all (when enabled)
    int label_column = -1;
};

/// Optional Apache Arrow / Parquet reader.
/// Default build: constructor throws; enable with -DNEXUSDATA_WITH_ARROW=ON (future).
class ParquetDataset : public Dataset {
public:
    ParquetDataset(const std::string& path, ParquetOptions opt = {});

    [[nodiscard]] std::size_t size() const override;
    [[nodiscard]] Sample get(std::size_t index) const override;

private:
    std::size_t n_ = 0;
};

[[nodiscard]] bool parquet_support_enabled();

} // namespace nexusdata
