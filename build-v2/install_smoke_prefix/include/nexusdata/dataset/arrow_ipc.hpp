#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "nexusdata/core/ndarray.hpp"
#include "nexusdata/dataset/dataset.hpp"

namespace nexusdata {

/// Options for ArrowIpcDataset. Column roles match ParquetOptions.
struct ArrowIpcOptions {
    /// Comma-separated column names to read. Empty = every column, in file order.
    std::string columns;
    /// Index into the projected column list used as Sample::label. -1 = none.
    /// Ignored when `label` is non-empty.
    int label_column = -1;
    std::string label{};
    std::vector<std::string> metadata_columns{};
    std::vector<std::string> extra_columns{};
};

/// Arrow IPC file and Feather v1/v2 reader (map-style).
///
/// Enabled with -DNEXUSDATA_WITH_ARROW=ON (same switch as ParquetDataset). The file
/// is memory-mapped. `column()` returns that column as one NDArray of shape
/// [rows, ...] sharing the mapping — no copy — when the column is a single
/// contiguous primitive buffer without nulls (not bool, which is bit-packed).
/// Otherwise the column is materialized and `column_is_zero_copy()` is false.
/// The returned array aliases dataset storage; do not write to it.
///
/// get() copies a single row, same layout rules as ParquetDataset.
/// Thread-safety: read-only after construction; concurrent get() / column() is safe.
class ArrowIpcDataset : public Dataset {
public:
    ArrowIpcDataset(const std::string& path, ArrowIpcOptions opt = {});
    ~ArrowIpcDataset() override;
    ArrowIpcDataset(const ArrowIpcDataset&) = delete;
    ArrowIpcDataset& operator=(const ArrowIpcDataset&) = delete;
    ArrowIpcDataset(ArrowIpcDataset&&) noexcept;
    ArrowIpcDataset& operator=(ArrowIpcDataset&&) noexcept;

    [[nodiscard]] std::size_t size() const override;
    [[nodiscard]] Sample get(std::size_t index) const override;

    [[nodiscard]] const std::vector<std::string>& column_names() const;

    /// Full projected column (all rows). Throws if `name` is not a numeric column.
    [[nodiscard]] NDArray column(std::string_view name) const;
    [[nodiscard]] bool column_is_zero_copy(std::string_view name) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

[[nodiscard]] bool arrow_ipc_support_enabled();

} // namespace nexusdata
