#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "nexusdata/dataset/dataset.hpp"

namespace nexusdata {

/// Comparison used by ParquetPredicate. Bounds are inclusive where the operator is.
enum class ParquetCompare : std::uint8_t { Eq, Ne, Lt, Le, Gt, Ge };

/// One predicate. Several predicates are combined with AND.
///
/// Numeric columns use `number` (`is_string == false`). Integers match exactly when
/// `number` is an integer in range; otherwise the comparison is done in double.
/// String / binary columns use `text` (`is_string == true`), ordered bytewise
/// (unsigned, the Parquet UTF-8 order). A null never matches.
struct ParquetPredicate {
    std::string column;
    ParquetCompare op = ParquetCompare::Eq;
    double number = 0;
    std::string text{};
    bool is_string = false;
};

/// Options for ParquetDataset. Existing fields stay first so `ParquetOptions{cols, label}`
/// and `ParquetDataset(path, {})` keep working.
struct ParquetOptions {
    /// Comma-separated column names to read. Empty = every column, in file order.
    std::string columns;
    /// Index into the projected column list used as Sample::label. -1 = none.
    /// Ignored when `label` is non-empty.
    int label_column = -1;
    /// Column name used as Sample::label. Wins over `label_column`.
    std::string label{};
    /// Numeric columns stored in Sample::metadata (formatted as text). String and
    /// binary columns always go to metadata.
    std::vector<std::string> metadata_columns{};
    /// Numeric columns stored in Sample::extra_tensors instead of Sample::input.
    /// Variable-length lists must be listed here.
    std::vector<std::string> extra_columns{};
    /// Row filter. A row group whose column statistics prove it has no match is not
    /// decoded (`predicate_pushdown`). Rows that are read are still filtered exactly.
    std::vector<ParquetPredicate> predicates{};
    bool predicate_pushdown = true;
    /// Parse columns of a row group in parallel (Arrow's IO pool).
    bool use_threads = true;
};

/// Apache Arrow / Parquet reader (map-style).
///
/// Enabled with -DNEXUSDATA_WITH_ARROW=ON. Without it the constructor throws
/// InvalidArgumentError, same as the other optional backends.
///
/// Projection: only the columns named in `options.columns` (plus any predicate
/// column) are decoded. Predicate pushdown drops a whole row group when its
/// min/max statistics cannot satisfy a predicate; `row_groups_skipped()` counts
/// those. The dataset then contains only rows that match, in file order.
///
/// Sample layout matches JsonDataset: one numeric input column keeps its shape
/// (scalars are [1], a fixed-length list is [len]); several input columns are
/// flattened and concatenated and must share a dtype. No label column => label
/// is the filtered row index (int64). Strings are metadata. Nulls in float32 /
/// float64 become NaN; a null in an integer or bool column is an error.
///
/// The projected columns are materialized at construction, so get() only copies
/// one row. Thread-safety: read-only after construction; concurrent get() is safe.
class ParquetDataset : public Dataset {
public:
    ParquetDataset(const std::string& path, ParquetOptions opt = {});
    ~ParquetDataset() override;
    ParquetDataset(const ParquetDataset&) = delete;
    ParquetDataset& operator=(const ParquetDataset&) = delete;
    ParquetDataset(ParquetDataset&&) noexcept;
    ParquetDataset& operator=(ParquetDataset&&) noexcept;

    [[nodiscard]] std::size_t size() const override;
    [[nodiscard]] Sample get(std::size_t index) const override;

    /// Projected columns, in sample order.
    [[nodiscard]] const std::vector<std::string>& column_names() const;

    [[nodiscard]] std::size_t row_groups_total() const noexcept;
    /// Row groups whose column pages were decoded.
    [[nodiscard]] std::size_t row_groups_read() const noexcept;
    /// Row groups skipped by predicate pushdown (statistics proved no match).
    [[nodiscard]] std::size_t row_groups_skipped() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

[[nodiscard]] bool parquet_support_enabled();

} // namespace nexusdata
