#pragma once

// Shared Arrow Table -> columnar Sample storage. Only included from translation
// units built with NEXUSDATA_WITH_ARROW.

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "nexusdata/dataset/sample.hpp"

#ifdef NEXUSDATA_WITH_ARROW

#include <memory>

namespace arrow {
class Table;
}

namespace nexusdata::arrow_detail {

enum class Role : std::uint8_t { Input, Label, Extra, Metadata };

struct ColumnPlan {
    /// Empty = every column of the table, in schema order.
    std::vector<std::string> names;
    std::string label_name;
    int label_index = -1;
    std::vector<std::string> metadata;
    std::vector<std::string> extra;
};

struct Column {
    std::string name;
    enum class Kind : std::uint8_t { Numeric, Strings, Ragged } kind = Kind::Numeric;
    Role role = Role::Input;
    DType dtype = DType::Float32;
    /// Shape of one sample's slice. Scalars are [1].
    Shape sample_shape{};
    std::size_t row_numel = 0;
    /// Numeric columns: [rows, ...values]. Scalars are stored as [rows].
    NDArray data;
    std::vector<std::string> strings;
    std::vector<NDArray> ragged;
    bool zero_copy = false;
};

struct TableColumns {
    std::size_t rows = 0;
    std::vector<Column> columns;
    std::vector<std::string> names;

    [[nodiscard]] Sample get(std::size_t index) const;
    [[nodiscard]] const Column* find(std::string_view name) const;
};

/// `keep` is one byte per table row (nonzero = keep), or nullptr to keep every row.
/// `zero_copy` views a single contiguous primitive buffer when that is possible
/// (no row filter, one chunk, no nulls). `where` is prefixed onto error messages.
[[nodiscard]] TableColumns materialize(const arrow::Table& table, const std::uint8_t* keep,
                                       const ColumnPlan& plan, bool zero_copy,
                                       const std::string& where);

[[nodiscard]] std::vector<std::string> split_names(const std::string& csv);

} // namespace nexusdata::arrow_detail

#endif
