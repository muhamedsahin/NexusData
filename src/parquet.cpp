#include "nexusdata/dataset/parquet.hpp"

#include <utility>

#ifdef NEXUSDATA_WITH_ARROW

#ifndef NOMINMAX
#define NOMINMAX
#endif

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/schema.h>
#include <parquet/metadata.h>
#include <parquet/statistics.h>
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string_view>

#include "arrow_columns.hpp"

#endif

namespace nexusdata {
namespace {

#ifdef NEXUSDATA_WITH_ARROW

void ok_arrow(const arrow::Status& st, const std::string& what) {
    if (!st.ok()) throw IOError(what + ": " + st.ToString());
}

template <typename T>
T take_arrow(arrow::Result<T> result, const std::string& what) {
    if (!result.ok()) throw IOError(what + ": " + result.status().ToString());
    return std::move(result).ValueOrDie();
}

void collect_leaves(const parquet::arrow::SchemaField& field, std::vector<int>& out) {
    if (field.column_index >= 0) out.push_back(field.column_index);
    for (const auto& child : field.children) collect_leaves(child, out);
}

const parquet::arrow::SchemaField* find_field(const std::vector<parquet::arrow::SchemaField>& fields,
                                              const std::string& name, const std::string& path) {
    const parquet::arrow::SchemaField* found = nullptr;
    for (const auto& field : fields) {
        if (field.field && field.field->name() == name) {
            if (found) throw InvalidArgumentError(path + ": duplicate column \"" + name + "\"");
            found = &field;
        }
    }
    if (!found) throw InvalidArgumentError(path + ": column \"" + name + "\" not found");
    return found;
}

template <typename T>
bool fits_exact(double number, T& out) {
    if (!std::isfinite(number)) return false;
    if (number < static_cast<double>(std::numeric_limits<T>::lowest())) return false;
    if (number > static_cast<double>(std::numeric_limits<T>::max())) return false;
    out = static_cast<T>(number);
    return static_cast<double>(out) == number;
}

template <typename T>
bool skip_ordered(T min, T max, ParquetCompare op, T bound) {
    switch (op) {
        case ParquetCompare::Eq: return bound < min || bound > max;
        case ParquetCompare::Ne: return min == max && min == bound;
        case ParquetCompare::Lt: return min >= bound;
        case ParquetCompare::Le: return min > bound;
        case ParquetCompare::Gt: return max <= bound;
        case ParquetCompare::Ge: return max < bound;
    }
    return false;
}

bool skip_int_stats(const parquet::Statistics& stats, ParquetCompare op, double number) {
    const auto* descr = stats.descr();
    if (descr) {
        const auto logical = descr->logical_type();
        if (logical && logical->type() == parquet::LogicalType::Type::INT) {
            const auto* as_int = dynamic_cast<const parquet::IntLogicalType*>(logical.get());
            if (as_int && !as_int->is_signed()) return false;
        }
    }
    switch (stats.physical_type()) {
        case parquet::Type::INT32: {
            const auto* typed = dynamic_cast<const parquet::Int32Statistics*>(&stats);
            if (!typed) return false;
            std::int32_t bound = 0;
            if (!fits_exact(number, bound)) return op == ParquetCompare::Eq;
            return skip_ordered(typed->min(), typed->max(), op, bound);
        }
        case parquet::Type::INT64: {
            const auto* typed = dynamic_cast<const parquet::Int64Statistics*>(&stats);
            if (!typed) return false;
            std::int64_t bound = 0;
            if (!fits_exact(number, bound)) return op == ParquetCompare::Eq;
            return skip_ordered(typed->min(), typed->max(), op, bound);
        }
        default: return false;
    }
}

bool skip_float_stats(double min, double max, ParquetCompare op, double bound) {
    if (std::isnan(min) || std::isnan(max) || std::isnan(bound)) return false;
    switch (op) {
        case ParquetCompare::Eq: return bound < min || bound > max;
        case ParquetCompare::Ne: return min == max && min == bound;
        case ParquetCompare::Lt: return min >= bound;
        case ParquetCompare::Le: return min > bound;
        case ParquetCompare::Gt: return max <= bound;
        case ParquetCompare::Ge: return max < bound;
    }
    return false;
}

int cmp_bytes(const parquet::ByteArray& left, std::string_view right) {
    const std::size_t n = std::min(static_cast<std::size_t>(left.len), right.size());
    if (n > 0) {
        const int c = std::memcmp(left.ptr, right.data(), n);
        if (c < 0) return -1;
        if (c > 0) return 1;
    }
    if (left.len < right.size()) return -1;
    if (left.len > right.size()) return 1;
    return 0;
}

bool skip_bytes(const parquet::ByteArray& min, const parquet::ByteArray& max, ParquetCompare op,
                std::string_view text) {
    const int cmin = cmp_bytes(min, text);
    const int cmax = cmp_bytes(max, text);
    switch (op) {
        case ParquetCompare::Eq: return cmin > 0 || cmax < 0;
        case ParquetCompare::Ne: return cmin == 0 && cmax == 0;
        case ParquetCompare::Lt: return cmin >= 0;
        case ParquetCompare::Le: return cmin > 0;
        case ParquetCompare::Gt: return cmax <= 0;
        case ParquetCompare::Ge: return cmax < 0;
    }
    return false;
}

/// True when statistics prove that no non-null value in the row group can match.
bool stats_reject(const parquet::Statistics& stats, const ParquetPredicate& pred, int64_t rows) {
    if (rows > 0 && stats.HasNullCount() && stats.null_count() >= rows) return true;
    if (!stats.HasMinMax()) return false;
    switch (stats.physical_type()) {
        case parquet::Type::BOOLEAN: {
            if (pred.is_string) return false;
            const auto* typed = dynamic_cast<const parquet::BoolStatistics*>(&stats);
            if (!typed) return false;
            const int min = typed->min() ? 1 : 0;
            const int max = typed->max() ? 1 : 0;
            int bound = 0;
            if (!fits_exact(pred.number, bound)) return pred.op == ParquetCompare::Eq;
            return skip_ordered(min, max, pred.op, bound);
        }
        case parquet::Type::INT32:
        case parquet::Type::INT64:
            if (pred.is_string) return false;
            return skip_int_stats(stats, pred.op, pred.number);
        case parquet::Type::FLOAT: {
            if (pred.is_string) return false;
            const auto* typed = dynamic_cast<const parquet::FloatStatistics*>(&stats);
            if (!typed) return false;
            return skip_float_stats(typed->min(), typed->max(), pred.op, pred.number);
        }
        case parquet::Type::DOUBLE: {
            if (pred.is_string) return false;
            const auto* typed = dynamic_cast<const parquet::DoubleStatistics*>(&stats);
            if (!typed) return false;
            return skip_float_stats(typed->min(), typed->max(), pred.op, pred.number);
        }
        case parquet::Type::BYTE_ARRAY: {
            if (!pred.is_string) return false;
            const auto* typed = dynamic_cast<const parquet::ByteArrayStatistics*>(&stats);
            if (!typed) return false;
            return skip_bytes(typed->min(), typed->max(), pred.op, pred.text);
        }
        default: return false;
    }
}

template <typename T>
bool match_int(T value, ParquetCompare op, double number) {
    T bound{};
    if (fits_exact(number, bound)) return !skip_ordered(value, value, op, bound);
    const double x = static_cast<double>(value);
    switch (op) {
        case ParquetCompare::Eq: return x == number;
        case ParquetCompare::Ne: return x != number;
        case ParquetCompare::Lt: return x < number;
        case ParquetCompare::Le: return x <= number;
        case ParquetCompare::Gt: return x > number;
        case ParquetCompare::Ge: return x >= number;
    }
    return false;
}

bool match_float(double value, ParquetCompare op, double number) {
    if (std::isnan(value) || std::isnan(number)) return false;
    switch (op) {
        case ParquetCompare::Eq: return value == number;
        case ParquetCompare::Ne: return value != number;
        case ParquetCompare::Lt: return value < number;
        case ParquetCompare::Le: return value <= number;
        case ParquetCompare::Gt: return value > number;
        case ParquetCompare::Ge: return value >= number;
    }
    return false;
}

int cmp_text(std::string_view left, std::string_view right) {
    const std::size_t n = std::min(left.size(), right.size());
    if (n > 0) {
        const int c = std::memcmp(left.data(), right.data(), n);
        if (c < 0) return -1;
        if (c > 0) return 1;
    }
    if (left.size() < right.size()) return -1;
    if (left.size() > right.size()) return 1;
    return 0;
}

bool match_text(std::string_view value, ParquetCompare op, std::string_view text) {
    const int c = cmp_text(value, text);
    switch (op) {
        case ParquetCompare::Eq: return c == 0;
        case ParquetCompare::Ne: return c != 0;
        case ParquetCompare::Lt: return c < 0;
        case ParquetCompare::Le: return c <= 0;
        case ParquetCompare::Gt: return c > 0;
        case ParquetCompare::Ge: return c >= 0;
    }
    return false;
}

template <typename ArrowType>
void mask_numeric(const arrow::ChunkedArray& column, const ParquetPredicate& pred,
                  std::uint8_t* keep) {
    using T = typename ArrowType::c_type;
    std::size_t row = 0;
    for (int c = 0; c < column.num_chunks(); ++c) {
        const auto& array = static_cast<const arrow::NumericArray<ArrowType>&>(*column.chunk(c));
        const T* values = array.raw_values();
        for (int64_t i = 0; i < array.length(); ++i, ++row) {
            if (!keep[row]) continue;
            if (!array.IsValid(i) || !match_int(values[i], pred.op, pred.number)) keep[row] = 0;
        }
    }
}

void mask_bool(const arrow::ChunkedArray& column, const ParquetPredicate& pred, std::uint8_t* keep) {
    std::size_t row = 0;
    for (int c = 0; c < column.num_chunks(); ++c) {
        const auto& array = static_cast<const arrow::BooleanArray&>(*column.chunk(c));
        for (int64_t i = 0; i < array.length(); ++i, ++row) {
            if (!keep[row]) continue;
            const int bit = array.IsValid(i) && array.Value(i) ? 1 : 0;
            if (!array.IsValid(i) || !match_int(bit, pred.op, pred.number)) keep[row] = 0;
        }
    }
}

void mask_float(const arrow::ChunkedArray& column, const ParquetPredicate& pred, std::uint8_t* keep,
                bool as_double) {
    std::size_t row = 0;
    for (int c = 0; c < column.num_chunks(); ++c) {
        const arrow::Array& array = *column.chunk(c);
        for (int64_t i = 0; i < array.length(); ++i, ++row) {
            if (!keep[row]) continue;
            if (!array.IsValid(i)) {
                keep[row] = 0;
                continue;
            }
            const double value = as_double ? static_cast<const arrow::DoubleArray&>(array).Value(i)
                                           : static_cast<const arrow::FloatArray&>(array).Value(i);
            if (!match_float(value, pred.op, pred.number)) keep[row] = 0;
        }
    }
}

void mask_text(const arrow::ChunkedArray& column, const ParquetPredicate& pred, std::uint8_t* keep,
               const std::string& path) {
    std::size_t row = 0;
    for (int c = 0; c < column.num_chunks(); ++c) {
        const arrow::Array& array = *column.chunk(c);
        if (array.type_id() == arrow::Type::DICTIONARY) {
            throw IOError(path + ": predicate on a dictionary-encoded column is not supported");
        }
        for (int64_t i = 0; i < array.length(); ++i, ++row) {
            if (!keep[row]) continue;
            if (!array.IsValid(i)) {
                keep[row] = 0;
                continue;
            }
            std::string value;
            switch (array.type_id()) {
                case arrow::Type::STRING:
                    value = static_cast<const arrow::StringArray&>(array).GetString(i);
                    break;
                case arrow::Type::LARGE_STRING:
                    value = static_cast<const arrow::LargeStringArray&>(array).GetString(i);
                    break;
                case arrow::Type::BINARY:
                    value = static_cast<const arrow::BinaryArray&>(array).GetString(i);
                    break;
                case arrow::Type::LARGE_BINARY:
                    value = static_cast<const arrow::LargeBinaryArray&>(array).GetString(i);
                    break;
                default: throw IOError(path + ": predicate column is not text");
            }
            if (!match_text(value, pred.op, pred.text)) keep[row] = 0;
        }
    }
}

void apply_predicate(const arrow::ChunkedArray& column, const ParquetPredicate& pred,
                     std::uint8_t* keep, const std::string& path) {
    const arrow::DataType* type = column.type().get();
    if (type->id() == arrow::Type::DICTIONARY) {
        type = static_cast<const arrow::DictionaryType&>(*type).value_type().get();
    }
    const bool text = type->id() == arrow::Type::STRING || type->id() == arrow::Type::LARGE_STRING ||
                      type->id() == arrow::Type::BINARY || type->id() == arrow::Type::LARGE_BINARY;
    if (pred.is_string != text) {
        throw InvalidArgumentError(path + ": predicate on \"" + pred.column + "\" does not match the column type");
    }
    if (text) {
        mask_text(column, pred, keep, path);
        return;
    }
    switch (type->id()) {
        case arrow::Type::BOOL: mask_bool(column, pred, keep); break;
        case arrow::Type::INT8: mask_numeric<arrow::Int8Type>(column, pred, keep); break;
        case arrow::Type::INT16: mask_numeric<arrow::Int16Type>(column, pred, keep); break;
        case arrow::Type::INT32: mask_numeric<arrow::Int32Type>(column, pred, keep); break;
        case arrow::Type::INT64: mask_numeric<arrow::Int64Type>(column, pred, keep); break;
        case arrow::Type::UINT8: mask_numeric<arrow::UInt8Type>(column, pred, keep); break;
        case arrow::Type::UINT16: mask_numeric<arrow::UInt16Type>(column, pred, keep); break;
        case arrow::Type::UINT32: mask_numeric<arrow::UInt32Type>(column, pred, keep); break;
        case arrow::Type::UINT64: mask_numeric<arrow::UInt64Type>(column, pred, keep); break;
        case arrow::Type::FLOAT: mask_float(column, pred, keep, false); break;
        case arrow::Type::DOUBLE: mask_float(column, pred, keep, true); break;
        case arrow::Type::DATE32: mask_numeric<arrow::Date32Type>(column, pred, keep); break;
        case arrow::Type::DATE64: mask_numeric<arrow::Date64Type>(column, pred, keep); break;
        case arrow::Type::TIMESTAMP: mask_numeric<arrow::TimestampType>(column, pred, keep); break;
        default:
            throw InvalidArgumentError(path + ": predicate column \"" + pred.column +
                                       "\" must be a scalar numeric or string");
    }
}

#endif

} // namespace

struct ParquetDataset::Impl {
#ifdef NEXUSDATA_WITH_ARROW
    arrow_detail::TableColumns table;
    std::size_t groups_total = 0;
    std::size_t groups_read = 0;
    std::size_t groups_skipped = 0;
#endif
};

ParquetDataset::~ParquetDataset() = default;
ParquetDataset::ParquetDataset(ParquetDataset&&) noexcept = default;
ParquetDataset& ParquetDataset::operator=(ParquetDataset&&) noexcept = default;

bool parquet_support_enabled() {
#if defined(NEXUSDATA_WITH_ARROW)
    return true;
#else
    return false;
#endif
}

ParquetDataset::ParquetDataset(const std::string& path, ParquetOptions opt) {
#if !defined(NEXUSDATA_WITH_ARROW)
    (void)path;
    (void)opt;
    throw InvalidArgumentError(
        "ParquetDataset: built without Arrow. Reconfigure with -DNEXUSDATA_WITH_ARROW=ON");
#else
    auto input = take_arrow(arrow::io::ReadableFile::Open(path), path);
    auto reader = take_arrow(parquet::arrow::OpenFile(input, arrow::default_memory_pool()), path);
    reader->set_use_threads(opt.use_threads);

    const auto metadata = reader->parquet_reader()->metadata();
    const auto& fields = reader->manifest().schema_fields;
    std::vector<std::string> names = arrow_detail::split_names(opt.columns);
    const bool project_all = names.empty();
    if (project_all) {
        for (const auto& field : fields) {
            if (field.field) names.push_back(field.field->name());
        }
    }
    std::vector<int> leaves;
    if (!project_all) {
        for (const auto& name : names) collect_leaves(*find_field(fields, name, path), leaves);
    }
    for (const auto& pred : opt.predicates) {
        const auto* field = find_field(fields, pred.column, path);
        if (field->column_index < 0) {
            throw InvalidArgumentError(path + ": predicate column \"" + pred.column +
                                       "\" must be a scalar column");
        }
        if (!project_all) leaves.push_back(field->column_index);
    }
    std::sort(leaves.begin(), leaves.end());
    leaves.erase(std::unique(leaves.begin(), leaves.end()), leaves.end());

    auto impl = std::make_unique<Impl>();
    impl->groups_total = static_cast<std::size_t>(metadata->num_row_groups());
    std::vector<int> row_groups;
    row_groups.reserve(impl->groups_total);
    for (int group = 0; group < metadata->num_row_groups(); ++group) {
        bool skip = false;
        if (opt.predicate_pushdown && !opt.predicates.empty()) {
            const auto row_group = metadata->RowGroup(group);
            for (const auto& pred : opt.predicates) {
                const int leaf = find_field(fields, pred.column, path)->column_index;
                const auto chunk = row_group->ColumnChunk(leaf);
                const auto stats = chunk->statistics();
                if (stats && stats_reject(*stats, pred, row_group->num_rows())) {
                    skip = true;
                    break;
                }
            }
        }
        if (skip) ++impl->groups_skipped;
        else row_groups.push_back(group);
    }
    impl->groups_read = row_groups.size();

    if (row_groups.empty()) {
        impl->table.names = names;
        impl->table.columns.reserve(names.size());
        for (const auto& name : names) {
            arrow_detail::Column column;
            column.name = name;
            impl->table.columns.push_back(std::move(column));
        }
        impl_ = std::move(impl);
        return;
    }

    std::shared_ptr<arrow::Table> table;
    if (project_all) {
        table = take_arrow(reader->ReadRowGroups(row_groups), path);
    } else {
        table = take_arrow(reader->ReadRowGroups(row_groups, leaves), path);
    }
    std::vector<std::uint8_t> keep(static_cast<std::size_t>(table->num_rows()), 1);
    for (const auto& pred : opt.predicates) {
        int index = -1;
        for (int i = 0; i < table->num_columns(); ++i) {
            if (table->field(i)->name() == pred.column) {
                index = i;
                break;
            }
        }
        if (index < 0) {
            throw IOError(path + ": predicate column \"" + pred.column + "\" was not read");
        }
        apply_predicate(*table->column(index), pred, keep.data(), path);
    }
    arrow_detail::ColumnPlan plan;
    plan.names = std::move(names);
    plan.label_name = opt.label;
    plan.label_index = opt.label_column;
    plan.metadata = opt.metadata_columns;
    plan.extra = opt.extra_columns;
    const std::uint8_t* mask = opt.predicates.empty() ? nullptr : keep.data();
    impl->table = arrow_detail::materialize(*table, mask, plan, false, path);
    impl_ = std::move(impl);
#endif
}

std::size_t ParquetDataset::size() const {
#if defined(NEXUSDATA_WITH_ARROW)
    return impl_ ? impl_->table.rows : 0;
#else
    return 0;
#endif
}

Sample ParquetDataset::get(std::size_t index) const {
#if !defined(NEXUSDATA_WITH_ARROW)
    (void)index;
    throw IndexError("ParquetDataset::get: not available");
#else
    if (!impl_ || index >= impl_->table.rows) {
        throw IndexError("ParquetDataset::get: index out of range");
    }
    return impl_->table.get(index);
#endif
}

const std::vector<std::string>& ParquetDataset::column_names() const {
#if !defined(NEXUSDATA_WITH_ARROW)
    throw InvalidArgumentError(
        "ParquetDataset: built without Arrow. Reconfigure with -DNEXUSDATA_WITH_ARROW=ON");
#else
    if (!impl_) throw InvalidArgumentError("ParquetDataset: empty");
    return impl_->table.names;
#endif
}

std::size_t ParquetDataset::row_groups_total() const noexcept {
#if defined(NEXUSDATA_WITH_ARROW)
    return impl_ ? impl_->groups_total : 0;
#else
    return 0;
#endif
}

std::size_t ParquetDataset::row_groups_read() const noexcept {
#if defined(NEXUSDATA_WITH_ARROW)
    return impl_ ? impl_->groups_read : 0;
#else
    return 0;
#endif
}

std::size_t ParquetDataset::row_groups_skipped() const noexcept {
#if defined(NEXUSDATA_WITH_ARROW)
    return impl_ ? impl_->groups_skipped : 0;
#else
    return 0;
#endif
}

} // namespace nexusdata
