#include "arrow_columns.hpp"

#ifdef NEXUSDATA_WITH_ARROW

#ifndef NOMINMAX
#define NOMINMAX
#endif

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif
#include <arrow/api.h>
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>

namespace nexusdata::arrow_detail {
namespace {

bool listed(const std::vector<std::string>& names, const std::string& name) {
    for (const auto& n : names) {
        if (n == name) return true;
    }
    return false;
}

[[nodiscard]] std::optional<DType> numeric_dtype(const arrow::DataType& type) {
    switch (type.id()) {
        case arrow::Type::BOOL: return DType::Bool;
        case arrow::Type::INT8: return DType::Int8;
        case arrow::Type::INT16: return DType::Int16;
        case arrow::Type::INT32: return DType::Int32;
        case arrow::Type::INT64: return DType::Int64;
        case arrow::Type::UINT8: return DType::UInt8;
        case arrow::Type::UINT16: return DType::UInt16;
        case arrow::Type::UINT32: return DType::UInt32;
        case arrow::Type::UINT64: return DType::UInt64;
        case arrow::Type::HALF_FLOAT: return DType::Float16;
        case arrow::Type::FLOAT: return DType::Float32;
        case arrow::Type::DOUBLE: return DType::Float64;
        case arrow::Type::DATE32: return DType::Int32;
        case arrow::Type::DATE64: return DType::Int64;
        case arrow::Type::TIME32: return DType::Int32;
        case arrow::Type::TIME64: return DType::Int64;
        case arrow::Type::TIMESTAMP: return DType::Int64;
        case arrow::Type::DURATION: return DType::Int64;
        default: return std::nullopt;
    }
}

bool is_stringish(const arrow::DataType& type) {
    switch (type.id()) {
        case arrow::Type::STRING:
        case arrow::Type::LARGE_STRING:
        case arrow::Type::BINARY:
        case arrow::Type::LARGE_BINARY: return true;
        default: return false;
    }
}

bool nulls_are_nan(DType dtype) {
    return dtype == DType::Float32 || dtype == DType::Float64;
}

[[noreturn]] void unsupported(const std::string& where, const std::string& name,
                              const arrow::DataType& type) {
    throw IOError(where + ": column \"" + name + "\" has unsupported type " + type.ToString());
}

int field_index(const arrow::Table& table, const std::string& name, const std::string& where) {
    int found = -1;
    for (int i = 0; i < table.num_columns(); ++i) {
        if (table.field(i)->name() == name) {
            if (found >= 0) {
                throw InvalidArgumentError(where + ": duplicate column \"" + name + "\"");
            }
            found = i;
        }
    }
    if (found < 0) {
        throw InvalidArgumentError(where + ": column \"" + name + "\" not found");
    }
    return found;
}

std::size_t count_kept(const std::uint8_t* keep, int64_t n) {
    if (!keep) return static_cast<std::size_t>(n);
    std::size_t c = 0;
    for (int64_t i = 0; i < n; ++i) c += keep[i] ? 1 : 0;
    return c;
}

const std::uint8_t* row_bytes(const Column& col, std::size_t index) {
    return static_cast<const std::uint8_t*>(col.data.data()) +
           index * col.row_numel * size_of(col.dtype);
}

NDArray copy_row(const Column& col, std::size_t index) {
    NDArray out(col.sample_shape, col.dtype);
    const std::size_t bytes = col.row_numel * size_of(col.dtype);
    if (bytes > 0) {
        std::memcpy(out.data(), row_bytes(col, index), bytes);
    }
    return out;
}

std::string string_at(const arrow::Array& array, int64_t i) {
    switch (array.type_id()) {
        case arrow::Type::STRING:
            return static_cast<const arrow::StringArray&>(array).GetString(i);
        case arrow::Type::LARGE_STRING:
            return static_cast<const arrow::LargeStringArray&>(array).GetString(i);
        case arrow::Type::BINARY:
            return static_cast<const arrow::BinaryArray&>(array).GetString(i);
        case arrow::Type::LARGE_BINARY:
            return static_cast<const arrow::LargeBinaryArray&>(array).GetString(i);
        default: throw IOError("internal: not a string array");
    }
}

std::optional<int64_t> dict_index(const arrow::Array& indices, int64_t i) {
    if (!indices.IsValid(i)) return std::nullopt;
    switch (indices.type_id()) {
        case arrow::Type::INT8: return static_cast<const arrow::Int8Array&>(indices).Value(i);
        case arrow::Type::INT16: return static_cast<const arrow::Int16Array&>(indices).Value(i);
        case arrow::Type::INT32: return static_cast<const arrow::Int32Array&>(indices).Value(i);
        case arrow::Type::INT64: return static_cast<const arrow::Int64Array&>(indices).Value(i);
        case arrow::Type::UINT8: return static_cast<const arrow::UInt8Array&>(indices).Value(i);
        case arrow::Type::UINT16: return static_cast<const arrow::UInt16Array&>(indices).Value(i);
        case arrow::Type::UINT32:
            return static_cast<int64_t>(static_cast<const arrow::UInt32Array&>(indices).Value(i));
        case arrow::Type::UINT64: {
            const auto v = static_cast<const arrow::UInt64Array&>(indices).Value(i);
            if (v > static_cast<std::uint64_t>(std::numeric_limits<int64_t>::max())) {
                throw IOError("dictionary index does not fit in int64");
            }
            return static_cast<int64_t>(v);
        }
        default: throw IOError("unsupported dictionary index type " + indices.type()->ToString());
    }
}

template <typename ArrowType>
typename ArrowType::c_type dict_value(const arrow::Array& dict, int64_t index) {
    const auto& values = static_cast<const arrow::NumericArray<ArrowType>&>(dict);
    if (index < 0 || index >= values.length() || !values.IsValid(index)) {
        throw IOError("dictionary index out of range");
    }
    return values.Value(index);
}

void null_value(DType dtype, const std::string& where, const std::string& name, void* dst) {
    if (dtype == DType::Float32) {
        *static_cast<float*>(dst) = std::numeric_limits<float>::quiet_NaN();
        return;
    }
    if (dtype == DType::Float64) {
        *static_cast<double*>(dst) = std::numeric_limits<double>::quiet_NaN();
        return;
    }
    throw IOError(where + ": column \"" + name + "\" has a null in a non-floating column");
}

template <typename ArrowType>
void consume_numeric_chunk(const arrow::Array& array, const std::uint8_t* keep, std::size_t& row,
                           typename ArrowType::c_type* dst, std::size_t& written, DType dtype,
                           const std::string& where, const std::string& name) {
    using T = typename ArrowType::c_type;
    const auto& a = static_cast<const arrow::NumericArray<ArrowType>&>(array);
    const T* src = a.raw_values();
    bool span = a.null_count() == 0;
    if (span && keep) {
        for (std::size_t r = row; r < row + static_cast<std::size_t>(a.length()); ++r) {
            if (!keep[r]) {
                span = false;
                break;
            }
        }
    }
    if (span) {
        if (a.length() > 0) {
            std::memcpy(dst + written, src, static_cast<std::size_t>(a.length()) * sizeof(T));
        }
        written += static_cast<std::size_t>(a.length());
        row += static_cast<std::size_t>(a.length());
        return;
    }
    for (int64_t i = 0; i < a.length(); ++i, ++row) {
        if (keep && !keep[row]) continue;
        if (!a.IsValid(i)) {
            null_value(dtype, where, name, dst + written);
        } else {
            dst[written] = src[i];
        }
        ++written;
    }
}

void consume_bool_chunk(const arrow::Array& array, const std::uint8_t* keep, std::size_t& row,
                        std::uint8_t* dst, std::size_t& written, const std::string& where,
                        const std::string& name) {
    const auto& a = static_cast<const arrow::BooleanArray&>(array);
    for (int64_t i = 0; i < a.length(); ++i, ++row) {
        if (keep && !keep[row]) continue;
        if (!a.IsValid(i)) {
            throw IOError(where + ": column \"" + name + "\" has a null in a non-floating column");
        }
        dst[written++] = a.Value(i) ? 1 : 0;
    }
}

template <typename ArrowType>
void consume_dict_numeric(const arrow::DictionaryArray& array, const std::uint8_t* keep,
                          std::size_t& row, typename ArrowType::c_type* dst, std::size_t& written,
                          DType dtype, const std::string& where, const std::string& name) {
    using T = typename ArrowType::c_type;
    const arrow::Array& indices = *array.indices();
    const arrow::Array& dict = *array.dictionary();
    for (int64_t i = 0; i < array.length(); ++i, ++row) {
        if (keep && !keep[row]) continue;
        const auto ix = dict_index(indices, i);
        if (!ix) {
            null_value(dtype, where, name, dst + written);
        } else {
            dst[written] = dict_value<ArrowType>(dict, *ix);
        }
        ++written;
    }
}

void consume_strings(const arrow::Array& array, const std::uint8_t* keep, std::size_t& row,
                     std::vector<std::string>& out) {
    for (int64_t i = 0; i < array.length(); ++i, ++row) {
        if (keep && !keep[row]) continue;
        if (!array.IsValid(i)) out.emplace_back();
        else out.push_back(string_at(array, i));
    }
}

void consume_dict_strings(const arrow::DictionaryArray& array, const std::uint8_t* keep,
                          std::size_t& row, std::vector<std::string>& out) {
    const arrow::Array& indices = *array.indices();
    const arrow::Array& dict = *array.dictionary();
    if (!is_stringish(*dict.type())) {
        throw IOError("unsupported dictionary value type " + dict.type()->ToString());
    }
    for (int64_t i = 0; i < array.length(); ++i, ++row) {
        if (keep && !keep[row]) continue;
        const auto ix = dict_index(indices, i);
        if (!ix) out.emplace_back();
        else out.push_back(string_at(dict, *ix));
    }
}

template <typename ArrowType>
bool view_chunk(const arrow::Array& array, Column& col) {
    const auto& a = static_cast<const arrow::NumericArray<ArrowType>&>(array);
    if (a.length() == 0 || !a.raw_values()) return false;
    auto held = a.data();
    const void* p = a.raw_values();
    col.sample_shape = Shape{1};
    col.row_numel = 1;
    col.data = NDArray::from_shared(AlignedBuffer(const_cast<void*>(p), [held](void*) {}),
                                    Shape{static_cast<std::size_t>(a.length())}, col.dtype,
                                    Device::host());
    col.zero_copy = true;
    return true;
}

bool try_view_primitive(const arrow::ChunkedArray& chunks, Column& col) {
    if (chunks.num_chunks() != 1) return false;
    const arrow::Array& array = *chunks.chunk(0);
    if (array.null_count() != 0 || array.length() == 0) return false;
    switch (array.type_id()) {
        case arrow::Type::INT8: return view_chunk<arrow::Int8Type>(array, col);
        case arrow::Type::INT16: return view_chunk<arrow::Int16Type>(array, col);
        case arrow::Type::INT32: return view_chunk<arrow::Int32Type>(array, col);
        case arrow::Type::INT64: return view_chunk<arrow::Int64Type>(array, col);
        case arrow::Type::UINT8: return view_chunk<arrow::UInt8Type>(array, col);
        case arrow::Type::UINT16: return view_chunk<arrow::UInt16Type>(array, col);
        case arrow::Type::UINT32: return view_chunk<arrow::UInt32Type>(array, col);
        case arrow::Type::UINT64: return view_chunk<arrow::UInt64Type>(array, col);
        case arrow::Type::HALF_FLOAT: return view_chunk<arrow::HalfFloatType>(array, col);
        case arrow::Type::FLOAT: return view_chunk<arrow::FloatType>(array, col);
        case arrow::Type::DOUBLE: return view_chunk<arrow::DoubleType>(array, col);
        case arrow::Type::DATE32: return view_chunk<arrow::Date32Type>(array, col);
        case arrow::Type::DATE64: return view_chunk<arrow::Date64Type>(array, col);
        case arrow::Type::TIME32: return view_chunk<arrow::Time32Type>(array, col);
        case arrow::Type::TIME64: return view_chunk<arrow::Time64Type>(array, col);
        case arrow::Type::TIMESTAMP: return view_chunk<arrow::TimestampType>(array, col);
        case arrow::Type::DURATION: return view_chunk<arrow::DurationType>(array, col);
        default: return false;
    }
}

bool try_view_fixed_list(const arrow::ChunkedArray& chunks, Column& col) {
    if (chunks.num_chunks() != 1) return false;
    const arrow::Array& array = *chunks.chunk(0);
    if (array.type_id() != arrow::Type::FIXED_SIZE_LIST || array.null_count() != 0 ||
        array.length() == 0 || array.offset() != 0) {
        return false;
    }
    const auto& list = static_cast<const arrow::FixedSizeListArray&>(array);
    const arrow::Array& child = *list.values();
    if (child.null_count() != 0 || child.offset() != 0) return false;
    const auto dt = numeric_dtype(*child.type());
    if (!dt || *dt == DType::Bool) return false;
    const int32_t width = list.value_length();
    if (width < 0 || child.length() != list.length() * static_cast<int64_t>(width)) return false;
    col.dtype = *dt;
    col.sample_shape = Shape{static_cast<std::size_t>(width)};
    col.row_numel = static_cast<std::size_t>(width);
    // Reuse the primitive view helper by pretending the child is the column, then
    // reshape. view_chunk sets a flat [n] shape, so build the shared buffer here.
    const void* p = nullptr;
    std::shared_ptr<arrow::ArrayData> held = child.data();
    switch (child.type_id()) {
        case arrow::Type::FLOAT:
            p = static_cast<const arrow::FloatArray&>(child).raw_values();
            break;
        case arrow::Type::DOUBLE:
            p = static_cast<const arrow::DoubleArray&>(child).raw_values();
            break;
        case arrow::Type::INT32:
            p = static_cast<const arrow::Int32Array&>(child).raw_values();
            break;
        case arrow::Type::INT64:
            p = static_cast<const arrow::Int64Array&>(child).raw_values();
            break;
        case arrow::Type::INT8:
            p = static_cast<const arrow::Int8Array&>(child).raw_values();
            break;
        case arrow::Type::INT16:
            p = static_cast<const arrow::Int16Array&>(child).raw_values();
            break;
        case arrow::Type::UINT8:
            p = static_cast<const arrow::UInt8Array&>(child).raw_values();
            break;
        default: return false;
    }
    if (!p) return false;
    col.data = NDArray::from_shared(
        AlignedBuffer(const_cast<void*>(p), [held](void*) {}),
        Shape{static_cast<std::size_t>(list.length()), static_cast<std::size_t>(width)}, col.dtype,
        Device::host());
    col.zero_copy = true;
    return true;
}

template <typename ArrowType>
void fill_numeric(const arrow::ChunkedArray& chunks, const std::uint8_t* keep, Column& col,
                  const std::string& where) {
    using T = typename ArrowType::c_type;
    auto* dst = reinterpret_cast<T*>(col.data.data<std::uint8_t>());
    std::size_t row = 0;
    std::size_t written = 0;
    for (int c = 0; c < chunks.num_chunks(); ++c) {
        const arrow::Array& chunk = *chunks.chunk(c);
        if (chunk.type_id() == arrow::Type::DICTIONARY) {
            consume_dict_numeric<ArrowType>(static_cast<const arrow::DictionaryArray&>(chunk), keep,
                                            row, dst, written, col.dtype, where, col.name);
        } else {
            consume_numeric_chunk<ArrowType>(chunk, keep, row, dst, written, col.dtype, where,
                                             col.name);
        }
    }
    if (written != static_cast<std::size_t>(col.data.shape()[0])) {
        throw IOError(where + ": internal row count mismatch in \"" + col.name + "\"");
    }
}

void fill_bool(const arrow::ChunkedArray& chunks, const std::uint8_t* keep, Column& col,
               const std::string& where) {
    auto* dst = col.data.data<std::uint8_t>();
    std::size_t row = 0;
    std::size_t written = 0;
    for (int c = 0; c < chunks.num_chunks(); ++c) {
        consume_bool_chunk(*chunks.chunk(c), keep, row, dst, written, where, col.name);
    }
    if (written != col.data.shape()[0]) {
        throw IOError(where + ": internal row count mismatch in \"" + col.name + "\"");
    }
}

void load_numeric(const arrow::ChunkedArray& chunks, const std::uint8_t* keep, Column& col,
                  std::size_t kept, bool zero_copy, const std::string& where) {
    const arrow::DataType* type = chunks.type().get();
    if (type->id() == arrow::Type::DICTIONARY) {
        type = static_cast<const arrow::DictionaryType&>(*type).value_type().get();
        zero_copy = false;
    }
    const auto dt = numeric_dtype(*type);
    if (!dt) unsupported(where, col.name, *chunks.type());
    col.dtype = *dt;
    col.sample_shape = Shape{1};
    col.row_numel = 1;
    if (zero_copy && try_view_primitive(chunks, col)) return;
    if (kept == 0) {
        col.data = NDArray(Shape{0}, col.dtype);
        return;
    }
    col.data = NDArray(Shape{kept}, col.dtype);
    switch (type->id()) {
        case arrow::Type::BOOL: fill_bool(chunks, keep, col, where); break;
        case arrow::Type::INT8: fill_numeric<arrow::Int8Type>(chunks, keep, col, where); break;
        case arrow::Type::INT16: fill_numeric<arrow::Int16Type>(chunks, keep, col, where); break;
        case arrow::Type::INT32: fill_numeric<arrow::Int32Type>(chunks, keep, col, where); break;
        case arrow::Type::INT64: fill_numeric<arrow::Int64Type>(chunks, keep, col, where); break;
        case arrow::Type::UINT8: fill_numeric<arrow::UInt8Type>(chunks, keep, col, where); break;
        case arrow::Type::UINT16: fill_numeric<arrow::UInt16Type>(chunks, keep, col, where); break;
        case arrow::Type::UINT32: fill_numeric<arrow::UInt32Type>(chunks, keep, col, where); break;
        case arrow::Type::UINT64: fill_numeric<arrow::UInt64Type>(chunks, keep, col, where); break;
        case arrow::Type::HALF_FLOAT:
            fill_numeric<arrow::HalfFloatType>(chunks, keep, col, where);
            break;
        case arrow::Type::FLOAT: fill_numeric<arrow::FloatType>(chunks, keep, col, where); break;
        case arrow::Type::DOUBLE: fill_numeric<arrow::DoubleType>(chunks, keep, col, where); break;
        case arrow::Type::DATE32: fill_numeric<arrow::Date32Type>(chunks, keep, col, where); break;
        case arrow::Type::DATE64: fill_numeric<arrow::Date64Type>(chunks, keep, col, where); break;
        case arrow::Type::TIME32: fill_numeric<arrow::Time32Type>(chunks, keep, col, where); break;
        case arrow::Type::TIME64: fill_numeric<arrow::Time64Type>(chunks, keep, col, where); break;
        case arrow::Type::TIMESTAMP:
            fill_numeric<arrow::TimestampType>(chunks, keep, col, where);
            break;
        case arrow::Type::DURATION:
            fill_numeric<arrow::DurationType>(chunks, keep, col, where);
            break;
        default: unsupported(where, col.name, *type);
    }
}

void load_strings(const arrow::ChunkedArray& chunks, const std::uint8_t* keep, Column& col,
                  std::size_t kept) {
    col.kind = Column::Kind::Strings;
    col.strings.reserve(kept);
    std::size_t row = 0;
    const bool dict = chunks.type()->id() == arrow::Type::DICTIONARY;
    for (int c = 0; c < chunks.num_chunks(); ++c) {
        const arrow::Array& chunk = *chunks.chunk(c);
        if (dict || chunk.type_id() == arrow::Type::DICTIONARY) {
            consume_dict_strings(static_cast<const arrow::DictionaryArray&>(chunk), keep, row,
                                 col.strings);
        } else {
            consume_strings(chunk, keep, row, col.strings);
        }
    }
    if (col.strings.size() != kept) {
        throw IOError("internal row count mismatch in \"" + col.name + "\"");
    }
}

template <typename ListArray, typename ArrowType>
const typename ArrowType::c_type* list_base(const ListArray& list) {
    return static_cast<const arrow::NumericArray<ArrowType>&>(*list.values()).raw_values();
}

template <typename ListArray, typename ArrowType>
void copy_one_list(const ListArray& list, int64_t i, typename ArrowType::c_type* dst, int64_t len,
                   DType dtype, const std::string& where, const std::string& name) {
    using T = typename ArrowType::c_type;
    if (!list.IsValid(i)) {
        if constexpr (std::is_floating_point_v<T>) {
            if (!nulls_are_nan(dtype)) {
                throw IOError(where + ": column \"" + name + "\" has a null in a non-floating column");
            }
            const T nan = std::numeric_limits<T>::quiet_NaN();
            for (int64_t j = 0; j < len; ++j) dst[j] = nan;
            return;
        } else {
            throw IOError(where + ": column \"" + name + "\" has a null in a non-floating column");
        }
    }
    const auto off = list.value_offset(i);
    const T* src = list_base<ListArray, ArrowType>(list) + off;
    const auto& child = static_cast<const arrow::NumericArray<ArrowType>&>(*list.values());
    if (child.null_count() == 0) {
        std::memcpy(dst, src, static_cast<std::size_t>(len) * sizeof(T));
        return;
    }
    for (int64_t j = 0; j < len; ++j) {
        if (!child.IsValid(off + j)) null_value(dtype, where, name, dst + j);
        else dst[j] = src[j];
    }
}

template <typename ListArray, typename ArrowType>
void load_list_typed(const arrow::ChunkedArray& chunks, const std::uint8_t* keep, Column& col,
                     std::size_t kept, bool fixed, int32_t fixed_width, const std::string& where) {
    using T = typename ArrowType::c_type;
    bool ragged = false;
    int64_t common = fixed ? fixed_width : -1;
    if (!fixed) {
        std::size_t row = 0;
        for (int c = 0; c < chunks.num_chunks(); ++c) {
            const auto& list = static_cast<const ListArray&>(*chunks.chunk(c));
            for (int64_t i = 0; i < list.length(); ++i, ++row) {
                if (keep && !keep[row]) continue;
                const int64_t len = list.IsValid(i) ? list.value_length(i) : 0;
                if (common < 0) common = len;
                else if (common != len) ragged = true;
            }
        }
        if (common < 0) common = 0;
    }
    if (ragged) {
        col.kind = Column::Kind::Ragged;
        col.ragged.reserve(kept);
        std::size_t row = 0;
        for (int c = 0; c < chunks.num_chunks(); ++c) {
            const auto& list = static_cast<const ListArray&>(*chunks.chunk(c));
            for (int64_t i = 0; i < list.length(); ++i, ++row) {
                if (keep && !keep[row]) continue;
                const int64_t len = list.IsValid(i) ? list.value_length(i) : 0;
                NDArray row_arr(Shape{static_cast<std::size_t>(len)}, col.dtype);
                if (len > 0) {
                    copy_one_list<ListArray, ArrowType>(list, i, row_arr.data<T>(), len, col.dtype,
                                                        where, col.name);
                }
                col.ragged.push_back(std::move(row_arr));
            }
        }
        return;
    }
    col.kind = Column::Kind::Numeric;
    col.sample_shape = Shape{static_cast<std::size_t>(common)};
    col.row_numel = static_cast<std::size_t>(common);
    if (kept == 0 || common == 0) {
        col.data = NDArray(Shape{kept, static_cast<std::size_t>(common)}, col.dtype);
        return;
    }
    col.data = NDArray(Shape{kept, static_cast<std::size_t>(common)}, col.dtype);
    auto* dst = reinterpret_cast<T*>(col.data.data<std::uint8_t>());
    std::size_t row = 0;
    std::size_t written = 0;
    for (int c = 0; c < chunks.num_chunks(); ++c) {
        const auto& list = static_cast<const ListArray&>(*chunks.chunk(c));
        for (int64_t i = 0; i < list.length(); ++i, ++row) {
            if (keep && !keep[row]) continue;
            copy_one_list<ListArray, ArrowType>(list, i, dst + written * static_cast<std::size_t>(common),
                                                common, col.dtype, where, col.name);
            ++written;
        }
    }
}

template <typename ListArray>
void load_list_as(const arrow::ChunkedArray& chunks, const std::uint8_t* keep, Column& col,
                  std::size_t kept, bool fixed, int32_t fixed_width, const arrow::DataType& values,
                  const std::string& where) {
    switch (values.id()) {
        case arrow::Type::INT8:
            load_list_typed<ListArray, arrow::Int8Type>(chunks, keep, col, kept, fixed, fixed_width,
                                                        where);
            break;
        case arrow::Type::INT16:
            load_list_typed<ListArray, arrow::Int16Type>(chunks, keep, col, kept, fixed, fixed_width,
                                                         where);
            break;
        case arrow::Type::INT32:
            load_list_typed<ListArray, arrow::Int32Type>(chunks, keep, col, kept, fixed, fixed_width,
                                                         where);
            break;
        case arrow::Type::INT64:
            load_list_typed<ListArray, arrow::Int64Type>(chunks, keep, col, kept, fixed, fixed_width,
                                                         where);
            break;
        case arrow::Type::UINT8:
            load_list_typed<ListArray, arrow::UInt8Type>(chunks, keep, col, kept, fixed, fixed_width,
                                                         where);
            break;
        case arrow::Type::UINT16:
            load_list_typed<ListArray, arrow::UInt16Type>(chunks, keep, col, kept, fixed, fixed_width,
                                                          where);
            break;
        case arrow::Type::UINT32:
            load_list_typed<ListArray, arrow::UInt32Type>(chunks, keep, col, kept, fixed, fixed_width,
                                                          where);
            break;
        case arrow::Type::UINT64:
            load_list_typed<ListArray, arrow::UInt64Type>(chunks, keep, col, kept, fixed, fixed_width,
                                                          where);
            break;
        case arrow::Type::FLOAT:
            load_list_typed<ListArray, arrow::FloatType>(chunks, keep, col, kept, fixed, fixed_width,
                                                         where);
            break;
        case arrow::Type::DOUBLE:
            load_list_typed<ListArray, arrow::DoubleType>(chunks, keep, col, kept, fixed, fixed_width,
                                                          where);
            break;
        default: unsupported(where, col.name, *chunks.type());
    }
}

void load_list(const arrow::ChunkedArray& chunks, const std::uint8_t* keep, Column& col,
               std::size_t kept, bool zero_copy, const std::string& where) {
    const arrow::DataType& type = *chunks.type();
    const arrow::DataType* values = nullptr;
    bool fixed = false;
    int32_t fixed_width = 0;
    if (type.id() == arrow::Type::LIST) {
        values = static_cast<const arrow::ListType&>(type).value_type().get();
    } else if (type.id() == arrow::Type::LARGE_LIST) {
        values = static_cast<const arrow::LargeListType&>(type).value_type().get();
    } else if (type.id() == arrow::Type::FIXED_SIZE_LIST) {
        const auto& ft = static_cast<const arrow::FixedSizeListType&>(type);
        values = ft.value_type().get();
        fixed = true;
        fixed_width = ft.list_size();
    } else {
        unsupported(where, col.name, type);
    }
    const auto dt = numeric_dtype(*values);
    if (!dt || *dt == DType::Bool) unsupported(where, col.name, type);
    col.dtype = *dt;
    if (zero_copy && fixed && try_view_fixed_list(chunks, col)) return;
    if (type.id() == arrow::Type::LIST) {
        load_list_as<arrow::ListArray>(chunks, keep, col, kept, fixed, fixed_width, *values, where);
    } else if (type.id() == arrow::Type::LARGE_LIST) {
        load_list_as<arrow::LargeListArray>(chunks, keep, col, kept, fixed, fixed_width, *values,
                                            where);
    } else {
        load_list_as<arrow::FixedSizeListArray>(chunks, keep, col, kept, fixed, fixed_width, *values,
                                                where);
    }
}

std::string format_scalar(const Column& col, std::size_t index) {
    const std::uint8_t* p = row_bytes(col, index);
    char buf[64];
    auto finish = [&](std::to_chars_result r) {
        if (r.ec != std::errc()) return std::string("0");
        return std::string(buf, r.ptr);
    };
    switch (col.dtype) {
        case DType::Bool: return p[0] ? "1" : "0";
        case DType::Int8: return finish(std::to_chars(buf, buf + sizeof(buf), *reinterpret_cast<const std::int8_t*>(p)));
        case DType::Int16:
            return finish(std::to_chars(buf, buf + sizeof(buf), *reinterpret_cast<const std::int16_t*>(p)));
        case DType::Int32:
            return finish(std::to_chars(buf, buf + sizeof(buf), *reinterpret_cast<const std::int32_t*>(p)));
        case DType::Int64:
            return finish(std::to_chars(buf, buf + sizeof(buf), *reinterpret_cast<const std::int64_t*>(p)));
        case DType::UInt8:
            return finish(std::to_chars(buf, buf + sizeof(buf), *reinterpret_cast<const std::uint8_t*>(p)));
        case DType::UInt16:
            return finish(std::to_chars(buf, buf + sizeof(buf), *reinterpret_cast<const std::uint16_t*>(p)));
        case DType::UInt32:
            return finish(std::to_chars(buf, buf + sizeof(buf), *reinterpret_cast<const std::uint32_t*>(p)));
        case DType::UInt64:
            return finish(std::to_chars(buf, buf + sizeof(buf), *reinterpret_cast<const std::uint64_t*>(p)));
        case DType::Float32: {
            float v;
            std::memcpy(&v, p, sizeof(v));
            auto r = std::to_chars(buf, buf + sizeof(buf), v, std::chars_format::general);
            if (r.ec != std::errc()) {
                std::snprintf(buf, sizeof(buf), "%.9g", static_cast<double>(v));
                return buf;
            }
            return std::string(buf, r.ptr);
        }
        case DType::Float64: {
            double v;
            std::memcpy(&v, p, sizeof(v));
            auto r = std::to_chars(buf, buf + sizeof(buf), v, std::chars_format::general);
            if (r.ec != std::errc()) {
                std::snprintf(buf, sizeof(buf), "%.17g", v);
                return buf;
            }
            return std::string(buf, r.ptr);
        }
        default: return {};
    }
}

void assign_roles(TableColumns& out, const ColumnPlan& plan, const std::string& where) {
    if (!plan.label_name.empty()) {
        bool found = false;
        for (const auto& c : out.columns) found = found || c.name == plan.label_name;
        if (!found) {
            throw InvalidArgumentError(where + ": label column \"" + plan.label_name +
                                       "\" is not in the projection");
        }
    } else if (plan.label_index >= static_cast<int>(out.columns.size())) {
        throw InvalidArgumentError(where + ": label_column is out of range");
    }
    for (const auto& name : plan.metadata) {
        if (!listed(out.names, name)) {
            throw InvalidArgumentError(where + ": metadata column \"" + name +
                                       "\" is not in the projection");
        }
    }
    for (const auto& name : plan.extra) {
        if (!listed(out.names, name)) {
            throw InvalidArgumentError(where + ": extra column \"" + name +
                                       "\" is not in the projection");
        }
    }
    std::optional<DType> input_dtype;
    for (std::size_t i = 0; i < out.columns.size(); ++i) {
        Column& col = out.columns[i];
        const bool is_label = !plan.label_name.empty() ? col.name == plan.label_name
                                                       : static_cast<int>(i) == plan.label_index &&
                                                             plan.label_index >= 0;
        if (col.kind == Column::Kind::Strings) {
            if (is_label) {
                throw InvalidArgumentError(where + ": label column \"" + col.name + "\" is not numeric");
            }
            if (listed(plan.extra, col.name)) {
                throw InvalidArgumentError(where + ": string column \"" + col.name +
                                           "\" is metadata, not an extra tensor");
            }
            col.role = Role::Metadata;
            continue;
        }
        if (col.kind == Column::Kind::Ragged) {
            if (!listed(plan.extra, col.name)) {
                throw InvalidArgumentError(where + ": column \"" + col.name +
                                           "\" has variable-length lists; add it to extra_columns");
            }
            col.role = Role::Extra;
            continue;
        }
        if (is_label) {
            col.role = Role::Label;
        } else if (listed(plan.metadata, col.name)) {
            if (col.row_numel != 1) {
                throw InvalidArgumentError(where + ": only scalar columns can be metadata (\"" +
                                           col.name + "\")");
            }
            col.strings.resize(out.rows);
            for (std::size_t r = 0; r < out.rows; ++r) col.strings[r] = format_scalar(col, r);
            col.kind = Column::Kind::Strings;
            col.data = NDArray();
            col.zero_copy = false;
            col.role = Role::Metadata;
        } else if (listed(plan.extra, col.name)) {
            col.role = Role::Extra;
        } else {
            if (input_dtype && *input_dtype != col.dtype) {
                throw InvalidArgumentError(where + ": input columns must share one dtype");
            }
            input_dtype = col.dtype;
            col.role = Role::Input;
        }
    }
}

} // namespace

std::vector<std::string> split_names(const std::string& csv) {
    std::vector<std::string> out;
    if (csv.empty()) return out;
    std::size_t i = 0;
    while (i <= csv.size()) {
        const std::size_t j = csv.find(',', i);
        const std::size_t end = j == std::string::npos ? csv.size() : j;
        const std::string raw = csv.substr(i, end - i);
        const auto first = raw.find_first_not_of(" \t");
        const auto last = raw.find_last_not_of(" \t");
        if (first == std::string::npos) {
            throw InvalidArgumentError("empty column name in column list");
        }
        out.push_back(raw.substr(first, last - first + 1));
        if (j == std::string::npos) break;
        i = j + 1;
    }
    return out;
}

TableColumns materialize(const arrow::Table& table, const std::uint8_t* keep, const ColumnPlan& plan,
                         bool zero_copy, const std::string& where) {
    std::vector<std::string> names = plan.names;
    if (names.empty()) {
        names.reserve(static_cast<std::size_t>(table.num_columns()));
        for (int i = 0; i < table.num_columns(); ++i) names.push_back(table.field(i)->name());
    }
    for (std::size_t a = 0; a < names.size(); ++a) {
        for (std::size_t b = a + 1; b < names.size(); ++b) {
            if (names[a] == names[b]) {
                throw InvalidArgumentError(where + ": column \"" + names[a] + "\" listed twice");
            }
        }
    }
    const std::size_t kept = count_kept(keep, table.num_rows());
    TableColumns out;
    out.rows = kept;
    out.names = names;
    out.columns.reserve(names.size());
    for (const auto& name : names) {
        const int idx = field_index(table, name, where);
        const arrow::ChunkedArray& chunks = *table.column(idx);
        Column col;
        col.name = name;
        const arrow::DataType& type = *chunks.type();
        const arrow::DataType& value =
            type.id() == arrow::Type::DICTIONARY
                ? *static_cast<const arrow::DictionaryType&>(type).value_type()
                : type;
        if (is_stringish(value)) load_strings(chunks, keep, col, kept);
        else if (value.id() == arrow::Type::LIST || value.id() == arrow::Type::LARGE_LIST ||
                 value.id() == arrow::Type::FIXED_SIZE_LIST) {
            if (type.id() == arrow::Type::DICTIONARY) unsupported(where, name, type);
            load_list(chunks, keep, col, kept, zero_copy && keep == nullptr, where);
        } else if (numeric_dtype(value)) {
            load_numeric(chunks, keep, col, kept, zero_copy && keep == nullptr, where);
        } else {
            unsupported(where, name, type);
        }
        out.columns.push_back(std::move(col));
    }
    ColumnPlan resolved = plan;
    resolved.names = names;
    assign_roles(out, resolved, where);
    return out;
}

const Column* TableColumns::find(std::string_view name) const {
    for (const auto& col : columns) {
        if (col.name == name) return &col;
    }
    return nullptr;
}

Sample TableColumns::get(std::size_t index) const {
    if (index >= rows) throw IndexError("row index out of range");
    Sample sample;
    std::vector<const Column*> inputs;
    const Column* label = nullptr;
    for (const auto& col : columns) {
        if (col.role == Role::Input) inputs.push_back(&col);
        else if (col.role == Role::Label) label = &col;
    }
    if (inputs.size() == 1) {
        sample.input = copy_row(*inputs[0], index);
    } else if (inputs.size() > 1) {
        std::size_t n = 0;
        for (const Column* col : inputs) n += col->row_numel;
        sample.input = NDArray(Shape{n}, inputs[0]->dtype);
        auto* dst = static_cast<std::uint8_t*>(sample.input.data());
        for (const Column* col : inputs) {
            const std::size_t bytes = col->row_numel * size_of(col->dtype);
            if (bytes > 0) std::memcpy(dst, row_bytes(*col, index), bytes);
            dst += bytes;
        }
    }
    if (label) {
        sample.label = copy_row(*label, index);
    } else {
        sample.label = NDArray(Shape{1}, DType::Int64);
        sample.label.data<std::int64_t>()[0] = static_cast<std::int64_t>(index);
    }
    for (const auto& col : columns) {
        if (col.role == Role::Extra) {
            if (col.kind == Column::Kind::Ragged) {
                sample.extra_tensors.set(col.name, col.ragged[index].clone());
            } else {
                sample.extra_tensors.set(col.name, copy_row(col, index));
            }
        } else if (col.role == Role::Metadata) {
            sample.metadata.set(col.name, col.strings[index]);
        }
    }
    return sample;
}

} // namespace nexusdata::arrow_detail

#endif
