#include "nexusdata/dataset/arrow_ipc.hpp"

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
#include <arrow/ipc/api.h>
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#include <string>
#include <vector>

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

std::shared_ptr<arrow::Table> read_ipc(const std::shared_ptr<arrow::io::RandomAccessFile>& file,
                                       const std::string& path) {
    std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
    auto reader = take_arrow(arrow::ipc::RecordBatchFileReader::Open(file), path);
    batches.reserve(static_cast<std::size_t>(reader->num_record_batches()));
    for (int i = 0; i < reader->num_record_batches(); ++i) {
        batches.push_back(take_arrow(reader->ReadRecordBatch(i), path));
    }
    if (batches.empty()) {
        throw IOError(path + ": Arrow IPC file contains no record batches");
    }
    return take_arrow(arrow::Table::FromRecordBatches(reader->schema(), batches), path);
}

std::shared_ptr<arrow::Table> read_feather_or_ipc(const std::string& path,
                                                  const std::vector<std::string>& names) {
    auto mapped = take_arrow(arrow::io::MemoryMappedFile::Open(path, arrow::io::FileMode::READ), path);
    auto feather = arrow::ipc::feather::Reader::Open(mapped);
    std::shared_ptr<arrow::Table> table;
    if (feather.ok()) {
        auto reader = std::move(feather).ValueOrDie();
        if (names.empty()) ok_arrow(reader->Read(&table), path);
        else ok_arrow(reader->Read(names, &table), path);
        return table;
    }
    auto reopened = take_arrow(arrow::io::MemoryMappedFile::Open(path, arrow::io::FileMode::READ), path);
    table = read_ipc(reopened, path);
    if (names.empty()) return table;
    std::vector<int> indices;
    indices.reserve(names.size());
    for (const auto& name : names) {
        const int index = table->schema()->GetFieldIndex(name);
        if (index < 0) throw InvalidArgumentError(path + ": column \"" + name + "\" not found");
        indices.push_back(index);
    }
    return take_arrow(table->SelectColumns(indices), path);
}

#endif

} // namespace

struct ArrowIpcDataset::Impl {
#ifdef NEXUSDATA_WITH_ARROW
    arrow_detail::TableColumns table;
#endif
};

ArrowIpcDataset::~ArrowIpcDataset() = default;
ArrowIpcDataset::ArrowIpcDataset(ArrowIpcDataset&&) noexcept = default;
ArrowIpcDataset& ArrowIpcDataset::operator=(ArrowIpcDataset&&) noexcept = default;

bool arrow_ipc_support_enabled() {
#if defined(NEXUSDATA_WITH_ARROW)
    return true;
#else
    return false;
#endif
}

ArrowIpcDataset::ArrowIpcDataset(const std::string& path, ArrowIpcOptions opt) {
#if !defined(NEXUSDATA_WITH_ARROW)
    (void)path;
    (void)opt;
    throw InvalidArgumentError(
        "ArrowIpcDataset: built without Arrow. Reconfigure with -DNEXUSDATA_WITH_ARROW=ON");
#else
    const std::vector<std::string> names = arrow_detail::split_names(opt.columns);
    auto table = read_feather_or_ipc(path, names);
    arrow_detail::ColumnPlan plan;
    plan.names = names;
    plan.label_name = opt.label;
    plan.label_index = opt.label_column;
    plan.metadata = opt.metadata_columns;
    plan.extra = opt.extra_columns;
    auto impl = std::make_unique<Impl>();
    impl->table = arrow_detail::materialize(*table, nullptr, plan, true, path);
    impl_ = std::move(impl);
#endif
}

std::size_t ArrowIpcDataset::size() const {
#if defined(NEXUSDATA_WITH_ARROW)
    return impl_ ? impl_->table.rows : 0;
#else
    return 0;
#endif
}

Sample ArrowIpcDataset::get(std::size_t index) const {
#if !defined(NEXUSDATA_WITH_ARROW)
    (void)index;
    throw IndexError("ArrowIpcDataset::get: not available");
#else
    if (!impl_ || index >= impl_->table.rows) {
        throw IndexError("ArrowIpcDataset::get: index out of range");
    }
    return impl_->table.get(index);
#endif
}

const std::vector<std::string>& ArrowIpcDataset::column_names() const {
#if !defined(NEXUSDATA_WITH_ARROW)
    throw InvalidArgumentError(
        "ArrowIpcDataset: built without Arrow. Reconfigure with -DNEXUSDATA_WITH_ARROW=ON");
#else
    if (!impl_) throw InvalidArgumentError("ArrowIpcDataset: empty");
    return impl_->table.names;
#endif
}

NDArray ArrowIpcDataset::column(std::string_view name) const {
#if !defined(NEXUSDATA_WITH_ARROW)
    (void)name;
    throw InvalidArgumentError(
        "ArrowIpcDataset: built without Arrow. Reconfigure with -DNEXUSDATA_WITH_ARROW=ON");
#else
    if (!impl_) throw InvalidArgumentError("ArrowIpcDataset: empty");
    const auto* col = impl_->table.find(name);
    if (!col || col->kind != arrow_detail::Column::Kind::Numeric) {
        throw InvalidArgumentError("ArrowIpcDataset::column: \"" + std::string(name) +
                                   "\" is not a numeric column");
    }
    return col->data;
#endif
}

bool ArrowIpcDataset::column_is_zero_copy(std::string_view name) const {
#if !defined(NEXUSDATA_WITH_ARROW)
    (void)name;
    return false;
#else
    if (!impl_) return false;
    const auto* col = impl_->table.find(name);
    return col && col->zero_copy;
#endif
}

} // namespace nexusdata
