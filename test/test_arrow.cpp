// v2.0 Phase 3: Parquet column projection, row-group pushdown, Arrow IPC / Feather.

#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "nexusdata/dataset/arrow_ipc.hpp"
#include "nexusdata/dataset/parquet.hpp"
#include "nexusdata/loading/dataloader.hpp"

using namespace nexusdata;

#if !defined(NEXUSDATA_WITH_ARROW)

TEST_CASE("Parquet and Arrow IPC stay disabled without Arrow") {
    CHECK_FALSE(parquet_support_enabled());
    CHECK_FALSE(arrow_ipc_support_enabled());
    CHECK_THROWS_AS(ParquetDataset("nope.parquet", {}), InvalidArgumentError);
    CHECK_THROWS_AS(ArrowIpcDataset("nope.feather", {}), InvalidArgumentError);
}

#else

#ifndef NOMINMAX
#define NOMINMAX
#endif
#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <parquet/arrow/writer.h>
#include <parquet/properties.h>
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#include <cstdio>
#include <memory>

namespace {

void must(const arrow::Status& status) {
    if (!status.ok()) throw std::runtime_error(status.ToString());
}

template <typename T>
T must_result(arrow::Result<T> result) {
    if (!result.ok()) throw std::runtime_error(result.status().ToString());
    return std::move(result).ValueOrDie();
}

std::shared_ptr<arrow::Array> finish(arrow::ArrayBuilder& builder) {
    std::shared_ptr<arrow::Array> out;
    must(builder.Finish(&out));
    return out;
}

void write_parquet(const std::string& path, const std::shared_ptr<arrow::Table>& table, int64_t rows_per_group,
                   bool statistics) {
    auto sink = must_result(arrow::io::FileOutputStream::Open(path));
    std::shared_ptr<parquet::WriterProperties> props = parquet::default_writer_properties();
    if (!statistics) {
        props = parquet::WriterProperties::Builder().disable_statistics()->build();
    }
    must(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), sink, rows_per_group, props));
    must(sink->Close());
}

void write_feather(const std::string& path, const std::shared_ptr<arrow::Table>& table) {
    auto sink = must_result(arrow::io::FileOutputStream::Open(path));
    must(arrow::ipc::feather::WriteTable(*table, sink.get()));
    must(sink->Close());
}

void write_ipc_batches(const std::string& path, const std::shared_ptr<arrow::Table>& table,
                       int64_t batch_rows) {
    auto sink = must_result(arrow::io::FileOutputStream::Open(path));
    auto writer = must_result(arrow::ipc::MakeFileWriter(sink.get(), table->schema()));
    arrow::TableBatchReader reader(*table);
    reader.set_chunksize(batch_rows);
    std::shared_ptr<arrow::RecordBatch> batch;
    while (true) {
        must(reader.ReadNext(&batch));
        if (!batch) break;
        must(writer->WriteRecordBatch(*batch));
    }
    must(writer->Close());
    must(sink->Close());
}

std::shared_ptr<arrow::Table> score_table() {
    arrow::FloatBuilder x;
    arrow::DoubleBuilder score;
    arrow::StringBuilder name;
    for (int i = 0; i < 20; ++i) {
        must(x.Append(static_cast<float>(i)));
        must(score.Append(static_cast<double>(i)));
        must(name.Append(i < 10 ? std::string("low") : std::string("high")));
    }
    auto schema = arrow::schema({arrow::field("x", arrow::float32()),
                                 arrow::field("score", arrow::float64()),
                                 arrow::field("name", arrow::utf8())});
    return arrow::Table::Make(schema, {finish(x), finish(score), finish(name)});
}

} // namespace

TEST_CASE("Parquet: projection, roles, and exact row-group pushdown") {
    const std::string path = "nd_p3_scores.parquet";
    write_parquet(path, score_table(), 10, true);

    ParquetOptions opt;
    opt.columns = "x, score, name";
    opt.label = "score";
    ParquetPredicate pred;
    pred.column = "score";
    pred.op = ParquetCompare::Ge;
    pred.number = 10;
    opt.predicates = {pred};

    ParquetDataset ds(path, opt);
    CHECK(ds.row_groups_total() == 2);
    CHECK(ds.row_groups_read() == 1);
    CHECK(ds.row_groups_skipped() == 1);
    CHECK(ds.size() == 10);
    CHECK(ds.column_names().size() == 3);
    const Sample first = ds.get(0);
    CHECK(first.input.data<float>()[0] == 10.0f);
    CHECK(first.label.data<double>()[0] == 10.0);
    CHECK(first.metadata.at("name") == "high");
    CHECK(ds.get(9).input.data<float>()[0] == 19.0f);

    ParquetOptions edge = opt;
    edge.predicates[0].number = 9;
    ParquetDataset kept(path, edge);
    CHECK(kept.row_groups_read() == 2);
    CHECK(kept.size() == 11);
    CHECK(kept.get(0).label.data<double>()[0] == 9.0);

    ParquetOptions none = opt;
    none.predicates[0].number = 100;
    ParquetDataset empty(path, none);
    CHECK(empty.size() == 0);
    CHECK(empty.row_groups_skipped() == 2);
    CHECK(empty.row_groups_read() == 0);
    CHECK(empty.column_names().size() == 3);

    ParquetPredicate text;
    text.column = "name";
    text.op = ParquetCompare::Eq;
    text.is_string = true;
    text.text = "high";
    ParquetOptions by_name;
    by_name.columns = "x, name";
    by_name.predicates = {text};
    ParquetDataset named(path, by_name);
    CHECK(named.row_groups_skipped() == 1);
    CHECK(named.size() == 10);
    CHECK(named.get(0).metadata.at("name") == "high");
    CHECK_THROWS_AS(static_cast<void>(named.get(0).metadata.at("score")), IndexError);

    std::remove(path.c_str());
}

TEST_CASE("Parquet: without statistics every group is read and rows are still filtered") {
    const std::string path = "nd_p3_nostats.parquet";
    write_parquet(path, score_table(), 10, false);
    ParquetOptions opt;
    opt.label = "score";
    ParquetPredicate pred;
    pred.column = "score";
    pred.op = ParquetCompare::Ge;
    pred.number = 10;
    opt.predicates = {pred};
    ParquetDataset ds(path, opt);
    CHECK(ds.row_groups_total() == 2);
    CHECK(ds.row_groups_read() == 2);
    CHECK(ds.row_groups_skipped() == 0);
    CHECK(ds.size() == 10);
    CHECK(ds.get(0).input.data<float>()[0] == 10.0f);
    std::remove(path.c_str());
}

TEST_CASE("Parquet: unread columns are not decoded, null integers fail, lists stay ragged") {
    const std::string path = "nd_p3_project.parquet";
    arrow::FloatBuilder x;
    arrow::FixedSizeBinaryBuilder blob(arrow::fixed_size_binary(3));
    must(x.Append(1.5f));
    must(x.Append(2.5f));
    must(blob.Append("abc"));
    must(blob.Append("de "));
    auto schema = arrow::schema({arrow::field("x", arrow::float32()),
                                 arrow::field("blob", arrow::fixed_size_binary(3))});
    write_parquet(path, arrow::Table::Make(schema, {finish(x), finish(blob)}), 2, true);
    ParquetOptions only_x;
    only_x.columns = "x";
    ParquetDataset ds(path, only_x);
    CHECK(ds.size() == 2);
    CHECK(ds.column_names().size() == 1);
    CHECK(ds.get(1).input.data<float>()[0] == 2.5f);
    CHECK_THROWS_AS(ParquetDataset(path, {}), IOError);
    ParquetOptions missing;
    missing.columns = "missing";
    CHECK_THROWS_AS(ParquetDataset(path, missing), InvalidArgumentError);
    std::remove(path.c_str());

    const std::string nulls = "nd_p3_nulls.parquet";
    arrow::Int32Builder ids;
    must(ids.Append(1));
    must(ids.AppendNull());
    write_parquet(nulls, arrow::Table::Make(arrow::schema({arrow::field("id", arrow::int32())}),
                                            {finish(ids)}),
                  2, true);
    CHECK_THROWS_AS(ParquetDataset(nulls, {}), IOError);
    std::remove(nulls.c_str());

    const std::string lists = "nd_p3_lists.parquet";
    auto pool = arrow::default_memory_pool();
    arrow::ListBuilder list(pool, std::make_shared<arrow::FloatBuilder>());
    auto* values = static_cast<arrow::FloatBuilder*>(list.value_builder());
    arrow::FloatBuilder feature;
    must(feature.Append(7.0f));
    must(feature.Append(8.0f));
    must(list.Append());
    must(values->Append(1.0f));
    must(values->Append(2.0f));
    must(list.Append());
    must(values->Append(3.0f));
    auto list_schema = arrow::schema(
        {arrow::field("x", arrow::float32()), arrow::field("box", arrow::list(arrow::float32()))});
    write_parquet(lists, arrow::Table::Make(list_schema, {finish(feature), finish(list)}), 2, true);
    ParquetOptions ragged;
    ragged.extra_columns = {"box"};
    ParquetDataset boxes(lists, ragged);
    CHECK(boxes.get(0).input.data<float>()[0] == 7.0f);
    CHECK(boxes.get(0).extra_tensors.at("box").shape() == Shape{2});
    CHECK(boxes.get(1).extra_tensors.at("box").shape() == Shape{1});
    CHECK(boxes.get(1).extra_tensors.at("box").data<float>()[0] == 3.0f);
    ParquetOptions as_input;
    CHECK_THROWS_AS(ParquetDataset(lists, as_input), InvalidArgumentError);
    std::remove(lists.c_str());

    const std::string fixed = "nd_p3_fixed.parquet";
    arrow::FloatBuilder flat;
    must(flat.AppendValues({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
    auto packed = must_result(arrow::FixedSizeListArray::FromArrays(finish(flat), 3));
    arrow::Int64Builder y;
    must(y.Append(0));
    must(y.Append(1));
    auto fixed_schema = arrow::schema({arrow::field("vec", arrow::fixed_size_list(arrow::float32(), 3)),
                                       arrow::field("y", arrow::int64())});
    write_parquet(fixed, arrow::Table::Make(fixed_schema, {packed, finish(y)}), 2, true);
    ParquetOptions vec;
    vec.label = "y";
    ParquetDataset vectors(fixed, vec);
    CHECK(vectors.get(1).input.shape() == Shape{3});
    CHECK(vectors.get(1).input.data<float>()[0] == 4.0f);
    CHECK(vectors.get(1).input.data<float>()[2] == 6.0f);
    CHECK(vectors.get(1).label.data<std::int64_t>()[0] == 1);
    std::remove(fixed.c_str());
}

TEST_CASE("Parquet + DataLoader stacks projected rows") {
    const std::string path = "nd_p3_loader.parquet";
    write_parquet(path, score_table(), 20, true);
    ParquetOptions opt;
    opt.columns = "x, score";
    opt.label_column = 1;
    auto ds = std::make_shared<ParquetDataset>(path, opt);
    DataLoaderOptions dl;
    dl.batch_size = 4;
    dl.num_workers = 2;
    DataLoader loader(DatasetConstPtr(ds), dl);
    std::size_t seen = 0;
    for (const Batch& batch : loader) {
        CHECK(batch.inputs.shape()[0] == 4);
        CHECK(batch.inputs.data<float>()[0] == static_cast<float>(seen));
        CHECK(batch.labels.data<double>()[0] == static_cast<double>(seen));
        seen += batch.inputs.shape()[0];
    }
    CHECK(seen == 20);
    std::remove(path.c_str());
}

TEST_CASE("Feather and Arrow IPC: zero-copy primitive columns, nulls materialize") {
    const std::string feather = "nd_p3.feather";
    arrow::FloatBuilder x;
    arrow::BooleanBuilder flag;
    arrow::FloatBuilder broken;
    arrow::StringBuilder name;
    for (int i = 0; i < 8; ++i) {
        must(x.Append(static_cast<float>(i + 1)));
        must(flag.Append(i % 2 == 0));
        must(name.Append("r" + std::to_string(i)));
        if (i == 3) must(broken.AppendNull());
        else must(broken.Append(static_cast<float>(i)));
    }
    auto schema = arrow::schema({arrow::field("x", arrow::float32()), arrow::field("flag", arrow::boolean()),
                                 arrow::field("broken", arrow::float32()),
                                 arrow::field("name", arrow::utf8())});
    auto table = arrow::Table::Make(schema, {finish(x), finish(flag), finish(broken), finish(name)});
    write_feather(feather, table);

    ArrowIpcOptions split;
    split.extra_columns = {"broken", "flag"};
    ArrowIpcDataset ds(feather, split);
    CHECK(ds.size() == 8);
    CHECK(ds.column_is_zero_copy("x"));
    CHECK_FALSE(ds.column_is_zero_copy("flag"));
    CHECK_FALSE(ds.column_is_zero_copy("broken"));
    NDArray view;
    {
        ArrowIpcDataset alive(feather, split);
        view = alive.column("x");
        CHECK(view.shape() == Shape{8});
        CHECK(view.data<float>()[7] == 8.0f);
    }
    CHECK(view.data<float>()[0] == 1.0f);
    CHECK(ds.column("flag").data<std::uint8_t>()[0] == 1);
    CHECK(ds.column("flag").data<std::uint8_t>()[1] == 0);
    ArrowIpcDataset one(feather, split);
    CHECK(one.column_is_zero_copy("x"));
    CHECK(one.get(3).input.data<float>()[0] == 4.0f);
    CHECK(std::isnan(one.get(3).extra_tensors.at("broken").data<float>()[0]));
    CHECK(one.get(2).metadata.at("name") == "r2");
    std::remove(feather.c_str());

    const std::string ipc = "nd_p3.arrow";
    write_ipc_batches(ipc, table, 3);
    ArrowIpcDataset batches(ipc, split);
    CHECK(batches.size() == 8);
    CHECK(batches.get(7).input.data<float>()[0] == 8.0f);
    CHECK(batches.get(7).metadata.at("name") == "r7");
    std::remove(ipc.c_str());
}

TEST_CASE("Arrow IPC rejects a truncated file") {
    const std::string path = "nd_p3_bad.arrow";
    std::FILE* f = std::fopen(path.c_str(), "wb");
    REQUIRE(f != nullptr);
    std::fputs("not arrow", f);
    std::fclose(f);
    CHECK_THROWS_AS(ArrowIpcDataset(path, {}), IOError);
    std::remove(path.c_str());
}

#endif
