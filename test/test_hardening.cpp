#include <doctest/doctest.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "nexusdata/core/error.hpp"
#include "nexusdata/core/ndarray.hpp"
#include "nexusdata/core/shape.hpp"
#include "nexusdata/dataset/audio.hpp"
#include "nexusdata/dataset/csv/csv_dataset.hpp"
#include "nexusdata/dataset/in_memory.hpp"
#include "nexusdata/dataset/npy.hpp"
#include "nexusdata/dataset/webdataset.hpp"
#include "nexusdata/fuzz/harness.hpp"
#include "nexusdata/loading/batch.hpp"
#include "nexusdata/loading/dataloader.hpp"

using namespace nexusdata;

namespace {

std::string write_temp(const std::string& name, const std::string& body) {
    std::ofstream out(name, std::ios::binary);
    out.write(body.data(), static_cast<std::streamsize>(body.size()));
    return name;
}

} // namespace

TEST_CASE("error message helpers") {
    CHECK(quote_path("a/b") == "\"a/b\"");
    const auto idx = format_index_error("Foo::get", 5, 3);
    CHECK(idx.find("index 5") != std::string::npos);
    CHECK(idx.find("[0, 3)") != std::string::npos);
    const auto loc = format_loc("f.csv", 2, 4, "bad");
    CHECK(loc == "f.csv:2:4 bad");
    CHECK(format_shape_error("stack", Shape{2, 3}, Shape{2, 4}).find("[2, 3]") !=
          std::string::npos);
}

TEST_CASE("index OOB messages") {
    NDArray X(Shape{2, 1}, DType::Float32);
    NDArray y(Shape{2}, DType::Int64);
    InMemoryDataset ds(X, y);
    try {
        ds.get(9);
        FAIL("expected IndexError");
    } catch (const IndexError& e) {
        const std::string msg = e.what();
        CHECK(msg.find("index 9") != std::string::npos);
        CHECK(msg.find("[0, 2)") != std::string::npos);
    }
}

TEST_CASE("empty and corrupt CSV") {
    const auto empty = write_temp("hard_empty.csv", "");
    CHECK_THROWS_AS(CSVDataset(empty, {}), IOError);
    std::remove(empty.c_str());

    const auto bad = write_temp("hard_bad.csv", "a,b\n1\n");
    CHECK_THROWS_AS(CSVDataset(bad, {}), IOError);
    std::remove(bad.c_str());

    const auto unclosed = write_temp("hard_quote.csv", "a,b\n\"1,2\n");
    try {
        CSVDataset(unclosed, {});
        FAIL("expected IOError");
    } catch (const IOError& e) {
        CHECK(std::string(e.what()).find("unclosed") != std::string::npos);
    }
    std::remove(unclosed.c_str());
}

TEST_CASE("corrupt npy wav buffers") {
    std::vector<std::uint8_t> junk(16, 0xAB);
    CHECK_THROWS_AS(load_npy_from_buffer(junk.data(), junk.size()), IOError);
    CHECK_THROWS_AS(load_wav_from_buffer(junk.data(), junk.size()), IOError);
    CHECK_THROWS_AS(load_npy_from_buffer(nullptr, 0), InvalidArgumentError);
    CHECK_THROWS_AS(load_wav_from_buffer(nullptr, 0), InvalidArgumentError);

    // Truncated RIFF
    std::string riff = "RIFF";
    riff.append(8, '\0');
    CHECK_THROWS_AS(load_wav_from_buffer(reinterpret_cast<const std::uint8_t*>(riff.data()),
                                         riff.size()),
                    IOError);
}

TEST_CASE("collate edge cases") {
    CHECK_THROWS_AS(collate_stack({}), InvalidArgumentError);
    CHECK_THROWS_AS(collate_pad_sequence({}), InvalidArgumentError);

    Sample a;
    a.input = NDArray(Shape{2}, DType::Float32);
    a.label = NDArray(Shape{1}, DType::Int64);
    Sample b;
    b.input = NDArray(Shape{2}, DType::Float64);
    b.label = NDArray(Shape{1}, DType::Int64);
    CHECK_THROWS_AS(collate_pad_sequence({a, b}), ShapeError);
}

TEST_CASE("DataLoader invalid options") {
    NDArray X(Shape{4, 1}, DType::Float32);
    NDArray y(Shape{4}, DType::Int64);
    InMemoryDataset ds(X, y);
    DataLoaderOptions lo;
    lo.batch_size = 0;
    CHECK_THROWS_AS(DataLoader(ds, lo), InvalidArgumentError);
}

TEST_CASE("fuzz harness does not abort on garbage") {
    std::vector<std::uint8_t> buf(64);
    for (std::size_t i = 0; i < buf.size(); ++i) {
        buf[i] = static_cast<std::uint8_t>(i * 17u);
    }
    (void)fuzz::try_npy(buf.data(), buf.size());
    (void)fuzz::try_wav(buf.data(), buf.size());

    const auto p = write_temp("hard_fuzz.csv", std::string(buf.begin(), buf.end()));
    (void)fuzz::try_csv_file(p);
    std::remove(p.c_str());

    const auto t = write_temp("hard_fuzz.tar", std::string(buf.begin(), buf.end()));
    (void)fuzz::try_webdataset_file(t);
    std::remove(t.c_str());
}
