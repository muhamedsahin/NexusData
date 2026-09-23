// v2.0 Phase 3: nested JSON (plan 7.1).

#include <doctest/doctest.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "nexusdata/dataset/json.hpp"
#include "nexusdata/io/compression.hpp"
#include "nexusdata/loading/dataloader.hpp"

using namespace nexusdata;

namespace {

std::string write_text(const std::string& name, const std::string& text) {
    std::ofstream f(name, std::ios::binary);
    f << text;
    return name;
}

// Detection-style records: nested object, numeric arrays, class name, variable box list.
std::string record(int i) {
    std::string boxes;
    for (int k = 0; k < i % 3; ++k) {
        if (k) boxes += ",";
        boxes += "{\"bbox\":[" + std::to_string(k) + "," + std::to_string(i) + ",10,20],\"iscrowd\":0}";
    }
    return "{\"id\":\"img_" + std::to_string(i) + "\",\"feat\":{\"a\":" + std::to_string(i) +
           ",\"b\":[" + std::to_string(i + 1) + ".5," + std::to_string(i + 2) +
           "]},\"cls\":\"" + (i % 2 ? "dog" : "cat") + "\",\"ann\":[" + boxes +
           "],\"ignored\":{\"deep\":[1,{\"x\":\"\\u00e7\"}]},\"note\":null}";
}

std::vector<JsonField> detection_fields() {
    std::vector<JsonField> f(5);
    f[0].path = "feat.a";
    f[1].path = "feat.b";
    f[2].path = "cls";
    f[2].role = JsonRole::Label;
    f[2].dtype = DType::Int64;
    f[2].categories = {"cat", "dog"};
    f[3].path = "ann[*].bbox";
    f[3].role = JsonRole::Extra;
    f[3].name = "bbox";
    f[4].path = "id";
    f[4].role = JsonRole::Metadata;
    return f;
}

} // namespace

TEST_CASE("json::Value: parse, typed access, paths, dump round trip") {
    const json::Value v = json::parse(
        R"({"a": 1, "b": [true, null, 2.5, -0.0, 12345678901234567], "s": "\u00e7\ud83d\ude00\n", "o": {"k": [10, 20]}})");
    CHECK(v.find("a")->as_int() == 1);
    CHECK(v.find("b")->as_array()[4].as_int() == 12345678901234567LL); // exact int64
    CHECK(v.find("s")->as_string() == "\xc3\xa7\xf0\x9f\x98\x80\n");
    CHECK(v.at_path("o.k[1]")->as_int() == 20);
    CHECK(v.at_path("o.missing") == nullptr);
    CHECK(json::parse(v.dump()) == v);
    CHECK(json::parse("2.0").type() == json::Value::Type::Double);
    CHECK(json::parse(json::Value(2.0).dump()).type() == json::Value::Type::Double);
    CHECK_THROWS_AS((void)v.find("a")->as_string(), InvalidArgumentError);

    json::Value cfg;
    cfg.set("name", "x");
    cfg.set("opt", nullptr).set("k", 1); // set() returns the member, so nesting chains
    cfg.set("name", "y");
    CHECK(cfg.dump() == R"({"name":"y","opt":{"k":1}})");
}

TEST_CASE("json::parse: rejects malformed input with a byte offset") {
    for (const char* bad : {"{", "[1,]", "{\"a\" 1}", "01", "1.", "\"\\x\"", "tru", "[1] 2",
                            "\"\x01\"", "\"\\ud800\""}) {
        CAPTURE(bad);
        CHECK_THROWS_AS((void)json::parse(bad), IOError);
    }
    try {
        (void)json::parse("[1, 2, oops]");
        FAIL("no throw");
    } catch (const IOError& e) {
        CHECK(std::string(e.what()).find("byte 7") != std::string::npos);
    }
}

TEST_CASE("JsonDataset: nested paths, categories, ragged [*] extras, metadata") {
    std::string text;
    for (int i = 0; i < 6; ++i) text += record(i) + "\n";
    JsonOptions opt;
    opt.fields = detection_fields();
    auto ds = JsonDataset::from_string(text, opt);
    REQUIRE(ds->size() == 6);

    const Sample s = ds->get(5);
    REQUIRE(s.input.shape() == Shape{3}); // feat.a ++ feat.b
    CHECK(s.input.data<float>()[0] == 5.0f);
    CHECK(s.input.data<float>()[1] == 6.5f);
    CHECK(s.input.data<float>()[2] == 7.0f);
    CHECK(s.label.data<std::int64_t>()[0] == 1); // "dog"
    const NDArray& boxes = s.extra_tensors.at("bbox");
    REQUIRE(boxes.shape() == Shape{2, 4});
    CHECK(boxes.data<float>()[4] == 1.0f);
    CHECK(boxes.data<float>()[5] == 5.0f);
    CHECK(s.metadata.at("id") == "img_5");
    CHECK(ds->get(3).extra_tensors.at("bbox").shape() == Shape{0});
}

TEST_CASE("JsonDataset: array documents, records_path and format detection") {
    const std::string doc = "{\"meta\": {\"v\": 2}, \"data\": {\"items\": [" + record(0) + ", " + record(1) + "]}}";
    JsonOptions opt;
    opt.fields = detection_fields();
    opt.records_path = "data.items";
    auto ds = JsonDataset::from_string(doc, opt);
    CHECK(ds->size() == 2);
    CHECK(ds->get(1).metadata.at("id") == "img_1");

    JsonField x;
    x.path = "[1]";
    JsonOptions arr_opt;
    arr_opt.fields = {x};
    // A top-level array document ...
    auto a = JsonDataset::from_string("[[1, 2], [3, 4], [5, 6]]", arr_opt);
    CHECK(a->size() == 3);
    CHECK(a->get(2).input.data<float>()[0] == 6.0f);
    // ... versus legacy array-per-line JSONL.
    auto l = JsonDataset::from_string("[1, 2]\n[3, 4]\n", arr_opt);
    CHECK(l->size() == 2);
    CHECK(l->get(1).input.data<float>()[0] == 4.0f);
}

TEST_CASE("JsonDataset: shapes, dtypes, defaults, nulls") {
    JsonField img;
    img.path = "img";
    img.dtype = DType::UInt8;
    JsonField flat;
    flat.path = "img";
    flat.role = JsonRole::Extra;
    flat.name = "flat";
    flat.shape = {4};
    JsonField score;
    score.path = "score";
    score.role = JsonRole::Label;
    score.dtype = DType::Float64;
    score.default_value = -1.0;
    JsonOptions opt;
    opt.fields = {img, flat, score};
    auto ds = JsonDataset::from_string(
        "{\"img\": [[1, 2], [3, 4]], \"score\": null}\n{\"img\": [[5, 6], [7, 8]]}\n", opt);
    REQUIRE(ds->size() == 2);
    const Sample a = ds->get(0);
    CHECK(a.input.shape() == Shape{2, 2});
    CHECK(a.input.dtype() == DType::UInt8);
    CHECK(a.input.data<std::uint8_t>()[3] == 4);
    CHECK(a.extra_tensors.at("flat").shape() == Shape{4});
    CHECK(std::isnan(a.label.data<double>()[0]));
    CHECK(ds->get(1).label.data<double>()[0] == -1.0); // absent -> default

    JsonField bad;
    bad.path = "v";
    bad.dtype = DType::Int32;
    JsonOptions strict;
    strict.fields = {bad};
    CHECK_THROWS_AS((void)JsonDataset::from_string("{\"v\": 1.5}", strict), IOError);
    CHECK_THROWS_AS((void)JsonDataset::from_string("{\"v\": [[1], [2, 3]]}", strict), IOError);
    CHECK_THROWS_AS((void)JsonDataset::from_string("{\"w\": 1}", strict), IOError);
}

TEST_CASE("JsonDataset: skip_invalid and error context") {
    JsonField v;
    v.path = "v";
    JsonOptions opt;
    opt.fields = {v};
    const std::string text = "{\"v\": 1}\n{\"v\": }\n{\"v\": \"str\"}\n{\"v\": 4}\n";
    try {
        (void)JsonDataset::from_string(text, opt);
        FAIL("no throw");
    } catch (const IOError& e) {
        CHECK(std::string(e.what()).find("record 1") != std::string::npos);
    }
    opt.skip_invalid = true;
    auto ds = JsonDataset::from_string(text, opt);
    CHECK(ds->size() == 2);
    CHECK(ds->skipped() == 2);
    CHECK(ds->get(1).input.data<float>()[0] == 4.0f);
}

TEST_CASE("field inference and JSON Schema mapping") {
    auto inferred = JsonDataset::from_string("{\"x\": 1, \"y\": [2, 3], \"name\": \"a\", \"o\": {}}\n");
    REQUIRE(inferred->fields().size() == 3);
    CHECK(inferred->get(0).input.numel() == 3);
    CHECK(inferred->get(0).metadata.at("name") == "a");

    const json::Value schema = json::parse(R"({
      "type": "object",
      "properties": {
        "pixels": {"type": "array", "minItems": 2, "maxItems": 2,
                   "items": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "integer"}},
                   "x-nexusdata-dtype": "uint8"},
        "label": {"type": "string", "enum": ["neg", "pos"], "x-nexusdata-role": "label"},
        "meta": {"type": "object", "properties": {"source": {"type": "string"}}},
        "weight": {"type": "number", "x-nexusdata-role": "extra", "default": 1.0}
      }})");
    const auto fields = json_fields_from_schema(schema);
    REQUIRE(fields.size() == 4);
    CHECK(fields[0].shape == Shape{2, 3});
    CHECK(fields[0].dtype == DType::UInt8);
    CHECK(fields[1].categories.size() == 2);
    CHECK(fields[2].path == "meta.source");
    CHECK(fields[2].role == JsonRole::Metadata);
    JsonOptions opt;
    opt.fields = fields;
    auto ds = JsonDataset::from_string(
        R"({"pixels": [[1,2,3],[4,5,6]], "label": "pos", "meta": {"source": "cam"}})", opt);
    const Sample s = ds->get(0);
    CHECK(s.input.shape() == Shape{2, 3});
    CHECK(s.label.data<std::int64_t>()[0] == 1);
    CHECK(s.metadata.at("meta.source") == "cam");
    CHECK(s.extra_tensors.at("weight").data<float>()[0] == 1.0f);
}

TEST_CASE("JsonDataset: parallel parse keeps record order") {
    std::string text;
    for (int i = 0; i < 20000; ++i) text += record(i) + "\n";
    JsonOptions opt;
    opt.fields = detection_fields();
    opt.num_threads = 8;
    auto ds = JsonDataset::from_string(text, opt);
    REQUIRE(ds->size() == 20000);
    for (std::size_t i : {std::size_t{0}, std::size_t{4999}, std::size_t{12345}, std::size_t{19999}}) {
        CHECK(ds->get(i).metadata.at("id") == "img_" + std::to_string(i));
        CHECK(ds->get(i).input.data<float>()[0] == static_cast<float>(i));
    }
}

TEST_CASE("JsonLinesIterable streams compressed files and matches JsonDataset") {
    std::string text;
    for (int i = 0; i < 500; ++i) text += record(i) + "\n\n";
    std::string path = "nd_stream.jsonl";
    if (codec_available(Codec::Zstd)) {
        const auto z = compress(Codec::Zstd, text.data(), text.size());
        path += ".zst";
        write_text(path, std::string(z.begin(), z.end()));
    } else {
        write_text(path, text);
    }
    JsonOptions opt;
    opt.fields = detection_fields();
    JsonDataset ds(path, opt);
    JsonLinesIterable it_ds(path, opt);
    auto it = it_ds.make_iterator();
    std::size_t n = 0;
    while (it->has_next()) {
        const Sample a = it->next();
        const Sample b = ds.get(n);
        CHECK(a.metadata.at("id") == b.metadata.at("id"));
        CHECK(a.extra_tensors.at("bbox").shape() == b.extra_tensors.at("bbox").shape());
        ++n;
    }
    CHECK(n == 500);
    std::remove(path.c_str());
}

TEST_CASE("JsonDataset + DataLoader: ragged boxes batch as lists") {
    std::string text;
    for (int i = 0; i < 12; ++i) text += record(i) + "\n";
    JsonOptions opt;
    opt.fields = detection_fields();
    auto ds = JsonDataset::from_string(text, opt);
    DataLoaderOptions dl_opt;
    dl_opt.batch_size = 4;
    dl_opt.num_workers = 2;
    DataLoader dl(DatasetConstPtr(ds), dl_opt, collate_stack_nd);
    std::size_t seen = 0;
    for (const Batch& b : dl) {
        CHECK(b.inputs.shape()[1] == 3);
        CHECK(b.ragged_extras.at("bbox").size() == b.inputs.shape()[0]);
        CHECK(b.metadata.at("id")[0] == "img_" + std::to_string(seen));
        seen += b.inputs.shape()[0];
    }
    CHECK(seen == 12);
}
