#include <doctest/doctest.h>

#include <cmath>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include "nexusdata/adapter/c_api.h"
#include "nexusdata/core/describe.hpp"
#include "nexusdata/core/metrics.hpp"
#include "nexusdata/core/plugin_loader.hpp"
#include "nexusdata/core/registry.hpp"
#include "nexusdata/dataset/function_dataset.hpp"
#include "nexusdata/pipeline/fluent.hpp"
#include "nexusdata/pipeline/pipeline_builder.hpp"
#include "nexusdata/pipeline/tabular.hpp"

using namespace nexusdata;

namespace {

struct Rec {
    int a = 0;
    std::string name;
    double z = 0;
};
NEXUSDATA_DESCRIBE_STRUCT(Rec, a, name, z)

std::string write_csv(const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    out << "x,y\n0.5,1\n2,0\n-1,3\n";
    return path;
}

std::string plugin_file() {
#ifdef _WIN32
    char buf[MAX_PATH];
    const DWORD n = GetModuleFileNameA(nullptr, buf, MAX_PATH);
    const auto dir = std::string(buf, n);
    const auto slash = dir.find_last_of("\\/");
    return dir.substr(0, slash + 1) + ND_DEMO_PLUGIN_NAME;
#else
    return ND_DEMO_PLUGIN_NAME;
#endif
}

class CountSink : public metrics::Sink {
public:
    void add(std::string_view name, double value) override {
        if (name == "rows") total += value;
        calls += 1;
    }
    double total = 0;
    int calls = 0;
};

} // namespace

TEST_CASE("registry opens csv and a function dataset needs no subclass") {
    const std::string path = write_csv("nd_ext.csv");
    DatasetPtr ds = DataSourceRegistry::instance().open("csv:" + path, R"({"has_header":true,"label_column":"y"})");
    CHECK(ds->size() == 3);
    CHECK(ds->get(0).input.data<float>()[0] == doctest::Approx(0.5f));
    CHECK(ds->get(0).label.data<float>()[0] == doctest::Approx(1.f));

    DatasetPtr fn = make_dataset(3, [](std::size_t i) {
        Sample s;
        s.input = NDArray(Shape{1}, DType::Int64);
        s.input.data<std::int64_t>()[0] = static_cast<std::int64_t>(i * 10);
        s.label = NDArray(Shape{1}, DType::Int64);
        s.label.data<std::int64_t>()[0] = static_cast<std::int64_t>(i);
        return s;
    });
    CHECK(fn->get(2).input.data<std::int64_t>()[0] == 20);

    int yielded = 0;
    GeneratorIterable gen([&] {
        auto left = std::make_shared<int>(2);
        return GeneratorIterable::NextFn([left]() -> std::optional<Sample> {
            if (*left == 0) return std::nullopt;
            --(*left);
            Sample s;
            s.input = NDArray(Shape{1}, DType::Float32);
            s.input.data<float>()[0] = 1.f;
            s.label = NDArray(Shape{1}, DType::Int64);
            s.label.data<std::int64_t>()[0] = 0;
            return s;
        });
    });
    auto it = gen.make_iterator();
    while (it->has_next()) {
        (void)it->next();
        ++yielded;
    }
    CHECK(yielded == 2);
    std::remove(path.c_str());
}

TEST_CASE("describe, fluent chain, json pipeline and metrics") {
    Rec rec;
    rec.a = 4;
    rec.name = "row";
    rec.z = 1.25;
    Sample described = to_sample(rec);
    CHECK(described.extra_tensors.at("a").data<std::int64_t>()[0] == 4);
    CHECK(described.metadata.at("name") == "row");
    CHECK(described.extra_tensors.at("z").data<double>()[0] == doctest::Approx(1.25));

    const std::string path = write_csv("nd_ext_fluent.csv");
    auto loader = from_csv(path).map(std::make_shared<ClipTransform>(0.0, 1.0)).batch(2);
    auto b = loader->begin();
    CHECK((*b).inputs.numel() == 2);
    CHECK((*b).inputs.data<float>()[0] == doctest::Approx(0.5f));
    CHECK((*b).inputs.data<float>()[1] == doctest::Approx(1.f));

    const std::string cfg = "nd_ext_pipe.json";
    {
        std::ofstream out(cfg);
        out << R"({"source":{"type":"csv","path":"nd_ext_fluent.csv","options":{"has_header":true,"label_column":"y"}},
"transforms":[{"type":"clip","params":{"min":0,"max":1}}],
"loader":{"batch_size":3,"shuffle":false,"num_workers":0,"device":"cpu"}})";
    }
    BuiltPipeline built = PipelineBuilder::from_file(cfg);
    CHECK(built.dataset->size() == 3);
    CHECK(built.loader->size() == 1);
    CHECK(built.dataset->get(2).input.data<float>()[0] == doctest::Approx(0.f));

    const std::string yml = "nd_ext_pipe.yaml";
    {
        std::ofstream out(yml);
        out << "source:\n"
               "  type: csv\n"
               "  path: nd_ext_fluent.csv\n"
               "  options: { has_header: true, label_column: \"y\" }\n"
               "transforms:\n"
               "  - type: clip\n"
               "    params: { min: 0, max: 1 }\n"
               "loader:\n"
               "  batch_size: 3\n"
               "  shuffle: false\n"
               "  num_workers: 0\n";
    }
    BuiltPipeline from_yaml = PipelineBuilder::from_file(yml);
    CHECK(from_yaml.dataset->get(1).input.data<float>()[0] == doctest::Approx(1.f));

    CountSink sink;
    metrics::set_sink(&sink);
    metrics::publish("rows", 3);
    metrics::add("rows", 9);
    CHECK(sink.total == doctest::Approx(3));
    CHECK(sink.calls == 1);
    CHECK_FALSE(metrics::enabled());
    metrics::set_sink(nullptr);

    std::remove(path.c_str());
    std::remove(cfg.c_str());
    std::remove(yml.c_str());
}

TEST_CASE("C API and a loaded plugin share the host registry") {
    const std::string path = write_csv("nd_ext_c.csv");
    NexusDataset* ds = nullptr;
    const std::string uri = "csv:" + path;
    CHECK(nexus_dataset_open(uri.c_str(), R"({"has_header":true,"label_column":"y"})", &ds) == NEXUS_OK);
    std::size_t n = 0;
    CHECK(nexus_dataset_size(ds, &n) == NEXUS_OK);
    CHECK(n == 3);
    NexusDataLoader* loader = nullptr;
    CHECK(nexus_dataloader_create(ds, R"({"batch_size":2})", &loader) == NEXUS_OK);
    NexusBatch* batch = nullptr;
    CHECK(nexus_dataloader_next_batch(loader, &batch) == NEXUS_OK);
    NexusNDArray* inputs = nullptr;
    CHECK(nexus_batch_inputs(batch, &inputs) == NEXUS_OK);
    std::size_t numel = 0;
    CHECK(nexus_ndarray_numel(inputs, &numel) == NEXUS_OK);
    CHECK(numel == 2);
    nexus_ndarray_destroy(inputs);
    nexus_batch_destroy(batch);
    NexusBatch* done = nullptr;
    CHECK(nexus_dataloader_next_batch(loader, &done) == NEXUS_OK);
    nexus_batch_destroy(done);
    CHECK(nexus_dataloader_next_batch(loader, &done) == NEXUS_END_OF_EPOCH);
    nexus_dataloader_destroy(loader);
    nexus_dataset_destroy(ds);

    load_plugin(plugin_file());
    DatasetPtr plugged = DataSourceRegistry::instance().open("plug:3", "{}");
    CHECK(plugged->size() == 3);
    CHECK(plugged->get(0).input.data<float>()[0] == doctest::Approx(1.f));
    CHECK(plugged->get(2).label.data<std::int64_t>()[0] == 2);
    std::remove(path.c_str());
}

TEST_CASE("open covers built-in schemes, records, and fitted preprocessors") {
    const std::string csv = write_csv("nd_ext_open.csv");
    DatasetPtr via_open = open("csv:" + csv, R"({"has_header":true,"label_column":"y"})");
    CHECK(via_open->size() == 3);
    CHECK(via_open->get(1).input.data<float>()[0] == doctest::Approx(2.f));

    const std::string lines = "nd_ext_lines.jsonl";
    {
        std::ofstream out(lines);
        out << "[1.0, 0]\n[3.0, 1]\n";
    }
    DatasetPtr jsonl = open("jsonl:" + lines, R"({"array_with_label":true})");
    CHECK(jsonl->size() == 2);
    CHECK(jsonl->get(1).input.data<float>()[0] == doctest::Approx(3.f));
    CHECK(jsonl->get(1).label.data<float>()[0] == doctest::Approx(1.f));

    const std::string text = "nd_ext_lines.txt";
    {
        std::ofstream out(text);
        out << "alpha\nbeta\n";
    }
    CHECK(open("text:" + text)->size() == 2);

    bool arrow = false;
    try {
        (void)open("parquet:missing.parquet");
    } catch (const InvalidArgumentError& e) {
        arrow = std::string(e.what()).find("NEXUSDATA_WITH_ARROW") != std::string::npos;
    }
    CHECK(arrow);
    CHECK_THROWS_AS(static_cast<void>(open("no-such-scheme:x")), InvalidArgumentError);

    std::vector<Rec> rows(2);
    rows[0].a = 1;
    rows[0].name = "a";
    rows[0].z = 0.5;
    rows[1].a = 2;
    rows[1].name = "b";
    rows[1].z = 1.5;
    DatasetPtr recs = records_dataset(rows);
    CHECK(recs->size() == 2);
    CHECK(recs->get(1).extra_tensors.at("a").data<std::int64_t>()[0] == 2);
    CHECK(recs->get(1).metadata.at("name") == "b");

    const std::string cfg = "nd_ext_pre.json";
    {
        std::ofstream out(cfg);
        out << R"({"source":{"type":"csv","path":"nd_ext_open.csv","options":{"has_header":true,"label_column":"y"}},
"preprocessors":[{"type":"standard_scaler"}],
"loader":{"batch_size":3,"shuffle":false,"num_workers":0}})";
    }
    BuiltPipeline scaled = PipelineBuilder::from_file(cfg);
    const double stddev = std::sqrt(1.5);
    CHECK(scaled.dataset->get(0).input.data<float>()[0] == doctest::Approx(0.f));
    CHECK(scaled.dataset->get(1).input.data<float>()[0] == doctest::Approx(1.5 / stddev));
    CHECK(scaled.dataset->get(2).input.data<float>()[0] == doctest::Approx(-1.5 / stddev));
    CHECK(scaled.dataset->get(0).label.data<float>()[0] == doctest::Approx(1.f));
    CHECK(TransformerRegistry::instance().contains("minmax_scaler"));

    std::remove(csv.c_str());
    std::remove(lines.c_str());
    std::remove(text.c_str());
    std::remove(cfg.c_str());
}
