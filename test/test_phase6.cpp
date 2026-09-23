#include <doctest/doctest.h>

#include <fstream>
#include <sstream>
#include <string>

#include "nexusdata/adapter/c_api.h"
#include "nexusdata/bench/harness.hpp"
#include "nexusdata/core/error.hpp"
#include "nexusdata/core/metrics.hpp"
#include "nexusdata/dataset/function_dataset.hpp"
#include "nexusdata/io/json.hpp"
#include "nexusdata/pipeline/fluent.hpp"
#include "nexusdata/pipeline/pipeline_builder.hpp"
#include "nexusdata/pipeline/tabular.hpp"

using namespace nexusdata;

namespace {

std::string write_csv() {
    std::ofstream out("nd_phase6.csv", std::ios::binary);
    out << "x,y\n0.5,1\n2,0\n-1,3\n";
    return "nd_phase6.csv";
}

class CountSink : public metrics::Sink {
public:
    void add(std::string_view, double value) override {
        total += value;
        ++calls;
    }
    double total = 0;
    int calls = 0;
};

} // namespace

TEST_CASE("fluent chain filters, batches, and names a bad pipeline step") {
    const std::string path = write_csv();
    CSVOptions opt;
    opt.label_column = "y";
    auto loader = from_csv(path, opt)
                      .map(ClipTransform(0.0, 1.0))
                      .filter([](const Sample& s) { return s.label.data<float>()[0] > 0.f; })
                      .shuffle(1)
                      .batches(2, false)
                      .prefetch(2)
                      .build();
    CHECK(loader->size() == 1);
    const Batch b = *loader->begin();
    CHECK(b.inputs.numel() == 2);
    CHECK(b.inputs.data<float>()[0] <= 1.f);

    DatasetPtr via = from_uri("csv:" + path, R"({"has_header":true,"label_column":"y"})").dataset();
    CHECK(via->size() == 3);

    bool named = false;
    try {
        (void)PipelineBuilder::from_json(
            R"({"source":{"type":"csv","path":"nd_phase6.csv","options":{"has_header":true}},
"transforms":[{"type":"not_a_transform"}]})");
    } catch (const InvalidArgumentError& e) {
        const std::string msg = e.what();
        named = msg.find("pipeline transform 'not_a_transform'") != std::string::npos;
    }
    CHECK(named);

    bool sourced = false;
    try {
        (void)PipelineBuilder::from_json(
            R"({"source":{"type":"csv","path":"nd_phase6_missing.csv","options":{"has_header":true}}})");
    } catch (const IOError& e) {
        const std::string msg = e.what();
        sourced = msg.find("pipeline source 'csv'") != std::string::npos;
    }
    CHECK(sourced);
    std::remove(path.c_str());
}

TEST_CASE("C API sample carries input and label") {
    const std::string path = write_csv();
    const std::string uri = "csv:" + path;
    NexusDataset* ds = nullptr;
    CHECK(nexus_dataset_open(uri.c_str(), R"({"has_header":true,"label_column":"y"})", &ds) == NEXUS_OK);
    NexusSample* sample = nullptr;
    CHECK(nexus_dataset_get(ds, 0, &sample) == NEXUS_OK);
    NexusNDArray* input = nullptr;
    NexusNDArray* label = nullptr;
    CHECK(nexus_sample_input(sample, &input) == NEXUS_OK);
    CHECK(nexus_sample_label(sample, &label) == NEXUS_OK);
    const float* in = nullptr;
    const float* lab = nullptr;
    CHECK(nexus_ndarray_data_const(input, reinterpret_cast<const void**>(&in)) == NEXUS_OK);
    CHECK(nexus_ndarray_data_const(label, reinterpret_cast<const void**>(&lab)) == NEXUS_OK);
    CHECK(in[0] == doctest::Approx(0.5f));
    CHECK(lab[0] == doctest::Approx(1.f));
    nexus_ndarray_destroy(input);
    nexus_ndarray_destroy(label);
    nexus_sample_destroy(sample);
    nexus_dataset_destroy(ds);
    std::remove(path.c_str());
}

TEST_CASE("metrics stay off by default and the bench budget holds") {
    CountSink sink;
    metrics::set_sink(&sink);
    metrics::add("dataloader.batches", 4);
    CHECK(sink.calls == 0);
    metrics::publish("dataloader.batches", 4);
    CHECK(sink.calls == 1);
    CHECK(sink.total == doctest::Approx(4));
    metrics::set_sink(nullptr);
    CHECK_FALSE(metrics::enabled());

    std::ifstream baseline_file;
    for (const char* candidate : {"bench/baseline.json", "../bench/baseline.json"}) {
        baseline_file.open(candidate, std::ios::binary);
        if (baseline_file) break;
        baseline_file.clear();
    }
    REQUIRE(baseline_file);
    std::ostringstream baseline_text;
    baseline_text << baseline_file.rdbuf();
    const json::Value baseline = json::parse(baseline_text.str());
    const double budget = baseline.find("budgets_seconds")->find("function_dataset_20k_gets")->as_double();
    const double slack = baseline.get_double("slack_ratio", 0.5);
    CHECK(bench::within_budget(0.05, budget, slack));
    CHECK_FALSE(bench::within_budget(10.0, budget, slack));

    constexpr std::size_t n = 20000;
    DatasetPtr ds = make_dataset(n, [](std::size_t i) {
        Sample s;
        s.input = NDArray(Shape{1}, DType::Float32);
        s.input.data<float>()[0] = static_cast<float>(i);
        s.label = NDArray(Shape{1}, DType::Int64);
        s.label.data<std::int64_t>()[0] = static_cast<std::int64_t>(i);
        return s;
    });
    const double seconds = bench::time_best([&] {
        float sum = 0;
        for (std::size_t i = 0; i < n; ++i) sum += ds->get(i).input.data<float>()[0];
        if (sum < 0) sink.total += sum;
    }, 1);
    CHECK(bench::within_budget(seconds, budget, slack));
}
