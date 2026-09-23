// One hello-world path per built-in reader that works without an optional SDK.
// parquet / hdf5 / sqlite / ffmpeg print the reconfigure message on this build.

#include <fstream>
#include <iostream>
#include <string>

#include "nexusdata/core/error.hpp"
#include "nexusdata/core/registry.hpp"
#include "nexusdata/pipeline/fluent.hpp"
#include "nexusdata/pipeline/pipeline_builder.hpp"
#include "nexusdata/pipeline/tabular.hpp"

using namespace nexusdata;

namespace {

void write_text(const std::string& path, const std::string& body) {
    std::ofstream out(path, std::ios::binary);
    out << body;
}

void show(const std::string& title, const Dataset& ds) {
    const Sample s = ds.get(0);
    std::cout << title << " rows=" << ds.size() << " input_numel=" << s.input.numel() << "\n";
}

void show_optional(const std::string& uri) {
    try {
        (void)open(uri);
        std::cout << uri << " opened\n";
    } catch (const Error& e) {
        std::cout << uri << " -> " << e.what() << "\n";
    }
}

} // namespace

int main() {
    write_text("hello.csv", "x,y\n1.5,0\n2.5,1\n");
    write_text("hello.jsonl", "[1.0, 0]\n[4.0, 1]\n");
    write_text("hello.txt", "alpha\nbeta\n");
    write_text("hello.json", "{\"x\": 1.25}\n{\"x\": 2.25}\n");
    write_text("hello_pipe.json",
               R"({"source":{"type":"csv","path":"hello.csv","options":{"has_header":true,"label_column":"y"}},
"transforms":[{"type":"clip","params":{"min":0,"max":2}}],
"loader":{"batch_size":2,"num_workers":0}})");

    auto chain = from_csv("hello.csv", CSVOptions{"y"}).map(ClipTransform(0.0, 2.0)).filter([](const Sample& s) {
        return s.input.data<float>()[0] > 0.f;
    });
    show("csv chain", *chain.dataset());

    show("jsonl", *from_uri("jsonl:hello.jsonl").dataset());
    show("json", *from_uri("json:hello.json").dataset());
    show("text", *from_uri("text:hello.txt").dataset());

    BuiltPipeline pipe = PipelineBuilder::from_file("hello_pipe.json");
    auto batch = pipe.loader->begin();
    std::cout << "pipeline batch numel=" << (*batch).inputs.numel() << "\n";

    show_optional("parquet:missing.parquet");
    show_optional("hdf5:missing.h5");
    show_optional("sqlite:missing.db");
    return 0;
}
