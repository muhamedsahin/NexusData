#include <string>

#include "nexusdata/core/plugin_loader.hpp"
#include "nexusdata/core/registry.hpp"
#include "nexusdata/dataset/function_dataset.hpp"

#if defined(_WIN32)
#define ND_PLUGIN_API __declspec(dllexport)
#else
#define ND_PLUGIN_API __attribute__((visibility("default")))
#endif

extern "C" ND_PLUGIN_API const char* nexusdata_plugin_abi_version() {
    return nexusdata::kPluginAbiVersion;
}

extern "C" ND_PLUGIN_API void nexusdata_plugin_register(nexusdata::DataSourceRegistry& registry) {
    registry.register_scheme("plug", [](const std::string& uri, const std::string&) {
        const int n = std::stoi(nexusdata::uri_path(uri));
        return nexusdata::make_dataset(static_cast<std::size_t>(n < 0 ? 0 : n), [](std::size_t i) {
            nexusdata::Sample s;
            s.input = nexusdata::NDArray(nexusdata::Shape{1}, nexusdata::DType::Float32);
            s.input.data<float>()[0] = static_cast<float>(i + 1);
            s.label = nexusdata::NDArray(nexusdata::Shape{1}, nexusdata::DType::Int64);
            s.label.data<std::int64_t>()[0] = static_cast<std::int64_t>(i);
            return s;
        });
    });
}
