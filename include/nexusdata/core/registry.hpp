#pragma once

#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

#include "nexusdata/core/error.hpp"
#include "nexusdata/dataset/dataset.hpp"

namespace nexusdata {

/// Factory registry for `scheme:path` datasets. Built-in schemes are registered
/// on first use: csv, json, jsonl, npy, text, wav, audio, webdataset, parquet,
/// feather, ipc, hdf5, sqlite. Plugins and user code add more without touching
/// the core. Optional backends still throw their "Reconfigure with
/// -DNEXUSDATA_WITH_X=ON" error when that scheme is opened.
class DataSourceRegistry {
public:
    using Factory = std::function<DatasetPtr(const std::string& uri, const std::string& config_json)>;

    static DataSourceRegistry& instance();

    void register_scheme(std::string scheme, Factory factory);

    /// `scheme:rest` or `scheme://rest`. Throws InvalidArgumentError for an
    /// unknown scheme or a URI with no colon.
    [[nodiscard]] DatasetPtr open(const std::string& uri, const std::string& config_json = "{}") const;

    [[nodiscard]] bool contains(std::string_view scheme) const;

private:
    std::unordered_map<std::string, Factory> schemes_;
};

/// Path portion of a `scheme:rest` / `scheme://rest` URI (the part factories receive
/// is still the full URI; this helper is for factory implementations).
[[nodiscard]] std::string uri_path(const std::string& uri);

/// `DataSourceRegistry::instance().open`. One entry point for built-in schemes
/// and anything a plugin registered.
[[nodiscard]] inline DatasetPtr open(const std::string& uri, const std::string& config_json = "{}") {
    return DataSourceRegistry::instance().open(uri, config_json);
}

} // namespace nexusdata

/// Static registration. `fn` is a DataSourceRegistry::Factory. The name must be
/// an identifier. Runs during dynamic initialization.
#define NEXUSDATA_REGISTER_DATASOURCE(scheme, fn)                                            \
    [[maybe_unused]] static const bool nexusdata_reg_##scheme = [] {                        \
        ::nexusdata::DataSourceRegistry::instance().register_scheme(#scheme, (fn));         \
        return true;                                                                        \
    }()
