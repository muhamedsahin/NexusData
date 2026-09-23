#include "nexusdata/core/registry.hpp"

#include "nexusdata/dataset/arrow_ipc.hpp"
#include "nexusdata/dataset/audio.hpp"
#include "nexusdata/dataset/csv/csv_dataset.hpp"
#include "nexusdata/dataset/hdf5.hpp"
#include "nexusdata/dataset/json.hpp"
#include "nexusdata/dataset/jsonl.hpp"
#include "nexusdata/dataset/npy.hpp"
#include "nexusdata/dataset/parquet.hpp"
#include "nexusdata/dataset/sqlite.hpp"
#include "nexusdata/dataset/text.hpp"
#include "nexusdata/dataset/webdataset.hpp"
#include "nexusdata/io/json.hpp"

namespace nexusdata {
namespace {

json::Value read_cfg(const std::string& config_json) {
    if (config_json.empty() || config_json == "{}") return json::Value{json::Value::Object{}};
    json::Value cfg = json::parse(config_json);
    if (!cfg.is_object()) {
        throw InvalidArgumentError("DataSourceRegistry: config JSON must be an object");
    }
    return cfg;
}

void register_builtins(DataSourceRegistry& reg) {
    if (reg.contains("csv")) return;

    reg.register_scheme("csv", [](const std::string& uri, const std::string& config_json) {
        const json::Value cfg = read_cfg(config_json);
        CSVOptions opt;
        opt.has_header = cfg.get_bool("has_header", true);
        opt.label_column = cfg.get_string("label_column", "");
        const std::string delim = cfg.get_string("delimiter", ",");
        if (!delim.empty()) opt.delimiter = delim[0];
        return std::make_shared<CSVDataset>(uri_path(uri), std::move(opt));
    });

    reg.register_scheme("json", [](const std::string& uri, const std::string& config_json) {
        const json::Value cfg = read_cfg(config_json);
        JsonOptions opt;
        const std::string format = cfg.get_string("format", "auto");
        if (format == "lines") {
            opt.format = JsonFormat::Lines;
        } else if (format == "array") {
            opt.format = JsonFormat::Array;
        } else if (format == "auto") {
            opt.format = JsonFormat::Auto;
        } else {
            throw InvalidArgumentError("json scheme: format must be auto, lines, or array");
        }
        opt.records_path = cfg.get_string("records_path", "");
        opt.skip_invalid = cfg.get_bool("skip_invalid", false);
        if (const json::Value* fields = cfg.find("fields")) {
            for (const json::Value& f : fields->as_array()) {
                JsonField field;
                field.path = f.get_string("path", "");
                if (field.path.empty()) {
                    throw InvalidArgumentError("json scheme: field.path is required");
                }
                const std::string role = f.get_string("role", "input");
                if (role == "label") {
                    field.role = JsonRole::Label;
                } else if (role == "extra") {
                    field.role = JsonRole::Extra;
                } else if (role == "metadata") {
                    field.role = JsonRole::Metadata;
                } else if (role == "input") {
                    field.role = JsonRole::Input;
                } else {
                    throw InvalidArgumentError("json scheme: unknown field role '" + role + "'");
                }
                field.name = f.get_string("name", "");
                opt.fields.push_back(std::move(field));
            }
        }
        return std::make_shared<JsonDataset>(uri_path(uri), std::move(opt));
    });

    reg.register_scheme("jsonl", [](const std::string& uri, const std::string& config_json) {
        const json::Value cfg = read_cfg(config_json);
        JSONLOptions opt;
        opt.array_with_label = cfg.get_bool("array_with_label", true);
        return std::make_shared<JSONLDataset>(uri_path(uri), opt);
    });

    reg.register_scheme("npy", [](const std::string& uri, const std::string& config_json) {
        const json::Value cfg = read_cfg(config_json);
        return std::make_shared<NpyDataset>(uri_path(uri), cfg.get_string("labels", ""));
    });

    reg.register_scheme("text", [](const std::string& uri, const std::string& config_json) {
        const json::Value cfg = read_cfg(config_json);
        TextOptions opt;
        opt.max_length = static_cast<std::size_t>(cfg.get_int("max_length", 0));
        opt.skip_empty = cfg.get_bool("skip_empty", true);
        return std::make_shared<TextLineDataset>(uri_path(uri), opt);
    });

    reg.register_scheme("wav", [](const std::string& uri, const std::string& config_json) {
        const json::Value cfg = read_cfg(config_json);
        WavOptions opt;
        opt.target_frames = static_cast<std::size_t>(cfg.get_int("target_frames", 0));
        return std::make_shared<WavDataset>(uri_path(uri), opt);
    });

    reg.register_scheme("audio", [](const std::string& uri, const std::string& config_json) {
        const json::Value cfg = read_cfg(config_json);
        AudioOptions opt;
        opt.target_frames = static_cast<std::size_t>(cfg.get_int("target_frames", 0));
        return std::make_shared<AudioDataset>(uri_path(uri), opt);
    });

    reg.register_scheme("webdataset", [](const std::string& uri, const std::string&) {
        return std::make_shared<WebDataset>(uri_path(uri));
    });

    reg.register_scheme("parquet", [](const std::string& uri, const std::string& config_json) {
        const json::Value cfg = read_cfg(config_json);
        ParquetOptions opt;
        opt.columns = cfg.get_string("columns", "");
        opt.label = cfg.get_string("label", "");
        return std::make_shared<ParquetDataset>(uri_path(uri), std::move(opt));
    });

    auto arrow_ipc = [](const std::string& uri, const std::string& config_json) {
        const json::Value cfg = read_cfg(config_json);
        ArrowIpcOptions opt;
        opt.columns = cfg.get_string("columns", "");
        opt.label = cfg.get_string("label", "");
        return std::make_shared<ArrowIpcDataset>(uri_path(uri), std::move(opt));
    };
    reg.register_scheme("feather", arrow_ipc);
    reg.register_scheme("ipc", arrow_ipc);

    reg.register_scheme("hdf5", [](const std::string& uri, const std::string& config_json) {
        const json::Value cfg = read_cfg(config_json);
        Hdf5Options opt;
        opt.dataset_path = cfg.get_string("dataset_path", "/data");
        opt.label_path = cfg.get_string("label_path", "");
        return std::make_shared<Hdf5Dataset>(uri_path(uri), std::move(opt));
    });

    reg.register_scheme("sqlite", [](const std::string& uri, const std::string& config_json) {
        const json::Value cfg = read_cfg(config_json);
        SqliteOptions opt;
        opt.sql = cfg.get_string("sql", "");
        opt.label_column = static_cast<int>(cfg.get_int("label_column", -1));
        return std::make_shared<SqliteDataset>(uri_path(uri), std::move(opt));
    });
}

} // namespace

DataSourceRegistry& DataSourceRegistry::instance() {
    static DataSourceRegistry reg;
    static const bool once = (register_builtins(reg), true);
    (void)once;
    return reg;
}

void DataSourceRegistry::register_scheme(std::string scheme, Factory factory) {
    if (scheme.empty() || !factory) {
        throw InvalidArgumentError("DataSourceRegistry: scheme and factory are required");
    }
    if (schemes_.count(scheme)) {
        throw InvalidArgumentError("DataSourceRegistry: scheme '" + scheme + "' is already registered");
    }
    schemes_.emplace(std::move(scheme), std::move(factory));
}

bool DataSourceRegistry::contains(std::string_view scheme) const {
    return schemes_.find(std::string(scheme)) != schemes_.end();
}

DatasetPtr DataSourceRegistry::open(const std::string& uri, const std::string& config_json) const {
    const auto colon = uri.find(':');
    if (colon == std::string::npos || colon == 0) {
        throw InvalidArgumentError("DataSourceRegistry::open: URI needs a scheme (scheme:path)");
    }
    const std::string scheme = uri.substr(0, colon);
    const auto it = schemes_.find(scheme);
    if (it == schemes_.end()) {
        throw InvalidArgumentError("DataSourceRegistry::open: unknown scheme '" + scheme + "'");
    }
    return it->second(uri, config_json);
}

std::string uri_path(const std::string& uri) {
    const auto colon = uri.find(':');
    if (colon == std::string::npos) {
        throw InvalidArgumentError("uri_path: URI needs a scheme");
    }
    std::string rest = uri.substr(colon + 1);
    if (rest.rfind("//", 0) == 0) rest.erase(0, 2);
    return rest;
}

} // namespace nexusdata
