#include "nexusdata/pipeline/pipeline_builder.hpp"

#include <cctype>
#include <fstream>
#include <sstream>
#include <utility>
#include <vector>

#include "nexusdata/backend/device.hpp"
#include "nexusdata/core/array_utils.hpp"
#include "nexusdata/core/error.hpp"
#include "nexusdata/core/registry.hpp"
#include "nexusdata/dataset/map.hpp"
#include "nexusdata/pipeline/tabular.hpp"
#include "nexusdata/preprocessing/scaler.hpp"
#include "nexusdata/simd/numeric_ops.hpp"

namespace nexusdata {
namespace {

json::Value parse_flow(std::string_view text) {
    text = text.substr(text.find('{') + 1);
    if (text.empty() || text.back() != '}') {
        throw IOError("pipeline YAML: unclosed flow map");
    }
    text.remove_suffix(1);
    json::Value::Object obj;
    std::size_t i = 0;
    while (i < text.size()) {
        while (i < text.size() && (text[i] == ' ' || text[i] == ',')) ++i;
        if (i >= text.size()) break;
        const auto colon = text.find(':', i);
        if (colon == std::string_view::npos) throw IOError("pipeline YAML: flow map needs key: value");
        std::string key(text.substr(i, colon - i));
        while (!key.empty() && key.front() == ' ') key.erase(key.begin());
        while (!key.empty() && key.back() == ' ') key.pop_back();
        i = colon + 1;
        while (i < text.size() && text[i] == ' ') ++i;
        std::string raw;
        if (i < text.size() && text[i] == '"') {
            const auto end = text.find('"', i + 1);
            if (end == std::string_view::npos) throw IOError("pipeline YAML: unclosed string");
            raw = std::string(text.substr(i, end - i + 1));
            i = end + 1;
        } else {
            const auto comma = text.find(',', i);
            raw = std::string(text.substr(i, comma == std::string_view::npos ? text.size() - i : comma - i));
            while (!raw.empty() && raw.back() == ' ') raw.pop_back();
            i = comma == std::string_view::npos ? text.size() : comma + 1;
        }
        if (raw == "true" || raw == "false") {
            obj.emplace_back(std::move(key), json::Value(raw == "true"));
        } else if (!raw.empty() && (std::isdigit(static_cast<unsigned char>(raw[0])) || raw[0] == '-' ||
                                    raw[0] == '"')) {
            obj.emplace_back(std::move(key), json::parse(raw == "true" ? "true" : (raw.front() == '"' ? raw : raw)));
        } else {
            obj.emplace_back(std::move(key), json::Value(raw));
        }
    }
    return json::Value(std::move(obj));
}

json::Value parse_scalar(std::string raw) {
    while (!raw.empty() && raw.front() == ' ') raw.erase(raw.begin());
    while (!raw.empty() && raw.back() == ' ') raw.pop_back();
    if (!raw.empty() && raw.front() == '{') return parse_flow(raw);
    if (raw == "true" || raw == "false" || raw == "null") return json::parse(raw);
    if (!raw.empty() && (raw.front() == '"' || raw.front() == '-' ||
                         std::isdigit(static_cast<unsigned char>(raw.front())))) {
        return json::parse(raw);
    }
    return json::Value(raw);
}

struct YLine {
    int indent = 0;
    std::string text;
};

std::vector<YLine> split_lines(std::string_view text) {
    std::vector<YLine> lines;
    std::size_t i = 0;
    while (i < text.size()) {
        auto eol = text.find('\n', i);
        if (eol == std::string_view::npos) eol = text.size();
        std::string_view line = text.substr(i, eol - i);
        if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
        i = eol < text.size() ? eol + 1 : text.size();
        int indent = 0;
        while (indent < static_cast<int>(line.size()) && line[static_cast<std::size_t>(indent)] == ' ') ++indent;
        std::string_view body = line.substr(static_cast<std::size_t>(indent));
        if (body.empty() || body.front() == '#') continue;
        lines.push_back(YLine{indent, std::string(body)});
    }
    return lines;
}

json::Value parse_block(const std::vector<YLine>& lines, std::size_t& i, int indent);

json::Value parse_block(const std::vector<YLine>& lines, std::size_t& i, int indent) {
    if (i >= lines.size() || lines[i].indent < indent) return json::Value(json::Value::Object{});
    if (lines[i].text.rfind("- ", 0) == 0) {
        json::Value::Array arr;
        while (i < lines.size() && lines[i].indent == indent && lines[i].text.rfind("- ", 0) == 0) {
            std::string rest = lines[i].text.substr(2);
            ++i;
            const auto colon = rest.find(':');
            if (colon == std::string::npos) {
                arr.emplace_back(parse_scalar(rest));
                continue;
            }
            json::Value::Object obj;
            std::string key = rest.substr(0, colon);
            std::string val = rest.substr(colon + 1);
            while (!key.empty() && key.back() == ' ') key.pop_back();
            if (val.find_first_not_of(' ') == std::string::npos) {
                if (i < lines.size() && lines[i].indent > indent) {
                    obj.emplace_back(std::move(key), parse_block(lines, i, lines[i].indent));
                } else {
                    obj.emplace_back(std::move(key), json::Value{});
                }
            } else {
                obj.emplace_back(std::move(key), parse_scalar(val));
            }
            while (i < lines.size() && lines[i].indent > indent && lines[i].text.rfind("- ", 0) != 0) {
                const auto c2 = lines[i].text.find(':');
                if (c2 == std::string::npos) throw IOError("pipeline YAML: expected key");
                std::string k2 = lines[i].text.substr(0, c2);
                std::string v2 = lines[i].text.substr(c2 + 1);
                const int child = lines[i].indent;
                ++i;
                if (v2.find_first_not_of(' ') == std::string::npos && i < lines.size() && lines[i].indent > child) {
                    obj.emplace_back(std::move(k2), parse_block(lines, i, lines[i].indent));
                } else {
                    obj.emplace_back(std::move(k2), parse_scalar(v2));
                }
            }
            arr.emplace_back(json::Value(std::move(obj)));
        }
        return json::Value(std::move(arr));
    }
    json::Value::Object obj;
    while (i < lines.size() && lines[i].indent == indent) {
        const auto colon = lines[i].text.find(':');
        if (colon == std::string::npos) throw IOError("pipeline YAML: expected key: value");
        std::string key = lines[i].text.substr(0, colon);
        std::string val = lines[i].text.substr(colon + 1);
        ++i;
        if (val.find_first_not_of(' ') == std::string::npos) {
            if (i < lines.size() && lines[i].indent > indent) {
                obj.emplace_back(std::move(key), parse_block(lines, i, lines[i].indent));
            } else {
                obj.emplace_back(std::move(key), json::Value{});
            }
        } else {
            obj.emplace_back(std::move(key), parse_scalar(val));
        }
    }
    return json::Value(std::move(obj));
}

json::Value parse_yaml(std::string_view text) {
    auto lines = split_lines(text);
    std::size_t i = 0;
    if (lines.empty()) return json::Value(json::Value::Object{});
    return parse_block(lines, i, lines[0].indent);
}

TransformConstPtr builtin_transform(const std::string& type, const json::Value& params) {
    if (type == "clip") {
        return std::make_shared<ClipTransform>(params.get_double("min", 0), params.get_double("max", 0));
    }
    if (type == "log1p") return std::make_shared<Log1pTransform>();
    if (type == "standardize") {
        const json::Value* mean = params.find("mean");
        const json::Value* scale = params.find("scale");
        if (!mean || !scale) {
            throw InvalidArgumentError("pipeline: standardize needs params.mean and params.scale");
        }
        std::vector<double> m, s;
        for (const json::Value& v : mean->as_array()) m.push_back(v.as_double());
        for (const json::Value& v : scale->as_array()) s.push_back(v.as_double());
        return std::make_shared<StandardizeTransform>(std::move(m), std::move(s));
    }
    return nullptr;
}

class FittedRowTransform : public Transform {
public:
    FittedRowTransform(std::shared_ptr<Transformer> model, std::size_t features)
        : model_(std::move(model)), features_(features) {}

    [[nodiscard]] Sample apply(const Sample& sample) const override {
        if (sample.input.numel() != features_) {
            throw ShapeError("pipeline preprocessor: row has " + std::to_string(sample.input.numel()) +
                             " values, fitted width is " + std::to_string(features_));
        }
        NDArray row(Shape{1, features_}, DType::Float64);
        auto* dst = row.data<double>();
        for (std::size_t j = 0; j < features_; ++j) dst[j] = detail::as_double(sample.input, j);
        NDArray scaled = model_->transform(row);
        NDArray flat = simd::cast(scaled, DType::Float32);
        flat.reshape(Shape{features_});
        return sample.with_input(std::move(flat));
    }

private:
    std::shared_ptr<Transformer> model_;
    std::size_t features_ = 0;
};

NDArray stack_inputs(const Dataset& ds) {
    if (ds.size() == 0) {
        throw InvalidArgumentError("pipeline: preprocessor needs at least one row");
    }
    const std::size_t features = ds.get(0).input.numel();
    if (features == 0) throw InvalidArgumentError("pipeline: preprocessor row is empty");
    NDArray stacked(Shape{ds.size(), features}, DType::Float64);
    auto* dst = stacked.data<double>();
    for (std::size_t i = 0; i < ds.size(); ++i) {
        const NDArray& in = ds.get(i).input;
        if (in.numel() != features) {
            throw ShapeError("pipeline: preprocessor rows have different widths");
        }
        for (std::size_t j = 0; j < features; ++j) {
            dst[i * features + j] = detail::as_double(in, j);
        }
    }
    return stacked;
}

DatasetPtr apply_preprocessors(const json::Value& doc, DatasetPtr ds) {
    const json::Value* list = doc.find("preprocessors");
    if (!list || !list->is_array() || list->as_array().empty()) return ds;
    NDArray matrix = stack_inputs(*ds);
    const std::size_t features = matrix.shape()[1];
    for (const json::Value& step : list->as_array()) {
        const std::string type = step.get_string("type", "");
        const json::Value* params = step.find("params");
        const json::Value empty{json::Value::Object{}};
        std::shared_ptr<Transformer> model =
            TransformerRegistry::instance().create(type, params ? *params : empty);
        model->fit(matrix);
        matrix = model->transform(matrix);
        ds = std::make_shared<MapDataset>(DatasetConstPtr(ds),
                                          std::make_shared<FittedRowTransform>(std::move(model), features));
    }
    return ds;
}

void register_transform_builtins(TransformRegistry& reg) {
    if (reg.contains("clip")) return;
    reg.register_type("clip", [](const json::Value& p) { return builtin_transform("clip", p); });
    reg.register_type("log1p", [](const json::Value& p) { return builtin_transform("log1p", p); });
    reg.register_type("standardize", [](const json::Value& p) { return builtin_transform("standardize", p); });
}

BuiltPipeline build_doc(const json::Value& doc) {
    const json::Value* source = doc.find("source");
    if (!source || !source->is_object()) {
        throw InvalidArgumentError("pipeline: source object is required");
    }
    const std::string type = source->get_string("type", "");
    const std::string path = source->get_string("path", "");
    if (type.empty() || path.empty()) {
        throw InvalidArgumentError("pipeline: source.type and source.path are required");
    }
    std::string cfg = "{}";
    if (const json::Value* o = source->find("options")) cfg = o->dump();
    DatasetPtr ds;
    try {
        ds = apply_preprocessors(doc, DataSourceRegistry::instance().open(type + ":" + path, cfg));
    } catch (const IOError& e) {
        throw IOError(with_context("pipeline source '" + type + "'", e));
    } catch (const InvalidArgumentError& e) {
        throw InvalidArgumentError(with_context("pipeline source '" + type + "'", e));
    } catch (const Error& e) {
        throw Error(with_context("pipeline source '" + type + "'", e));
    }

    std::vector<TransformConstPtr> steps;
    if (const json::Value* list = doc.find("transforms")) {
        for (const json::Value& step : list->as_array()) {
            const std::string t = step.get_string("type", "");
            const json::Value* params = step.find("params");
            const json::Value empty{json::Value::Object{}};
            try {
                steps.push_back(TransformRegistry::instance().create(t, params ? *params : empty));
            } catch (const Error& e) {
                throw InvalidArgumentError(with_context("pipeline transform '" + t + "'", e));
            }
        }
    }
    TransformConstPtr chain;
    DatasetPtr mapped = ds;
    if (!steps.empty()) {
        chain = steps.size() == 1 ? steps[0]
                                  : std::static_pointer_cast<const Transform>(std::make_shared<Compose>(steps));
        mapped = std::make_shared<MapDataset>(DatasetConstPtr(ds), chain);
    }

    DataLoaderOptions opt;
    if (const json::Value* loader = doc.find("loader")) {
        opt.batch_size = static_cast<std::size_t>(loader->get_int("batch_size", 1));
        opt.shuffle = loader->get_bool("shuffle", false);
        opt.num_workers = static_cast<int>(loader->get_int("num_workers", 0));
        opt.drop_last = loader->get_bool("drop_last", false);
        opt.seed = static_cast<std::uint64_t>(loader->get_int("seed", 0));
        opt.prefetch_factor = static_cast<int>(loader->get_int("prefetch_factor", opt.prefetch_factor));
        const std::string device = loader->get_string("device", "cpu");
        if (device == "auto") {
            opt.device = Device::auto_select();
        } else if (device == "cpu" || device == "host") {
            opt.device = Device::host();
        } else {
            throw InvalidArgumentError("pipeline: unknown device '" + device + "'");
        }
    }
    BuiltPipeline built;
    built.dataset = std::move(mapped);
    built.transforms = std::move(chain);
    built.loader = std::make_unique<DataLoader>(DatasetConstPtr(built.dataset), opt);
    return built;
}

} // namespace

TransformRegistry& TransformRegistry::instance() {
    static TransformRegistry reg;
    static const bool once = (register_transform_builtins(reg), true);
    (void)once;
    return reg;
}

void TransformRegistry::register_type(std::string type, Factory factory) {
    if (type.empty() || !factory) {
        throw InvalidArgumentError("TransformRegistry: type and factory are required");
    }
    if (types_.count(type)) {
        throw InvalidArgumentError("TransformRegistry: type '" + type + "' is already registered");
    }
    types_.emplace(std::move(type), std::move(factory));
}

bool TransformRegistry::contains(std::string_view type) const {
    return types_.find(std::string(type)) != types_.end();
}

TransformConstPtr TransformRegistry::create(const std::string& type, const json::Value& params) const {
    const auto it = types_.find(type);
    if (it == types_.end()) {
        throw InvalidArgumentError("pipeline: unknown transform '" + type + "'");
    }
    TransformConstPtr t = it->second(params);
    if (!t) throw InvalidArgumentError("pipeline: transform '" + type + "' produced nothing");
    return t;
}

TransformerRegistry& TransformerRegistry::instance() {
    static TransformerRegistry reg;
    static const bool once = [] {
        reg.register_type("standard_scaler", [](const json::Value& p) {
            return std::make_shared<StandardScaler>(p.get_bool("with_mean", true),
                                                    p.get_bool("with_std", true));
        });
        reg.register_type("minmax_scaler", [](const json::Value& p) {
            return std::make_shared<MinMaxScaler>(p.get_double("feature_min", 0.0),
                                                  p.get_double("feature_max", 1.0));
        });
        return true;
    }();
    (void)once;
    return reg;
}

void TransformerRegistry::register_type(std::string type, Factory factory) {
    if (type.empty() || !factory) {
        throw InvalidArgumentError("TransformerRegistry: type and factory are required");
    }
    if (types_.count(type)) {
        throw InvalidArgumentError("TransformerRegistry: type '" + type + "' is already registered");
    }
    types_.emplace(std::move(type), std::move(factory));
}

bool TransformerRegistry::contains(std::string_view type) const {
    return types_.find(std::string(type)) != types_.end();
}

std::shared_ptr<Transformer> TransformerRegistry::create(const std::string& type,
                                                         const json::Value& params) const {
    const auto it = types_.find(type);
    if (it == types_.end()) {
        throw InvalidArgumentError("pipeline: unknown preprocessor '" + type + "'");
    }
    std::shared_ptr<Transformer> model = it->second(params);
    if (!model) throw InvalidArgumentError("pipeline: preprocessor '" + type + "' produced nothing");
    return model;
}

BuiltPipeline PipelineBuilder::from_json(std::string_view text) {
    return build_doc(json::parse(text));
}

BuiltPipeline PipelineBuilder::from_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw IOError("pipeline: cannot open " + quote_path(path));
    std::ostringstream ss;
    ss << in.rdbuf();
    const std::string text = ss.str();
    std::size_t i = 0;
    while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i]))) ++i;
    if (i < text.size() && text[i] == '{') return from_json(text);
    return build_doc(parse_yaml(text));
}

} // namespace nexusdata
