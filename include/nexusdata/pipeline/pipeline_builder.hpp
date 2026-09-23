#pragma once

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>

#include "nexusdata/dataset/dataset.hpp"
#include "nexusdata/io/json.hpp"
#include "nexusdata/loading/dataloader.hpp"
#include "nexusdata/pipeline/transform.hpp"
#include "nexusdata/preprocessing/transformer.hpp"

namespace nexusdata {

/// Dataset plus the loader described by a JSON (or the small YAML subset) document.
struct BuiltPipeline {
    DatasetPtr dataset;
    TransformConstPtr transforms;
    std::unique_ptr<DataLoader> loader;
};

/// `source.type` is any scheme in DataSourceRegistry (`csv`, `json`, `jsonl`, ...).
/// `preprocessors[].type` is `standard_scaler`, `minmax_scaler`, or a type
/// registered with TransformerRegistry. Each one is fit on the rows produced by
/// the previous step, then applied per sample.
/// `transforms[].type` is `clip`, `log1p`, `standardize`, or a TransformRegistry
/// type. `loader` maps onto DataLoaderOptions.
class PipelineBuilder {
public:
    [[nodiscard]] static BuiltPipeline from_json(std::string_view text);
    [[nodiscard]] static BuiltPipeline from_file(const std::string& path);
};

/// Custom transform steps referenced from a pipeline document.
class TransformRegistry {
public:
    using Factory = std::function<TransformConstPtr(const json::Value& params)>;

    static TransformRegistry& instance();
    void register_type(std::string type, Factory factory);
    [[nodiscard]] TransformConstPtr create(const std::string& type, const json::Value& params) const;
    [[nodiscard]] bool contains(std::string_view type) const;

private:
    std::unordered_map<std::string, Factory> types_;
};

/// Fit/transform steps referenced from a pipeline `preprocessors` list.
/// Built-ins: `standard_scaler` (with_mean, with_std), `minmax_scaler`
/// (feature_min, feature_max).
class TransformerRegistry {
public:
    using Factory = std::function<std::shared_ptr<Transformer>(const json::Value& params)>;

    static TransformerRegistry& instance();
    void register_type(std::string type, Factory factory);
    [[nodiscard]] std::shared_ptr<Transformer> create(const std::string& type,
                                                      const json::Value& params) const;
    [[nodiscard]] bool contains(std::string_view type) const;

private:
    std::unordered_map<std::string, Factory> types_;
};

} // namespace nexusdata
