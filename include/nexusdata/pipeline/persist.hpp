#pragma once

#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "nexusdata/core/error.hpp"
#include "nexusdata/pipeline/tabular.hpp"
#include "nexusdata/pipeline/transform.hpp"
#include "nexusdata/preprocessing/transformer.hpp"

namespace nexusdata {

[[nodiscard]] std::shared_ptr<Transformer> make_transformer_by_type(const std::string& type);

/// Versioned pipeline snapshot: fitted Transformers + serializable tabular Compose steps.
/// Text format tag: NEXUSDATA_PIPELINE 1
class Pipeline {
public:
    void add_transformer(std::shared_ptr<Transformer> t) {
        if (!t) {
            throw InvalidArgumentError("Pipeline::add_transformer: null");
        }
        transformers_.push_back(std::move(t));
    }

    void set_compose(TransformConstPtr compose) { compose_ = std::move(compose); }

    /// Register a serializable compose step (e.g. "Clip -1 1", "Log1p").
    void add_compose_step(std::string step) { compose_steps_.push_back(std::move(step)); }

    [[nodiscard]] const std::vector<std::shared_ptr<Transformer>>& transformers() const {
        return transformers_;
    }
    [[nodiscard]] TransformConstPtr compose() const { return compose_; }
    [[nodiscard]] const std::vector<std::string>& compose_steps() const { return compose_steps_; }

    [[nodiscard]] NDArray transform_matrix(const NDArray& X) const {
        NDArray cur = X;
        for (const auto& t : transformers_) {
            if (!t || !t->is_fitted()) {
                throw InvalidArgumentError("Pipeline: transformer not fitted");
            }
            cur = t->transform(cur);
        }
        return cur;
    }

    [[nodiscard]] Sample apply(const Sample& sample) const {
        if (!compose_) {
            return sample;
        }
        return compose_->apply(sample);
    }

    void save(std::ostream& out) const;
    void load(std::istream& in,
              const std::function<std::shared_ptr<Transformer>(const std::string&)>& factory =
                  make_transformer_by_type);

    void save_file(const std::string& path) const {
        std::ofstream out(path);
        if (!out) {
            throw IOError("Pipeline::save_file: cannot write \"" + path + "\"");
        }
        save(out);
    }

    void load_file(const std::string& path,
                   const std::function<std::shared_ptr<Transformer>(const std::string&)>& factory =
                       make_transformer_by_type) {
        std::ifstream in(path);
        if (!in) {
            throw IOError("Pipeline::load_file: cannot read \"" + path + "\"");
        }
        load(in, factory);
    }

    void rebuild_compose_from_steps();

private:
    std::vector<std::shared_ptr<Transformer>> transformers_;
    TransformConstPtr compose_;
    std::vector<std::string> compose_steps_;
};

} // namespace nexusdata
