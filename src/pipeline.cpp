#include "nexusdata/pipeline/persist.hpp"

#include <sstream>

#include "nexusdata/preprocessing/encoder.hpp"
#include "nexusdata/preprocessing/imputer.hpp"
#include "nexusdata/preprocessing/scaler.hpp"

namespace nexusdata {

std::shared_ptr<Transformer> make_transformer_by_type(const std::string& type) {
    if (type == "StandardScaler") {
        return std::make_shared<StandardScaler>();
    }
    if (type == "MinMaxScaler") {
        return std::make_shared<MinMaxScaler>();
    }
    if (type == "RobustScaler") {
        return std::make_shared<RobustScaler>();
    }
    if (type == "MaxAbsScaler") {
        return std::make_shared<MaxAbsScaler>();
    }
    if (type == "LabelEncoder") {
        return std::make_shared<LabelEncoder>();
    }
    if (type == "OneHotEncoder") {
        return std::make_shared<OneHotEncoder>();
    }
    if (type == "OrdinalEncoder") {
        return std::make_shared<OrdinalEncoder>();
    }
    if (type == "Imputer") {
        return std::make_shared<Imputer>();
    }
    if (type == "Binarizer") {
        return std::make_shared<Binarizer>();
    }
    if (type == "Normalizer") {
        return std::make_shared<Normalizer>();
    }
    return nullptr;
}

void Pipeline::rebuild_compose_from_steps() {
    std::vector<TransformConstPtr> steps;
    for (const auto& s : compose_steps_) {
        std::istringstream iss(s);
        std::string kind;
        iss >> kind;
        if (kind == "Clip") {
            double a = 0, b = 0;
            iss >> a >> b;
            steps.push_back(std::make_shared<ClipTransform>(a, b));
        } else if (kind == "Log1p") {
            steps.push_back(std::make_shared<Log1pTransform>());
        } else if (kind == "Standardize") {
            std::size_t f = 0;
            iss >> f;
            std::vector<double> mean(f), scale(f);
            for (std::size_t i = 0; i < f; ++i) {
                iss >> mean[i];
            }
            for (std::size_t i = 0; i < f; ++i) {
                iss >> scale[i];
            }
            steps.push_back(
                std::make_shared<StandardizeTransform>(std::move(mean), std::move(scale)));
        } else if (kind == "MinMax") {
            std::size_t f = 0;
            double fmin = 0, fmax = 1;
            iss >> f >> fmin >> fmax;
            std::vector<double> dmin(f), dmax(f);
            for (std::size_t i = 0; i < f; ++i) {
                iss >> dmin[i];
            }
            for (std::size_t i = 0; i < f; ++i) {
                iss >> dmax[i];
            }
            steps.push_back(std::make_shared<MinMaxTransform>(std::move(dmin), std::move(dmax),
                                                              fmin, fmax));
        } else {
            throw IOError("Pipeline: unknown compose step \"" + kind + "\"");
        }
    }
    if (steps.empty()) {
        compose_.reset();
    } else {
        compose_ = std::make_shared<Compose>(std::move(steps));
    }
}

void Pipeline::save(std::ostream& out) const {
    out << "NEXUSDATA_PIPELINE 1\n";
    out << "TRANSFORMERS " << transformers_.size() << "\n";
    for (const auto& t : transformers_) {
        if (!t) {
            throw InvalidArgumentError("Pipeline::save: null transformer");
        }
        t->save(out);
        out << "END_TRANSFORMER\n";
    }
    out << "COMPOSE " << compose_steps_.size() << "\n";
    for (const auto& step : compose_steps_) {
        out << step << "\n";
    }
}

void Pipeline::load(std::istream& in,
                    const std::function<std::shared_ptr<Transformer>(const std::string&)>& factory) {
    std::string tag;
    int ver = 0;
    in >> tag >> ver;
    if (tag != "NEXUSDATA_PIPELINE" || ver != 1) {
        throw IOError("Pipeline::load: bad header");
    }
    std::string key;
    std::size_t n = 0;
    in >> key >> n;
    if (key != "TRANSFORMERS") {
        throw IOError("Pipeline::load: expected TRANSFORMERS");
    }
    transformers_.clear();
    std::string line;
    std::getline(in, line); // consume EOL after TRANSFORMERS n
    for (std::size_t i = 0; i < n; ++i) {
        std::ostringstream block;
        while (std::getline(in, line)) {
            if (line == "END_TRANSFORMER") {
                break;
            }
            block << line << "\n";
        }
        std::istringstream peek(block.str());
        std::string ptag;
        int pver = 0;
        peek >> ptag >> pver;
        std::string type_key, type_name;
        peek >> type_key >> type_name;
        if (ptag != "NEXUSDATA_PREPROCESSOR" || type_key != "TYPE") {
            throw IOError("Pipeline::load: malformed transformer block");
        }
        auto t = factory(type_name);
        if (!t) {
            throw IOError("Pipeline::load: unknown type " + type_name);
        }
        std::istringstream full(block.str());
        t->load(full);
        transformers_.push_back(std::move(t));
    }
    in >> key >> n;
    if (key != "COMPOSE") {
        throw IOError("Pipeline::load: expected COMPOSE");
    }
    compose_steps_.clear();
    std::getline(in, line);
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::getline(in, line)) {
            throw IOError("Pipeline::load: truncated compose steps");
        }
        compose_steps_.push_back(line);
    }
    rebuild_compose_from_steps();
}

} // namespace nexusdata
