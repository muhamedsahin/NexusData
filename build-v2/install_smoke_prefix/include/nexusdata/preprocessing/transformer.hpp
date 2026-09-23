#pragma once

#include <fstream>
#include <iostream>
#include <string>

#include "nexusdata/core/error.hpp"
#include "nexusdata/core/ndarray.hpp"

namespace nexusdata {

/// Sklearn-style fit / transform API for tabular NDArray [N, F].
/// Thread-safety: fit() not thread-safe; transform() of a fitted immutable
/// instance is safe for concurrent reads.
class Transformer {
public:
    virtual ~Transformer() = default;

    virtual void fit(const NDArray& X) = 0;
    [[nodiscard]] virtual NDArray transform(const NDArray& X) const = 0;

    NDArray fit_transform(const NDArray& X) {
        fit(X);
        return transform(X);
    }

    [[nodiscard]] virtual bool is_fitted() const = 0;
    [[nodiscard]] virtual std::string name() const = 0;

    /// Persist fitted state (text format, versioned).
    virtual void save(std::ostream& out) const = 0;
    virtual void load(std::istream& in) = 0;

    void save_file(const std::string& path) const {
        std::ofstream out(path);
        if (!out) {
            throw IOError("Transformer::save_file: cannot write \"" + path + "\"");
        }
        save(out);
    }

    void load_file(const std::string& path) {
        std::ifstream in(path);
        if (!in) {
            throw IOError("Transformer::load_file: cannot read \"" + path + "\"");
        }
        load(in);
    }
};

} // namespace nexusdata
