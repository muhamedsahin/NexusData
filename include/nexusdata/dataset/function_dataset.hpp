#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "nexusdata/core/error.hpp"
#include "nexusdata/dataset/dataset.hpp"
#include "nexusdata/dataset/iterable.hpp"

namespace nexusdata {

/// Map-style dataset backed by a lambda. No subclass required.
class FunctionDataset : public Dataset {
public:
    using GetFn = std::function<Sample(std::size_t)>;

    FunctionDataset(std::size_t n, GetFn fn) : n_(n), fn_(std::move(fn)) {
        if (!fn_) {
            throw InvalidArgumentError("FunctionDataset: null getter");
        }
    }

    [[nodiscard]] std::size_t size() const override { return n_; }

    [[nodiscard]] Sample get(std::size_t index) const override {
        if (index >= n_) {
            throw IndexError(format_index_error("FunctionDataset::get", index, n_));
        }
        return fn_(index);
    }

private:
    std::size_t n_ = 0;
    GetFn fn_;
};

[[nodiscard]] inline DatasetPtr make_dataset(std::size_t n, FunctionDataset::GetFn fn) {
    return std::make_shared<FunctionDataset>(n, std::move(fn));
}

/// One Sample per element. `T` must have `to_sample(const T&)`, normally from
/// `NEXUSDATA_DESCRIBE_STRUCT`. Arithmetic fields land in extra_tensors, strings
/// in metadata.
template <class T>
[[nodiscard]] DatasetPtr records_dataset(const std::vector<T>& rows) {
    return make_dataset(rows.size(), [rows](std::size_t i) { return to_sample(rows[i]); });
}

/// Streaming dataset. Each iterator calls @p factory once to get a fresh
/// `std::function<std::optional<Sample>()>`. nullopt ends the stream.
class GeneratorIterable : public IterableDataset {
public:
    using NextFn = std::function<std::optional<Sample>()>;
    using Factory = std::function<NextFn()>;

    explicit GeneratorIterable(Factory factory) : factory_(std::move(factory)) {
        if (!factory_) {
            throw InvalidArgumentError("GeneratorIterable: null factory");
        }
    }

    class It : public Iterator {
    public:
        explicit It(NextFn next) : next_(std::move(next)) { pull(); }
        [[nodiscard]] bool has_next() const override { return has_; }
        [[nodiscard]] Sample next() override {
            if (!has_) throw IndexError("GeneratorIterable: exhausted");
            Sample s = std::move(pending_);
            pull();
            return s;
        }

    private:
        void pull() {
            if (!next_) {
                has_ = false;
                return;
            }
            std::optional<Sample> s = next_();
            has_ = s.has_value();
            if (has_) pending_ = std::move(*s);
        }
        NextFn next_;
        Sample pending_{};
        bool has_ = false;
    };

    [[nodiscard]] std::unique_ptr<Iterator> make_iterator() const override {
        NextFn fn = factory_();
        if (!fn) throw InvalidArgumentError("GeneratorIterable: factory returned an empty function");
        return std::make_unique<It>(std::move(fn));
    }

private:
    Factory factory_;
};

} // namespace nexusdata
