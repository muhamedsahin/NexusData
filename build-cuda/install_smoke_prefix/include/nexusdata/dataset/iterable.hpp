#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include "nexusdata/core/error.hpp"
#include "nexusdata/core/random.hpp"
#include "nexusdata/dataset/dataset.hpp"
#include "nexusdata/dataset/sample.hpp"

namespace nexusdata {

/// Streaming / iterable dataset: no random access, yields samples sequentially.
/// Thread-safety: iterators are not thread-safe; create one per worker.
class IterableDataset {
public:
    virtual ~IterableDataset() = default;

    class Iterator {
    public:
        virtual ~Iterator() = default;
        [[nodiscard]] virtual bool has_next() const = 0;
        [[nodiscard]] virtual Sample next() = 0;
    };

    [[nodiscard]] virtual std::unique_ptr<Iterator> make_iterator() const = 0;
};

using IterableDatasetPtr = std::shared_ptr<IterableDataset>;
using IterableDatasetConstPtr = std::shared_ptr<const IterableDataset>;

/// Adapt a map-style Dataset into an IterableDataset (sequential 0..n-1).
class IterableFromMap : public IterableDataset {
public:
    explicit IterableFromMap(DatasetConstPtr ds) : ds_(std::move(ds)) {
        if (!ds_) {
            throw InvalidArgumentError("IterableFromMap: null dataset");
        }
    }

    class It : public Iterator {
    public:
        explicit It(DatasetConstPtr ds) : ds_(std::move(ds)), n_(ds_->size()) {}
        [[nodiscard]] bool has_next() const override { return i_ < n_; }
        [[nodiscard]] Sample next() override {
            if (!has_next()) {
                throw IndexError("IterableFromMap: exhausted");
            }
            return ds_->get(i_++);
        }

    private:
        DatasetConstPtr ds_;
        std::size_t n_ = 0;
        std::size_t i_ = 0;
    };

    [[nodiscard]] std::unique_ptr<Iterator> make_iterator() const override {
        return std::make_unique<It>(ds_);
    }

private:
    DatasetConstPtr ds_;
};

/// Shuffle buffer for streaming (reservoir-like window shuffle).
class ShuffleBufferIterable : public IterableDataset {
public:
    ShuffleBufferIterable(IterableDatasetConstPtr inner, std::size_t buf_size, std::uint64_t seed)
        : inner_(std::move(inner)), buf_size_(buf_size), seed_(seed) {
        if (!inner_) {
            throw InvalidArgumentError("ShuffleBufferIterable: null");
        }
        if (buf_size_ == 0) {
            throw InvalidArgumentError("ShuffleBufferIterable: buf_size > 0 required");
        }
    }

    class It : public Iterator {
    public:
        It(IterableDatasetConstPtr inner, std::size_t buf_size, std::uint64_t seed)
            : it_(inner->make_iterator()), buf_size_(buf_size), rng_(seed) {
            fill();
        }
        [[nodiscard]] bool has_next() const override { return !buf_.empty() || it_->has_next(); }
        [[nodiscard]] Sample next() override {
            if (!has_next()) {
                throw IndexError("ShuffleBufferIterable: exhausted");
            }
            if (buf_.empty()) {
                fill();
            }
            const std::size_t j = rng_.next_bounded(static_cast<std::uint32_t>(buf_.size()));
            Sample out = std::move(buf_[j]);
            buf_[j] = std::move(buf_.back());
            buf_.pop_back();
            fill();
            return out;
        }

    private:
        void fill() {
            while (buf_.size() < buf_size_ && it_->has_next()) {
                buf_.push_back(it_->next());
            }
        }
        std::unique_ptr<Iterator> it_;
        std::size_t buf_size_ = 0;
        PCG32 rng_;
        std::vector<Sample> buf_;
    };

    [[nodiscard]] std::unique_ptr<Iterator> make_iterator() const override {
        return std::make_unique<It>(inner_, buf_size_, seed_);
    }

private:
    IterableDatasetConstPtr inner_;
    std::size_t buf_size_ = 0;
    std::uint64_t seed_ = 0;
};

/// Concatenate IterableDatasets sequentially.
class ChainIterable : public IterableDataset {
public:
    explicit ChainIterable(std::vector<IterableDatasetConstPtr> parts) : parts_(std::move(parts)) {
        if (parts_.empty()) {
            throw InvalidArgumentError("ChainIterable: empty");
        }
        for (const auto& p : parts_) {
            if (!p) {
                throw InvalidArgumentError("ChainIterable: null part");
            }
        }
    }

    class It : public Iterator {
    public:
        explicit It(std::vector<IterableDatasetConstPtr> parts) : parts_(std::move(parts)) {
            advance();
        }
        [[nodiscard]] bool has_next() const override {
            return cur_ && cur_->has_next();
        }
        [[nodiscard]] Sample next() override {
            if (!has_next()) {
                throw IndexError("ChainIterable: exhausted");
            }
            Sample s = cur_->next();
            if (!cur_->has_next()) {
                advance();
            }
            return s;
        }

    private:
        void advance() {
            while (idx_ < parts_.size()) {
                cur_ = parts_[idx_++]->make_iterator();
                if (cur_->has_next()) {
                    return;
                }
            }
            cur_.reset();
        }
        std::vector<IterableDatasetConstPtr> parts_;
        std::size_t idx_ = 0;
        std::unique_ptr<Iterator> cur_;
    };

    [[nodiscard]] std::unique_ptr<Iterator> make_iterator() const override {
        return std::make_unique<It>(parts_);
    }

private:
    std::vector<IterableDatasetConstPtr> parts_;
};

} // namespace nexusdata
