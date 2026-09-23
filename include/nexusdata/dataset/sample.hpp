#pragma once

#include <algorithm>
#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "nexusdata/core/error.hpp"
#include "nexusdata/core/ndarray.hpp"

namespace nexusdata {

/// Small insertion-ordered string-keyed map for per-sample / per-batch fields.
/// Vector-backed: an empty map owns no heap memory on any standard library
/// (MSVC's std::unordered_map allocates even when empty), and linear lookup over
/// the handful of keys a sample carries beats hashing.
template <class V>
class FieldMap {
public:
    using value_type = std::pair<std::string, V>;
    using iterator = typename std::vector<value_type>::iterator;
    using const_iterator = typename std::vector<value_type>::const_iterator;

    FieldMap() = default;

    [[nodiscard]] bool empty() const noexcept { return items_.empty(); }
    [[nodiscard]] std::size_t size() const noexcept { return items_.size(); }
    void clear() noexcept { items_.clear(); }
    void reserve(std::size_t n) { items_.reserve(n); }

    [[nodiscard]] iterator begin() noexcept { return items_.begin(); }
    [[nodiscard]] iterator end() noexcept { return items_.end(); }
    [[nodiscard]] const_iterator begin() const noexcept { return items_.begin(); }
    [[nodiscard]] const_iterator end() const noexcept { return items_.end(); }

    [[nodiscard]] V* find(std::string_view key) noexcept {
        for (auto& kv : items_) {
            if (kv.first == key) return &kv.second;
        }
        return nullptr;
    }
    [[nodiscard]] const V* find(std::string_view key) const noexcept {
        for (const auto& kv : items_) {
            if (kv.first == key) return &kv.second;
        }
        return nullptr;
    }
    [[nodiscard]] bool contains(std::string_view key) const noexcept { return find(key) != nullptr; }

    /// Throws IndexError when @p key is absent.
    [[nodiscard]] const V& at(std::string_view key) const {
        if (const V* v = find(key)) return *v;
        throw IndexError("FieldMap: no field '" + std::string(key) + "'");
    }
    [[nodiscard]] V& at(std::string_view key) {
        if (V* v = find(key)) return *v;
        throw IndexError("FieldMap: no field '" + std::string(key) + "'");
    }

    /// Inserts a default value when @p key is absent.
    V& operator[](std::string_view key) {
        if (V* v = find(key)) return *v;
        items_.emplace_back(std::string(key), V{});
        return items_.back().second;
    }

    /// Insert or overwrite.
    V& set(std::string key, V value) {
        if (V* v = find(key)) {
            *v = std::move(value);
            return *v;
        }
        items_.emplace_back(std::move(key), std::move(value));
        return items_.back().second;
    }

    bool erase(std::string_view key) {
        const auto it = std::find_if(items_.begin(), items_.end(),
                                     [&](const value_type& kv) { return kv.first == key; });
        if (it == items_.end()) return false;
        items_.erase(it);
        return true;
    }

private:
    std::vector<value_type> items_;
};

/// Single sample.
/// v0.1: input + label. v2.0: optional named tensors (e.g. "bbox", "mask",
/// "attention_mask") and string metadata (e.g. source path / id); both stay
/// allocation-free while empty. Collate functions batch `extra_tensors` by key.
/// Thread-safety: same as NDArray (shared storage).
struct Sample {
    NDArray input;
    NDArray label;
    // Default member initializers keep `Sample{input, label}` free of
    // -Wmissing-field-initializers in client code built with -Wextra.
    FieldMap<NDArray> extra_tensors{};
    FieldMap<std::string> metadata{};

    /// Copy of this sample with a new input: transforms that replace the input
    /// use this so label, extra tensors and metadata travel along.
    [[nodiscard]] Sample with_input(NDArray new_input) const {
        Sample s;
        s.input = std::move(new_input);
        s.label = label;
        s.extra_tensors = extra_tensors;
        s.metadata = metadata;
        return s;
    }
};

} // namespace nexusdata
