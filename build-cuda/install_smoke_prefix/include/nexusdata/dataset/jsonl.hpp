#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "nexusdata/dataset/dataset.hpp"
#include "nexusdata/dataset/in_memory.hpp"
#include "nexusdata/io/mapped_file.hpp"

namespace nexusdata {

struct JSONLOptions {
    /// If true, each line is a JSON array: [f1, f2, ..., label]
    /// Nested objects are not supported in v0.4.
    bool array_with_label = true;
    DType feature_dtype = DType::Float32;
    DType label_dtype = DType::Float32;
    int num_threads = 0; // 0 => hardware concurrency
};

/// JSONL / NDJSON reader for numeric JSON arrays (one sample per line).
/// Uses mmap + newline SIMD/scalar scan + parallel line parse.
class JSONLDataset : public Dataset {
public:
    explicit JSONLDataset(const std::string& path, JSONLOptions options = {});

    [[nodiscard]] std::size_t size() const override;
    [[nodiscard]] Sample get(std::size_t index) const override;

    [[nodiscard]] std::size_t num_features() const noexcept { return n_features_; }

private:
    InMemoryDataset impl_;
    std::size_t n_features_ = 0;
};

} // namespace nexusdata
