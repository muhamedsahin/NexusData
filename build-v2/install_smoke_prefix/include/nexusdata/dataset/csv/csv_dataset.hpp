#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "nexusdata/core/dtype.hpp"
#include "nexusdata/core/ndarray.hpp"
#include "nexusdata/dataset/dataset.hpp"
#include "nexusdata/dataset/in_memory.hpp"

namespace nexusdata {

enum class CSVErrorPolicy {
    Throw, // throw IOError with file:line:col context
    Skip,  // skip bad rows
};

struct CSVOptions {
    std::string label_column;          // empty => last column is label
    std::optional<std::size_t> label_column_index;
    char delimiter = ',';
    bool has_header = true;
    bool skip_blank_lines = true;
    char comment_prefix = '\0';       // e.g. '#'; '\0' disables
    DType feature_dtype = DType::Float32;
    DType label_dtype = DType::Float32;
    CSVErrorPolicy error_policy = CSVErrorPolicy::Throw;
    std::vector<std::string> na_values{"", "NA", "NaN", "null", "NULL"};
    /// Memory-map the file (recommended for large CSVs).
    bool use_mmap = true;
    /// Parallel record parse threads (0 = hardware concurrency when mmap path used).
    int num_threads = 0;
};

/// Tabular CSV/TSV reader loaded fully into memory for v0.1.
/// Supports: header, custom delimiter, RFC4180 quotes (`""` escape),
/// CRLF/LF, UTF-8 BOM, label by name or index, contextual parse errors.
///
/// Thread-safety: read-only after construction; concurrent get() is safe.
class CSVDataset : public Dataset {
public:
    explicit CSVDataset(const std::string& path, CSVOptions options = {});

    [[nodiscard]] std::size_t size() const override;
    [[nodiscard]] Sample get(std::size_t index) const override;

    [[nodiscard]] const std::vector<std::string>& column_names() const noexcept {
        return column_names_;
    }
    [[nodiscard]] std::size_t num_features() const noexcept { return n_features_; }
    [[nodiscard]] const NDArray& features() const noexcept { return impl_.features(); }
    [[nodiscard]] const NDArray& labels() const noexcept { return impl_.labels(); }

private:
    InMemoryDataset impl_;
    std::vector<std::string> column_names_;
    std::size_t n_features_ = 0;
};

} // namespace nexusdata
