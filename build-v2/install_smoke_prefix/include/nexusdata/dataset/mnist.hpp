#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "nexusdata/dataset/dataset.hpp"
#include "nexusdata/io/mapped_file.hpp"

namespace nexusdata {

struct MNISTOptions {
    bool train = true;
    /// If empty, expects standard filenames in root:
    /// train-images-idx3-ubyte, train-labels-idx1-ubyte (or t10k-*).
    std::string images_file;
    std::string labels_file;
};

/// MNIST / Fashion-MNIST IDX reader. Images returned as HWC UInt8 [28,28,1].
/// Zero-copy view over mmap for image bytes; label copied per sample.
class MNIST : public Dataset {
public:
    explicit MNIST(std::string root, MNISTOptions options = {});

    [[nodiscard]] std::size_t size() const override;
    [[nodiscard]] Sample get(std::size_t index) const override;

    [[nodiscard]] int image_rows() const noexcept { return rows_; }
    [[nodiscard]] int image_cols() const noexcept { return cols_; }

private:
    MappedFile images_map_;
    MappedFile labels_map_;
    const std::uint8_t* images_data_ = nullptr;
    const std::uint8_t* labels_data_ = nullptr;
    std::size_t n_ = 0;
    int rows_ = 0;
    int cols_ = 0;
};

} // namespace nexusdata
