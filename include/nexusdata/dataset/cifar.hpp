#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "nexusdata/dataset/dataset.hpp"
#include "nexusdata/io/mapped_file.hpp"

namespace nexusdata {

struct CIFAROptions {
    bool cifar100 = false;
    /// Batch files relative to root; empty => standard names.
    std::vector<std::string> batch_files;
};

/// CIFAR-10 / CIFAR-100 binary reader. Images as HWC UInt8 [32,32,3].
class CIFAR : public Dataset {
public:
    explicit CIFAR(std::string root, CIFAROptions options = {});

    [[nodiscard]] std::size_t size() const override;
    [[nodiscard]] Sample get(std::size_t index) const override;

private:
    struct Batch {
        MappedFile map;
        const std::uint8_t* data = nullptr;
        std::size_t count = 0;
        std::size_t record_size = 0;
    };

    std::vector<Batch> batches_;
    std::size_t n_ = 0;
    bool cifar100_ = false;
    std::size_t label_bytes_ = 1;
};

} // namespace nexusdata
