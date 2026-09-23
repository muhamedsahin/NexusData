#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "nexusdata/dataset/dataset.hpp"
#include "nexusdata/image/image.hpp"

namespace nexusdata {

struct ImageFolderOptions {
    ImageDecodeOptions decode;
    /// File extensions to accept (lowercase, with dot), empty = defaults.
    std::vector<std::string> extensions{};
    bool recursive = false; // only one level of class dirs by default
};

/// Directory layout: root/class_name/*.jpg — class folder name = label id (sorted).
/// Lazy decode on get(i). Corrupt files: skipped at scan if policy Skip, else throw on get.
/// Thread-safety: concurrent get() OK if filesystem reads are OK (decode allocates per call).
class ImageFolder : public Dataset {
public:
    explicit ImageFolder(std::string root, ImageFolderOptions options = {});

    [[nodiscard]] std::size_t size() const override;
    [[nodiscard]] Sample get(std::size_t index) const override;

    [[nodiscard]] const std::vector<std::string>& classes() const noexcept { return classes_; }
    [[nodiscard]] const std::string& root() const noexcept { return root_; }

private:
    struct Entry {
        std::string path;
        std::int64_t label = 0;
    };

    std::string root_;
    ImageFolderOptions options_;
    std::vector<std::string> classes_;
    std::vector<Entry> entries_;
};

} // namespace nexusdata
