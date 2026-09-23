#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "nexusdata/dataset/dataset.hpp"
#include "nexusdata/image/image.hpp"
#include "nexusdata/io/mapped_file.hpp"

namespace nexusdata {

struct WebDatasetOptions {
    ImageDecodeOptions image_decode{};
    /// Preferred image suffix when multiple image-like members exist (informational).
    std::string image_suffix = ".jpg";
};

/// Minimal WebDataset-style tar shard reader (ustar). Groups members by key prefix
/// (filename without extension). Expects <key>.jpg/png (+ optional <key>.cls with integer label).
class WebDataset : public Dataset {
public:
    explicit WebDataset(const std::string& tar_path, WebDatasetOptions opt = {});

    [[nodiscard]] std::size_t size() const override;
    [[nodiscard]] Sample get(std::size_t index) const override;

private:
    struct Member {
        std::string name;
        std::uint64_t offset = 0;
        std::uint64_t size = 0;
    };
    struct SampleRef {
        std::string key;
        std::size_t image_member = static_cast<std::size_t>(-1);
        std::int64_t label = 0;
        bool has_label = false;
    };

    std::string path_;
    WebDatasetOptions opt_;
    std::vector<Member> members_;
    std::vector<SampleRef> samples_;
    MappedFile map_;
};

} // namespace nexusdata
