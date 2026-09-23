#include "nexusdata/dataset/image_folder.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <string>

#include "nexusdata/core/error.hpp"

namespace fs = std::filesystem;

namespace nexusdata {

namespace {

bool has_ext(const fs::path& p, const std::vector<std::string>& exts) {
    std::string e = p.extension().string();
    std::transform(e.begin(), e.end(), e.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    for (const auto& x : exts) {
        if (e == x) {
            return true;
        }
    }
    return false;
}

std::vector<std::string> default_exts() {
    return {".jpg", ".jpeg", ".png", ".bmp", ".tga", ".gif", ".ppm", ".pgm"};
}

} // namespace

ImageFolder::ImageFolder(std::string root, ImageFolderOptions options)
    : root_(std::move(root)), options_(std::move(options)) {
    if (options_.extensions.empty()) {
        options_.extensions = default_exts();
    }
    const fs::path root_path(root_);
    if (!fs::exists(root_path) || !fs::is_directory(root_path)) {
        throw IOError("ImageFolder: root is not a directory: \"" + root_ + "\"");
    }

    for (const auto& ent : fs::directory_iterator(root_path)) {
        if (!ent.is_directory()) {
            continue;
        }
        classes_.push_back(ent.path().filename().string());
    }
    std::sort(classes_.begin(), classes_.end());

    for (std::size_t ci = 0; ci < classes_.size(); ++ci) {
        const fs::path class_dir = root_path / classes_[ci];
        auto collect = [&](const fs::path& file) {
            if (!fs::is_regular_file(file)) {
                return;
            }
            if (!has_ext(file, options_.extensions)) {
                return;
            }
            if (options_.decode.on_corrupt == CorruptImagePolicy::Skip) {
                // Probe lightly: defer full decode to get(); still record path.
            }
            entries_.push_back(Entry{file.string(), static_cast<std::int64_t>(ci)});
        };

        if (options_.recursive) {
            for (const auto& f : fs::recursive_directory_iterator(class_dir)) {
                collect(f.path());
            }
        } else {
            for (const auto& f : fs::directory_iterator(class_dir)) {
                collect(f.path());
            }
        }
    }

    std::sort(entries_.begin(), entries_.end(),
              [](const Entry& a, const Entry& b) { return a.path < b.path; });

    if (entries_.empty()) {
        throw IOError("ImageFolder: no images found under \"" + root_ + "\"");
    }
}

std::size_t ImageFolder::size() const {
    return entries_.size();
}

Sample ImageFolder::get(std::size_t index) const {
    if (index >= entries_.size()) {
        throw IndexError("ImageFolder::get: index out of range");
    }
    const Entry& e = entries_[index];
    Image img = load_image(e.path, options_.decode);
    if (img.empty()) {
        throw IOError("ImageFolder::get: failed to decode \"" + e.path + "\"");
    }
    Sample s;
    s.input = std::move(img.data);
    s.label = NDArray(Shape{1}, DType::Int64);
    s.label.data<std::int64_t>()[0] = e.label;
    return s;
}

} // namespace nexusdata
