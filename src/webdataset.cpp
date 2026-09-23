#include "nexusdata/dataset/webdataset.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <unordered_map>

#include "nexusdata/core/error.hpp"
#include "nexusdata/io/mapped_file.hpp"

namespace nexusdata {

namespace {

std::uint64_t parse_octal(const char* p, std::size_t n) {
    std::uint64_t v = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const char c = p[i];
        if (c == '\0' || c == ' ') {
            continue;
        }
        if (c < '0' || c > '7') {
            break;
        }
        v = (v << 3) + static_cast<std::uint64_t>(c - '0');
    }
    return v;
}

std::string strip_ext(const std::string& name) {
    const auto slash = name.find_last_of("/\\");
    const std::string base = (slash == std::string::npos) ? name : name.substr(slash + 1);
    const auto dot = base.find_last_of('.');
    if (dot == std::string::npos) {
        return base;
    }
    return base.substr(0, dot);
}

std::string lower_ext(const std::string& name) {
    const auto dot = name.find_last_of('.');
    if (dot == std::string::npos) {
        return {};
    }
    std::string e = name.substr(dot);
    for (char& c : e) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return e;
}

bool is_image_ext(const std::string& ext) {
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".tga";
}

} // namespace

WebDataset::WebDataset(const std::string& tar_path, WebDatasetOptions opt)
    : path_(tar_path), opt_(std::move(opt)) {
    map_ = read_input(path_);
    if (map_.size() < 512) {
        throw IOError("WebDataset: tar too small");
    }
    std::size_t off = 0;
    while (off + 512 <= map_.size()) {
        const char* hdr = reinterpret_cast<const char*>(map_.data() + off);
        bool all_zero = true;
        for (int i = 0; i < 512; ++i) {
            if (hdr[i] != 0) {
                all_zero = false;
                break;
            }
        }
        if (all_zero) {
            break;
        }
        // ustar magic at 257
        if (std::memcmp(hdr + 257, "ustar", 5) != 0 && hdr[0] == '\0') {
            break;
        }
        char namebuf[101] = {};
        std::memcpy(namebuf, hdr, 100);
        const std::uint64_t sz = parse_octal(hdr + 124, 12);
        const char typeflag = hdr[156];
        const std::size_t data_off = off + 512;
        if (typeflag == '0' || typeflag == '\0') {
            if (data_off > map_.size() || sz > map_.size() - data_off) {
                throw IOError(std::string("WebDataset: truncated member \"") + namebuf + "\"");
            }
            Member m;
            m.name = namebuf;
            m.offset = data_off;
            m.size = sz;
            members_.push_back(std::move(m));
        }
        if (sz > (std::numeric_limits<std::uint64_t>::max() - 511)) {
            throw IOError("WebDataset: member size overflow");
        }
        const std::uint64_t padded = ((sz + 511) / 512) * 512;
        if (padded > map_.size() - data_off) {
            break;
        }
        off = data_off + static_cast<std::size_t>(padded);
    }

    std::unordered_map<std::string, SampleRef> by_key;
    for (std::size_t i = 0; i < members_.size(); ++i) {
        const auto& m = members_[i];
        const std::string key = strip_ext(m.name);
        const std::string ext = lower_ext(m.name);
        auto& ref = by_key[key];
        ref.key = key;
        if (is_image_ext(ext) || ext == opt_.image_suffix) {
            ref.image_member = i;
        } else if (ext == ".cls") {
            std::string txt(reinterpret_cast<const char*>(map_.data() + m.offset),
                            static_cast<std::size_t>(m.size));
            while (!txt.empty() && (txt.back() == '\n' || txt.back() == '\r' || txt.back() == ' ')) {
                txt.pop_back();
            }
            ref.label = std::stoll(txt);
            ref.has_label = true;
        }
    }
    for (auto& kv : by_key) {
        if (kv.second.image_member != static_cast<std::size_t>(-1) &&
            is_image_ext(lower_ext(members_[kv.second.image_member].name))) {
            samples_.push_back(kv.second);
        }
    }
    std::sort(samples_.begin(), samples_.end(),
              [](const SampleRef& a, const SampleRef& b) { return a.key < b.key; });
    if (samples_.empty()) {
        throw IOError("WebDataset: no image samples found in tar");
    }
}

std::size_t WebDataset::size() const {
    return samples_.size();
}

Sample WebDataset::get(std::size_t index) const {
    if (index >= samples_.size()) {
        throw IndexError(format_index_error("WebDataset::get", index, samples_.size()));
    }
    const auto& ref = samples_[index];
    const auto& m = members_[ref.image_member];
    if (m.offset + m.size > map_.size()) {
        throw IOError("WebDataset: truncated member");
    }
    Image img = decode_image(map_.data() + m.offset, static_cast<std::size_t>(m.size),
                             opt_.image_decode);
    Sample s;
    s.input = std::move(img.data);
    s.label = NDArray(Shape{1}, DType::Int64);
    s.label.data<std::int64_t>()[0] = ref.has_label ? ref.label : static_cast<std::int64_t>(index);
    s.metadata.set("__key__", ref.key);
    return s;
}

} // namespace nexusdata
