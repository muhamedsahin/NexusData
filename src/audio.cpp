#include "nexusdata/dataset/audio.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#include "nexusdata/core/error.hpp"

namespace fs = std::filesystem;

namespace nexusdata {

namespace {

std::uint32_t rd_u32(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}
std::uint16_t rd_u16(const std::uint8_t* p) {
    return static_cast<std::uint16_t>(p[0] | (p[1] << 8));
}

std::int16_t rd_i16(const std::uint8_t* p) {
    return static_cast<std::int16_t>(rd_u16(p));
}

float rd_f32(const std::uint8_t* p) {
    float v = 0.0f;
    std::memcpy(&v, p, sizeof(float));
    return v;
}

} // namespace

NDArray load_wav_from_buffer(const std::uint8_t* data, std::size_t size, WavInfo* info) {
    if (data == nullptr) {
        throw InvalidArgumentError("load_wav_from_buffer: null data");
    }
    if (size < 44) {
        throw IOError("load_wav: buffer too small (" + std::to_string(size) + " bytes)");
    }
    if (std::memcmp(data, "RIFF", 4) != 0 || std::memcmp(data + 8, "WAVE", 4) != 0) {
        throw IOError("load_wav: not a RIFF/WAVE file");
    }
    std::size_t off = 12;
    int audio_format = 0, channels = 0, rate = 0, bps = 0;
    const std::uint8_t* data_ptr = nullptr;
    std::size_t data_sz = 0;
    while (off + 8 <= size) {
        const char* id = reinterpret_cast<const char*>(data + off);
        const std::uint32_t sz = rd_u32(data + off + 4);
        off += 8;
        if (sz > size - off) {
            throw IOError("load_wav: truncated chunk at offset " + std::to_string(off));
        }
        if (std::memcmp(id, "fmt ", 4) == 0) {
            if (sz < 16) {
                throw IOError("load_wav: fmt chunk too small");
            }
            audio_format = rd_u16(data + off);
            channels = rd_u16(data + off + 2);
            rate = static_cast<int>(rd_u32(data + off + 4));
            bps = rd_u16(data + off + 14);
        } else if (std::memcmp(id, "data", 4) == 0) {
            data_ptr = data + off;
            data_sz = sz;
        }
        off += sz + (sz & 1); // word align
    }
    if (!data_ptr || channels <= 0) {
        throw IOError("load_wav: missing fmt/data");
    }
    if (bps <= 0 || (bps % 8) != 0) {
        throw IOError("load_wav: invalid bits_per_sample " + std::to_string(bps));
    }
    if (audio_format != 1 && audio_format != 3) {
        throw IOError("load_wav: only PCM (1) or IEEE float (3) supported, got " +
                      std::to_string(audio_format));
    }
    const std::size_t bytes_per_samp = static_cast<std::size_t>(bps / 8);
    const std::size_t frame_bytes = bytes_per_samp * static_cast<std::size_t>(channels);
    if (frame_bytes == 0) {
        throw IOError("load_wav: invalid frame size");
    }
    const std::size_t frames = data_sz / frame_bytes;
    if (info) {
        info->sample_rate = rate;
        info->channels = channels;
        info->bits_per_sample = bps;
        info->num_frames = frames;
    }
    NDArray out(Shape{frames, static_cast<std::size_t>(channels)}, DType::Float32);
    auto* dst = out.data<float>();
    const std::size_t n = frames * static_cast<std::size_t>(channels);
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint8_t* sp = data_ptr + i * bytes_per_samp;
        if (audio_format == 1 && bps == 16) {
            dst[i] = static_cast<float>(rd_i16(sp)) / 32768.0f;
        } else if (audio_format == 3 && bps == 32) {
            dst[i] = rd_f32(sp);
        } else {
            throw IOError("load_wav: unsupported PCM layout (format=" +
                          std::to_string(audio_format) + ", bps=" + std::to_string(bps) + ")");
        }
    }
    return out;
}

NDArray load_wav(const std::string& path, WavInfo* info) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw IOError("load_wav: cannot open " + quote_path(path));
    }
    std::vector<std::uint8_t> buf((std::istreambuf_iterator<char>(in)), {});
    return load_wav_from_buffer(buf.data(), buf.size(), info);
}

WavDataset::WavDataset(const std::string& path_or_dir, WavOptions opt) : opt_(opt) {
    fs::path p(path_or_dir);
    std::error_code ec;
    if (fs::is_directory(p, ec)) {
        for (const auto& e : fs::directory_iterator(p, ec)) {
            if (ec || !e.is_regular_file()) {
                continue;
            }
            auto ext = e.path().extension().string();
            for (char& c : ext) {
                c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            }
            if (ext == ".wav") {
                paths_.push_back(e.path().string());
            }
        }
        std::sort(paths_.begin(), paths_.end());
    } else {
        paths_.push_back(path_or_dir);
    }
    if (paths_.empty()) {
        throw IOError("WavDataset: no wav files in " + quote_path(path_or_dir));
    }
}

std::size_t WavDataset::size() const {
    return paths_.size();
}

Sample WavDataset::get(std::size_t index) const {
    if (index >= paths_.size()) {
        throw IndexError(format_index_error("WavDataset::get", index, paths_.size()));
    }
    WavInfo info{};
    NDArray audio = load_wav(paths_[index], &info);
    if (opt_.target_frames > 0) {
        const std::size_t ch = audio.shape()[1];
        NDArray fixed(Shape{opt_.target_frames, ch}, DType::Float32);
        std::memset(fixed.data(), 0, fixed.nbytes());
        if (opt_.pad_value != 0.0f) {
            auto* p = fixed.data<float>();
            for (std::size_t i = 0; i < fixed.numel(); ++i) {
                p[i] = opt_.pad_value;
            }
        }
        const std::size_t copy_frames = std::min(opt_.target_frames, audio.shape()[0]);
        std::memcpy(fixed.data(), audio.data(), copy_frames * ch * sizeof(float));
        audio = std::move(fixed);
    }
    Sample s;
    s.input = std::move(audio);
    s.label = NDArray(Shape{1}, DType::Int64);
    s.label.data<std::int64_t>()[0] = static_cast<std::int64_t>(index);
    return s;
}

} // namespace nexusdata
