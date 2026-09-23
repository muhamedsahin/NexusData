#include "nexusdata/dataset/audio.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#include "nd_audio_c.h"
#include "nexusdata/core/error.hpp"
#include "nexusdata/image/media.hpp"

namespace fs = std::filesystem;

namespace nexusdata {
namespace {

NDArray fit_frames(NDArray audio, const AudioOptions& opt) {
    if (opt.target_frames == 0) return audio;
    if (audio.shape().size() != 2) {
        throw IOError("load_audio: expected shape [frames, channels]");
    }
    const std::size_t ch = audio.shape()[1];
    NDArray fixed(Shape{opt.target_frames, ch}, DType::Float32);
    auto* dst = fixed.data<float>();
    for (std::size_t i = 0; i < fixed.numel(); ++i) dst[i] = opt.pad_value;
    const std::size_t copy_frames = std::min(opt.target_frames, audio.shape()[0]);
    std::memcpy(dst, audio.data(), copy_frames * ch * sizeof(float));
    return fixed;
}

NDArray from_pcm(NdPcm& pcm, AudioInfo* info) {
    struct Guard {
        NdPcm* p;
        ~Guard() { nd_pcm_free(p); }
    } guard{&pcm};
    if (!pcm.samples || pcm.frames <= 0 || pcm.channels <= 0) {
        throw IOError("load_audio: decoder produced no samples");
    }
    if (info) {
        info->sample_rate = pcm.sample_rate;
        info->channels = pcm.channels;
        info->num_frames = static_cast<std::size_t>(pcm.frames);
    }
    const std::size_t frames = static_cast<std::size_t>(pcm.frames);
    const std::size_t channels = static_cast<std::size_t>(pcm.channels);
    NDArray out(Shape{frames, channels}, DType::Float32);
    std::memcpy(out.data(), pcm.samples, frames * channels * sizeof(float));
    return out;
}

bool audio_ext(const fs::path& path) {
    auto ext = path.extension().string();
    for (char& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return ext == ".wav" || ext == ".mp3" || ext == ".flac" || ext == ".ogg" || ext == ".oga";
}

} // namespace

NDArray load_audio_from_buffer(const std::uint8_t* data, std::size_t size, AudioInfo* info) {
    if (data == nullptr || size == 0) {
        throw InvalidArgumentError("load_audio_from_buffer: empty buffer");
    }
    switch (sniff_media(data, size)) {
        case MediaFormat::Wav: {
            WavInfo wav{};
            NDArray out = load_wav_from_buffer(data, size, &wav);
            if (info) {
                info->sample_rate = wav.sample_rate;
                info->channels = wav.channels;
                info->num_frames = wav.num_frames;
            }
            return out;
        }
        case MediaFormat::Mp3: {
            NdPcm pcm{};
            if (nd_mp3_decode(data, size, &pcm) != 0) {
                nd_pcm_free(&pcm);
                throw IOError("load_audio: MP3 decode failed");
            }
            return from_pcm(pcm, info);
        }
        case MediaFormat::Flac: {
            NdPcm pcm{};
            if (nd_flac_decode(data, size, &pcm) != 0) {
                nd_pcm_free(&pcm);
                throw IOError("load_audio: FLAC decode failed");
            }
            return from_pcm(pcm, info);
        }
        case MediaFormat::Ogg: {
            NdPcm pcm{};
            if (nd_vorbis_decode(data, size, &pcm) != 0) {
                nd_pcm_free(&pcm);
                throw IOError("load_audio: Ogg/Vorbis decode failed");
            }
            return from_pcm(pcm, info);
        }
        default:
            throw IOError("load_audio: unrecognized audio container");
    }
}

NDArray load_audio(const std::string& path, AudioInfo* info) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw IOError("load_audio: cannot open " + quote_path(path));
    std::vector<std::uint8_t> buf((std::istreambuf_iterator<char>(in)), {});
    return load_audio_from_buffer(buf.data(), buf.size(), info);
}

AudioDataset::AudioDataset(const std::string& path_or_dir, AudioOptions opt) : opt_(opt) {
    fs::path p(path_or_dir);
    std::error_code ec;
    if (fs::is_directory(p, ec)) {
        for (const auto& e : fs::directory_iterator(p, ec)) {
            if (ec || !e.is_regular_file()) continue;
            if (audio_ext(e.path())) paths_.push_back(e.path().string());
        }
        std::sort(paths_.begin(), paths_.end());
    } else {
        paths_.push_back(path_or_dir);
    }
    if (paths_.empty()) {
        throw IOError("AudioDataset: no audio files in " + quote_path(path_or_dir));
    }
}

std::size_t AudioDataset::size() const { return paths_.size(); }

Sample AudioDataset::get(std::size_t index) const {
    if (index >= paths_.size()) {
        throw IndexError(format_index_error("AudioDataset::get", index, paths_.size()));
    }
    Sample s;
    s.input = fit_frames(load_audio(paths_[index], nullptr), opt_);
    s.label = NDArray(Shape{1}, DType::Int64);
    s.label.data<std::int64_t>()[0] = static_cast<std::int64_t>(index);
    return s;
}

} // namespace nexusdata
