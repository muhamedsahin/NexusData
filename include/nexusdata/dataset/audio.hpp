#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "nexusdata/dataset/dataset.hpp"

namespace nexusdata {

struct WavInfo {
    int sample_rate = 0;
    int channels = 0;
    int bits_per_sample = 0;
    std::size_t num_frames = 0;
};

struct WavOptions {
    /// If > 0, crop/pad to this many frames per channel interleaved.
    std::size_t target_frames = 0;
    float pad_value = 0.0f;
};

/// PCM WAV reader (16-bit or 32-bit float, little-endian). Output Float32 [frames, channels].
class WavDataset : public Dataset {
public:
    /// Single file → dataset of size 1, or directory of .wav files.
    explicit WavDataset(const std::string& path_or_dir, WavOptions opt = {});

    [[nodiscard]] std::size_t size() const override;
    [[nodiscard]] Sample get(std::size_t index) const override;

    [[nodiscard]] const std::vector<std::string>& paths() const noexcept { return paths_; }

private:
    std::vector<std::string> paths_;
    WavOptions opt_;
};

[[nodiscard]] NDArray load_wav(const std::string& path, WavInfo* info = nullptr);

/// Parse PCM WAV bytes in memory (for fuzzing / embedded payloads).
[[nodiscard]] NDArray load_wav_from_buffer(const std::uint8_t* data, std::size_t size,
                                           WavInfo* info = nullptr);

struct AudioInfo {
    int sample_rate = 0;
    int channels = 0;
    std::size_t num_frames = 0;
};

struct AudioOptions {
    /// If > 0, crop or pad to this many frames. Layout stays [frames, channels].
    std::size_t target_frames = 0;
    float pad_value = 0.0f;
};

/// Sniff and decode WAV, MP3, FLAC, or Ogg/Vorbis.
/// MP3/FLAC/Ogg use vendored single-file decoders (minimp3, dr_flac, stb_vorbis)
/// rather than libsndfile, so the default build has no extra system library.
/// Output matches WAV: Float32, shape [frames, channels], samples in [-1, 1].
[[nodiscard]] NDArray load_audio_from_buffer(const std::uint8_t* data, std::size_t size,
                                             AudioInfo* info = nullptr);
[[nodiscard]] NDArray load_audio(const std::string& path, AudioInfo* info = nullptr);

/// One file, or a directory of .wav/.mp3/.flac/.ogg/.oga files.
class AudioDataset : public Dataset {
public:
    explicit AudioDataset(const std::string& path_or_dir, AudioOptions opt = {});

    [[nodiscard]] std::size_t size() const override;
    [[nodiscard]] Sample get(std::size_t index) const override;

    [[nodiscard]] const std::vector<std::string>& paths() const noexcept { return paths_; }

private:
    std::vector<std::string> paths_;
    AudioOptions opt_;
};

} // namespace nexusdata
