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

} // namespace nexusdata
