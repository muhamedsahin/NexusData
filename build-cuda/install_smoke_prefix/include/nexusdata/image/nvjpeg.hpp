#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "nexusdata/core/ndarray.hpp"

namespace nexusdata {

/// GPU JPEG decoder (nvJPEG): compressed bytes go to the device and are decoded
/// there, so only the (much smaller) JPEG crosses PCIe instead of raw pixels.
/// Output is interleaved RGB [H, W, 3] uint8, like decode_image() with
/// ImageChannels::RGB (grayscale / CMYK files are converted to RGB). Pixel
/// values follow nvJPEG's IDCT, which may differ from stb_image by a few levels.
///
/// Requires a build with -DNEXUSDATA_WITH_NVJPEG=ON (implies CUDA).
/// Thread-safety: one decode at a time per decoder (internally serialized);
/// use one decoder per worker thread for parallel decoding.
class NvJpegDecoder {
public:
    /// Throws InvalidArgumentError when available() is false.
    explicit NvJpegDecoder(int device = 0);
    ~NvJpegDecoder();
    NvJpegDecoder(const NvJpegDecoder&) = delete;
    NvJpegDecoder& operator=(const NvJpegDecoder&) = delete;

    /// Built with nvJPEG and a CUDA device is present.
    [[nodiscard]] static bool available() noexcept;

    /// Decode one JPEG. @p out = CUDA device (default: this decoder's) or host.
    [[nodiscard]] NDArray decode(const std::uint8_t* data, std::size_t size,
                                 Device out = Device::auto_select());

    /// Decode several JPEGs on the decoder's stream.
    [[nodiscard]] std::vector<NDArray> decode_batch(
        const std::vector<std::vector<std::uint8_t>>& files, Device out = Device::auto_select());

    [[nodiscard]] int device() const noexcept { return device_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    int device_ = 0;
};

} // namespace nexusdata
