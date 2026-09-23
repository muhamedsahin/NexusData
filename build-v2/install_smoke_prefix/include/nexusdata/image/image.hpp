#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "nexusdata/core/ndarray.hpp"

namespace nexusdata {

enum class ImageChannels {
    Keep = 0,   // as stored in file
    Gray = 1,
    RGB = 3,
    RGBA = 4,
};

enum class CorruptImagePolicy {
    Throw,
    Skip, // used by ImageFolder scanner
};

struct ImageDecodeOptions {
    ImageChannels channels = ImageChannels::RGB;
    bool apply_exif_orientation = true; // best-effort via stb (limited)
    CorruptImagePolicy on_corrupt = CorruptImagePolicy::Throw;
};

/// Decoded image as HWC contiguous NDArray (UInt8), shape [H, W, C].
struct Image {
    NDArray data; // [H,W,C] UInt8
    int height = 0;
    int width = 0;
    int channels = 0;

    [[nodiscard]] bool empty() const noexcept { return height <= 0 || width <= 0; }
};

/// Decode image from file path (JPEG/PNG/BMP/TGA/GIF via stb_image).
/// Rationale: stb_image is header-only and covers JPEG/PNG/BMP/TGA/GIF.
/// WebP, uncompressed TIFF, and optional turbojpeg/AVIF go through decode_media_image.
[[nodiscard]] Image load_image(const std::string& path,
                               ImageDecodeOptions options = {});

/// Decode from memory buffer.
[[nodiscard]] Image decode_image(const std::uint8_t* bytes,
                                 std::size_t size,
                                 ImageDecodeOptions options = {});

/// Convert HWC UInt8 image NDArray to float CHW in [0,1] (ToTensor helper).
[[nodiscard]] NDArray image_to_tensor(const NDArray& hwc_u8);

} // namespace nexusdata
