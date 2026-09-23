#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "nexusdata/image/image.hpp"

namespace nexusdata {

enum class MediaFormat {
    Unknown,
    Jpeg,
    Png,
    Gif,
    Bmp,
    Webp,
    Avif,
    Tiff,
    Wav,
    Mp3,
    Flac,
    Ogg,
    Mp4,
};

/// Magic-byte sniff. Does not decode.
[[nodiscard]] MediaFormat sniff_media(const std::uint8_t* data, std::size_t size);
[[nodiscard]] std::string_view media_format_name(MediaFormat format);

/// JPEG/PNG/GIF/BMP/TGA go through stb (or libjpeg-turbo when
/// NEXUSDATA_WITH_JPEGTURBO=ON). WebP is linked by default. Uncompressed
/// 8-bit TIFF is built in. AVIF throws InvalidArgumentError naming
/// NEXUSDATA_WITH_AVIF unless that flag is on.
[[nodiscard]] Image decode_media_image(const std::uint8_t* data, std::size_t size,
                                       ImageDecodeOptions options = {});

} // namespace nexusdata
