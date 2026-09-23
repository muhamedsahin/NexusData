#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include "stb_image.h"

#include "nexusdata/image/image.hpp"

#include <cstring>
#include <string>

#include "nexusdata/core/error.hpp"
#include "nexusdata/simd/image_ops.hpp"

namespace nexusdata {

namespace {

Image from_stbi(unsigned char* pixels, int w, int h, int c, const std::string& ctx) {
    if (!pixels) {
        throw IOError("image decode failed (" + ctx + "): " +
                      std::string(stbi_failure_reason() ? stbi_failure_reason() : "unknown"));
    }
    Image img;
    img.width = w;
    img.height = h;
    img.channels = c;
    img.data = NDArray(Shape{static_cast<std::size_t>(h), static_cast<std::size_t>(w),
                             static_cast<std::size_t>(c)},
                       DType::UInt8);
    std::memcpy(img.data.data(), pixels, static_cast<std::size_t>(w) * h * c);
    stbi_image_free(pixels);
    return img;
}

int desired_channels(ImageChannels ch) {
    switch (ch) {
        case ImageChannels::Keep: return 0;
        case ImageChannels::Gray: return 1;
        case ImageChannels::RGB:  return 3;
        case ImageChannels::RGBA: return 4;
    }
    return 3;
}

} // namespace

Image load_image(const std::string& path, ImageDecodeOptions options) {
    int w = 0, h = 0, c = 0;
    // stb does not fully honour EXIF for all formats; flag reserved for future.
    (void)options.apply_exif_orientation;
    unsigned char* pixels =
        stbi_load(path.c_str(), &w, &h, &c, desired_channels(options.channels));
    if (!pixels) {
        if (options.on_corrupt == CorruptImagePolicy::Skip) {
            return {};
        }
        throw IOError("load_image(\"" + path + "\"): " +
                      std::string(stbi_failure_reason() ? stbi_failure_reason() : "unknown"));
    }
    const int out_c = desired_channels(options.channels) == 0 ? c
                                                              : desired_channels(options.channels);
    return from_stbi(pixels, w, h, out_c, path);
}

Image decode_image(const std::uint8_t* bytes, std::size_t size, ImageDecodeOptions options) {
    if (!bytes || size == 0) {
        throw InvalidArgumentError("decode_image: empty buffer");
    }
    int w = 0, h = 0, c = 0;
    (void)options.apply_exif_orientation;
    unsigned char* pixels = stbi_load_from_memory(
        bytes, static_cast<int>(size), &w, &h, &c, desired_channels(options.channels));
    if (!pixels) {
        if (options.on_corrupt == CorruptImagePolicy::Skip) {
            return {};
        }
        throw IOError(std::string("decode_image: ") +
                      (stbi_failure_reason() ? stbi_failure_reason() : "unknown"));
    }
    const int out_c = desired_channels(options.channels) == 0 ? c
                                                              : desired_channels(options.channels);
    return from_stbi(pixels, w, h, out_c, "memory");
}

NDArray image_to_tensor(const NDArray& hwc_u8) {
    if (hwc_u8.dtype() != DType::UInt8 || hwc_u8.shape().size() != 3) {
        throw ShapeError("image_to_tensor: expected HWC UInt8");
    }
    const std::size_t h = hwc_u8.shape()[0];
    const std::size_t w = hwc_u8.shape()[1];
    const std::size_t c = hwc_u8.shape()[2];
    NDArray out(Shape{c, h, w}, DType::Float32);
    simd::hwc_u8_to_chw_f32(hwc_u8.data<std::uint8_t>(), out.data<float>(), h, w, c);
    return out;
}

} // namespace nexusdata
