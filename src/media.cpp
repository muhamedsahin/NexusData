#include "nexusdata/image/media.hpp"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "nexusdata/core/error.hpp"

#if defined(NEXUSDATA_WITH_WEBP)
extern "C" {
#include "src/webp/decode.h"
}
#endif
#if defined(NEXUSDATA_WITH_JPEGTURBO)
#include <turbojpeg.h>
#endif
#if defined(NEXUSDATA_WITH_AVIF)
#include <avif/avif.h>
#endif

namespace nexusdata {
namespace {

bool starts(const std::uint8_t* p, std::size_t n, const char* lit, std::size_t ln) {
    return n >= ln && std::memcmp(p, lit, ln) == 0;
}

Image image_from_packed(int w, int h, int src_c, const std::uint8_t* src, ImageChannels want) {
    if (w <= 0 || h <= 0 || !src || (src_c != 1 && src_c != 3 && src_c != 4)) {
        throw IOError("decode_media_image: empty image");
    }
    int dst_c = src_c;
    switch (want) {
        case ImageChannels::Keep: dst_c = src_c; break;
        case ImageChannels::Gray: dst_c = 1; break;
        case ImageChannels::RGB: dst_c = 3; break;
        case ImageChannels::RGBA: dst_c = 4; break;
    }
    Image img;
    img.width = w;
    img.height = h;
    img.channels = dst_c;
    img.data = NDArray(Shape{static_cast<std::size_t>(h), static_cast<std::size_t>(w),
                             static_cast<std::size_t>(dst_c)},
                       DType::UInt8);
    auto* dst = img.data.data<std::uint8_t>();
    const std::size_t n = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint8_t* p = src + i * static_cast<std::size_t>(src_c);
        std::uint8_t ch[4] = {0, 0, 0, 255};
        if (src_c == 1) {
            ch[0] = ch[1] = ch[2] = p[0];
        } else {
            ch[0] = p[0];
            ch[1] = p[1];
            ch[2] = src_c > 2 ? p[2] : p[0];
            if (src_c > 3) ch[3] = p[3];
        }
        std::uint8_t* o = dst + i * static_cast<std::size_t>(dst_c);
        if (dst_c == 1) {
            o[0] = static_cast<std::uint8_t>((static_cast<int>(ch[0]) + ch[1] + ch[2]) / 3);
        } else if (dst_c == 3) {
            o[0] = ch[0];
            o[1] = ch[1];
            o[2] = ch[2];
        } else {
            o[0] = ch[0];
            o[1] = ch[1];
            o[2] = ch[2];
            o[3] = ch[3];
        }
    }
    return img;
}

Image decode_tiff(const std::uint8_t* data, std::size_t size, ImageDecodeOptions options) {
    if (!data || size < 8) throw IOError("tiff: buffer too small");
    const bool le = data[0] == 'I' && data[1] == 'I' && data[2] == 0x2A && data[3] == 0x00;
    const bool be = data[0] == 'M' && data[1] == 'M' && data[2] == 0x00 && data[3] == 0x2A;
    if (!le && !be) throw IOError("tiff: bad magic");
    auto ru16 = [&](std::size_t o) -> std::uint16_t {
        if (o + 2 > size) throw IOError("tiff: truncated");
        return le ? static_cast<std::uint16_t>(data[o] | (data[o + 1] << 8))
                  : static_cast<std::uint16_t>((data[o] << 8) | data[o + 1]);
    };
    auto ru32 = [&](std::size_t o) -> std::uint32_t {
        if (o + 4 > size) throw IOError("tiff: truncated");
        if (le) {
            return static_cast<std::uint32_t>(data[o]) | (static_cast<std::uint32_t>(data[o + 1]) << 8) |
                   (static_cast<std::uint32_t>(data[o + 2]) << 16) |
                   (static_cast<std::uint32_t>(data[o + 3]) << 24);
        }
        return (static_cast<std::uint32_t>(data[o]) << 24) | (static_cast<std::uint32_t>(data[o + 1]) << 16) |
               (static_cast<std::uint32_t>(data[o + 2]) << 8) | data[o + 3];
    };
    const std::uint32_t ifd = ru32(4);
    if (static_cast<std::size_t>(ifd) + 2 > size) throw IOError("tiff: bad IFD offset");
    const std::uint16_t count = ru16(ifd);
    const std::size_t ent = static_cast<std::size_t>(ifd) + 2;
    if (ent + static_cast<std::size_t>(count) * 12 > size) throw IOError("tiff: IFD truncated");

    std::uint32_t width = 0, height = 0, compression = 1, photometric = 1, spp = 1, rows = 0, planar = 1;
    std::vector<std::uint32_t> offsets, counts;
    for (std::uint16_t i = 0; i < count; ++i) {
        const std::size_t e = ent + static_cast<std::size_t>(i) * 12;
        const std::uint16_t tag = ru16(e);
        const std::uint16_t type = ru16(e + 2);
        const std::uint32_t n = ru32(e + 4);
        const std::uint32_t raw = ru32(e + 8);
        if (type != 3 && type != 4) continue;
        const std::size_t es = type == 3 ? 2 : 4;
        std::vector<std::uint32_t> vals;
        vals.reserve(n);
        const std::size_t nbytes = es * static_cast<std::size_t>(n);
        if (nbytes <= 4) {
            for (std::uint32_t k = 0; k < n; ++k) {
                vals.push_back(type == 3 ? ru16(e + 8 + static_cast<std::size_t>(k) * 2) : raw);
            }
        } else {
            if (static_cast<std::size_t>(raw) + nbytes > size) throw IOError("tiff: value out of range");
            for (std::uint32_t k = 0; k < n; ++k) {
                const std::size_t at = static_cast<std::size_t>(raw) + static_cast<std::size_t>(k) * es;
                vals.push_back(type == 3 ? ru16(at) : ru32(at));
            }
        }
        if (vals.empty()) continue;
        switch (tag) {
            case 256: width = vals[0]; break;
            case 257: height = vals[0]; break;
            case 258:
                for (std::uint32_t b : vals) {
                    if (b != 8) throw IOError("tiff: only 8-bit samples");
                }
                break;
            case 259: compression = vals[0]; break;
            case 262: photometric = vals[0]; break;
            case 273: offsets = std::move(vals); break;
            case 277: spp = vals[0]; break;
            case 278: rows = vals[0]; break;
            case 279: counts = std::move(vals); break;
            case 284: planar = vals[0]; break;
            default: break;
        }
    }
    if (compression != 1) throw IOError("tiff: only uncompressed strips");
    if (spp != 1 && spp != 3) throw IOError("tiff: only gray or RGB");
    if (planar != 1) throw IOError("tiff: only chunky planar configuration");
    if (photometric > 2) throw IOError("tiff: photometric interpretation is not gray or RGB");
    if (width == 0 || height == 0 || offsets.empty()) throw IOError("tiff: missing geometry");
    if (static_cast<std::uint64_t>(width) * height > 32000000ull) throw IOError("tiff: image too large");
    if (rows == 0) rows = height;
    if (counts.empty()) {
        std::uint32_t row = 0;
        for (std::size_t s = 0; s < offsets.size() && row < height; ++s) {
            const std::uint32_t rows_here = std::min(rows, height - row);
            counts.push_back(rows_here * width * spp);
            row += rows_here;
        }
    }
    std::vector<std::uint8_t> pix(static_cast<std::size_t>(width) * height * spp);
    std::uint32_t row = 0;
    for (std::size_t s = 0; s < offsets.size() && row < height; ++s) {
        const std::uint32_t rows_here = std::min(rows, height - row);
        const std::size_t need = static_cast<std::size_t>(rows_here) * width * spp;
        const std::size_t got = s < counts.size() ? counts[s] : 0;
        if (got < need) throw IOError("tiff: short strip");
        if (static_cast<std::size_t>(offsets[s]) + need > size) throw IOError("tiff: strip out of range");
        std::memcpy(pix.data() + static_cast<std::size_t>(row) * width * spp, data + offsets[s], need);
        row += rows_here;
    }
    if (row < height) throw IOError("tiff: strips do not cover the image");
    if (photometric == 0 && spp == 1) {
        for (std::uint8_t& px : pix) px = static_cast<std::uint8_t>(255 - px);
    }
    return image_from_packed(static_cast<int>(width), static_cast<int>(height), static_cast<int>(spp),
                             pix.data(), options.channels);
}

#if defined(NEXUSDATA_WITH_WEBP)
Image decode_webp(const std::uint8_t* data, std::size_t size, ImageDecodeOptions options) {
    if (size > static_cast<std::size_t>(0x7fffffff)) throw IOError("webp: buffer too large");
    WebPBitstreamFeatures feat;
    if (WebPGetFeatures(data, size, &feat) != VP8_STATUS_OK || feat.width <= 0 || feat.height <= 0) {
        throw IOError("webp: not a decodable bitstream");
    }
    int w = 0, h = 0;
    std::uint8_t* px = feat.has_alpha ? WebPDecodeRGBA(data, size, &w, &h) : WebPDecodeRGB(data, size, &w, &h);
    if (!px) throw IOError("webp: decode failed");
    try {
        Image img = image_from_packed(w, h, feat.has_alpha ? 4 : 3, px, options.channels);
        WebPFree(px);
        return img;
    } catch (...) {
        WebPFree(px);
        throw;
    }
}
#endif

#if defined(NEXUSDATA_WITH_JPEGTURBO)
Image decode_jpeg_turbo(const std::uint8_t* data, std::size_t size, ImageDecodeOptions options) {
    tjhandle handle = tjInitDecompress();
    if (!handle) throw IOError("jpeg-turbo: tjInitDecompress failed");
    int w = 0, h = 0, subsamp = 0, colorspace = 0;
    if (tjDecompressHeader3(handle, data, static_cast<unsigned long>(size), &w, &h, &subsamp, &colorspace) != 0) {
        const std::string why = tjGetErrorStr2(handle);
        tjDestroy(handle);
        throw IOError("jpeg-turbo: " + why);
    }
    int fmt = TJPF_RGB;
    int ch = 3;
    if (options.channels == ImageChannels::Gray) {
        fmt = TJPF_GRAY;
        ch = 1;
    } else if (options.channels == ImageChannels::RGBA) {
        fmt = TJPF_RGBA;
        ch = 4;
    }
    std::vector<std::uint8_t> px(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * static_cast<std::size_t>(ch));
    if (tjDecompress2(handle, data, static_cast<unsigned long>(size), px.data(), w, 0, h, fmt, TJFLAG_FASTDCT) != 0) {
        const std::string why = tjGetErrorStr2(handle);
        tjDestroy(handle);
        throw IOError("jpeg-turbo: " + why);
    }
    tjDestroy(handle);
    return image_from_packed(w, h, ch, px.data(), options.channels == ImageChannels::Keep ? ImageChannels::RGB
                                                                                          : options.channels);
}
#endif

#if defined(NEXUSDATA_WITH_AVIF)
Image decode_avif(const std::uint8_t* data, std::size_t size, ImageDecodeOptions options) {
    avifDecoder* dec = avifDecoderCreate();
    if (!dec) throw IOError("avif: decoder allocation failed");
    if (avifDecoderSetIOMemory(dec, data, size) != AVIF_RESULT_OK ||
        avifDecoderParse(dec) != AVIF_RESULT_OK || avifDecoderNextImage(dec) != AVIF_RESULT_OK ||
        !dec->image) {
        avifDecoderDestroy(dec);
        throw IOError("avif: decode failed");
    }
    avifRGBImage rgb;
    avifRGBImageSetDefaults(&rgb, dec->image);
    const bool alpha = options.channels == ImageChannels::RGBA ||
                       (options.channels == ImageChannels::Keep && dec->image->alphaPlane != nullptr);
    rgb.format = alpha ? AVIF_RGB_FORMAT_RGBA : AVIF_RGB_FORMAT_RGB;
    if (avifRGBImageAllocatePixels(&rgb) != AVIF_RESULT_OK || avifImageYUVToRGB(dec->image, &rgb) != AVIF_RESULT_OK) {
        avifRGBImageFreePixels(&rgb);
        avifDecoderDestroy(dec);
        throw IOError("avif: YUV conversion failed");
    }
    const int ch = alpha ? 4 : 3;
    std::vector<std::uint8_t> packed(static_cast<std::size_t>(rgb.width) * rgb.height * static_cast<std::size_t>(ch));
    for (uint32_t y = 0; y < rgb.height; ++y) {
        std::memcpy(packed.data() + static_cast<std::size_t>(y) * rgb.width * static_cast<std::size_t>(ch),
                    rgb.pixels + static_cast<std::size_t>(y) * rgb.rowBytes,
                    static_cast<std::size_t>(rgb.width) * static_cast<std::size_t>(ch));
    }
    avifRGBImageFreePixels(&rgb);
    avifDecoderDestroy(dec);
    ImageChannels want = options.channels;
    if (want == ImageChannels::Keep) want = alpha ? ImageChannels::RGBA : ImageChannels::RGB;
    return image_from_packed(static_cast<int>(rgb.width), static_cast<int>(rgb.height), ch, packed.data(), want);
}
#endif

} // namespace

MediaFormat sniff_media(const std::uint8_t* data, std::size_t size) {
    if (!data || size < 4) return MediaFormat::Unknown;
    if (size >= 3 && data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF) return MediaFormat::Jpeg;
    if (starts(data, size, "\x89PNG", 4)) return MediaFormat::Png;
    if (starts(data, size, "GIF8", 4)) return MediaFormat::Gif;
    if (data[0] == 'B' && data[1] == 'M') return MediaFormat::Bmp;
    if (size >= 12 && std::memcmp(data, "RIFF", 4) == 0 && std::memcmp(data + 8, "WEBP", 4) == 0) {
        return MediaFormat::Webp;
    }
    if (size >= 12 && std::memcmp(data + 4, "ftyp", 4) == 0) {
        if (std::memcmp(data + 8, "avif", 4) == 0 || std::memcmp(data + 8, "avis", 4) == 0) {
            return MediaFormat::Avif;
        }
        return MediaFormat::Mp4;
    }
    const std::uint8_t tiff_le[] = {'I', 'I', 0x2A, 0x00};
    const std::uint8_t tiff_be[] = {'M', 'M', 0x00, 0x2A};
    if (size >= 4 && (std::memcmp(data, tiff_le, 4) == 0 || std::memcmp(data, tiff_be, 4) == 0)) {
        return MediaFormat::Tiff;
    }
    if (size >= 12 && std::memcmp(data, "RIFF", 4) == 0 && std::memcmp(data + 8, "WAVE", 4) == 0) {
        return MediaFormat::Wav;
    }
    if (starts(data, size, "fLaC", 4)) return MediaFormat::Flac;
    if (starts(data, size, "OggS", 4)) return MediaFormat::Ogg;
    if (size >= 2 && data[0] == 0xFF && (data[1] & 0xE0) == 0xE0) return MediaFormat::Mp3;
    if (starts(data, size, "ID3", 3)) return MediaFormat::Mp3;
    return MediaFormat::Unknown;
}

std::string_view media_format_name(MediaFormat format) {
    switch (format) {
        case MediaFormat::Jpeg: return "jpeg";
        case MediaFormat::Png: return "png";
        case MediaFormat::Gif: return "gif";
        case MediaFormat::Bmp: return "bmp";
        case MediaFormat::Webp: return "webp";
        case MediaFormat::Avif: return "avif";
        case MediaFormat::Tiff: return "tiff";
        case MediaFormat::Wav: return "wav";
        case MediaFormat::Mp3: return "mp3";
        case MediaFormat::Flac: return "flac";
        case MediaFormat::Ogg: return "ogg";
        case MediaFormat::Mp4: return "mp4";
        case MediaFormat::Unknown: return "unknown";
    }
    return "unknown";
}

Image decode_media_image(const std::uint8_t* data, std::size_t size, ImageDecodeOptions options) {
    try {
        switch (sniff_media(data, size)) {
            case MediaFormat::Jpeg:
#if defined(NEXUSDATA_WITH_JPEGTURBO)
                return decode_jpeg_turbo(data, size, options);
#else
                return decode_image(data, size, options);
#endif
            case MediaFormat::Png:
            case MediaFormat::Gif:
            case MediaFormat::Bmp:
                return decode_image(data, size, options);
            case MediaFormat::Webp:
#if defined(NEXUSDATA_WITH_WEBP)
                return decode_webp(data, size, options);
#else
                throw InvalidArgumentError(
                    "decode_media_image: WebP is not linked. Reconfigure with -DNEXUSDATA_WITH_WEBP=ON");
#endif
            case MediaFormat::Avif:
#if defined(NEXUSDATA_WITH_AVIF)
                return decode_avif(data, size, options);
#else
                throw InvalidArgumentError(
                    "decode_media_image: AVIF is not linked. Reconfigure with -DNEXUSDATA_WITH_AVIF=ON");
#endif
            case MediaFormat::Tiff:
                return decode_tiff(data, size, options);
            default:
                throw IOError("decode_media_image: not an image buffer");
        }
    } catch (const IOError&) {
        if (options.on_corrupt == CorruptImagePolicy::Skip) return {};
        throw;
    }
}

} // namespace nexusdata
