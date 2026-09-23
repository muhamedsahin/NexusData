#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "nexusdata/dataset/audio.hpp"
#include "nexusdata/dataset/text.hpp"
#include "nexusdata/dataset/video.hpp"
#include "nexusdata/image/media.hpp"

using namespace nexusdata;
namespace fs = std::filesystem;

namespace {

extern "C" {
std::size_t WebPEncodeLosslessRGB(const std::uint8_t* rgb, int width, int height, int stride,
                                  std::uint8_t** output);
void WebPFree(void* ptr);
}

void push_u16_le(std::vector<std::uint8_t>& b, std::uint16_t v) {
    b.push_back(static_cast<std::uint8_t>(v));
    b.push_back(static_cast<std::uint8_t>(v >> 8));
}
void push_u32_le(std::vector<std::uint8_t>& b, std::uint32_t v) {
    b.push_back(static_cast<std::uint8_t>(v));
    b.push_back(static_cast<std::uint8_t>(v >> 8));
    b.push_back(static_cast<std::uint8_t>(v >> 16));
    b.push_back(static_cast<std::uint8_t>(v >> 24));
}

std::vector<std::uint8_t> tiny_wav() {
    std::vector<std::uint8_t> b;
    b.insert(b.end(), {'R', 'I', 'F', 'F'});
    push_u32_le(b, 38);
    b.insert(b.end(), {'W', 'A', 'V', 'E', 'f', 'm', 't', ' '});
    push_u32_le(b, 16);
    push_u16_le(b, 1);
    push_u16_le(b, 1);
    push_u32_le(b, 8000);
    push_u32_le(b, 16000);
    push_u16_le(b, 2);
    push_u16_le(b, 16);
    b.insert(b.end(), {'d', 'a', 't', 'a'});
    push_u32_le(b, 2);
    push_u16_le(b, 0);
    return b;
}

std::uint8_t crc8(const std::uint8_t* p, std::size_t n) {
    std::uint8_t c = 0;
    for (std::size_t i = 0; i < n; ++i) {
        c = static_cast<std::uint8_t>(c ^ p[i]);
        for (int b = 0; b < 8; ++b) {
            c = (c & 0x80) ? static_cast<std::uint8_t>((c << 1) ^ 0x07)
                           : static_cast<std::uint8_t>(c << 1);
        }
    }
    return c;
}

std::uint16_t crc16(const std::uint8_t* p, std::size_t n) {
    std::uint16_t c = 0;
    for (std::size_t i = 0; i < n; ++i) {
        c = static_cast<std::uint16_t>(c ^ (static_cast<std::uint16_t>(p[i]) << 8));
        for (int b = 0; b < 8; ++b) {
            c = (c & 0x8000) ? static_cast<std::uint16_t>((c << 1) ^ 0x8005)
                             : static_cast<std::uint16_t>(c << 1);
        }
    }
    return c;
}

void push_be16(std::vector<std::uint8_t>& b, std::uint16_t v) {
    b.push_back(static_cast<std::uint8_t>(v >> 8));
    b.push_back(static_cast<std::uint8_t>(v));
}

/// 16-sample mono 16-bit 44100 Hz constant-0 FLAC (STREAMINFO + one frame).
std::vector<std::uint8_t> tiny_flac() {
    std::vector<std::uint8_t> b = {'f', 'L', 'a', 'C', 0x80, 0x00, 0x00, 0x22};
    push_be16(b, 16);
    push_be16(b, 16);
    b.insert(b.end(), {0, 0, 0, 0, 0, 0});
    const std::uint64_t packed = (static_cast<std::uint64_t>(44100) << 44) |
                                 (static_cast<std::uint64_t>(15) << 36) | 16ull;
    for (int i = 7; i >= 0; --i) {
        b.push_back(static_cast<std::uint8_t>((packed >> (i * 8)) & 0xFF));
    }
    b.insert(b.end(), 16, 0);
    std::vector<std::uint8_t> frame = {0xFF, 0xF8, 0x69, 0x08, 0x00, 0x0F};
    frame.push_back(crc8(frame.data(), frame.size()));
    frame.insert(frame.end(), {0x00, 0x00, 0x00});
    const std::uint16_t c = crc16(frame.data(), frame.size());
    push_be16(frame, c);
    b.insert(b.end(), frame.begin(), frame.end());
    return b;
}

void write_file(const fs::path& path, const std::string& text) {
    std::ofstream out(path, std::ios::binary);
    out << text;
}

struct TiffEnt {
    std::uint16_t tag;
    std::uint16_t type;
    std::uint32_t count;
    std::uint32_t value;
};

std::vector<std::uint8_t> make_tiff(bool le, const std::vector<TiffEnt>& ents,
                                    const std::vector<std::uint8_t>& pixels, std::uint32_t& pixel_at) {
    const std::size_t ifd = 8;
    pixel_at = static_cast<std::uint32_t>(ifd + 2 + ents.size() * 12 + 4);
    std::vector<std::uint8_t> b(pixel_at + pixels.size(), 0);
    auto wu16 = [&](std::size_t o, std::uint16_t v) {
        if (le) {
            b[o] = static_cast<std::uint8_t>(v);
            b[o + 1] = static_cast<std::uint8_t>(v >> 8);
        } else {
            b[o] = static_cast<std::uint8_t>(v >> 8);
            b[o + 1] = static_cast<std::uint8_t>(v);
        }
    };
    auto wu32 = [&](std::size_t o, std::uint32_t v) {
        if (le) {
            b[o] = static_cast<std::uint8_t>(v);
            b[o + 1] = static_cast<std::uint8_t>(v >> 8);
            b[o + 2] = static_cast<std::uint8_t>(v >> 16);
            b[o + 3] = static_cast<std::uint8_t>(v >> 24);
        } else {
            b[o] = static_cast<std::uint8_t>(v >> 24);
            b[o + 1] = static_cast<std::uint8_t>(v >> 16);
            b[o + 2] = static_cast<std::uint8_t>(v >> 8);
            b[o + 3] = static_cast<std::uint8_t>(v);
        }
    };
    if (le) {
        b[0] = 'I';
        b[1] = 'I';
        b[2] = 0x2A;
    } else {
        b[0] = 'M';
        b[1] = 'M';
        b[3] = 0x2A;
    }
    wu32(4, static_cast<std::uint32_t>(ifd));
    wu16(ifd, static_cast<std::uint16_t>(ents.size()));
    for (std::size_t i = 0; i < ents.size(); ++i) {
        const std::size_t e = ifd + 2 + i * 12;
        wu16(e, ents[i].tag);
        wu16(e + 2, ents[i].type);
        wu32(e + 4, ents[i].count);
        if (ents[i].type == 3) {
            wu16(e + 8, static_cast<std::uint16_t>(ents[i].value));
        } else {
            wu32(e + 8, ents[i].value);
        }
    }
    std::memcpy(b.data() + pixel_at, pixels.data(), pixels.size());
    return b;
}

} // namespace

TEST_CASE("audio buffer: wav, flac silence, and rejected mp3/ogg") {
    const auto wav = tiny_wav();
    AudioInfo info{};
    const NDArray a = load_audio_from_buffer(wav.data(), wav.size(), &info);
    CHECK(info.sample_rate == 8000);
    CHECK(info.channels == 1);
    CHECK(info.num_frames == 1);
    CHECK(a.shape()[0] == 1);
    CHECK(a.shape()[1] == 1);
    CHECK(a.data<float>()[0] == 0.0f);

    const auto flac = tiny_flac();
    CHECK(sniff_media(flac.data(), flac.size()) == MediaFormat::Flac);
    AudioInfo fi{};
    const NDArray f = load_audio_from_buffer(flac.data(), flac.size(), &fi);
    CHECK(fi.sample_rate == 44100);
    CHECK(fi.channels == 1);
    CHECK(fi.num_frames == 16);
    CHECK(f.shape()[0] == 16);
    CHECK(f.data<float>()[0] == 0.0f);
    CHECK(f.data<float>()[15] == 0.0f);

    const std::uint8_t mp3[] = {'I', 'D', '3', 0, 0, 0, 0, 0, 0, 0};
    CHECK_THROWS_AS(static_cast<void>(load_audio_from_buffer(mp3, sizeof(mp3), nullptr)), IOError);
    const std::uint8_t ogg[] = {'O', 'g', 'g', 'S', 0, 0, 0, 0};
    CHECK_THROWS_AS(static_cast<void>(load_audio_from_buffer(ogg, sizeof(ogg), nullptr)), IOError);
}

TEST_CASE("AudioDataset pads wav frames") {
    const fs::path dir = fs::temp_directory_path() / "nexusdata_phase4_wav";
    fs::create_directories(dir);
    const fs::path file = dir / "a.wav";
    const auto wav = tiny_wav();
    std::ofstream out(file, std::ios::binary);
    out.write(reinterpret_cast<const char*>(wav.data()), static_cast<std::streamsize>(wav.size()));
    out.close();
    AudioOptions opt;
    opt.target_frames = 4;
    AudioDataset ds(file.string(), opt);
    CHECK(ds.size() == 1);
    const Sample s = ds.get(0);
    CHECK(s.input.shape()[0] == 4);
    CHECK(s.input.shape()[1] == 1);
    CHECK(s.label.data<std::int64_t>()[0] == 0);
    fs::remove_all(dir);
}

TEST_CASE("webp lossless roundtrip and uncompressed tiff") {
    const std::uint8_t rgb[12] = {255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255};
    std::uint8_t* encoded = nullptr;
    const std::size_t n = WebPEncodeLosslessRGB(rgb, 2, 2, 6, &encoded);
    CHECK(n > 12);
    CHECK(encoded != nullptr);
    const Image img = decode_media_image(encoded, n);
    CHECK(img.width == 2);
    CHECK(img.height == 2);
    CHECK(img.channels == 3);
    CHECK(std::memcmp(img.data.data<std::uint8_t>(), rgb, 12) == 0);
    WebPFree(encoded);

    std::uint32_t pixel_at = 0;
    std::vector<TiffEnt> gray = {
        {256, 4, 1, 2}, {257, 4, 1, 2}, {258, 3, 1, 8}, {259, 3, 1, 1},
        {262, 3, 1, 1}, {273, 4, 1, 0}, {277, 3, 1, 1}, {278, 4, 1, 2}, {279, 4, 1, 4},
    };
    auto g = make_tiff(true, gray, {1, 2, 3, 4}, pixel_at);
    // StripOffsets is the 6th entry (index 5); its value field starts at entry+8.
    const std::size_t value = 8 + 2 + 5 * 12 + 8;
    g[value] = static_cast<std::uint8_t>(pixel_at);
    g[value + 1] = static_cast<std::uint8_t>(pixel_at >> 8);
    g[value + 2] = static_cast<std::uint8_t>(pixel_at >> 16);
    g[value + 3] = static_cast<std::uint8_t>(pixel_at >> 24);
    CHECK(sniff_media(g.data(), g.size()) == MediaFormat::Tiff);
    const Image tg = decode_media_image(g.data(), g.size());
    CHECK(tg.width == 2);
    CHECK(tg.height == 2);
    CHECK(tg.channels == 3);
    CHECK(tg.data.data<std::uint8_t>()[0] == 1);
    CHECK(tg.data.data<std::uint8_t>()[3] == 2);
    CHECK(tg.data.data<std::uint8_t>()[9] == 4);

    std::vector<TiffEnt> rgb_ent = {
        {256, 4, 1, 1}, {257, 4, 1, 1}, {258, 3, 1, 8}, {259, 3, 1, 1},
        {262, 3, 1, 2}, {273, 4, 1, 0}, {277, 3, 1, 3}, {278, 4, 1, 1}, {279, 4, 1, 3},
    };
    auto be = make_tiff(false, rgb_ent, {10, 20, 30}, pixel_at);
    const std::size_t be_value = 8 + 2 + 5 * 12 + 8;
    be[be_value] = static_cast<std::uint8_t>(pixel_at >> 24);
    be[be_value + 1] = static_cast<std::uint8_t>(pixel_at >> 16);
    be[be_value + 2] = static_cast<std::uint8_t>(pixel_at >> 8);
    be[be_value + 3] = static_cast<std::uint8_t>(pixel_at);
    const Image tr = decode_media_image(be.data(), be.size());
    CHECK(tr.width == 1);
    CHECK(tr.channels == 3);
    CHECK(tr.data.data<std::uint8_t>()[0] == 10);
    CHECK(tr.data.data<std::uint8_t>()[1] == 20);
    CHECK(tr.data.data<std::uint8_t>()[2] == 30);
}

TEST_CASE("bpe wordpiece and unigram tokenizers") {
    const fs::path dir = fs::temp_directory_path() / "nexusdata_phase4_tok";
    fs::create_directories(dir);
    write_file(dir / "vocab.txt", "a\nb\nc\nab\nabc\n");
    write_file(dir / "merges.txt", "#version: 0.2\na b\nab c\n");
    BpeTokenizer bpe((dir / "vocab.txt").string(), (dir / "merges.txt").string());
    const auto ids = bpe.encode("abc");
    CHECK(ids.size() == 1);
    CHECK(ids[0] == 4);
    CHECK(bpe.decode(ids) == "abc");

    write_file(dir / "vocab.json", "{\"a\":0,\"b\":1,\"ab\":2}");
    write_file(dir / "merges2.txt", "a b\n");
    BpeTokenizer bpe_json((dir / "vocab.json").string(), (dir / "merges2.txt").string());
    CHECK(bpe_json.encode("ab") == std::vector<std::int64_t>{2});
    CHECK(bpe_json.decode({2}) == "ab");

    write_file(dir / "wp.txt", "[UNK]\nun\n##want\n##ed\n");
    WordPieceTokenizer wp((dir / "wp.txt").string());
    const auto wp_ids = wp.encode("unwanted");
    CHECK(wp_ids.size() == 3);
    CHECK(wp.decode(wp_ids) == "unwanted");
    CHECK(wp.encode("zzz") == std::vector<std::int64_t>{0});

    write_file(dir / "uni.txt", "a\t-1.0\nb\t-1.0\nc\t-1.0\nab\t-0.6\nabc\t-0.2\n");
    UnigramTokenizer uni((dir / "uni.txt").string());
    const auto u = uni.encode("abc");
    CHECK(u.size() == 1);
    CHECK(u[0] == 4);
    CHECK(uni.decode(u) == "abc");

    WhitespaceHashTokenizer hash(100);
    CHECK(hash.encode("a b").size() == 2);
    fs::remove_all(dir);
}

TEST_CASE("video iterable names the missing ffmpeg and nvdec flags") {
    bool ffmpeg = false;
    try {
        (void)VideoIterable("missing.mp4");
    } catch (const InvalidArgumentError& e) {
        ffmpeg = std::string(e.what()).find("NEXUSDATA_WITH_FFMPEG") != std::string::npos;
    }
    CHECK(ffmpeg);
    VideoOptions opt;
    opt.hw_decode = true;
    bool nvdec = false;
    try {
        (void)VideoIterable("missing.mp4", opt);
    } catch (const InvalidArgumentError& e) {
        nvdec = std::string(e.what()).find("NEXUSDATA_WITH_NVDEC") != std::string::npos;
    }
    CHECK(nvdec);
}
