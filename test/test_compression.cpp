// v2.0 Phase 3: transparent compression layer (plan 7.5).

#include <doctest/doctest.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "nexusdata/dataset/csv/csv_dataset.hpp"
#include "nexusdata/dataset/jsonl.hpp"
#include "nexusdata/dataset/npy.hpp"
#include "nexusdata/dataset/webdataset.hpp"
#include "nexusdata/io/byte_source.hpp"
#include "nexusdata/io/compression.hpp"

using namespace nexusdata;

namespace {

const Codec kCodecs[] = {Codec::Gzip, Codec::Zlib, Codec::Zstd, Codec::Lz4, Codec::Bzip2, Codec::Xz};

std::vector<std::uint8_t> bytes_of(const std::string& s) {
    return {s.begin(), s.end()};
}

/// Compressible but not trivial: repeated records with a varying counter.
std::vector<std::uint8_t> corpus(std::size_t n) {
    std::vector<std::uint8_t> v;
    v.reserve(n + 64);
    std::uint32_t x = 12345;
    while (v.size() < n) {
        x = x * 1103515245u + 12345u;
        const std::string rec = "row," + std::to_string(v.size()) + "," + std::to_string(x % 1000) + "\n";
        v.insert(v.end(), rec.begin(), rec.end());
    }
    v.resize(n);
    return v;
}

std::string write_file(const std::string& name, const std::vector<std::uint8_t>& data) {
    std::ofstream f(name, std::ios::binary);
    f.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    return name;
}

std::vector<std::uint8_t> drain(ByteSource& src, std::size_t read_size) {
    std::vector<std::uint8_t> out;
    std::vector<std::uint8_t> buf(read_size);
    for (;;) {
        const std::size_t k = src.read(buf.data(), buf.size());
        if (k == 0) break;
        out.insert(out.end(), buf.begin(), buf.begin() + static_cast<std::ptrdiff_t>(k));
    }
    return out;
}

/// Hands out at most `step` bytes per read: exercises every frame-boundary split.
class TrickleSource final : public ByteSource {
public:
    TrickleSource(std::vector<std::uint8_t> data, std::size_t step) : d_(std::move(data)), step_(step) {}
    std::size_t read(void* dst, std::size_t n) override {
        const std::size_t k = std::min({n, step_, d_.size() - off_});
        std::memcpy(dst, d_.data() + off_, k);
        off_ += k;
        return k;
    }

private:
    std::vector<std::uint8_t> d_;
    std::size_t step_;
    std::size_t off_ = 0;
};

void write_ustar_member(std::string& tar, const std::string& name, const std::string& data) {
    char hdr[512] = {};
    std::memcpy(hdr, name.c_str(), std::min<std::size_t>(name.size(), 100));
    std::snprintf(hdr + 100, 8, "%07o", 0644);
    std::snprintf(hdr + 108, 8, "%07o", 0);
    std::snprintf(hdr + 116, 8, "%07o", 0);
    std::snprintf(hdr + 124, 12, "%011o", static_cast<unsigned>(data.size()));
    std::snprintf(hdr + 136, 12, "%011o", 0);
    hdr[156] = '0';
    std::memcpy(hdr + 257, "ustar", 5);
    hdr[262] = '0';
    hdr[263] = '0';
    std::memset(hdr + 148, ' ', 8);
    unsigned sum = 0;
    for (int i = 0; i < 512; ++i) sum += static_cast<unsigned char>(hdr[i]);
    std::snprintf(hdr + 148, 8, "%06o", sum);
    tar.append(hdr, 512);
    tar.append(data);
    while (tar.size() % 512 != 0) tar.push_back('\0');
}

std::string tiny_bmp() {
    const unsigned char bmp[] = {
        0x42, 0x4D, 0x3A, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x36, 0x00, 0x00, 0x00, 0x28, 0x00,
        0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x18, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x13, 0x0B, 0x00, 0x00, 0x13, 0x0B, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0x00};
    return std::string(reinterpret_cast<const char*>(bmp), sizeof(bmp));
}

// Produced by CPython 3.11 gzip/zlib/bz2/lzma from "NexusData golden\n" * 3.
const std::string kGoldenText = "NexusData golden\nNexusData golden\nNexusData golden\n";
const std::uint8_t kGzipTwoMembers[] = {  // + a second member "second member\n"
    0x1f,0x8b,0x08,0x00,0x00,0x00,0x00,0x00,0x02,0x0a,0xf3,0x4b,0xad,0x28,0x2d,0x76,0x49,0x2c,0x49,
    0x54,0x48,0xcf,0xcf,0x49,0x49,0xcd,0xe3,0xf2,0x23,0x28,0x00,0x00,0x23,0x98,0x70,0x8d,0x33,0x00,
    0x00,0x00,0x1f,0x8b,0x08,0x00,0x00,0x00,0x00,0x00,0x02,0x0a,0x2b,0x4e,0x4d,0xce,0xcf,0x4b,0x51,
    0xc8,0x4d,0xcd,0x4d,0x4a,0x2d,0xe2,0x02,0x00,0x36,0x18,0x4b,0x0e,0x0e,0x00,0x00,0x00};
const std::uint8_t kZlib[] = {
    0x78,0x9c,0xf3,0x4b,0xad,0x28,0x2d,0x76,0x49,0x2c,0x49,0x54,0x48,0xcf,0xcf,0x49,0x49,0xcd,0xe3,
    0xf2,0x23,0x28,0x00,0x00,0xea,0x83,0x12,0x91};
const std::uint8_t kBzip2[] = {
    0x42,0x5a,0x68,0x39,0x31,0x41,0x59,0x26,0x53,0x59,0xc4,0x4b,0xb1,0xf9,0x00,0x00,0x05,0xd5,0x80,
    0x00,0x10,0x40,0x00,0x04,0x01,0x26,0x85,0x8e,0x40,0x20,0x00,0x23,0x3f,0xf5,0x52,0x69,0xa6,0x6a,
    0x79,0x26,0x29,0x80,0x00,0x86,0x12,0xba,0x9e,0xa5,0x08,0x42,0x92,0xa6,0xd4,0xfd,0xe3,0x82,0xee,
    0x48,0xa7,0x0a,0x12,0x18,0x89,0x76,0x3f,0x20};
const std::uint8_t kXz[] = {
    0xfd,0x37,0x7a,0x58,0x5a,0x00,0x00,0x04,0xe6,0xd6,0xb4,0x46,0x02,0x00,0x21,0x01,0x16,0x00,0x00,
    0x00,0x74,0x2f,0xe5,0xa3,0xe0,0x00,0x32,0x00,0x18,0x5d,0x00,0x27,0x19,0x4b,0x07,0x7a,0x54,0x19,
    0x2b,0xec,0x19,0x93,0x8d,0xd8,0x54,0x96,0x5b,0xaa,0xc3,0x8f,0x11,0x14,0xb4,0x00,0x00,0x00,0xe2,
    0xa4,0xa3,0xbe,0x2a,0x15,0xb8,0x83,0x00,0x01,0x34,0x33,0xca,0x27,0x7c,0x24,0x1f,0xb6,0xf3,0x7d,
    0x01,0x00,0x00,0x00,0x00,0x04,0x59,0x5a};

// numpy 1.24 np.savez_compressed(a=arange(6,'<f4').reshape(2,3), b=array([7,8,9],'<i8')):
// ZIP64 local headers with deflated members.
const std::uint8_t kNpzCompressed[] = {
    0x50,0x4b,0x03,0x04,0x2d,0x00,0x00,0x00,0x08,0x00,0x00,0x00,0x21,0x00,0x4f,0xe9,0x00,0x2a,0xff,
    0xff,0xff,0xff,0xff,0xff,0xff,0xff,0x05,0x00,0x14,0x00,0x61,0x2e,0x6e,0x70,0x79,0x01,0x00,0x10,
    0x00,0x98,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x55,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x9b,0xec,
    0x17,0xea,0x1b,0x10,0xc9,0xc8,0x50,0xc6,0x50,0xad,0x9e,0x92,0x5a,0x9c,0x5c,0xa4,0x6e,0xa5,0xa0,
    0x6e,0x93,0x66,0xa2,0xae,0xa3,0xa0,0x9e,0x96,0x5f,0x54,0x52,0x94,0x98,0x17,0x9f,0x5f,0x94,0x92,
    0x0a,0x12,0x77,0x4b,0xcc,0x29,0x4e,0x05,0x8a,0x17,0x67,0x24,0x16,0xa4,0x02,0xf9,0x1a,0x46,0x3a,
    0x0a,0xc6,0x9a,0x3a,0x0a,0xb5,0x0a,0x64,0x03,0x2e,0x06,0x30,0x68,0xb0,0x07,0x12,0x0e,0x40,0x04,
    0xc4,0x0d,0x40,0xbc,0xc0,0x01,0x00,0x50,0x4b,0x03,0x04,0x2d,0x00,0x00,0x00,0x08,0x00,0x00,0x00,
    0x21,0x00,0x08,0x65,0x1b,0x5b,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0x05,0x00,0x14,0x00,0x62,
    0x2e,0x6e,0x70,0x79,0x01,0x00,0x10,0x00,0x98,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x4d,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x9b,0xec,0x17,0xea,0x1b,0x10,0xc9,0xc8,0x50,0xc6,0x50,0xad,0x9e,0x92,
    0x5a,0x9c,0x5c,0xa4,0x6e,0xa5,0xa0,0x6e,0x93,0x69,0xa1,0xae,0xa3,0xa0,0x9e,0x96,0x5f,0x54,0x52,
    0x94,0x98,0x17,0x9f,0x5f,0x94,0x92,0x0a,0x12,0x77,0x4b,0xcc,0x29,0x4e,0x05,0x8a,0x17,0x67,0x24,
    0x16,0xa4,0x02,0xf9,0x1a,0xc6,0x3a,0x9a,0x3a,0x0a,0xb5,0x0a,0x14,0x00,0x2e,0x76,0x06,0x08,0xe0,
    0x80,0xd2,0x9c,0x50,0x1a,0x00,0x50,0x4b,0x01,0x02,0x2d,0x00,0x2d,0x00,0x00,0x00,0x08,0x00,0x00,
    0x00,0x21,0x00,0x4f,0xe9,0x00,0x2a,0x55,0x00,0x00,0x00,0x98,0x00,0x00,0x00,0x05,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x80,0x01,0x00,0x00,0x00,0x00,0x61,0x2e,0x6e,0x70,0x79,
    0x50,0x4b,0x01,0x02,0x2d,0x00,0x2d,0x00,0x00,0x00,0x08,0x00,0x00,0x00,0x21,0x00,0x08,0x65,0x1b,
    0x5b,0x4d,0x00,0x00,0x00,0x98,0x00,0x00,0x00,0x05,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x80,0x01,0x8c,0x00,0x00,0x00,0x62,0x2e,0x6e,0x70,0x79,0x50,0x4b,0x05,0x06,0x00,0x00,
    0x00,0x00,0x02,0x00,0x02,0x00,0x66,0x00,0x00,0x00,0x10,0x01,0x00,0x00,0x00,0x00};

} // namespace

TEST_CASE("codec detection: suffixes, magic bytes, suffix stripping") {
    CHECK(codec_from_path("data/train.CSV.GZ") == Codec::Gzip);
    CHECK(codec_from_path("shard-000.tgz") == Codec::Gzip);
    CHECK(codec_from_path("a.jsonl.zst") == Codec::Zstd);
    CHECK(codec_from_path("a.lz4") == Codec::Lz4);
    CHECK(codec_from_path("a.tbz2") == Codec::Bzip2);
    CHECK(codec_from_path("a.txz") == Codec::Xz);
    CHECK(codec_from_path("a.csv") == Codec::None);
    CHECK(codec_from_path("dir.gz/file") == Codec::None);

    CHECK(strip_codec_suffix("x/train.csv.gz") == "x/train.csv");
    CHECK(strip_codec_suffix("shard.tgz") == "shard.tar");
    CHECK(strip_codec_suffix("plain.csv") == "plain.csv");

    CHECK(sniff_codec(kGzipTwoMembers, sizeof(kGzipTwoMembers)) == Codec::Gzip);
    CHECK(sniff_codec(kBzip2, sizeof(kBzip2)) == Codec::Bzip2);
    CHECK(sniff_codec(kXz, sizeof(kXz)) == Codec::Xz);
    CHECK(sniff_codec(kZlib, sizeof(kZlib)) == Codec::None); // no reliable magic
    CHECK(sniff_codec("plain", 5) == Codec::None);
    for (Codec c : {Codec::Zstd, Codec::Lz4}) {
        if (!codec_available(c)) continue;
        const auto z = compress(c, "abc", 3);
        CHECK(sniff_codec(z.data(), z.size()) == c);
    }
}

TEST_CASE("golden streams from reference implementations decode") {
    struct Golden {
        Codec codec;
        const std::uint8_t* data;
        std::size_t size;
        std::string expect;
    };
    const Golden goldens[] = {
        {Codec::Gzip, kGzipTwoMembers, sizeof(kGzipTwoMembers), kGoldenText + "second member\n"},
        {Codec::Zlib, kZlib, sizeof(kZlib), kGoldenText},
        {Codec::Bzip2, kBzip2, sizeof(kBzip2), kGoldenText},
        {Codec::Xz, kXz, sizeof(kXz), kGoldenText},
    };
    for (const Golden& g : goldens) {
        if (!codec_available(g.codec)) continue;
        CAPTURE(codec_name(g.codec));
        const auto out = decompress(g.codec, g.data, g.size);
        CHECK(std::string(out.begin(), out.end()) == g.expect);
    }
}

TEST_CASE("round trip: every codec, sizes, levels and multi-frame output") {
    const std::size_t sizes[] = {0, 1, 4095, 300000};
    for (Codec c : kCodecs) {
        if (!codec_available(c)) continue;
        CAPTURE(codec_name(c));
        for (std::size_t n : sizes) {
            CAPTURE(n);
            const auto data = corpus(n);
            const auto z = compress(c, data.data(), data.size());
            if (n == 300000) CHECK(z.size() < n * 3 / 4); // lz4 trades ratio for speed
            CHECK(decompress(c, z.data(), z.size()) == data);

            CompressOptions multi;
            multi.frame_bytes = 7000;
            multi.level = 1;
            const auto zm = compress(c, data.data(), data.size(), multi);
            CHECK(decompress(c, zm.data(), zm.size()) == data);
        }
    }
}

TEST_CASE("DecompressingSource: arbitrary input splits and output sizes") {
    const auto data = corpus(100000);
    for (Codec c : kCodecs) {
        if (!codec_available(c)) continue;
        CAPTURE(codec_name(c));
        CompressOptions multi;
        multi.frame_bytes = 30000;
        const auto z = compress(c, data.data(), data.size(), multi);
        for (std::size_t step : {std::size_t{1}, std::size_t{7}, std::size_t{4096}}) {
            CAPTURE(step);
            DecompressingSource src(std::make_unique<TrickleSource>(z, step), c, 4096);
            CHECK(drain(src, step == 1 ? 3 : 65536) == data);
        }
    }
}

TEST_CASE("corrupt, truncated and trailing-garbage input throw IOError") {
    const auto data = corpus(50000);
    for (Codec c : kCodecs) {
        if (!codec_available(c)) continue;
        CAPTURE(codec_name(c));
        auto z = compress(c, data.data(), data.size());
        const std::vector<std::uint8_t> truncated(z.begin(), z.begin() + static_cast<std::ptrdiff_t>(z.size() / 2));
        CHECK_THROWS_AS((void)decompress(c, truncated.data(), truncated.size()), IOError);
        z[z.size() / 2] ^= 0x5a;
        z[z.size() / 2 + 1] ^= 0xa5;
        // zlib's adler32 / gzip's crc32 / zstd+lz4 checksums / bzip2 CRC / xz CRC64 all catch this.
        bool threw = false;
        try {
            const auto out = decompress(c, z.data(), z.size());
            threw = out != data; // lz4 without content checksum may decode garbage
        } catch (const IOError&) {
            threw = true;
        }
        CHECK(threw);
    }
    if (codec_available(Codec::Gzip)) {
        std::vector<std::uint8_t> g(std::begin(kGzipTwoMembers), std::end(kGzipTwoMembers));
        g.resize(g.size() + 512, 0); // tar-style zero padding is accepted
        CHECK(decompress(Codec::Gzip, g.data(), g.size()).size() == kGoldenText.size() + 14);
        g.push_back('x');
        CHECK_THROWS_AS((void)decompress(Codec::Gzip, g.data(), g.size()), IOError);
    }
}

TEST_CASE("read_input: mapped when plain, decoded when compressed, parallel zstd") {
    const auto data = corpus(2'000'000);
    const std::string plain = write_file("nd_ri_plain.bin", data);
    const InputBytes raw = read_input(plain);
    CHECK(raw.codec() == Codec::None);
    CHECK(std::equal(raw.data(), raw.data() + raw.size(), data.begin(), data.end()));

    for (Codec c : kCodecs) {
        if (!codec_available(c) || c == Codec::Zlib) continue;
        CAPTURE(codec_name(c));
        CompressOptions opt;
        opt.frame_bytes = c == Codec::Zstd ? 256 << 10 : 0; // multi-frame: parallel path
        // No suffix: the codec must come from the magic bytes.
        const std::string path = write_file("nd_ri_packed.bin", compress(c, data.data(), data.size(), opt));
        const InputBytes in = read_input(path, std::nullopt, 4);
        CHECK(in.codec() == c);
        REQUIRE(in.size() == data.size());
        CHECK(std::memcmp(in.data(), data.data(), data.size()) == 0);
        std::remove(path.c_str());
    }
    std::remove(plain.c_str());
}

TEST_CASE("LineReader: CRLF, empty lines, missing final newline, lines longer than a chunk") {
    std::string text = "a\r\n\nlong:";
    text += std::string(10000, 'x');
    text += "\nlast";
    LineReader lr(std::make_unique<TrickleSource>(bytes_of(text), 333), 4096);
    std::vector<std::string> lines;
    std::string_view v;
    while (lr.next(v)) lines.emplace_back(v);
    REQUIRE(lines.size() == 4);
    CHECK(lines[0] == "a");
    CHECK(lines[1].empty());
    CHECK(lines[2].size() == 10005);
    CHECK(lines[3] == "last");
    CHECK(lr.line_number() == 4);
}

TEST_CASE("readers accept compressed files transparently") {
    const std::string csv = "f1,f2,label\n1,2,0\n3,4,1\n5,6,0\n";
    const std::string jsonl = "[1.5, 2.5, 1]\n[3.5, 4.5, 0]\n";
    for (Codec c : {Codec::Gzip, Codec::Zstd, Codec::Bzip2}) {
        if (!codec_available(c)) continue;
        CAPTURE(codec_name(c));
        const std::string ext(c == Codec::Gzip ? ".gz" : c == Codec::Zstd ? ".zst" : ".bz2");

        const std::string csv_path =
            write_file("nd_c.csv" + ext, compress(c, csv.data(), csv.size()));
        CSVDataset ds(csv_path);
        REQUIRE(ds.size() == 3);
        CHECK(ds.get(2).input.data<float>()[1] == 6.0f);

        CSVOptions stream_opt;
        stream_opt.use_mmap = false;
        CHECK(CSVDataset(csv_path, stream_opt).size() == 3);

        const std::string jl_path =
            write_file("nd_c.jsonl" + ext, compress(c, jsonl.data(), jsonl.size()));
        JSONLDataset jds(jl_path);
        REQUIRE(jds.size() == 2);
        CHECK(jds.get(1).input.data<float>()[0] == 3.5f);

        std::string tar;
        write_ustar_member(tar, "k0.bmp", tiny_bmp());
        write_ustar_member(tar, "k0.cls", "5\n");
        tar.append(1024, '\0');
        const std::string tar_path =
            write_file("nd_c.tar" + ext, compress(c, tar.data(), tar.size()));
        WebDataset wds(tar_path);
        REQUIRE(wds.size() == 1);
        const Sample s = wds.get(0);
        CHECK(s.label.data<std::int64_t>()[0] == 5);
        CHECK(s.metadata.at("__key__") == "k0");

        std::remove(csv_path.c_str());
        std::remove(jl_path.c_str());
        std::remove(tar_path.c_str());
    }
}

TEST_CASE("NPY: .npy.gz and numpy's savez_compressed (ZIP64 + deflate)") {
    if (!codec_available(Codec::Gzip)) return;
    const std::string npz = write_file(
        "nd_c.npz", std::vector<std::uint8_t>(std::begin(kNpzCompressed), std::end(kNpzCompressed)));
    NpzFile z(npz);
    CHECK(z.keys().size() == 2);
    const NDArray a = z.get("a.npy");
    REQUIRE(a.shape() == Shape{2, 3});
    CHECK(a.data<float>()[5] == 5.0f);
    const NDArray b = z.get("b.npy");
    REQUIRE(b.shape() == Shape{3});
    CHECK(b.data<std::int64_t>()[2] == 9);

    // The same .npy payload, gzip-compressed on disk.
    const std::vector<std::uint8_t> npy_bytes =
        inflate_raw(kNpzCompressed + 55, 0x55, 0x98); // member "a.npy": 30 + 5 + 20 header bytes
    const std::string gz = write_file("nd_c.npy.gz", compress(Codec::Gzip, npy_bytes.data(), npy_bytes.size()));
    NpyFile f(gz);
    CHECK(f.array().shape() == Shape{2, 3});
    CHECK(f.array().data<float>()[4] == 4.0f);
    CHECK(load_npy(gz).view.data<float>()[3] == 3.0f);
    std::remove(npz.c_str());
    std::remove(gz.c_str());
}
