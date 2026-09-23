#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "nexusdata/dataset/npy.hpp"

using namespace nexusdata;
namespace fs = std::filesystem;

namespace {

void write_npy_f32(const fs::path& path, const Shape& shape, const std::vector<float>& data) {
    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i) header += ", ";
        header += std::to_string(shape[i]);
    }
    if (shape.size() == 1) {
        header += ",";
    }
    header += "), }";
    // pad header to 16-byte alignment for v1.0: 10 + header_len divisible by 16
    while ((10 + header.size()) % 16 != 0) {
        header.push_back(' ');
    }
    std::ofstream out(path, std::ios::binary);
    const unsigned char magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
    out.write(reinterpret_cast<const char*>(magic), 8);
    const auto hlen = static_cast<std::uint16_t>(header.size());
    out.put(static_cast<char>(hlen & 0xff));
    out.put(static_cast<char>((hlen >> 8) & 0xff));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    out.write(reinterpret_cast<const char*>(data.data()),
              static_cast<std::streamsize>(data.size() * sizeof(float)));
}

void write_npz_stored(const fs::path& path,
                        const std::string& member_name,
                        const std::vector<char>& npy_bytes) {
    // Minimal ZIP with one stored member.
    std::ofstream out(path, std::ios::binary);
    const std::uint32_t sig_local = 0x04034b50u;
    const std::uint16_t ver = 20;
    const std::uint16_t flags = 0;
    const std::uint16_t method = 0;
    const std::uint16_t time = 0, date = 0;
    const std::uint32_t crc = 0;
    const auto comp = static_cast<std::uint32_t>(npy_bytes.size());
    const auto uncomp = comp;
    const auto name_len = static_cast<std::uint16_t>(member_name.size());
    const std::uint16_t extra = 0;

    auto put_u16 = [&](std::uint16_t v) {
        out.put(static_cast<char>(v & 0xff));
        out.put(static_cast<char>((v >> 8) & 0xff));
    };
    auto put_u32 = [&](std::uint32_t v) {
        out.put(static_cast<char>(v & 0xff));
        out.put(static_cast<char>((v >> 8) & 0xff));
        out.put(static_cast<char>((v >> 16) & 0xff));
        out.put(static_cast<char>((v >> 24) & 0xff));
    };

    const std::uint32_t local_offset = 0;
    put_u32(sig_local);
    put_u16(ver);
    put_u16(flags);
    put_u16(method);
    put_u16(time);
    put_u16(date);
    put_u32(crc);
    put_u32(comp);
    put_u32(uncomp);
    put_u16(name_len);
    put_u16(extra);
    out.write(member_name.data(), name_len);
    out.write(npy_bytes.data(), static_cast<std::streamsize>(npy_bytes.size()));

    const auto central_offset = static_cast<std::uint32_t>(out.tellp());
    put_u32(0x02014b50u);
    put_u16(ver);
    put_u16(ver);
    put_u16(flags);
    put_u16(method);
    put_u16(time);
    put_u16(date);
    put_u32(crc);
    put_u32(comp);
    put_u32(uncomp);
    put_u16(name_len);
    put_u16(extra);
    put_u16(0); // comment
    put_u16(0);
    put_u16(0);
    put_u32(0);
    put_u32(local_offset);
    out.write(member_name.data(), name_len);

    const auto end_offset = static_cast<std::uint32_t>(out.tellp());
    put_u32(0x06054b50u);
    put_u16(0);
    put_u16(0);
    put_u16(1);
    put_u16(1);
    put_u32(end_offset - central_offset);
    put_u32(central_offset);
    put_u16(0);
}

std::vector<char> read_file_bytes(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    return std::vector<char>(std::istreambuf_iterator<char>(in), {});
}

} // namespace

TEST_CASE("load_npy float32 matrix") {
    const fs::path path = "test_arr.npy";
    write_npy_f32(path, {2, 3}, {1, 2, 3, 4, 5, 6});
    auto arr = load_npy(path.string(), true);
    CHECK(arr.view.shape() == Shape{2, 3});
    CHECK(arr.view.dtype() == DType::Float32);
    CHECK(arr.view.data<float>()[5] == doctest::Approx(6.0f));
    fs::remove(path);
}

TEST_CASE("NpyFile zero-copy view stays valid") {
    const fs::path path = "test_zc.npy";
    write_npy_f32(path, {4}, {10, 20, 30, 40});
    {
        NpyFile file(path.string());
        CHECK(file.array().data<float>()[2] == doctest::Approx(30.0f));
    }
    fs::remove(path);
}

TEST_CASE("NpyDataset get rows") {
    const fs::path fx = "test_feat.npy";
    const fs::path ly = "test_lab.npy";
    write_npy_f32(fx, {3, 2}, {1, 2, 3, 4, 5, 6});
    write_npy_f32(ly, {3}, {0, 1, 0});
    {
        NpyDataset ds(fx.string(), ly.string());
        CHECK(ds.size() == 3);
        Sample s = ds.get(1);
        CHECK(s.input.data<float>()[0] == doctest::Approx(3.0f));
        CHECK(s.input.data<float>()[1] == doctest::Approx(4.0f));
        CHECK(s.label.data<float>()[0] == doctest::Approx(1.0f));
    }
    fs::remove(fx);
    fs::remove(ly);
}

TEST_CASE("NpzFile stored member") {
    const fs::path npy = "tmp_member.npy";
    const fs::path npz = "test_arr.npz";
    write_npy_f32(npy, {2}, {7, 8});
    auto bytes = read_file_bytes(npy);
    write_npz_stored(npz, "a.npy", bytes);
    {
        NpzFile z(npz.string());
        CHECK(z.contains("a.npy"));
        NDArray a = z.get("a.npy");
        CHECK(a.shape() == Shape{2});
        CHECK(a.data<float>()[1] == doctest::Approx(8.0f));
    }
    fs::remove(npy);
    fs::remove(npz);
}
