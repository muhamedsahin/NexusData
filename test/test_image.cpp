#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "nexusdata/dataset/image_folder.hpp"
#include "nexusdata/dataset/map.hpp"
#include "nexusdata/image/image.hpp"
#include "nexusdata/pipeline/image.hpp"

using namespace nexusdata;
namespace fs = std::filesystem;

namespace {

void write_ppm(const fs::path& path, int w, int h, std::uint8_t r, std::uint8_t g, std::uint8_t b) {
    std::ofstream out(path, std::ios::binary);
    out << "P6\n" << w << " " << h << "\n255\n";
    for (int i = 0; i < w * h; ++i) {
        out.put(static_cast<char>(r));
        out.put(static_cast<char>(g));
        out.put(static_cast<char>(b));
    }
}

Sample make_hwc(int h, int w, int c, std::uint8_t fill) {
    Sample s;
    s.input = NDArray(Shape{static_cast<std::size_t>(h), static_cast<std::size_t>(w),
                            static_cast<std::size_t>(c)},
                      DType::UInt8);
    std::memset(s.input.data(), fill, s.input.nbytes());
    // gradient on first channel
    auto* p = s.input.data<std::uint8_t>();
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            p[(y * w + x) * c] = static_cast<std::uint8_t>((x + y) & 255);
        }
    }
    s.label = NDArray(Shape{1}, DType::Int64);
    s.label.data<std::int64_t>()[0] = 0;
    return s;
}

} // namespace

TEST_CASE("load_image PPM via stb") {
    const fs::path dir = "test_img_tmp";
    fs::create_directories(dir);
    const auto path = dir / "red.ppm";
    write_ppm(path, 4, 3, 255, 0, 0);
    Image img = load_image(path.string());
    CHECK(img.width == 4);
    CHECK(img.height == 3);
    CHECK(img.channels == 3);
    CHECK(img.data.data<std::uint8_t>()[0] == 255);
    CHECK(img.data.data<std::uint8_t>()[1] == 0);
    fs::remove_all(dir);
}

TEST_CASE("ImageFolder class labels from folder names") {
    const fs::path root = "test_imagefolder";
    fs::remove_all(root);
    fs::create_directories(root / "cat");
    fs::create_directories(root / "dog");
    write_ppm(root / "cat" / "a.ppm", 2, 2, 1, 2, 3);
    write_ppm(root / "dog" / "b.ppm", 2, 2, 4, 5, 6);
    write_ppm(root / "cat" / "c.ppm", 2, 2, 7, 8, 9);

    ImageFolder ds(root.string());
    CHECK(ds.classes().size() == 2);
    CHECK(ds.classes()[0] == "cat");
    CHECK(ds.classes()[1] == "dog");
    CHECK(ds.size() == 3);

    std::size_t cats = 0, dogs = 0;
    for (std::size_t i = 0; i < ds.size(); ++i) {
        Sample s = ds.get(i);
        CHECK(s.input.shape() == Shape{2, 2, 3});
        if (s.label.data<std::int64_t>()[0] == 0) ++cats;
        if (s.label.data<std::int64_t>()[0] == 1) ++dogs;
    }
    CHECK(cats == 2);
    CHECK(dogs == 1);
    fs::remove_all(root);
}

TEST_CASE("Resize CenterCrop ToTensor Normalize") {
    Sample s = make_hwc(8, 8, 3, 0);
    Resize resize(4, 4, ResizeInterp::Nearest);
    Sample r = resize(s);
    CHECK(r.input.shape() == Shape{4, 4, 3});

    CenterCrop crop(2, 2);
    Sample c = crop(r);
    CHECK(c.input.shape() == Shape{2, 2, 3});

    ToTensor to_tensor;
    Sample t = to_tensor(c);
    CHECK(t.input.shape() == Shape{3, 2, 2});
    CHECK(t.input.dtype() == DType::Float32);

    NormalizeImage norm({0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f});
    Sample n = norm(t);
    CHECK(n.input.shape() == Shape{3, 2, 2});
}

TEST_CASE("RandomHorizontalFlip determinism") {
    Sample s = make_hwc(2, 2, 1, 0);
    RandomHorizontalFlip a(1.0f, 99);
    RandomHorizontalFlip b(1.0f, 99);
    auto xa = a(s).input.data<std::uint8_t>()[0];
    auto xb = b(s).input.data<std::uint8_t>()[0];
    CHECK(xa == xb);
}

TEST_CASE("Grayscale and Pad") {
    Sample s = make_hwc(3, 3, 3, 10);
    Grayscale g(1);
    Sample gray = g(s);
    CHECK(gray.input.shape() == Shape{3, 3, 1});
    Pad pad(1, 0);
    Sample p = pad(gray);
    CHECK(p.input.shape() == Shape{5, 5, 1});
}

TEST_CASE("MapDataset with image transform") {
    const fs::path root = "test_map_img";
    fs::remove_all(root);
    fs::create_directories(root / "a");
    write_ppm(root / "a" / "x.ppm", 8, 8, 10, 20, 30);
    ImageFolder folder(root.string());
    auto tf = std::make_shared<Resize>(4, 4);
    MapDataset mapped(folder, tf);
    Sample s = mapped.get(0);
    CHECK(s.input.shape() == Shape{4, 4, 3});
    fs::remove_all(root);
}
