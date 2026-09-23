#include <doctest/doctest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

#include "nexusdata/dataset/cifar.hpp"
#include "nexusdata/dataset/mnist.hpp"

using namespace nexusdata;
namespace fs = std::filesystem;

namespace {

void write_be_u32(std::ostream& out, std::uint32_t v) {
    out.put(static_cast<char>((v >> 24) & 0xff));
    out.put(static_cast<char>((v >> 16) & 0xff));
    out.put(static_cast<char>((v >> 8) & 0xff));
    out.put(static_cast<char>(v & 0xff));
}

void write_mnist(const fs::path& root, std::size_t n, int rows, int cols) {
    fs::create_directories(root);
    {
        std::ofstream img((root / "train-images-idx3-ubyte").string(), std::ios::binary);
        write_be_u32(img, 2051);
        write_be_u32(img, static_cast<std::uint32_t>(n));
        write_be_u32(img, static_cast<std::uint32_t>(rows));
        write_be_u32(img, static_cast<std::uint32_t>(cols));
        for (std::size_t i = 0; i < n; ++i) {
            for (int p = 0; p < rows * cols; ++p) {
                img.put(static_cast<char>((i + p) & 255));
            }
        }
    }
    {
        std::ofstream lab((root / "train-labels-idx1-ubyte").string(), std::ios::binary);
        write_be_u32(lab, 2049);
        write_be_u32(lab, static_cast<std::uint32_t>(n));
        for (std::size_t i = 0; i < n; ++i) {
            lab.put(static_cast<char>(i % 10));
        }
    }
}

void write_cifar10_batch(const fs::path& path, std::size_t n) {
    std::ofstream out(path, std::ios::binary);
    for (std::size_t i = 0; i < n; ++i) {
        out.put(static_cast<char>(i % 10)); // label
        for (int p = 0; p < 3072; ++p) {
            out.put(static_cast<char>((i + p) & 255));
        }
    }
}

} // namespace

TEST_CASE("MNIST IDX reader") {
    const fs::path root = "test_mnist_data";
    fs::remove_all(root);
    write_mnist(root, 5, 28, 28);
    {
        MNIST ds(root.string());
        CHECK(ds.size() == 5);
        CHECK(ds.image_rows() == 28);
        Sample s = ds.get(2);
        CHECK(s.input.shape() == Shape{28, 28, 1});
        CHECK(s.label.data<std::int64_t>()[0] == 2);
        CHECK(s.input.data<std::uint8_t>()[0] == static_cast<std::uint8_t>((2 + 0) & 255));
    }
    fs::remove_all(root);
}

TEST_CASE("CIFAR-10 binary reader") {
    const fs::path root = "test_cifar_data";
    fs::remove_all(root);
    fs::create_directories(root);
    write_cifar10_batch(root / "data_batch_1.bin", 4);
    {
        CIFAROptions opt;
        opt.batch_files = {"data_batch_1.bin"};
        CIFAR ds(root.string(), opt);
        CHECK(ds.size() == 4);
        Sample s = ds.get(1);
        CHECK(s.input.shape() == Shape{32, 32, 3});
        CHECK(s.label.data<std::int64_t>()[0] == 1);
    }
    fs::remove_all(root);
}
