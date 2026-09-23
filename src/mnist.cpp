#include "nexusdata/dataset/mnist.hpp"

#include <cstring>
#include <filesystem>

#include "nexusdata/core/error.hpp"

namespace fs = std::filesystem;

namespace nexusdata {

namespace {

std::uint32_t read_be_u32(const std::uint8_t* p) {
    return (static_cast<std::uint32_t>(p[0]) << 24) |
           (static_cast<std::uint32_t>(p[1]) << 16) |
           (static_cast<std::uint32_t>(p[2]) << 8) |
           static_cast<std::uint32_t>(p[3]);
}

} // namespace

MNIST::MNIST(std::string root, MNISTOptions options) {
    fs::path base(root);
    std::string img_name = options.images_file;
    std::string lab_name = options.labels_file;
    if (img_name.empty()) {
        img_name = options.train ? "train-images-idx3-ubyte" : "t10k-images-idx3-ubyte";
    }
    if (lab_name.empty()) {
        lab_name = options.train ? "train-labels-idx1-ubyte" : "t10k-labels-idx1-ubyte";
    }

    images_map_ = MappedFile((base / img_name).string());
    labels_map_ = MappedFile((base / lab_name).string());

    if (images_map_.size() < 16) {
        throw IOError("MNIST: images file too small");
    }
    if (labels_map_.size() < 8) {
        throw IOError("MNIST: labels file too small");
    }

    const auto* img = images_map_.data();
    const auto* lab = labels_map_.data();
    const std::uint32_t magic_img = read_be_u32(img);
    const std::uint32_t magic_lab = read_be_u32(lab);
    if (magic_img != 2051) {
        throw IOError("MNIST: bad images magic " + std::to_string(magic_img));
    }
    if (magic_lab != 2049) {
        throw IOError("MNIST: bad labels magic " + std::to_string(magic_lab));
    }
    n_ = read_be_u32(img + 4);
    rows_ = static_cast<int>(read_be_u32(img + 8));
    cols_ = static_cast<int>(read_be_u32(img + 12));
    const std::uint32_t n_lab = read_be_u32(lab + 4);
    if (n_ != n_lab) {
        throw IOError("MNIST: images/labels count mismatch");
    }
    const std::size_t need_img = 16 + n_ * static_cast<std::size_t>(rows_) * cols_;
    const std::size_t need_lab = 8 + n_;
    if (images_map_.size() < need_img || labels_map_.size() < need_lab) {
        throw IOError("MNIST: truncated file");
    }
    images_data_ = img + 16;
    labels_data_ = lab + 8;
}

std::size_t MNIST::size() const {
    return n_;
}

Sample MNIST::get(std::size_t index) const {
    if (index >= n_) {
        throw IndexError("MNIST::get: index out of range");
    }
    const std::size_t pix = static_cast<std::size_t>(rows_) * cols_;
    Sample s;
    s.input = NDArray(Shape{static_cast<std::size_t>(rows_), static_cast<std::size_t>(cols_), 1},
                      DType::UInt8);
    std::memcpy(s.input.data(), images_data_ + index * pix, pix);
    s.label = NDArray(Shape{1}, DType::Int64);
    s.label.data<std::int64_t>()[0] = static_cast<std::int64_t>(labels_data_[index]);
    return s;
}

} // namespace nexusdata
