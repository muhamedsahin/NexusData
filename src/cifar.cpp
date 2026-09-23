#include "nexusdata/dataset/cifar.hpp"

#include <cstring>
#include <filesystem>

#include "nexusdata/core/error.hpp"

namespace fs = std::filesystem;

namespace nexusdata {

CIFAR::CIFAR(std::string root, CIFAROptions options) : cifar100_(options.cifar100) {
    label_bytes_ = cifar100_ ? 2 : 1; // coarse+fine vs single label; we use fine (last byte)
    const std::size_t record_size = label_bytes_ + 3072;
    fs::path base(root);

    std::vector<std::string> files = options.batch_files;
    if (files.empty()) {
        if (cifar100_) {
            files = {"train.bin"}; // caller may also pass test.bin
        } else {
            files = {"data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin",
                     "data_batch_4.bin", "data_batch_5.bin"};
        }
    }

    for (const auto& name : files) {
        const fs::path p = base / name;
        if (!fs::exists(p)) {
            throw IOError("CIFAR: missing batch file \"" + p.string() + "\"");
        }
        Batch b;
        b.map = MappedFile(p.string());
        if (b.map.size() % record_size != 0) {
            throw IOError("CIFAR: unexpected file size for \"" + p.string() + "\"");
        }
        b.count = b.map.size() / record_size;
        b.record_size = record_size;
        b.data = b.map.data();
        n_ += b.count;
        batches_.push_back(std::move(b));
    }
    if (n_ == 0) {
        throw IOError("CIFAR: no samples");
    }
}

std::size_t CIFAR::size() const {
    return n_;
}

Sample CIFAR::get(std::size_t index) const {
    if (index >= n_) {
        throw IndexError("CIFAR::get: index out of range");
    }
    std::size_t remaining = index;
    const Batch* batch = nullptr;
    for (const auto& b : batches_) {
        if (remaining < b.count) {
            batch = &b;
            break;
        }
        remaining -= b.count;
    }
    if (!batch) {
        throw IndexError("CIFAR::get: internal indexing error");
    }

    const std::uint8_t* rec = batch->data + remaining * batch->record_size;
    std::int64_t label = 0;
    if (cifar100_) {
        label = static_cast<std::int64_t>(rec[1]); // fine label
    } else {
        label = static_cast<std::int64_t>(rec[0]);
    }
    const std::uint8_t* pixels = rec + label_bytes_;

    // CIFAR stored as CHW planar RGB 32x32. Convert to HWC.
    Sample s;
    s.input = NDArray(Shape{32, 32, 3}, DType::UInt8);
    auto* dst = s.input.data<std::uint8_t>();
    for (int y = 0; y < 32; ++y) {
        for (int x = 0; x < 32; ++x) {
            const int plane = y * 32 + x;
            dst[(y * 32 + x) * 3 + 0] = pixels[0 * 1024 + plane];
            dst[(y * 32 + x) * 3 + 1] = pixels[1 * 1024 + plane];
            dst[(y * 32 + x) * 3 + 2] = pixels[2 * 1024 + plane];
        }
    }
    s.label = NDArray(Shape{1}, DType::Int64);
    s.label.data<std::int64_t>()[0] = label;
    return s;
}

} // namespace nexusdata
