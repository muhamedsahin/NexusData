#include <doctest/doctest.h>

#include <vector>

#include "nexusdata/backend/cuda_api.hpp"
#include "nexusdata/backend/device.hpp"
#include "nexusdata/backend/philox.hpp"
#include "nexusdata/backend/pinned.hpp"
#include "nexusdata/core/ndarray.hpp"
#include "nexusdata/loading/dataloader.hpp"
#include "nexusdata/dataset/in_memory.hpp"

using namespace nexusdata;

TEST_CASE("Philox golden deterministic") {
    auto a = Philox4x32::from_sample(42, 7, 1);
    auto b = Philox4x32::from_sample(42, 7, 1);
    std::uint32_t oa[4], ob[4];
    a.next(oa);
    b.next(ob);
    CHECK(oa[0] == ob[0]);
    CHECK(oa[1] == ob[1]);
    CHECK(oa[2] == ob[2]);
    CHECK(oa[3] == ob[3]);

    auto c = Philox4x32::from_sample(42, 8, 1);
    c.next(ob);
    CHECK(oa[0] != ob[0]); // different sample index ⇒ different draw
}

TEST_CASE("resolve_device Auto without CUDA stays host") {
    AutoDevicePolicy p;
    p.allow_cuda = true;
    Device d = resolve_device(Device::auto_select(), GpuOpKind::ImageAugment, 1u << 30, p);
    if (!cuda_runtime_available()) {
        CHECK(d.is_host());
    }
}

TEST_CASE("resolve_device TabularSmall always host") {
    Device d = resolve_device(Device::auto_select(), GpuOpKind::TabularSmall, 1u << 30);
    CHECK(d.is_host());
}

TEST_CASE("pinned alloc works without CUDA") {
    void* p = pinned_alloc(1024);
    CHECK(p != nullptr);
    pinned_free(p);
    auto buf = make_pinned_buffer(256);
    CHECK(buf.get() != nullptr);
}

TEST_CASE("NDArray::pinned and batch_to_tensor_normalize host") {
    NDArray img(Shape{2, 4, 4, 3}, DType::UInt8);
    auto* px = img.data<std::uint8_t>();
    for (std::size_t i = 0; i < img.numel(); ++i) {
        px[i] = static_cast<std::uint8_t>(i & 255);
    }
    auto out = batch_to_tensor_normalize(img, {0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f},
                                         Device::host());
    CHECK(out.shape() == Shape{2, 3, 4, 4});
    CHECK(out.device().is_host());
    CHECK(out.dtype() == DType::Float32);
}

TEST_CASE("DataLoader pin_memory on host") {
    NDArray features({8, 2}, DType::Float32);
    NDArray labels({8}, DType::Int64);
    for (int i = 0; i < 8; ++i) {
        features.data<float>()[i * 2] = static_cast<float>(i);
        labels.data<int64_t>()[i] = i;
    }
    InMemoryDataset ds(std::move(features), std::move(labels));
    DataLoaderOptions opt;
    opt.batch_size = 4;
    opt.pin_memory = true;
    opt.device = Device::host();
    DataLoader loader(ds, opt);
    for (const Batch& b : loader) {
        CHECK(b.inputs.device().is_host());
        CHECK(b.inputs.shape()[0] == 4);
    }
}

TEST_CASE("cuda_api available flag matches build") {
#if defined(NEXUSDATA_WITH_CUDA)
    // May still be false if no GPU present
    (void)cuda_api::available();
#else
    CHECK_FALSE(cuda_api::available());
    CHECK(cuda_api::device_count() == 0);
#endif
}
