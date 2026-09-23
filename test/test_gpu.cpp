// v2.0 Phase 2: CUDA kernels, GPU executor, GPU transforms, CostModel feedback,
// telemetry and nvJPEG. Host-side contracts run in every build; GPU cases are
// skipped (with a message) when no CUDA device is usable.

#include <doctest/doctest.h>

#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

#include "nexusdata/backend/cost_model.hpp"
#include "nexusdata/backend/cuda_api.hpp"
#include "nexusdata/backend/detail/kernel_math.hpp"
#include "nexusdata/backend/device.hpp"
#include "nexusdata/backend/gpu_executor.hpp"
#include "nexusdata/backend/gpu_telemetry.hpp"
#include "nexusdata/backend/kernels.hpp"
#include "nexusdata/core/error.hpp"
#include "nexusdata/core/random.hpp"
#include "nexusdata/dataset/dataset.hpp"
#include "nexusdata/image/image.hpp"
#include "nexusdata/image/nvjpeg.hpp"
#include "nexusdata/loading/dataloader.hpp"
#include "nexusdata/pipeline/augment.hpp"
#include "nexusdata/pipeline/fusion.hpp"
#include "nexusdata/pipeline/image.hpp"
#include "nexusdata/pipeline/tabular.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include "stb_image_write.h"

using namespace nexusdata;

namespace {

bool gpu() {
    static const bool ok = GpuExecutor::usable(0);
    return ok;
}

#define SKIP_WITHOUT_GPU()                                   \
    do {                                                     \
        if (!gpu()) {                                        \
            MESSAGE("no usable CUDA device: GPU case skipped"); \
            return;                                          \
        }                                                    \
    } while (0)

NDArray random_images(std::size_t n, std::size_t h, std::size_t w, std::size_t c,
                      std::uint64_t seed) {
    NDArray x(Shape{n, h, w, c}, DType::UInt8);
    PCG32 rng(seed);
    auto* p = x.data<std::uint8_t>();
    for (std::size_t i = 0; i < x.numel(); ++i) {
        p[i] = static_cast<std::uint8_t>(rng.next_u32());
    }
    return x;
}

NDArray row(const NDArray& batch, std::size_t i) {
    Shape s(batch.shape().begin() + 1, batch.shape().end());
    NDArray r(s, batch.dtype());
    const std::size_t bytes = r.nbytes();
    std::memcpy(r.data(), static_cast<const std::uint8_t*>(batch.data()) + i * bytes, bytes);
    return r;
}

template <typename T>
bool same_value(T a, T b) {
    if constexpr (std::is_floating_point_v<T>) {
        if (std::isnan(a) || std::isnan(b)) {
            return std::isnan(a) && std::isnan(b); // NaN payloads are not portable
        }
    }
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}

/// Bitwise equality (NaN-aware); both arrays are brought to the host.
bool identical(const NDArray& a_in, const NDArray& b_in) {
    const NDArray a = a_in.cpu();
    const NDArray b = b_in.cpu();
    if (a.shape() != b.shape() || a.dtype() != b.dtype()) {
        return false;
    }
    const std::size_t n = a.numel();
    switch (a.dtype()) {
        case DType::Float32:
            for (std::size_t i = 0; i < n; ++i)
                if (!same_value(a.data<float>()[i], b.data<float>()[i])) return false;
            return true;
        case DType::Float64:
            for (std::size_t i = 0; i < n; ++i)
                if (!same_value(a.data<double>()[i], b.data<double>()[i])) return false;
            return true;
        default:
            return std::memcmp(a.data(), b.data(), a.nbytes()) == 0;
    }
}

Batch image_batch(NDArray x) {
    Batch b;
    b.labels = NDArray(Shape{x.shape()[0]}, DType::Int64);
    b.inputs = std::move(x);
    return b;
}

Batch run_gpu(const Transform& t, const Batch& b, bool on_device) {
    GpuExecutor& ex = GpuExecutor::instance(0);
    auto lease = ex.acquire();
    GpuBatchContext ctx;
    ctx.executor = &ex;
    ctx.lane = &lease.lane();
    ctx.output_on_device = on_device;
    return t.apply_batch_gpu(b, ctx);
}

/// Unfused v1 chain for one sample with explicit crop origin / flips.
NDArray reference_augment(const NDArray& img, const AugmentSpec& spec,
                          const kernels::AugmentSample& a) {
    Sample s;
    s.input = img;
    const std::size_t w = img.shape()[1], c = img.shape()[2];
    if (spec.crop != AugmentSpec::Crop::None) {
        NDArray crop(Shape{spec.crop_height, spec.crop_width, c}, DType::UInt8);
        for (std::size_t y = 0; y < spec.crop_height; ++y) {
            std::memcpy(crop.data<std::uint8_t>() + y * spec.crop_width * c,
                        img.data<std::uint8_t>() +
                            ((static_cast<std::size_t>(a.crop_y) + y) * w +
                             static_cast<std::size_t>(a.crop_x)) * c,
                        spec.crop_width * c);
        }
        s.input = crop;
    }
    if (spec.out_height != 0 &&
        (spec.out_height != s.input.shape()[0] || spec.out_width != s.input.shape()[1])) {
        s = Resize(static_cast<int>(spec.out_height), static_cast<int>(spec.out_width),
                   spec.interp == simd::ResampleMode::Nearest ? ResizeInterp::Nearest
                                                              : ResizeInterp::Bilinear)
                .apply(s);
    }
    if (a.hflip) s = RandomHorizontalFlip(1.0f).apply(s);
    if (a.vflip) s = RandomVerticalFlip(1.0f).apply(s);
    if (spec.to_tensor) {
        s = ToTensor().apply(s);
        if (!spec.mean.empty()) s = NormalizeImage(spec.mean, spec.stddev).apply(s);
    }
    return s.input;
}

AugmentSpec imagenet_like() {
    AugmentSpec spec;
    spec.crop = AugmentSpec::Crop::Random;
    spec.crop_height = 45;
    spec.crop_width = 51;
    spec.out_height = 32;
    spec.out_width = 29;
    spec.hflip_p = 0.5f;
    spec.vflip_p = 0.3f;
    spec.mean = {0.485f, 0.456f, 0.406f};
    spec.stddev = {0.229f, 0.224f, 0.225f};
    return spec;
}

/// [N, H, W, C] image dataset (samples keep their shape; use collate_stack_nd).
class ImageDataset : public Dataset {
public:
    explicit ImageDataset(NDArray images) : images_(std::move(images)) {}
    [[nodiscard]] std::size_t size() const override { return images_.shape()[0]; }
    [[nodiscard]] Sample get(std::size_t i) const override {
        Sample s;
        s.input = row(images_, i);
        s.label = NDArray(Shape{1}, DType::Int64);
        s.label.data<std::int64_t>()[0] = static_cast<std::int64_t>(i);
        return s;
    }

private:
    NDArray images_;
};

NDArray tabular(std::size_t rows, std::size_t cols, DType dt, std::uint64_t seed) {
    NDArray x(Shape{rows, cols}, dt);
    PCG32 rng(seed);
    for (std::size_t i = 0; i < x.numel(); ++i) {
        const double v = (static_cast<double>(rng.next_u32()) / 4294967296.0 - 0.3) * 20.0;
        if (dt == DType::Float32) x.data<float>()[i] = static_cast<float>(v);
        else x.data<double>()[i] = v;
    }
    // Special values in the first row.
    const double specials[] = {std::numeric_limits<double>::quiet_NaN(),
                               std::numeric_limits<double>::infinity(),
                               -std::numeric_limits<double>::infinity(), -0.0, 1e-310};
    for (std::size_t k = 0; k < std::min<std::size_t>(cols, 5); ++k) {
        if (dt == DType::Float32) x.data<float>()[k] = static_cast<float>(specials[k]);
        else x.data<double>()[k] = specials[k];
    }
    return x;
}

TransformConstPtr tabular_chain(std::size_t cols) {
    std::vector<double> mean(cols), scale(cols), lo(cols), hi(cols);
    for (std::size_t j = 0; j < cols; ++j) {
        mean[j] = 0.37 * static_cast<double>(j) - 1.0;
        scale[j] = 1.0 + 0.11 * static_cast<double>(j);
        lo[j] = -2.0 - static_cast<double>(j);
        hi[j] = 3.0 + 0.5 * static_cast<double>(j);
    }
    return std::make_shared<Compose>(std::vector<TransformConstPtr>{
        std::make_shared<ClipTransform>(-8.0, 9.5),
        std::make_shared<StandardizeTransform>(mean, scale),
        std::make_shared<MinMaxTransform>(lo, hi, -1.0, 1.0),
    });
}

} // namespace

// --- host contracts (every build) ------------------------------------------------------

TEST_CASE("kernel math: integer nearest tap equals the v1 double formula") {
    for (int src = 1; src <= 300; src += 7) {
        for (int dst = 1; dst <= 300; dst += 5) {
            for (int k = 0; k < dst; ++k) {
                const auto v1 = std::min<std::size_t>(
                    static_cast<std::size_t>(src) - 1,
                    static_cast<std::size_t>(k * static_cast<double>(src) / dst));
                REQUIRE(static_cast<std::size_t>(kmath::nearest_tap(k, src, dst)) == v1);
            }
        }
    }
    CHECK(kmath::nearest_tap(4095, 4096, 4096) == 4095);
}

TEST_CASE("FusedAugment CPU equals the unfused crop/resize/flip/normalize chain") {
    const NDArray imgs = random_images(6, 60, 70, 3, 11);
    for (const auto interp : {simd::ResampleMode::Bilinear, simd::ResampleMode::Nearest}) {
        AugmentSpec spec = imagenet_like();
        spec.interp = interp;
        FusedAugment probe(spec, 99);
        std::vector<kernels::AugmentSample> geo;
        (void)probe.plan(6, 60, 70, 3, geo);

        FusedAugment aug(spec, 99);
        const Batch out = aug.apply_batch(image_batch(imgs));
        REQUIRE(out.inputs.shape() == Shape{6, 3, 32, 29});
        bool flipped = false;
        for (std::size_t i = 0; i < 6; ++i) {
            flipped = flipped || geo[i].hflip || geo[i].vflip;
            CHECK(identical(row(out.inputs, i), reference_augment(row(imgs, i), spec, geo[i])));
        }
        CHECK(flipped);
    }
}

TEST_CASE("FusedAugment: center crop, uint8 output and per-sample apply") {
    const NDArray imgs = random_images(3, 40, 33, 4, 5);
    AugmentSpec spec;
    spec.crop = AugmentSpec::Crop::Center;
    spec.crop_height = 21;
    spec.crop_width = 17;
    spec.hflip_p = 1.0f;
    spec.to_tensor = false;
    FusedAugment aug(spec);
    CHECK_FALSE(aug.is_random());
    const Batch out = aug.apply_batch(image_batch(imgs));
    REQUIRE(out.inputs.shape() == Shape{3, 21, 17, 4});
    for (std::size_t i = 0; i < 3; ++i) {
        Sample s;
        s.input = row(imgs, i);
        const NDArray ref =
            RandomHorizontalFlip(1.0f).apply(CenterCrop(21, 17).apply(s)).input;
        CHECK(identical(row(out.inputs, i), ref));
        CHECK(identical(aug.apply(s).input, ref));
    }
    AugmentSpec bad = spec;
    bad.crop_height = 41;
    CHECK_THROWS_AS((void)FusedAugment(bad).apply_batch(image_batch(imgs)), ShapeError);
}

TEST_CASE("FusedAugment: seeded draws are reproducible and batch == per-sample") {
    const NDArray imgs = random_images(5, 48, 56, 3, 3);
    const AugmentSpec spec = imagenet_like();
    FusedAugment a(spec, 7), b(spec, 7);
    const Batch ba = a.apply_batch(image_batch(imgs));
    for (std::size_t i = 0; i < 5; ++i) {
        Sample s;
        s.input = row(imgs, i);
        CHECK(identical(row(ba.inputs, i), b.apply(s).input));
    }
    a.set_seed(7);
    CHECK(identical(a.apply_batch(image_batch(imgs)).inputs, ba.inputs));
}

TEST_CASE("FusedImageTransform batch path equals per-sample path") {
    const NDArray imgs = random_images(4, 37, 23, 3, 21);
    Compose chain({std::make_shared<Resize>(19, 31), std::make_shared<ToTensor>(),
                   std::make_shared<NormalizeImage>(std::vector<float>{0.5f, 0.4f, 0.3f},
                                                    std::vector<float>{0.2f, 0.25f, 0.3f})});
    REQUIRE(chain.plan().size() == 1);
    CHECK(chain.supports_batch());
    CHECK(chain.supports_gpu_batch());
    CHECK(chain.batch_op_kind() == GpuOpKind::ImageAugment);
    const Batch out = chain.apply_batch(image_batch(imgs));
    REQUIRE(out.inputs.shape() == Shape{4, 3, 19, 31});
    for (std::size_t i = 0; i < 4; ++i) {
        Sample s;
        s.input = row(imgs, i);
        CHECK(identical(row(out.inputs, i), chain.apply(s).input));
    }
}

TEST_CASE("GPU capability flags of the built-in transforms") {
    CHECK(ClipTransform(0, 1).supports_gpu_batch());
    CHECK_FALSE(Log1pTransform().supports_gpu_batch()); // CUDA log1p is not bit-exact
    auto custom = FuseStep::make_custom("x2", [](double* v, std::size_t n, std::size_t) {
        for (std::size_t i = 0; i < n; ++i) v[i] *= 2;
    });
    CHECK_FALSE(FusedTransform({FuseStep::clip(0, 1), custom}).supports_gpu_batch());
    CHECK(FusedTransform({FuseStep::clip(0, 1), FuseStep::scalar(simd::BinaryOp::Mul, 3)})
              .supports_gpu_batch());
    CHECK(tabular_chain(4)->supports_gpu_batch());
    CHECK(tabular_chain(4)->batch_op_kind() == GpuOpKind::TabularSmall);
    CHECK_FALSE(Compose({std::make_shared<ToTensor>(), std::make_shared<Log1pTransform>()})
                    .supports_gpu_batch());
}

TEST_CASE("collate_stack_nd keeps sample shape") {
    std::vector<Sample> ss(2);
    for (auto& s : ss) {
        s.input = NDArray(Shape{2, 3, 4}, DType::UInt8);
        s.label = NDArray(Shape{1}, DType::Int64);
    }
    const Batch b = collate_stack_nd(ss);
    CHECK(b.inputs.shape() == Shape{2, 2, 3, 4});
    CHECK(collate_stack(ss).inputs.shape() == Shape{2, 24});
}

TEST_CASE("DataLoader runs a fused image batch transform on the host") {
    auto ds = std::make_shared<ImageDataset>(random_images(10, 24, 20, 3, 2));
    DataLoaderOptions opt;
    opt.batch_size = 4;
    opt.num_workers = 2;
    opt.transform_device = Device::host();
    AugmentSpec spec;
    spec.out_height = 12;
    spec.out_width = 10;
    opt.batch_transform = std::make_shared<FusedAugment>(spec);
    DataLoader dl(DatasetConstPtr(ds), opt, collate_stack_nd);
    std::size_t rows = 0;
    for (const Batch& b : dl) {
        CHECK(b.inputs.shape()[1] == 3);
        CHECK(b.inputs.shape()[2] == 12);
        rows += b.inputs.shape()[0];
    }
    CHECK(rows == 10);
}

TEST_CASE("GPU entry points report unavailability cleanly") {
    if (gpu()) {
        CHECK(GpuExecutor::instance(0).device() == 0);
        return;
    }
    CHECK_FALSE(GpuExecutor::usable(0));
    CHECK_THROWS_AS((void)GpuExecutor::instance(0), InvalidArgumentError);
    CHECK_FALSE(query_gpu_telemetry(0).has_value());
    CHECK_FALSE(NvJpegDecoder::available());
    CHECK_THROWS_AS(NvJpegDecoder(0), InvalidArgumentError);
    CHECK_THROWS_AS((void)run_gpu(ClipTransform(0, 1), Batch{}, false), InvalidArgumentError);
}

// --- CUDA kernels vs CPU (bit-exact) --------------------------------------------------------

TEST_CASE("CUDA hwc->chw normalize matches the CPU kernel for odd sizes") {
    SKIP_WITHOUT_GPU();
    for (std::size_t c : {1u, 3u, 4u, 5u}) {
        const NDArray imgs = random_images(3, 17, 29, c, 40 + c);
        std::vector<float> mean(c), stdv(c);
        for (std::size_t k = 0; k < c; ++k) {
            mean[k] = 0.1f * static_cast<float>(k + 1);
            stdv[k] = 0.2f + 0.05f * static_cast<float>(k);
        }
        const NDArray cpu = batch_to_tensor_normalize(imgs, mean, stdv, Device::host());
        const NDArray dev = batch_to_tensor_normalize(imgs, mean, stdv, Device::cuda(0));
        CHECK(dev.device().is_cuda());
        CHECK(identical(dev, cpu));

        // Legacy launcher with device-resident mean/std.
        const NDArray d_in = imgs.cuda(0);
        NDArray d_out(cpu.shape(), DType::Float32, Device::cuda(0));
        NDArray mean_h(Shape{c}, DType::Float32), std_h(Shape{c}, DType::Float32);
        std::memcpy(mean_h.data(), mean.data(), c * sizeof(float));
        std::memcpy(std_h.data(), stdv.data(), c * sizeof(float));
        const NDArray d_mean = mean_h.cuda(0), d_std = std_h.cuda(0);
        cuda_api::launch_hwc_u8_to_chw_f32_normalize(
            static_cast<const std::uint8_t*>(d_in.data()), static_cast<float*>(d_out.data()), 3,
            17, 29, static_cast<int>(c), static_cast<const float*>(d_mean.data()),
            static_cast<const float*>(d_std.data()), nullptr);
        cuda_api::stream_synchronize(nullptr);
        CHECK(identical(d_out, cpu));
    }
}

TEST_CASE("CUDA resize / flip launchers match simd kernels") {
    SKIP_WITHOUT_GPU();
    const NDArray img = random_images(2, 31, 45, 3, 8);
    const NDArray d_img = img.cuda(0);
    const auto* d_src = static_cast<const std::uint8_t*>(d_img.data());

    NDArray d_bil(Shape{2, 20, 67, 3}, DType::UInt8, Device::cuda(0));
    cuda_api::launch_resize_bilinear_u8(d_src, static_cast<std::uint8_t*>(d_bil.data()), 2, 31, 45,
                                        20, 67, 3, nullptr);
    NDArray d_near(Shape{20, 67, 3}, DType::UInt8, Device::cuda(0));
    cuda_api::launch_resize_nearest_u8(d_src, static_cast<std::uint8_t*>(d_near.data()), 31, 45,
                                       20, 67, 3, nullptr);
    NDArray d_h(Shape{31, 45, 3}, DType::UInt8, Device::cuda(0));
    NDArray d_v(Shape{31, 45, 3}, DType::UInt8, Device::cuda(0));
    cuda_api::launch_hflip_u8(d_src, static_cast<std::uint8_t*>(d_h.data()), 31, 45, 3, nullptr);
    cuda_api::launch_vflip_u8(d_src, static_cast<std::uint8_t*>(d_v.data()), 31, 45, 3, nullptr);
    cuda_api::stream_synchronize(nullptr);

    for (std::size_t i = 0; i < 2; ++i) {
        NDArray ref(Shape{20, 67, 3}, DType::UInt8);
        simd::resize_hwc_u8(row(img, i).data<std::uint8_t>(), 31, 45, ref.data<std::uint8_t>(), 20,
                            67, 3, simd::ResampleMode::Bilinear);
        CHECK(identical(row(d_bil.cpu(), i), ref));
    }
    NDArray near_ref(Shape{20, 67, 3}, DType::UInt8);
    simd::resize_hwc_u8(row(img, 0).data<std::uint8_t>(), 31, 45, near_ref.data<std::uint8_t>(), 20,
                        67, 3, simd::ResampleMode::Nearest);
    CHECK(identical(d_near, near_ref));
    NDArray h_ref(Shape{31, 45, 3}, DType::UInt8), v_ref(Shape{31, 45, 3}, DType::UInt8);
    simd::hflip_hwc_u8(row(img, 0).data<std::uint8_t>(), h_ref.data<std::uint8_t>(), 31, 45, 3);
    simd::vflip_hwc_u8(row(img, 0).data<std::uint8_t>(), v_ref.data<std::uint8_t>(), 31, 45, 3);
    CHECK(identical(d_h, h_ref));
    CHECK(identical(d_v, v_ref));
}

TEST_CASE("FusedAugment GPU == CPU (bilinear/nearest, CHW/HWC, chunked, host/device output)") {
    SKIP_WITHOUT_GPU();
    // 128 x 96 x 96 x 3 = 3.4 MiB: three pipelined chunks at the 1 MiB chunk floor.
    const NDArray imgs = random_images(128, 96, 96, 3, 17);
    for (const auto interp : {simd::ResampleMode::Bilinear, simd::ResampleMode::Nearest}) {
        for (const bool to_tensor : {true, false}) {
            AugmentSpec spec = imagenet_like();
            spec.crop_height = 80;
            spec.crop_width = 72;
            spec.out_height = 64;
            spec.out_width = 64;
            spec.interp = interp;
            spec.to_tensor = to_tensor;
            if (!to_tensor) {
                spec.mean.clear();
                spec.stddev.clear();
            }
            FusedAugment cpu_aug(spec, 5), gpu_aug(spec, 5), dev_aug(spec, 5);
            const Batch cpu = cpu_aug.apply_batch(image_batch(imgs));
            const Batch host_out = run_gpu(gpu_aug, image_batch(imgs), /*on_device=*/false);
            const Batch dev_out = run_gpu(dev_aug, image_batch(imgs.cuda(0)), /*on_device=*/true);
            CHECK(host_out.inputs.device().is_host());
            CHECK(cuda_api::is_host_pinned(host_out.inputs.data()));
            CHECK(dev_out.inputs.device().is_cuda());
            CHECK(identical(host_out.inputs, cpu.inputs));
            CHECK(identical(dev_out.inputs, cpu.inputs));
        }
    }
}

TEST_CASE("FusedImageTransform GPU == CPU (resize + normalize and plain ToTensor)") {
    SKIP_WITHOUT_GPU();
    const NDArray imgs = random_images(9, 41, 57, 3, 23);
    const std::vector<float> mean{0.5f, 0.4f, 0.3f}, stdv{0.2f, 0.25f, 0.3f};
    Compose resize_norm({std::make_shared<Resize>(33, 29), std::make_shared<ToTensor>(),
                         std::make_shared<NormalizeImage>(mean, stdv)});
    Compose nearest({std::make_shared<Resize>(80, 70, ResizeInterp::Nearest),
                     std::make_shared<ToTensor>()});
    Compose plain({std::make_shared<ToTensor>(), std::make_shared<NormalizeImage>(mean, stdv)});
    for (const Compose* c : {&resize_norm, &nearest, &plain}) {
        const Batch cpu = c->apply_batch(image_batch(imgs));
        CHECK(identical(run_gpu(*c, image_batch(imgs), false).inputs, cpu.inputs));
        CHECK(identical(run_gpu(*c, image_batch(imgs), true).inputs, cpu.inputs));
    }
}

TEST_CASE("FusedTransform GPU == CPU for float32 / float64 incl. NaN, Inf, -0, denormals") {
    SKIP_WITHOUT_GPU();
    for (const DType dt : {DType::Float32, DType::Float64}) {
        for (const std::size_t rows : {std::size_t{3}, std::size_t{200000}}) {
            const NDArray x = tabular(rows, 7, dt, 9);
            const TransformConstPtr chain = tabular_chain(7);
            Batch b;
            b.inputs = x;
            const Batch cpu = chain->apply_batch(b);
            CHECK(identical(run_gpu(*chain, b, false).inputs, cpu.inputs));
            CHECK(identical(run_gpu(*chain, b, true).inputs, cpu.inputs));
        }
    }
    // Scalar ops and a lone transform (Transform::apply_batch_gpu default).
    FusedTransform scal({FuseStep::scalar(simd::BinaryOp::Sub, 0.25),
                         FuseStep::scalar(simd::BinaryOp::Div, 3.0),
                         FuseStep::scalar(simd::BinaryOp::Mul, 1.5),
                         FuseStep::scalar(simd::BinaryOp::Add, -2.0)});
    Batch b;
    b.inputs = tabular(1000, 3, DType::Float32, 4);
    CHECK(identical(run_gpu(scal, b, false).inputs, scal.apply_batch(b).inputs));
    ClipTransform clip(-1.0, 1.0);
    CHECK(identical(run_gpu(clip, b, false).inputs, clip.apply_batch(b).inputs));
    // Unsupported dtype: transparent CPU fallback, output still placed on the device.
    Batch ib;
    ib.inputs = NDArray(Shape{4, 3}, DType::Int32);
    for (int i = 0; i < 12; ++i) ib.inputs.data<std::int32_t>()[i] = i - 6;
    const Batch ig = run_gpu(clip, ib, true);
    CHECK(ig.inputs.device().is_cuda());
    CHECK(identical(ig.inputs, clip.apply_batch(ib).inputs));
    // Feature-count mismatch is still a ShapeError.
    Batch wrong;
    wrong.inputs = tabular(4, 6, DType::Float32, 1);
    CHECK_THROWS_AS((void)run_gpu(*tabular_chain(7), wrong, false), ShapeError);
}

TEST_CASE("CUDA log1p kernel stays within 1 ulp of the host libm") {
    SKIP_WITHOUT_GPU();
    NDArray x(Shape{4096, 1}, DType::Float64);
    for (std::size_t i = 0; i < 4096; ++i) {
        x.data<double>()[i] = std::ldexp(static_cast<double>(i) + 0.5, static_cast<int>(i % 40) - 20);
    }
    GpuExecutor& ex = GpuExecutor::instance(0);
    auto lease = ex.acquire();
    GpuRowJob job;
    job.input = &x;
    job.output_shape = x.shape();
    job.output_dtype = DType::Float64;
    job.kernel = [](std::size_t, std::size_t rows, const void* in, void* out, CudaStreamHandle s) {
        kernels::TabularParams p;
        p.src = in;
        p.dst = out;
        p.dtype = kernels::TabularDType::F64;
        p.n = rows;
        p.row_len = 1;
        p.num_steps = 1;
        p.steps[0].kind = kernels::TabularStepKind::Log1p;
        kernels::launch_tabular_cuda(p, s);
    };
    const NDArray y = ex.run_rows(lease.lane(), job, false);
    for (std::size_t i = 0; i < 4096; ++i) {
        const double ref = std::log1p(x.data<double>()[i]);
        const double got = y.data<double>()[i];
        CHECK((got == ref || std::nextafter(ref, got) == got));
    }
}

// --- executor ------------------------------------------------------------------------------

TEST_CASE("GpuExecutor: concurrent lanes, buffer reuse and queue depth") {
    SKIP_WITHOUT_GPU();
    GpuExecutorOptions o;
    o.max_lanes = 3;
    o.min_chunk_bytes = 64 * 1024;
    GpuExecutor ex(o);
    const TransformConstPtr chain = tabular_chain(5);
    Batch b;
    b.inputs = tabular(50000, 5, DType::Float32, 2);
    const NDArray ref = chain->apply_batch(b).inputs;
    std::atomic<int> bad{0};
    std::atomic<std::size_t> max_busy{0};
    std::vector<std::thread> ts;
    for (int t = 0; t < 6; ++t) {
        ts.emplace_back([&] {
            for (int k = 0; k < 5; ++k) {
                auto lease = ex.acquire();
                std::size_t seen = ex.busy_lanes();
                std::size_t prev = max_busy.load();
                while (seen > prev && !max_busy.compare_exchange_weak(prev, seen)) {
                }
                GpuBatchContext ctx;
                ctx.executor = &ex;
                ctx.lane = &lease.lane();
                ctx.output_on_device = (k % 2) == 1;
                if (!identical(chain->apply_batch_gpu(b, ctx).inputs, ref)) ++bad;
            }
        });
    }
    for (auto& t : ts) t.join();
    CHECK(bad.load() == 0);
    CHECK(ex.busy_lanes() == 0);
    CHECK(ex.queue_depth() == 0);
    CHECK(max_busy.load() <= 3);
    {
        // Free lanes mean no queueing; only the batch that would block counts.
        auto a = ex.acquire();
        auto b = ex.acquire();
        CHECK(ex.queue_depth() == 0);
        auto c = ex.acquire();
        CHECK(ex.queue_depth() == 1);
    }
    CHECK(ex.queue_depth() == 0);
    const auto st = ex.stats();
    CHECK(st.lanes_created <= 3);
    CHECK(st.device_hits > st.device_misses); // steady state recycles device memory
}

// --- DataLoader + CostModel -------------------------------------------------------------------

TEST_CASE("DataLoader: GPU batch transform output equals CPU output") {
    SKIP_WITHOUT_GPU();
    const NDArray imgs = random_images(40, 64, 48, 3, 31);
    auto ds = std::make_shared<ImageDataset>(imgs);
    AugmentSpec spec;
    spec.crop = AugmentSpec::Crop::Center;
    spec.crop_height = 56;
    spec.crop_width = 40;
    spec.out_height = 32;
    spec.out_width = 32;
    spec.mean = {0.5f, 0.5f, 0.5f};
    spec.stddev = {0.25f, 0.25f, 0.25f};

    auto run = [&](Device transform_device, Device out_device, int workers) {
        DataLoaderOptions opt;
        opt.batch_size = 8;
        opt.num_workers = workers;
        opt.transform_workers = 2;
        opt.transform_device = transform_device;
        opt.device = out_device;
        opt.batch_transform = std::make_shared<FusedAugment>(spec);
        opt.cost_model = std::make_shared<CostModel>();
        DataLoader dl(DatasetConstPtr(ds), opt, collate_stack_nd);
        std::vector<NDArray> outs;
        for (const Batch& b : dl) {
            CHECK(b.inputs.device() == (out_device.is_cuda() ? Device::cuda(0) : Device::host()));
            CHECK(b.labels.device() == b.inputs.device());
            outs.push_back(b.inputs.cpu());
        }
        const CostOpStats st = opt.cost_model->stats(GpuOpKind::ImageAugment);
        return std::make_pair(outs, st);
    };
    const auto [cpu, cpu_st] = run(Device::host(), Device::host(), 2);
    const auto [gpu_host, gpu_st] = run(Device::cuda(0), Device::host(), 2);
    const auto [gpu_dev, dev_st] = run(Device::cuda(0), Device::cuda(0), 0);
    // Auto warmup alternates paths; CPU-path batches must be uploaded (and timed) too.
    const auto [auto_dev, auto_st] = run(Device::auto_select(), Device::cuda(0), 0);
    REQUIRE(cpu.size() == 5);
    REQUIRE(gpu_host.size() == 5);
    REQUIRE(gpu_dev.size() == 5);
    REQUIRE(auto_dev.size() == 5);
    for (std::size_t i = 0; i < 5; ++i) {
        CHECK(identical(gpu_host[i], cpu[i]));
        CHECK(identical(gpu_dev[i], cpu[i]));
        CHECK(identical(auto_dev[i], cpu[i]));
    }
    CHECK(auto_st.cpu_samples + auto_st.gpu_samples == 5);
    CHECK(auto_st.cpu_samples > 0);
    CHECK(cpu_st.cpu_samples == 5);
    CHECK(cpu_st.gpu_samples == 0);
    CHECK(gpu_st.gpu_samples == 5); // real GPU timings reach the model
    CHECK(dev_st.gpu_samples == 5);
}

TEST_CASE("CostModel learns from real timings: tiny tabular batches go to the CPU") {
    SKIP_WITHOUT_GPU();
    CostModelOptions o;
    o.warmup_calls = 16;
    o.auto_telemetry = false;
    auto model = std::make_shared<CostModel>(o);
    const TransformConstPtr chain = tabular_chain(8);
    Batch b;
    b.inputs = tabular(32, 8, DType::Float32, 3); // 1 KiB: launch + PCIe latency dominate
    GpuExecutor& ex = GpuExecutor::instance(0);
    for (int i = 0; i < 40; ++i) {
        const CostDecision d = model->decide(GpuOpKind::TabularSmall, b.inputs.nbytes(), 0);
        if (d.use_gpu) {
            auto lease = ex.acquire();
            GpuBatchContext ctx;
            ctx.executor = &ex;
            ctx.lane = &lease.lane();
            ScopedCostTimer t(*model, GpuOpKind::TabularSmall, ExecPath::Gpu, b.inputs.nbytes());
            (void)chain->apply_batch_gpu(b, ctx);
        } else {
            ScopedCostTimer t(*model, GpuOpKind::TabularSmall, ExecPath::Cpu, b.inputs.nbytes());
            (void)chain->apply_batch(b);
        }
    }
    const CostOpStats st = model->stats(GpuOpKind::TabularSmall);
    CHECK(st.gpu_samples >= 8);
    CHECK(st.cpu_samples >= 8);
    const CostDecision d = model->decide(GpuOpKind::TabularSmall, b.inputs.nbytes(), 0);
    CHECK_FALSE(d.use_gpu);
    CHECK(d.predicted_gpu_us > d.predicted_cpu_us);
}

TEST_CASE("telemetry and calibration read the real device") {
    SKIP_WITHOUT_GPU();
    const auto t = query_gpu_telemetry(0);
    REQUIRE(t.has_value());
    REQUIRE(t->free_vram_bytes.has_value());
    CHECK(*t->free_vram_bytes > 0);
    if (nvml_available()) {
        CHECK(t->utilization >= 0.0);
        CHECK(t->utilization <= 1.0);
    }
    auto provider = make_cuda_telemetry_provider(0);
    CHECK(provider().has_value());

    const CalibrationReport rep = calibrate_auto_device_policy();
    CHECK(rep.cuda_available);
    CHECK(rep.h2d_gib_s > 0.5);
    CHECK(rep.d2h_gib_s > 0.5);
    CHECK(rep.launch_overhead_us > 0.0);
    CHECK(rep.launch_overhead_us < 5000.0);
    const auto props = cuda_api::device_properties(0);
    CHECK(props.multiprocessors > 0);
    CHECK_FALSE(props.pci_bus_id.empty());
}

// --- nvJPEG ---------------------------------------------------------------------------------

namespace {
void append_bytes(void* ctx, void* data, int size) {
    auto* v = static_cast<std::vector<std::uint8_t>*>(ctx);
    const auto* p = static_cast<const std::uint8_t*>(data);
    v->insert(v->end(), p, p + size);
}

std::vector<std::uint8_t> smooth_jpeg(int w, int h, int phase) {
    std::vector<std::uint8_t> px(static_cast<std::size_t>(w) * h * 3);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            std::uint8_t* p = &px[(static_cast<std::size_t>(y) * w + x) * 3];
            p[0] = static_cast<std::uint8_t>((x * 255) / std::max(1, w - 1));
            p[1] = static_cast<std::uint8_t>((y * 255) / std::max(1, h - 1));
            p[2] = static_cast<std::uint8_t>(128 + 100 * std::sin((x + y + phase) * 0.05));
        }
    }
    std::vector<std::uint8_t> jpg;
    stbi_write_jpg_to_func(append_bytes, &jpg, w, h, 3, px.data(), 95);
    return jpg;
}
} // namespace

TEST_CASE("nvJPEG decode matches stb_image within JPEG IDCT tolerance") {
    if (!NvJpegDecoder::available()) {
        MESSAGE("nvJPEG not built / no device: skipped");
        return;
    }
    NvJpegDecoder dec(0);
    const std::vector<std::uint8_t> jpg = smooth_jpeg(173, 91, 0);
    const Image host = decode_image(jpg.data(), jpg.size());
    const NDArray d = dec.decode(jpg.data(), jpg.size());
    CHECK(d.device().is_cuda());
    const NDArray g = d.cpu();
    REQUIRE(g.shape() == host.data.shape());
    double sum = 0.0;
    int worst = 0;
    for (std::size_t i = 0; i < g.numel(); ++i) {
        const int diff = std::abs(int(g.data<std::uint8_t>()[i]) - int(host.data.data<std::uint8_t>()[i]));
        sum += diff;
        worst = std::max(worst, diff);
    }
    CHECK(sum / static_cast<double>(g.numel()) < 2.0);
    CHECK(worst <= 24);

    const auto batch = dec.decode_batch({smooth_jpeg(64, 48, 1), smooth_jpeg(33, 70, 2)}, Device::host());
    REQUIRE(batch.size() == 2);
    CHECK(batch[0].shape() == Shape{48, 64, 3});
    CHECK(batch[1].shape() == Shape{70, 33, 3});
    CHECK(batch[1].device().is_host());
    std::vector<std::uint8_t> garbage(100, 0x42);
    CHECK_THROWS((void)dec.decode(garbage.data(), garbage.size()));
}
