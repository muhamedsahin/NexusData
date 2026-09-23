#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "nexusdata/backend/kernels.hpp"
#include "nexusdata/pipeline/transform.hpp"
#include "nexusdata/simd/image_ops.hpp"

namespace nexusdata {

/// Geometry + normalization of FusedAugment.
struct AugmentSpec {
    enum class Crop : std::uint8_t { None, Center, Random };

    Crop crop = Crop::None;
    std::size_t crop_height = 0;
    std::size_t crop_width = 0;
    /// Resize target after cropping; 0 x 0 keeps the crop size.
    std::size_t out_height = 0;
    std::size_t out_width = 0;
    simd::ResampleMode interp = simd::ResampleMode::Bilinear;
    float hflip_p = 0.0f;
    float vflip_p = 0.0f;
    /// true: CHW float32 (ToTensor, plus NormalizeImage when mean is set); false: HWC uint8.
    bool to_tensor = true;
    std::vector<float> mean;
    std::vector<float> stddev;
};

/// Crop -> Resize -> HFlip -> VFlip -> ToTensor -> NormalizeImage on HWC uint8
/// images as one kernel: a single read of the source pixels and a single write
/// of the result, on the CPU (apply / apply_batch) or on CUDA (apply_batch_gpu).
///
/// Values equal the unfused chain CenterCrop / RandomCrop -> Resize ->
/// flips -> ToTensor -> NormalizeImage for the same crop offsets and flips.
/// Random parameters come from a counter-based Philox stream (seed, call index),
/// drawn on the host before dispatch, so the CPU and GPU paths produce identical
/// batches. Thread-safe (the call counter is atomic).
class FusedAugment : public Transform {
public:
    explicit FusedAugment(AugmentSpec spec, std::uint64_t seed = 0);

    /// [H, W, C] uint8 sample.
    [[nodiscard]] Sample apply(const Sample& sample) const override;

    [[nodiscard]] bool supports_batch() const override { return true; }
    /// [N, H, W, C] uint8 batch; samples run in parallel on default_thread_pool().
    [[nodiscard]] Batch apply_batch(const Batch& batch) const override;

    [[nodiscard]] bool supports_gpu_batch() const override { return true; }
    [[nodiscard]] Batch apply_batch_gpu(const Batch& batch, GpuBatchContext& ctx) const override;
    [[nodiscard]] GpuOpKind batch_op_kind() const override { return GpuOpKind::ImageAugment; }

    [[nodiscard]] bool is_random() const override;
    void set_seed(std::uint64_t seed) override;

    [[nodiscard]] const AugmentSpec& spec() const noexcept { return spec_; }

    /// Kernel parameters for an [n, sh, sw, c] input; per-sample geometry is
    /// drawn into @p samples (consumes n random draws).
    [[nodiscard]] kernels::ImageAugmentParams plan(int n, int sh, int sw, int c,
                                                   std::vector<kernels::AugmentSample>& samples) const;

private:
    AugmentSpec spec_;
    std::uint64_t seed_ = 0;
    mutable std::atomic<std::uint64_t> counter_{0};
};

} // namespace nexusdata
