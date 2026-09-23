#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "nexusdata/core/ndarray.hpp"
#include "nexusdata/pipeline/transform.hpp"
#include "nexusdata/simd/image_ops.hpp"
#include "nexusdata/simd/numeric_ops.hpp"

namespace nexusdata {

/// Element-wise operation understood by the fusion compiler.
///
/// Steps operate on values widened to double; FusedTransform rounds back to the
/// storage dtype between steps so a fused chain is bit-identical to applying
/// the original transforms one after another.
struct FuseStep {
    enum class Kind : std::uint8_t { Clip, Log1p, Standardize, MinMax, ScalarOp, Custom };

    Kind kind = Kind::Custom;
    /// Clip: lo / hi. MinMax: mul (span) / add (feature_min). ScalarOp: a = operand.
    double a = 0.0;
    double b = 0.0;
    simd::BinaryOp op = simd::BinaryOp::Add;
    /// Standardize: mean / scale. MinMax: data_min / range (zero range already replaced by 1).
    std::vector<double> p0;
    std::vector<double> p1;
    /// Per-sample feature count for column-dependent steps; 0 = element-wise.
    std::size_t period = 0;
    /// Custom kernel: v[k] belongs to column (col0 + k) % period (col0 == 0 when period == 0).
    std::function<void(double* v, std::size_t n, std::size_t col0)> custom;
    /// Reported in errors ("<name>: feature count mismatch").
    std::string name;

    [[nodiscard]] static std::shared_ptr<const FuseStep> clip(double lo, double hi,
                                                              std::string name = "ClipTransform");
    [[nodiscard]] static std::shared_ptr<const FuseStep> log1p(std::string name = "Log1pTransform");
    [[nodiscard]] static std::shared_ptr<const FuseStep> standardize(
        std::vector<double> mean, std::vector<double> scale,
        std::string name = "StandardizeTransform");
    [[nodiscard]] static std::shared_ptr<const FuseStep> minmax(
        const std::vector<double>& data_min, const std::vector<double>& data_max,
        double feature_min, double feature_max, std::string name = "MinMaxTransform");
    [[nodiscard]] static std::shared_ptr<const FuseStep> scalar(simd::BinaryOp op, double a,
                                                                std::string name = "ScalarOp");
    /// User-defined fusable kernel (see Transform::fuse_step()).
    [[nodiscard]] static std::shared_ptr<const FuseStep> make_custom(
        std::string name, std::function<void(double*, std::size_t, std::size_t)> fn,
        std::size_t period = 0);
};

using FuseStepPtr = std::shared_ptr<const FuseStep>;

/// Apply one step with per-sample semantics (what a lone tabular transform does).
[[nodiscard]] Sample apply_fuse_step(const FuseStepPtr& step, const Sample& sample);

/// Compiled chain of FuseSteps executed in a single pass over the data
/// (one output allocation, one read, one write; steps run on an L1-resident tile).
class FusedTransform : public Transform {
public:
    explicit FusedTransform(std::vector<FuseStepPtr> steps);

    [[nodiscard]] Sample apply(const Sample& sample) const override;
    [[nodiscard]] bool is_fusable() const override { return true; }
    [[nodiscard]] bool supports_batch() const override { return true; }
    /// Rows are processed in parallel on default_thread_pool() for large batches.
    [[nodiscard]] Batch apply_batch(const Batch& batch) const override;

    /// False for chains with Custom steps (host-only kernels), Log1p (CUDA's log1p
    /// may differ from the host libm by 1 ulp, and the CPU/GPU choice must never
    /// change results) or more than 16 steps.
    [[nodiscard]] bool supports_gpu_batch() const override;
    /// Float32 / Float64 run as one CUDA kernel, bit-identical to apply_batch().
    /// Other dtypes run on the CPU.
    [[nodiscard]] Batch apply_batch_gpu(const Batch& batch, GpuBatchContext& ctx) const override;

    /// Run the chain on @p x where each sample occupies @p row_len consecutive elements.
    [[nodiscard]] NDArray run(const NDArray& x, std::size_t row_len) const;

    [[nodiscard]] const std::vector<FuseStepPtr>& steps() const noexcept { return steps_; }
    /// Feature count required by column-dependent steps (0 = none).
    [[nodiscard]] std::size_t period() const noexcept { return period_; }

    /// Elements per batch above which apply_batch() splits work across threads.
    static constexpr std::size_t kParallelMinElements = std::size_t{1} << 20;

private:
    void check_row(std::size_t row_len) const;

    std::vector<FuseStepPtr> steps_;
    std::size_t period_ = 0;
};

/// Fused [Resize] -> ToTensor / NormalizeImage chain for HWC uint8 images:
/// one pass writing CHW float32, no intermediate uint8 / float tensors.
/// Inputs that are not HWC uint8 fall back to the original transforms.
class FusedImageTransform : public Transform {
public:
    struct Spec {
        bool resize = false;
        std::size_t height = 0;
        std::size_t width = 0;
        simd::ResampleMode mode = simd::ResampleMode::Bilinear;
        std::vector<float> mean;   ///< empty: ToTensor only
        std::vector<float> stddev;
    };

    FusedImageTransform(Spec spec, std::vector<TransformConstPtr> originals);

    [[nodiscard]] Sample apply(const Sample& sample) const override;
    [[nodiscard]] bool is_fusable() const override { return true; }
    /// v2.0: [N, H, W, C] uint8 -> [N, C, dh, dw] float32, samples in parallel.
    [[nodiscard]] bool supports_batch() const override { return true; }
    [[nodiscard]] Batch apply_batch(const Batch& batch) const override;
    /// v2.0: one CUDA kernel per chunk, bit-identical to apply_batch().
    [[nodiscard]] bool supports_gpu_batch() const override { return true; }
    [[nodiscard]] Batch apply_batch_gpu(const Batch& batch, GpuBatchContext& ctx) const override;
    [[nodiscard]] GpuOpKind batch_op_kind() const override { return GpuOpKind::ImageAugment; }
    [[nodiscard]] const Spec& spec() const noexcept { return spec_; }

private:
    [[nodiscard]] bool batch_fast_path(const NDArray& x) const;

    Spec spec_;
    std::vector<TransformConstPtr> originals_;
};

namespace detail {
/// Apply @p t to every row of batch.inputs and restack ([B, ...out]); the
/// generic batch fallback for inputs a batch kernel cannot handle.
[[nodiscard]] Batch apply_rowwise(const Transform& t, const Batch& batch);

/// Run @p t.apply_batch() on the host and place the result per @p ctx (fallback
/// for inputs a GPU kernel cannot handle).
[[nodiscard]] Batch apply_batch_on_host(const Transform& t, const Batch& batch,
                                        const GpuBatchContext& ctx);
} // namespace detail

} // namespace nexusdata
