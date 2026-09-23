#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "nexusdata/backend/device.hpp"
#include "nexusdata/core/error.hpp"
#include "nexusdata/core/random.hpp"
#include "nexusdata/dataset/sample.hpp"
#include "nexusdata/loading/batch.hpp"

namespace nexusdata {

struct FuseStep;
struct GpuBatchContext;

/// Sample-level transform. Deterministic unless is_random() and seeded.
/// Thread-safety: immutable after construction except set_seed on random ones.
class Transform {
public:
    virtual ~Transform() = default;

    [[nodiscard]] virtual Sample apply(const Sample& sample) const = 0;

    Sample operator()(const Sample& sample) const { return apply(sample); }

    [[nodiscard]] virtual bool is_random() const { return false; }
    virtual void set_seed(std::uint64_t /*seed*/) {}

    /// Hint for future GPU fusion (v0.5).
    [[nodiscard]] virtual bool is_fusable() const { return false; }

    /// v2.0: element-wise description of this transform for the fusion compiler
    /// (pipeline/fusion.hpp). Transforms returning a step are merged with their
    /// neighbours by Compose into one single-pass kernel. Default: not fusable.
    [[nodiscard]] virtual std::shared_ptr<const FuseStep> fuse_step() const { return nullptr; }

    /// v2.0: true when apply_batch() can process a collated batch directly
    /// (rows = samples), e.g. inside the DataLoader transform stage.
    [[nodiscard]] virtual bool supports_batch() const { return fuse_step() != nullptr; }

    /// v2.0: batch-level application, equivalent to apply() on every row of
    /// batch.inputs (labels / mask pass through). The default runs fuse_step()
    /// over the whole batch; throws InvalidArgumentError when unsupported.
    [[nodiscard]] virtual Batch apply_batch(const Batch& batch) const;

    /// v2.0: true when apply_batch_gpu() runs this transform with CUDA kernels.
    /// Default: true for built-in (non-Custom) fuse_step() kinds.
    [[nodiscard]] virtual bool supports_gpu_batch() const;

    /// v2.0: GPU twin of apply_batch() (same values; see each override for exceptions).
    /// batch.inputs may live on the host or on ctx's device; the result is left on
    /// the device or downloaded to pinned host memory per ctx.output_on_device.
    /// Default: runs fuse_step() as a FusedTransform; otherwise throws InvalidArgumentError.
    [[nodiscard]] virtual Batch apply_batch_gpu(const Batch& batch, GpuBatchContext& ctx) const;

    /// v2.0: operation class the CostModel uses to learn CPU/GPU cost of apply_batch.
    [[nodiscard]] virtual GpuOpKind batch_op_kind() const { return GpuOpKind::TabularSmall; }
};

using TransformPtr = std::shared_ptr<Transform>;
using TransformConstPtr = std::shared_ptr<const Transform>;

namespace detail {
/// Build Compose's execution plan: fusable runs become FusedTransform /
/// fused image kernels, everything else is kept as-is (defined in fusion.cpp).
[[nodiscard]] std::vector<TransformConstPtr> compile_transform_plan(
    const std::vector<TransformConstPtr>& transforms);
} // namespace detail

/// Apply transforms in order.
///
/// v2.0: the chain is compiled once at construction; consecutive fusable steps
/// run as one pass with a single output allocation (results are bit-identical
/// to the unfused chain). Pass fuse = false to execute transforms one by one.
class Compose : public Transform {
public:
    explicit Compose(std::vector<TransformConstPtr> transforms, bool fuse = true)
        : transforms_(std::move(transforms)) {
        plan_ = fuse ? detail::compile_transform_plan(transforms_) : transforms_;
    }

    Compose(std::initializer_list<TransformConstPtr> transforms)
        : Compose(std::vector<TransformConstPtr>(transforms)) {}

    [[nodiscard]] Sample apply(const Sample& sample) const override {
        Sample out = sample;
        for (const auto& t : plan_) {
            if (!t) {
                continue;
            }
            out = t->apply(out);
        }
        return out;
    }

    void set_seed(std::uint64_t seed) override {
        seed_ = seed;
        // Re-seed random children if we hold non-const pointers via const_cast of known mutables.
        // Random transforms are never fused, so plan_ shares these same objects.
        for (auto& t : transforms_) {
            if (t && t->is_random()) {
                const_cast<Transform&>(*t).set_seed(seed_);
                // Derive per-transform stream for independence.
                seed_ += 0x9E3779B97F4A7C15ULL;
            }
        }
    }

    [[nodiscard]] bool is_random() const override {
        for (const auto& t : transforms_) {
            if (t && t->is_random()) {
                return true;
            }
        }
        return false;
    }

    [[nodiscard]] bool is_fusable() const override {
        bool any = false;
        for (const auto& t : transforms_) {
            if (!t) {
                continue;
            }
            if (!t->is_fusable()) {
                return false;
            }
            any = true;
        }
        return any;
    }

    [[nodiscard]] bool supports_batch() const override {
        bool any = false;
        for (const auto& t : plan_) {
            if (!t) {
                continue;
            }
            if (!t->supports_batch()) {
                return false;
            }
            any = true;
        }
        return any;
    }

    [[nodiscard]] Batch apply_batch(const Batch& batch) const override {
        if (!supports_batch()) {
            throw InvalidArgumentError("Compose::apply_batch: a step does not support batches");
        }
        Batch out = batch;
        for (const auto& t : plan_) {
            if (t) {
                out = t->apply_batch(out);
            }
        }
        return out;
    }

    [[nodiscard]] bool supports_gpu_batch() const override {
        bool any = false;
        for (const auto& t : plan_) {
            if (!t) {
                continue;
            }
            if (!t->supports_gpu_batch()) {
                return false;
            }
            any = true;
        }
        return any;
    }

    /// Intermediate results stay on the device between steps.
    [[nodiscard]] Batch apply_batch_gpu(const Batch& batch, GpuBatchContext& ctx) const override;

    /// Image kernels dominate a mixed chain; otherwise the first step's kind.
    [[nodiscard]] GpuOpKind batch_op_kind() const override {
        GpuOpKind kind = GpuOpKind::TabularSmall;
        bool first = true;
        for (const auto& t : plan_) {
            if (!t) {
                continue;
            }
            const GpuOpKind k = t->batch_op_kind();
            if (k == GpuOpKind::ImageAugment || k == GpuOpKind::ImageDecode) {
                return k;
            }
            if (first) {
                kind = k;
                first = false;
            }
        }
        return kind;
    }

    /// Original transform list.
    [[nodiscard]] const std::vector<TransformConstPtr>& transforms() const noexcept {
        return transforms_;
    }
    /// Compiled execution plan (fused groups replace runs of transforms).
    [[nodiscard]] const std::vector<TransformConstPtr>& plan() const noexcept { return plan_; }

private:
    std::vector<TransformConstPtr> transforms_;
    std::vector<TransformConstPtr> plan_;
    mutable std::uint64_t seed_ = 0;
};

/// With probability @p p apply @p transform, else identity.
class RandomApply : public Transform {
public:
    RandomApply(TransformConstPtr transform, float p, std::uint64_t seed = 0)
        : transform_(std::move(transform)), p_(p), rng_(seed) {
        if (p_ < 0.0f || p_ > 1.0f) {
            throw InvalidArgumentError("RandomApply: p must be in [0, 1]");
        }
    }

    [[nodiscard]] Sample apply(const Sample& sample) const override {
        if (!transform_) {
            return sample;
        }
        // Mutable RNG behind const apply — document as single-threaded use.
        auto& rng = const_cast<PCG32&>(rng_);
        if (rng.next_float() < p_) {
            return transform_->apply(sample);
        }
        return sample;
    }

    [[nodiscard]] bool is_random() const override { return true; }

    void set_seed(std::uint64_t seed) override {
        rng_ = PCG32(seed);
    }

private:
    TransformConstPtr transform_;
    float p_ = 0.5f;
    PCG32 rng_;
};

/// Choose exactly one transform uniformly (or by weights).
class OneOf : public Transform {
public:
    explicit OneOf(std::vector<TransformConstPtr> transforms, std::uint64_t seed = 0)
        : transforms_(std::move(transforms)), rng_(seed) {
        if (transforms_.empty()) {
            throw InvalidArgumentError("OneOf: empty transform list");
        }
    }

    [[nodiscard]] Sample apply(const Sample& sample) const override {
        auto& rng = const_cast<PCG32&>(rng_);
        const auto i = rng.next_bounded(static_cast<std::uint32_t>(transforms_.size()));
        return transforms_[i]->apply(sample);
    }

    [[nodiscard]] bool is_random() const override { return true; }

    void set_seed(std::uint64_t seed) override { rng_ = PCG32(seed); }

private:
    std::vector<TransformConstPtr> transforms_;
    PCG32 rng_;
};

} // namespace nexusdata
