#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>

#include "nexusdata/backend/device.hpp"

namespace nexusdata {

enum class ExecPath : std::uint8_t { Cpu = 0, Gpu = 1 };

/// Live GPU load snapshot (e.g. from NVML / cudaMemGetInfo). Unknown fields stay negative / unset.
struct GpuTelemetry {
    /// Busy fraction in [0, 1]; < 0 means unknown.
    double utilization = -1.0;
    std::optional<std::size_t> free_vram_bytes;
};

/// Returns nullopt when telemetry is unavailable. Called once per decision; keep it cheap
/// (cache inside the provider if the underlying query is slow).
using GpuTelemetryProvider = std::function<std::optional<GpuTelemetry>()>;

struct CostModelOptions {
    /// Per-op A/B profiling phase: the first N decisions alternate CPU/GPU.
    std::size_t warmup_calls = 32;
    /// After warmup, every N decisions re-probe the non-preferred path ...
    std::size_t recalibrate_every = 10000;
    /// ... this many times (workloads drift: batch size, resolution).
    std::size_t recalibrate_probes = 4;
    /// Weight of the newest measurement in the decayed fit (effective window ~ 1/alpha).
    double ewma_alpha = 0.1;
    /// GPU must be predicted this many times faster before it is chosen (hysteresis).
    double gpu_advantage = 1.10;
    /// Skip the GPU when its utilization is at or above this (training already saturates it).
    double gpu_busy_utilization = 0.90;
    /// Skip the GPU when free VRAM would drop below this (or below 2x the batch).
    std::size_t min_free_vram_bytes = std::size_t{256} << 20;
    /// Transfer / launch priors (replaced by calibrate_auto_device_policy() measurements).
    double h2d_gib_s = 6.0;
    double d2h_gib_s = 6.0;
    double launch_overhead_us = 40.0;
    /// Thresholds used until both paths of an op have measurements.
    AutoDevicePolicy fallback_policy{};
    /// Install make_cuda_telemetry_provider(0) when a CUDA device is detected
    /// (live utilization / free VRAM gates). set_telemetry_provider() overrides it.
    bool auto_telemetry = true;
};

/// Outcome of one batch-level decision.
struct CostDecision {
    bool use_gpu = false;
    /// True when the choice was forced for profiling (warmup / recalibration).
    bool exploring = false;
    double predicted_cpu_us = -1.0; ///< < 0: no estimate
    double predicted_gpu_us = -1.0; ///< < 0: no estimate (includes queue-depth penalty)
    /// Static string: "no_gpu", "gpu_busy", "low_vram", "warmup", "recalibrate",
    /// "heuristic", "model".
    const char* reason = "";
};

struct CostOpStats {
    std::uint64_t decisions = 0;
    std::uint64_t gpu_decisions = 0;
    std::uint64_t cpu_samples = 0;
    std::uint64_t gpu_samples = 0;
    /// Decayed mean of recorded wall time per MiB (< 0 without samples).
    double cpu_us_per_mib = -1.0;
    double gpu_us_per_mib = -1.0;
};

/// Learned CPU-vs-GPU cost model (JIT device scheduler).
///
/// Callers time whichever path actually ran and feed it back with record();
/// GPU timings are expected to be end-to-end (H2D + kernels + D2H if needed).
/// decide() is meant to be called once per batch, not per sample.
///
/// Hard rule: when no CUDA device is usable (or the build has no CUDA),
/// decide() always returns use_gpu == false.
///
/// Thread-safety: all methods are thread-safe.
class CostModel {
public:
    explicit CostModel(CostModelOptions options = {});

    CostModel(const CostModel&) = delete;
    CostModel& operator=(const CostModel&) = delete;

    /// Override GPU availability (simulation / tests). nullopt restores auto-detection.
    /// Note: resolve_device() additionally requires a real CUDA runtime.
    void set_gpu_available(std::optional<bool> available);
    [[nodiscard]] bool gpu_available() const;

    void set_telemetry_provider(GpuTelemetryProvider provider);

    /// Adopt measured bandwidth / launch overhead and fallback thresholds.
    void apply_calibration(const CalibrationReport& report);

    /// Feed one measured execution of @p op on @p path.
    void record(GpuOpKind op, ExecPath path, std::size_t nbytes, double micros);

    [[nodiscard]] CostDecision decide(GpuOpKind op, std::size_t nbytes,
                                      std::size_t gpu_queue_depth = 0);

    [[nodiscard]] bool should_use_gpu(GpuOpKind op, std::size_t nbytes,
                                      std::size_t current_queue_depth = 0) {
        return decide(op, nbytes, current_queue_depth).use_gpu;
    }

    /// Model estimate in microseconds (< 0 when the path has no samples yet).
    [[nodiscard]] double predict_us(GpuOpKind op, ExecPath path, std::size_t nbytes) const;

    [[nodiscard]] CostOpStats stats(GpuOpKind op) const;
    [[nodiscard]] const CostModelOptions& options() const noexcept { return opt_; }

    /// Forget all measurements and counters.
    void reset();

    /// Process-wide model shared by loaders that do not bring their own.
    [[nodiscard]] static CostModel& global();

private:
    struct Fit {
        double w = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
        std::uint64_t n = 0;
        void add(double x, double y, double alpha) noexcept;
        [[nodiscard]] double predict(double x, double intercept_floor) const noexcept;
    };
    struct OpState {
        std::array<Fit, 2> fit{};
        std::uint64_t decisions = 0;
        std::uint64_t gpu_decisions = 0;
        /// Last model preference (for recalibration probes).
        bool prefers_gpu = false;
    };

    [[nodiscard]] double predict_locked(const OpState& s, ExecPath path, double mib) const;
    [[nodiscard]] bool heuristic_gpu(GpuOpKind op, std::size_t nbytes) const;

    CostModelOptions opt_;
    mutable std::mutex mu_;
    std::array<OpState, 256> ops_{};
    std::optional<bool> gpu_override_;
    bool gpu_detected_ = false;
    GpuTelemetryProvider telemetry_;
};

/// Records the wall time of its scope into a CostModel (cancel() to discard).
class ScopedCostTimer {
public:
    ScopedCostTimer(CostModel& model, GpuOpKind op, ExecPath path, std::size_t nbytes)
        : model_(&model), op_(op), path_(path), nbytes_(nbytes),
          t0_(std::chrono::steady_clock::now()) {}
    ~ScopedCostTimer() {
        if (model_ != nullptr) {
            const double us = std::chrono::duration<double, std::micro>(
                                  std::chrono::steady_clock::now() - t0_)
                                  .count();
            model_->record(op_, path_, nbytes_, us);
        }
    }
    ScopedCostTimer(const ScopedCostTimer&) = delete;
    ScopedCostTimer& operator=(const ScopedCostTimer&) = delete;

    void cancel() noexcept { model_ = nullptr; }

private:
    CostModel* model_;
    GpuOpKind op_;
    ExecPath path_;
    std::size_t nbytes_;
    std::chrono::steady_clock::time_point t0_;
};

} // namespace nexusdata
