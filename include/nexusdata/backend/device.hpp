#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace nexusdata {

enum class DeviceKind : std::uint8_t {
    Auto = 0,
    Host = 1,
    CUDA = 2,
};

/// Device selector. Auto resolves at use-site via policy thresholds.
struct Device {
    DeviceKind kind = DeviceKind::Host;
    int index = 0; // CUDA device index

    [[nodiscard]] static Device host() { return Device{DeviceKind::Host, -1}; }
    [[nodiscard]] static Device cuda(int i = 0) { return Device{DeviceKind::CUDA, i}; }
    [[nodiscard]] static Device auto_select() { return Device{DeviceKind::Auto, 0}; }

    [[nodiscard]] bool is_host() const noexcept { return kind == DeviceKind::Host; }
    [[nodiscard]] bool is_cuda() const noexcept { return kind == DeviceKind::CUDA; }
    [[nodiscard]] bool is_auto() const noexcept { return kind == DeviceKind::Auto; }

    [[nodiscard]] std::string to_string() const {
        if (kind == DeviceKind::Host) return "cpu";
        if (kind == DeviceKind::Auto) return "auto";
        return "cuda:" + std::to_string(index);
    }

    friend bool operator==(const Device& a, const Device& b) noexcept {
        return a.kind == b.kind && a.index == b.index;
    }
    friend bool operator!=(const Device& a, const Device& b) noexcept { return !(a == b); }
};

/// Operation classes for Auto device decisions.
///
/// Values >= Custom are user-defined kinds obtained from register_custom_op_kind().
enum class GpuOpKind : std::uint8_t {
    TabularSmall = 0,
    ImageDecode,
    ImageAugment,
    NormalizeCollate,
    GatherLarge,
    // v2.0
    TextTokenize,
    AudioResample,
    JsonParse,
    Compression,
    /// Generic user operation; register_custom_op_kind() hands out Custom + 1 ... 255.
    Custom = 128,
};

/// Register a named user operation kind so the CostModel tracks it separately.
/// Idempotent per name. Throws InvalidArgumentError when all 127 slots are used.
[[nodiscard]] GpuOpKind register_custom_op_kind(const std::string& name);

/// Human-readable name ("image_augment", a registered custom name, "custom:<n>").
[[nodiscard]] std::string op_kind_name(GpuOpKind op);

[[nodiscard]] inline bool is_custom_op_kind(GpuOpKind op) noexcept {
    return static_cast<std::uint8_t>(op) >= static_cast<std::uint8_t>(GpuOpKind::Custom);
}

struct AutoDevicePolicy {
    /// Prefer CUDA for image augment when batch nbytes >= this.
    std::size_t image_nbytes_threshold = 1u << 20; // 1 MiB (pre-calibration default)
    std::size_t tabular_nbytes_threshold = 1u << 22; // 4 MiB
    bool allow_cuda = true;

    /// Estimated host memcpy bandwidth (GiB/s) from last calibration; 0 = unknown.
    double host_memcpy_gib_s = 0.0;
    /// Assumed effective host↔device bandwidth for break-even model (GiB/s).
    double assumed_pcie_gib_s = 6.0;
    /// Fixed launch/sync overhead used in break-even (microseconds).
    double launch_overhead_us = 40.0;
};

struct CalibrationReport {
    AutoDevicePolicy policy;
    double host_memcpy_gib_s = 0.0;
    std::size_t probe_bytes = 0;
    bool cuda_available = false;
    /// v2.0: measured pinned host->device / device->host bandwidth (0 without CUDA).
    double h2d_gib_s = 0.0;
    double d2h_gib_s = 0.0;
    /// v2.0: measured round-trip of a tiny async copy + stream sync (0 without CUDA).
    double launch_overhead_us = 0.0;
};

/// Measure host memcpy bandwidth and set Auto thresholds via a break-even model.
/// Without CUDA, thresholds remain high enough that Auto stays on Host for typical batches.
[[nodiscard]] CalibrationReport calibrate_auto_device_policy(AutoDevicePolicy base = {});

/// Resolve Auto → Host or CUDA. Without CUDA build / no device → Host.
[[nodiscard]] Device resolve_device(Device requested,
                                    GpuOpKind op,
                                    std::size_t nbytes,
                                    const AutoDevicePolicy& policy = {});

class CostModel;

/// Resolve Auto via a learned CostModel (batch-level decision). Explicit Host /
/// CUDA requests behave exactly like the policy overload; without a usable GPU
/// the result is always Host.
[[nodiscard]] Device resolve_device(Device requested,
                                    GpuOpKind op,
                                    std::size_t nbytes,
                                    CostModel& model,
                                    std::size_t gpu_queue_depth = 0);

[[nodiscard]] int cuda_device_count();
[[nodiscard]] bool cuda_runtime_available();

} // namespace nexusdata
