#pragma once

#include <chrono>
#include <optional>

#include "nexusdata/backend/cost_model.hpp"

namespace nexusdata {

/// True when the NVIDIA management library (nvml.dll / libnvidia-ml.so.1) could
/// be loaded. It ships with the driver; NexusData loads it at runtime, so there
/// is no link-time dependency.
[[nodiscard]] bool nvml_available();

/// Live utilization (NVML) and free VRAM (NVML, else cudaMemGetInfo) of CUDA
/// device @p device. std::nullopt when neither source works.
[[nodiscard]] std::optional<GpuTelemetry> query_gpu_telemetry(int device = 0);

/// CostModel provider that polls query_gpu_telemetry() at most every @p refresh
/// (decisions are per batch; NVML queries cost ~10-50 us).
[[nodiscard]] GpuTelemetryProvider make_cuda_telemetry_provider(
    int device = 0, std::chrono::milliseconds refresh = std::chrono::milliseconds(100));

} // namespace nexusdata
