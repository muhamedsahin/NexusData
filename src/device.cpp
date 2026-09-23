#include "nexusdata/backend/device.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "nexusdata/backend/cuda_api.hpp"
#include "nexusdata/backend/kernels.hpp"
#include "nexusdata/core/error.hpp"

namespace nexusdata {

bool cuda_runtime_available() {
    return cuda_api::available();
}

int cuda_device_count() {
    return cuda_api::device_count();
}

namespace {

struct CustomOpRegistry {
    std::mutex mu;
    std::vector<std::string> names; // index k => GpuOpKind(Custom + 1 + k)
};

CustomOpRegistry& custom_ops() {
    static CustomOpRegistry reg;
    return reg;
}

double measure_host_memcpy_gib_s(std::size_t bytes) {
    std::vector<std::uint8_t> src(bytes);
    std::vector<std::uint8_t> dst(bytes);
    for (std::size_t i = 0; i < bytes; ++i) {
        src[i] = static_cast<std::uint8_t>(i);
    }
    // warmup
    std::memcpy(dst.data(), src.data(), bytes);
    volatile std::uint64_t sink = 0;
    const int iters = 16;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        std::memcpy(dst.data(), src.data(), bytes);
        sink += dst[0] + dst[bytes / 2] + dst[bytes - 1];
    }
    const auto t1 = std::chrono::steady_clock::now();
    (void)sink;
    const double sec = std::chrono::duration<double>(t1 - t0).count() / static_cast<double>(iters);
    if (sec < 1e-9) {
        return 0.0;
    }
    return (static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0)) / sec;
}

struct CudaProbe {
    double h2d_gib_s = 0.0;
    double d2h_gib_s = 0.0;
    double launch_us = 0.0;
};

/// Pinned H2D / D2H bandwidth and small-copy round-trip latency. Only called
/// when a CUDA device is present; any failure yields zeros (callers fall back
/// to assumed values).
CudaProbe measure_cuda(std::size_t bytes) {
    CudaProbe out;
    void* host = nullptr;
    void* dev = nullptr;
    CudaStreamHandle stream = nullptr;
    try {
        host = cuda_api::host_alloc_pinned(bytes);
        dev = cuda_api::device_alloc(bytes);
        stream = cuda_api::stream_create();
        std::memset(host, 1, bytes);
        cuda_api::memcpy_h2d(dev, host, bytes, stream);
        cuda_api::stream_synchronize(stream);

        const int iters = 8;
        using clock = std::chrono::steady_clock;
        auto t0 = clock::now();
        for (int i = 0; i < iters; ++i) {
            cuda_api::memcpy_h2d(dev, host, bytes, stream);
        }
        cuda_api::stream_synchronize(stream);
        double sec = std::chrono::duration<double>(clock::now() - t0).count() / iters;
        const double gib = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
        out.h2d_gib_s = sec > 0 ? gib / sec : 0.0;

        t0 = clock::now();
        for (int i = 0; i < iters; ++i) {
            cuda_api::memcpy_d2h(host, dev, bytes, stream);
        }
        cuda_api::stream_synchronize(stream);
        sec = std::chrono::duration<double>(clock::now() - t0).count() / iters;
        out.d2h_gib_s = sec > 0 ? gib / sec : 0.0;

        // Launch + completion round trip of an empty kernel: the fixed cost every
        // GPU batch pays on top of transfers and compute.
        const int small_iters = 64;
        kernels::launch_noop_cuda(stream);
        cuda_api::stream_synchronize(stream);
        t0 = clock::now();
        for (int i = 0; i < small_iters; ++i) {
            kernels::launch_noop_cuda(stream);
            cuda_api::stream_synchronize(stream);
        }
        out.launch_us =
            std::chrono::duration<double, std::micro>(clock::now() - t0).count() / small_iters;
    } catch (...) {
        out = CudaProbe{};
    }
    if (stream) cuda_api::stream_destroy(stream);
    if (dev) cuda_api::device_free(dev);
    if (host) cuda_api::host_free_pinned(host);
    return out;
}

std::size_t break_even_nbytes(double pcie_gib_s, double launch_us) {
    // Model: prefer device when transfer_time + launch < 2 * transfer_time_at_threshold
    // Rough: nbytes / pcie > launch_overhead ⇒ nbytes > pcie * launch
    // Use 2x safety so small batches stay on host.
    if (pcie_gib_s <= 0.0) {
        return 1u << 22;
    }
    const double bytes =
        pcie_gib_s * (1024.0 * 1024.0 * 1024.0) * (launch_us * 1e-6) * 2.0;
    auto n = static_cast<std::size_t>(bytes);
    n = std::max<std::size_t>(n, 256 * 1024);       // >= 256 KiB
    n = std::min<std::size_t>(n, 64u * 1024 * 1024); // <= 64 MiB
    return n;
}

} // namespace

GpuOpKind register_custom_op_kind(const std::string& name) {
    if (name.empty()) {
        throw InvalidArgumentError("register_custom_op_kind: empty name");
    }
    auto& reg = custom_ops();
    std::lock_guard lock(reg.mu);
    for (std::size_t k = 0; k < reg.names.size(); ++k) {
        if (reg.names[k] == name) {
            return static_cast<GpuOpKind>(static_cast<std::size_t>(GpuOpKind::Custom) + 1 + k);
        }
    }
    constexpr std::size_t kMaxCustom = 255 - static_cast<std::size_t>(GpuOpKind::Custom);
    if (reg.names.size() >= kMaxCustom) {
        throw InvalidArgumentError("register_custom_op_kind: all " + std::to_string(kMaxCustom) +
                                   " custom op slots are in use");
    }
    reg.names.push_back(name);
    return static_cast<GpuOpKind>(static_cast<std::size_t>(GpuOpKind::Custom) + reg.names.size());
}

std::string op_kind_name(GpuOpKind op) {
    switch (op) {
        case GpuOpKind::TabularSmall:     return "tabular_small";
        case GpuOpKind::ImageDecode:      return "image_decode";
        case GpuOpKind::ImageAugment:     return "image_augment";
        case GpuOpKind::NormalizeCollate: return "normalize_collate";
        case GpuOpKind::GatherLarge:      return "gather_large";
        case GpuOpKind::TextTokenize:     return "text_tokenize";
        case GpuOpKind::AudioResample:    return "audio_resample";
        case GpuOpKind::JsonParse:        return "json_parse";
        case GpuOpKind::Compression:      return "compression";
        case GpuOpKind::Custom:           return "custom";
        default:
            break;
    }
    const auto v = static_cast<std::size_t>(op);
    if (v > static_cast<std::size_t>(GpuOpKind::Custom)) {
        auto& reg = custom_ops();
        std::lock_guard lock(reg.mu);
        const std::size_t k = v - static_cast<std::size_t>(GpuOpKind::Custom) - 1;
        if (k < reg.names.size()) {
            return reg.names[k];
        }
    }
    return "custom:" + std::to_string(v);
}

CalibrationReport calibrate_auto_device_policy(AutoDevicePolicy base) {
    CalibrationReport rep;
    rep.probe_bytes = 16u * 1024 * 1024; // 16 MiB
    rep.cuda_available = cuda_runtime_available() && cuda_device_count() > 0;
    rep.host_memcpy_gib_s = measure_host_memcpy_gib_s(rep.probe_bytes);

    AutoDevicePolicy p = base;
    p.host_memcpy_gib_s = rep.host_memcpy_gib_s;

    if (rep.cuda_available) {
        const CudaProbe probe = measure_cuda(rep.probe_bytes);
        rep.h2d_gib_s = probe.h2d_gib_s;
        rep.d2h_gib_s = probe.d2h_gib_s;
        rep.launch_overhead_us = probe.launch_us;
        if (probe.h2d_gib_s > 0.0) {
            p.assumed_pcie_gib_s = probe.h2d_gib_s;
        }
        if (probe.launch_us > 0.0) {
            p.launch_overhead_us = probe.launch_us;
        }
    }

    const std::size_t be = break_even_nbytes(p.assumed_pcie_gib_s, p.launch_overhead_us);

    if (!rep.cuda_available) {
        // Keep Auto on Host for typical work: inflate thresholds.
        p.image_nbytes_threshold = std::max(be, std::size_t{1u << 24});    // >= 16 MiB
        p.tabular_nbytes_threshold = std::max(be * 2, std::size_t{1u << 25}); // >= 32 MiB
        p.allow_cuda = base.allow_cuda;
    } else {
        p.image_nbytes_threshold = be;
        p.tabular_nbytes_threshold = std::max(be * 2, be + (1u << 20));
    }
    rep.policy = p;
    return rep;
}

Device resolve_device(Device requested, GpuOpKind op, std::size_t nbytes,
                      const AutoDevicePolicy& policy) {
    if (requested.is_host()) {
        return Device::host();
    }
    if (requested.is_cuda()) {
        if (!policy.allow_cuda || !cuda_runtime_available() || cuda_device_count() <= 0) {
            return Device::host();
        }
        const int idx = requested.index < 0 ? 0 : requested.index;
        if (idx >= cuda_device_count()) {
            return Device::host();
        }
        return Device::cuda(idx);
    }
    // Auto
    if (!policy.allow_cuda || !cuda_runtime_available() || cuda_device_count() <= 0) {
        return Device::host();
    }
    std::size_t thr = policy.tabular_nbytes_threshold;
    switch (op) {
        case GpuOpKind::ImageDecode:
        case GpuOpKind::ImageAugment:
        case GpuOpKind::NormalizeCollate:
            thr = policy.image_nbytes_threshold;
            break;
        case GpuOpKind::GatherLarge:
            thr = policy.tabular_nbytes_threshold;
            break;
        case GpuOpKind::TabularSmall:
            return Device::host();
        default:
            // No static GPU heuristic for the v2.0 kinds: text/audio/json/compression
            // and custom ops need a CostModel with measurements to go to CUDA.
            return Device::host();
    }
    if (nbytes >= thr) {
        return Device::cuda(0);
    }
    return Device::host();
}

} // namespace nexusdata
