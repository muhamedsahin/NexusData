#include "nexusdata/backend/gpu_telemetry.hpp"

#include <memory>
#include <mutex>
#include <string>

#include "nexusdata/backend/cuda_api.hpp"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace nexusdata {
namespace {

// Minimal NVML ABI (stable since NVML 1.0; see nvml.h).
using nvmlReturn_t = int;
using nvmlDevice_t = void*;
struct nvmlUtilization_t {
    unsigned int gpu;
    unsigned int memory;
};
struct nvmlMemory_t {
    unsigned long long total;
    unsigned long long free;
    unsigned long long used;
};
constexpr nvmlReturn_t kNvmlSuccess = 0;

struct Nvml {
    using InitFn = nvmlReturn_t (*)();
    using ByPciFn = nvmlReturn_t (*)(const char*, nvmlDevice_t*);
    using ByIndexFn = nvmlReturn_t (*)(unsigned int, nvmlDevice_t*);
    using UtilFn = nvmlReturn_t (*)(nvmlDevice_t, nvmlUtilization_t*);
    using MemFn = nvmlReturn_t (*)(nvmlDevice_t, nvmlMemory_t*);

    bool ok = false;
    ByPciFn by_pci = nullptr;
    ByIndexFn by_index = nullptr;
    UtilFn utilization = nullptr;
    MemFn memory = nullptr;

    Nvml() {
#if defined(_WIN32)
        HMODULE lib = LoadLibraryA("nvml.dll");
        if (lib == nullptr) {
            return;
        }
        auto sym = [&](const char* name) {
            return reinterpret_cast<void*>(GetProcAddress(lib, name));
        };
#else
        void* lib = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
        if (lib == nullptr) {
            return;
        }
        auto sym = [&](const char* name) { return dlsym(lib, name); };
#endif
        // Library intentionally stays loaded for the process lifetime.
        auto init = reinterpret_cast<InitFn>(sym("nvmlInit_v2"));
        by_pci = reinterpret_cast<ByPciFn>(sym("nvmlDeviceGetHandleByPciBusId_v2"));
        by_index = reinterpret_cast<ByIndexFn>(sym("nvmlDeviceGetHandleByIndex_v2"));
        utilization = reinterpret_cast<UtilFn>(sym("nvmlDeviceGetUtilizationRates"));
        memory = reinterpret_cast<MemFn>(sym("nvmlDeviceGetMemoryInfo"));
        ok = init != nullptr && utilization != nullptr && (by_pci != nullptr || by_index != nullptr) &&
             init() == kNvmlSuccess;
    }

    /// CUDA and NVML enumerate devices differently; the PCI bus id is the shared key.
    nvmlDevice_t handle(int cuda_device) const {
        nvmlDevice_t h = nullptr;
        if (by_pci != nullptr) {
            const std::string bus = cuda_api::device_properties(cuda_device).pci_bus_id;
            if (!bus.empty() && by_pci(bus.c_str(), &h) == kNvmlSuccess) {
                return h;
            }
        }
        if (by_index != nullptr &&
            by_index(static_cast<unsigned int>(cuda_device), &h) == kNvmlSuccess) {
            return h;
        }
        return nullptr;
    }
};

const Nvml& nvml() {
    static const Nvml lib;
    return lib;
}

} // namespace

bool nvml_available() { return nvml().ok; }

std::optional<GpuTelemetry> query_gpu_telemetry(int device) {
    if (!cuda_api::available() || device < 0 || device >= cuda_api::device_count()) {
        return std::nullopt;
    }
    GpuTelemetry t;
    bool any = false;
    const Nvml& lib = nvml();
    if (lib.ok) {
        if (nvmlDevice_t h = lib.handle(device)) {
            nvmlUtilization_t u{};
            if (lib.utilization(h, &u) == kNvmlSuccess) {
                t.utilization = static_cast<double>(u.gpu) / 100.0;
                any = true;
            }
            nvmlMemory_t m{};
            if (lib.memory != nullptr && lib.memory(h, &m) == kNvmlSuccess) {
                t.free_vram_bytes = static_cast<std::size_t>(m.free);
                any = true;
            }
        }
    }
    if (!t.free_vram_bytes) {
        try {
            cuda_api::set_device(device);
            std::size_t free_b = 0, total_b = 0;
            if (cuda_api::mem_get_info(free_b, total_b)) {
                t.free_vram_bytes = free_b;
                any = true;
            }
        } catch (...) {
        }
    }
    if (!any) {
        return std::nullopt;
    }
    return t;
}

GpuTelemetryProvider make_cuda_telemetry_provider(int device, std::chrono::milliseconds refresh) {
    struct Cache {
        std::mutex mu;
        std::chrono::steady_clock::time_point at{};
        bool valid = false;
        std::optional<GpuTelemetry> value;
    };
    auto cache = std::make_shared<Cache>();
    return [cache, device, refresh]() -> std::optional<GpuTelemetry> {
        const auto now = std::chrono::steady_clock::now();
        std::lock_guard lock(cache->mu);
        if (!cache->valid || now - cache->at >= refresh) {
            cache->value = query_gpu_telemetry(device);
            cache->at = now;
            cache->valid = true;
        }
        return cache->value;
    };
}

} // namespace nexusdata
