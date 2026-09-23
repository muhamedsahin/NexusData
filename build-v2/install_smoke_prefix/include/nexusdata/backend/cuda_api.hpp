#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "nexusdata/core/error.hpp"
#include "nexusdata/core/ndarray.hpp"

namespace nexusdata {

/// Opaque CUDA stream handle (void* = cudaStream_t when CUDA enabled).
using CudaStreamHandle = void*;
/// Opaque CUDA event handle (cudaEvent_t when CUDA is enabled).
using CudaEventHandle = void*;

struct CudaStream {
    CudaStreamHandle handle = nullptr;
    int device = 0;
};

/// Low-level CUDA helpers. When built without CUDA, functions throw or no-op as documented.
namespace cuda_api {

[[nodiscard]] bool available();
[[nodiscard]] int device_count();
void set_device(int index);
[[nodiscard]] std::string device_name(int index);

[[nodiscard]] void* device_alloc(std::size_t bytes);
void device_free(void* ptr) noexcept;

[[nodiscard]] void* host_alloc_pinned(std::size_t bytes);
void host_free_pinned(void* ptr) noexcept;

void memcpy_h2d(void* dst, const void* src, std::size_t bytes, CudaStreamHandle stream = nullptr);
void memcpy_d2h(void* dst, const void* src, std::size_t bytes, CudaStreamHandle stream = nullptr);
void memcpy_d2d(void* dst, const void* src, std::size_t bytes, CudaStreamHandle stream = nullptr);

[[nodiscard]] CudaStreamHandle stream_create();
void stream_destroy(CudaStreamHandle s) noexcept;
void stream_synchronize(CudaStreamHandle s);

void check_last_error(const char* what);

// --- v2.0 runtime additions -------------------------------------------------------

/// @p timing = false creates a cheaper event usable only for ordering.
[[nodiscard]] CudaEventHandle event_create(bool timing = false);
void event_destroy(CudaEventHandle e) noexcept;
void event_record(CudaEventHandle e, CudaStreamHandle stream);
/// Make @p stream wait for @p e (no host blocking).
void stream_wait_event(CudaStreamHandle stream, CudaEventHandle e);
void event_synchronize(CudaEventHandle e);
/// Milliseconds between two recorded timing events.
[[nodiscard]] float event_elapsed_ms(CudaEventHandle start, CudaEventHandle end);

/// Stream-ordered allocation (cudaMallocAsync); falls back to device_alloc when unsupported.
[[nodiscard]] void* device_alloc_async(std::size_t bytes, CudaStreamHandle stream);
void device_free_async(void* ptr, CudaStreamHandle stream) noexcept;

/// True when @p host_ptr lies in page-locked (cudaMallocHost / registered) memory.
[[nodiscard]] bool is_host_pinned(const void* host_ptr) noexcept;

[[nodiscard]] int current_device();
void device_synchronize();

/// Free / total device memory of the current device. False on failure.
[[nodiscard]] bool mem_get_info(std::size_t& free_bytes, std::size_t& total_bytes);

struct DeviceProperties {
    std::string name;
    int multiprocessors = 0;
    int compute_major = 0;
    int compute_minor = 0;
    std::size_t total_memory = 0;
    /// "0000:01:00.0"-style PCI bus id (matches NVML's device lookup).
    std::string pci_bus_id;
};
[[nodiscard]] DeviceProperties device_properties(int index);

/// Fused HWC uint8 → CHW float32 normalize: out = (x/255 - mean) / std
void launch_hwc_u8_to_chw_f32_normalize(const std::uint8_t* d_hwc,
                                        float* d_chw,
                                        int n, int h, int w, int c,
                                        const float* d_mean,
                                        const float* d_std,
                                        CudaStreamHandle stream);

/// v2.0: same kernel with mean/std passed by value (no device buffers needed).
/// Pass nullptr for plain ToTensor (x / 255). Channels must be <= 16.
void launch_hwc_u8_to_chw_f32(const std::uint8_t* d_hwc, float* d_chw, int n, int h, int w, int c,
                              const float* host_mean, const float* host_std,
                              CudaStreamHandle stream);

/// Nearest-neighbor resize HWC uint8 on device.
void launch_resize_nearest_u8(const std::uint8_t* d_src, std::uint8_t* d_dst,
                              int sh, int sw, int dh, int dw, int c,
                              CudaStreamHandle stream);

/// v2.0: bilinear resize of a batch [n, sh, sw, c] -> [n, dh, dw, c] (bit-identical to Resize).
void launch_resize_bilinear_u8(const std::uint8_t* d_src, std::uint8_t* d_dst, int n,
                               int sh, int sw, int dh, int dw, int c, CudaStreamHandle stream);

/// Horizontal flip HWC uint8 (out-of-place).
void launch_hflip_u8(const std::uint8_t* d_src, std::uint8_t* d_dst,
                     int h, int w, int c, CudaStreamHandle stream);

/// v2.0: vertical flip HWC uint8 (out-of-place).
void launch_vflip_u8(const std::uint8_t* d_src, std::uint8_t* d_dst,
                     int h, int w, int c, CudaStreamHandle stream);

} // namespace cuda_api

/// High-level host API: convert batch of HWC uint8 images to CHW float on target device.
[[nodiscard]] NDArray batch_to_tensor_normalize(const NDArray& hwc_u8_batch,
                                                const std::vector<float>& mean,
                                                const std::vector<float>& stddev,
                                                Device device,
                                                CudaStreamHandle stream = nullptr);

} // namespace nexusdata
