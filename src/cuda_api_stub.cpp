#include "nexusdata/backend/cuda_api.hpp"
#include "nexusdata/backend/kernels.hpp"

#include "nexusdata/core/error.hpp"

#if defined(NEXUSDATA_WITH_CUDA)
#error "cuda_api_stub.cpp must not be compiled when NEXUSDATA_WITH_CUDA is defined"
#endif

namespace nexusdata {

namespace {
[[noreturn]] void no_cuda(const char* what) {
    throw InvalidArgumentError(std::string(what) +
                               ": CUDA not enabled (reconfigure with -DNEXUSDATA_WITH_CUDA=ON)");
}
} // namespace

namespace cuda_api {

bool available() { return false; }
int device_count() { return 0; }
void set_device(int) {}
std::string device_name(int) { return {}; }

void* device_alloc(std::size_t) { no_cuda("device_alloc"); }
void device_free(void*) noexcept {}

void* host_alloc_pinned(std::size_t) { no_cuda("host_alloc_pinned"); }
void host_free_pinned(void*) noexcept {}

void memcpy_h2d(void*, const void*, std::size_t, CudaStreamHandle) { no_cuda("memcpy_h2d"); }
void memcpy_d2h(void*, const void*, std::size_t, CudaStreamHandle) { no_cuda("memcpy_d2h"); }
void memcpy_d2d(void*, const void*, std::size_t, CudaStreamHandle) { no_cuda("memcpy_d2d"); }

CudaStreamHandle stream_create() { return nullptr; }
void stream_destroy(CudaStreamHandle) noexcept {}
void stream_synchronize(CudaStreamHandle) {}
void check_last_error(const char*) {}

CudaEventHandle event_create(bool) { return nullptr; }
void event_destroy(CudaEventHandle) noexcept {}
void event_record(CudaEventHandle, CudaStreamHandle) {}
void stream_wait_event(CudaStreamHandle, CudaEventHandle) {}
void event_synchronize(CudaEventHandle) {}
float event_elapsed_ms(CudaEventHandle, CudaEventHandle) { return 0.0f; }

void* device_alloc_async(std::size_t, CudaStreamHandle) { no_cuda("device_alloc_async"); }
void device_free_async(void*, CudaStreamHandle) noexcept {}

bool is_host_pinned(const void*) noexcept { return false; }
int current_device() { return 0; }
void device_synchronize() {}
bool mem_get_info(std::size_t&, std::size_t&) { return false; }
DeviceProperties device_properties(int) { return {}; }

void launch_hwc_u8_to_chw_f32_normalize(const std::uint8_t*, float*, int, int, int, int,
                                        const float*, const float*, CudaStreamHandle) {
    no_cuda("launch_hwc_u8_to_chw_f32_normalize");
}
void launch_hwc_u8_to_chw_f32(const std::uint8_t*, float*, int, int, int, int, const float*,
                              const float*, CudaStreamHandle) {
    no_cuda("launch_hwc_u8_to_chw_f32");
}
void launch_resize_nearest_u8(const std::uint8_t*, std::uint8_t*, int, int, int, int, int,
                              CudaStreamHandle) {
    no_cuda("launch_resize_nearest_u8");
}
void launch_resize_bilinear_u8(const std::uint8_t*, std::uint8_t*, int, int, int, int, int, int,
                               CudaStreamHandle) {
    no_cuda("launch_resize_bilinear_u8");
}
void launch_hflip_u8(const std::uint8_t*, std::uint8_t*, int, int, int, CudaStreamHandle) {
    no_cuda("launch_hflip_u8");
}
void launch_vflip_u8(const std::uint8_t*, std::uint8_t*, int, int, int, CudaStreamHandle) {
    no_cuda("launch_vflip_u8");
}

} // namespace cuda_api

namespace kernels {

void launch_image_augment_cuda(const ImageAugmentParams&, CudaStreamHandle) {
    no_cuda("launch_image_augment_cuda");
}
void launch_tabular_cuda(const TabularParams&, CudaStreamHandle) { no_cuda("launch_tabular_cuda"); }
void launch_noop_cuda(CudaStreamHandle) { no_cuda("launch_noop_cuda"); }

} // namespace kernels

} // namespace nexusdata
