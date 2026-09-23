// CUDA runtime wrappers (compiled only when NEXUSDATA_WITH_CUDA=ON).
// Kernels live in cuda_kernels.cu.

#include "nexusdata/backend/cuda_api.hpp"

#include <cuda_runtime.h>

#include <string>

#include "nexusdata/core/error.hpp"

namespace nexusdata {
namespace cuda_api {
namespace {

[[noreturn]] void throw_cuda(const char* what, cudaError_t err) {
    throw Error(std::string("CUDA ") + what + ": " + cudaGetErrorString(err));
}

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        throw_cuda(what, e);
    }
}

cudaStream_t as_stream(CudaStreamHandle s) { return static_cast<cudaStream_t>(s); }
cudaEvent_t as_event(CudaEventHandle e) { return static_cast<cudaEvent_t>(e); }

} // namespace

bool available() {
    // cudaGetDeviceCount is cheap after the first call; cache anyway (called per batch).
    static const bool ok = [] {
        int n = 0;
        return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
    }();
    return ok;
}

int device_count() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) {
        return 0;
    }
    return n;
}

void set_device(int index) { check(cudaSetDevice(index), "cudaSetDevice"); }

std::string device_name(int index) {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, index) != cudaSuccess) {
        return {};
    }
    return prop.name;
}

void* device_alloc(std::size_t bytes) {
    void* p = nullptr;
    check(cudaMalloc(&p, bytes), "cudaMalloc");
    return p;
}

void device_free(void* ptr) noexcept {
    if (ptr) {
        cudaFree(ptr);
    }
}

void* host_alloc_pinned(std::size_t bytes) {
    void* p = nullptr;
    check(cudaMallocHost(&p, bytes), "cudaMallocHost");
    return p;
}

void host_free_pinned(void* ptr) noexcept {
    if (ptr) {
        cudaFreeHost(ptr);
    }
}

void memcpy_h2d(void* dst, const void* src, std::size_t bytes, CudaStreamHandle stream) {
    check(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, as_stream(stream)), "memcpy_h2d");
}

void memcpy_d2h(void* dst, const void* src, std::size_t bytes, CudaStreamHandle stream) {
    check(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, as_stream(stream)), "memcpy_d2h");
}

void memcpy_d2d(void* dst, const void* src, std::size_t bytes, CudaStreamHandle stream) {
    check(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, as_stream(stream)),
          "memcpy_d2d");
}

CudaStreamHandle stream_create() {
    cudaStream_t s = nullptr;
    check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "cudaStreamCreate");
    return s;
}

void stream_destroy(CudaStreamHandle s) noexcept {
    if (s) {
        cudaStreamDestroy(as_stream(s));
    }
}

void stream_synchronize(CudaStreamHandle s) {
    check(cudaStreamSynchronize(as_stream(s)), "cudaStreamSynchronize");
}

void check_last_error(const char* what) { check(cudaGetLastError(), what); }

CudaEventHandle event_create(bool timing) {
    cudaEvent_t e = nullptr;
    check(cudaEventCreateWithFlags(&e, timing ? cudaEventDefault : cudaEventDisableTiming),
          "cudaEventCreate");
    return e;
}

void event_destroy(CudaEventHandle e) noexcept {
    if (e) {
        cudaEventDestroy(as_event(e));
    }
}

void event_record(CudaEventHandle e, CudaStreamHandle stream) {
    check(cudaEventRecord(as_event(e), as_stream(stream)), "cudaEventRecord");
}

void stream_wait_event(CudaStreamHandle stream, CudaEventHandle e) {
    check(cudaStreamWaitEvent(as_stream(stream), as_event(e), 0), "cudaStreamWaitEvent");
}

void event_synchronize(CudaEventHandle e) {
    check(cudaEventSynchronize(as_event(e)), "cudaEventSynchronize");
}

float event_elapsed_ms(CudaEventHandle start, CudaEventHandle end) {
    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, as_event(start), as_event(end)), "cudaEventElapsedTime");
    return ms;
}

namespace {
bool async_alloc_supported() {
    static const bool ok = [] {
        int dev = 0;
        int v = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) return false;
        return cudaDeviceGetAttribute(&v, cudaDevAttrMemoryPoolsSupported, dev) == cudaSuccess &&
               v != 0;
    }();
    return ok;
}
} // namespace

void* device_alloc_async(std::size_t bytes, CudaStreamHandle stream) {
    if (!async_alloc_supported()) {
        return device_alloc(bytes);
    }
    void* p = nullptr;
    check(cudaMallocAsync(&p, bytes, as_stream(stream)), "cudaMallocAsync");
    return p;
}

void device_free_async(void* ptr, CudaStreamHandle stream) noexcept {
    if (!ptr) {
        return;
    }
    if (!async_alloc_supported()) {
        cudaFree(ptr);
        return;
    }
    cudaFreeAsync(ptr, as_stream(stream));
}

bool is_host_pinned(const void* host_ptr) noexcept {
    if (host_ptr == nullptr) {
        return false;
    }
    cudaPointerAttributes attr{};
    if (cudaPointerGetAttributes(&attr, host_ptr) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return attr.type == cudaMemoryTypeHost;
}

int current_device() {
    int d = 0;
    check(cudaGetDevice(&d), "cudaGetDevice");
    return d;
}

void device_synchronize() { check(cudaDeviceSynchronize(), "cudaDeviceSynchronize"); }

bool mem_get_info(std::size_t& free_bytes, std::size_t& total_bytes) {
    std::size_t f = 0, t = 0;
    if (cudaMemGetInfo(&f, &t) != cudaSuccess) {
        cudaGetLastError(); // clear sticky-free error state
        return false;
    }
    free_bytes = f;
    total_bytes = t;
    return true;
}

DeviceProperties device_properties(int index) {
    DeviceProperties out;
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, index) != cudaSuccess) {
        return out;
    }
    out.name = prop.name;
    out.multiprocessors = prop.multiProcessorCount;
    out.compute_major = prop.major;
    out.compute_minor = prop.minor;
    out.total_memory = prop.totalGlobalMem;
    char bus[32] = {};
    if (cudaDeviceGetPCIBusId(bus, static_cast<int>(sizeof(bus)), index) == cudaSuccess) {
        out.pci_bus_id = bus;
    }
    return out;
}

} // namespace cuda_api
} // namespace nexusdata
