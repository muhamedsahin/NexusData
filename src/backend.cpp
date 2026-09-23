#include "nexusdata/backend/backend.hpp"

#include "nexusdata/backend/cuda_api.hpp"
#include "nexusdata/core/error.hpp"

namespace nexusdata {

NDArray CpuBackend::to_device(const NDArray& src) const {
    if (src.device().is_host()) {
        return src;
    }
    // Copy device → host would require CUDA; without it, reject.
    throw InvalidArgumentError("CpuBackend::to_device: source is not on host");
}

std::shared_ptr<Backend> make_backend(Device device) {
    Device r = resolve_device(device, GpuOpKind::NormalizeCollate, /*nbytes=*/SIZE_MAX);
#if defined(NEXUSDATA_WITH_CUDA)
    if (r.is_cuda()) {
        return std::make_shared<CudaBackend>(r.index);
    }
#else
    (void)r;
#endif
    return std::make_shared<CpuBackend>();
}

#if defined(NEXUSDATA_WITH_CUDA)
CudaBackend::CudaBackend(int device_index) : index_(device_index) {
    cuda_api::set_device(index_);
    stream_ = cuda_api::stream_create();
}

CudaBackend::~CudaBackend() {
    if (stream_) {
        cuda_api::stream_destroy(stream_);
        stream_ = nullptr;
    }
}

NDArray CudaBackend::to_device(const NDArray& src) const {
    cuda_api::set_device(index_);
    if (src.device().is_cuda() && src.device().index == index_) {
        return src;
    }
    NDArray out(src.shape(), src.dtype(), Device::cuda(index_));
    if (src.nbytes() == 0) {
        return out;
    }
    if (src.device().is_host()) {
        cuda_api::memcpy_h2d(out.data(), src.data(), src.nbytes(), stream_);
    } else {
        cuda_api::memcpy_d2d(out.data(), src.data(), src.nbytes(), stream_);
    }
    cuda_api::stream_synchronize(stream_);
    return out;
}

void CudaBackend::synchronize() const {
    cuda_api::stream_synchronize(stream_);
}
#endif

} // namespace nexusdata
