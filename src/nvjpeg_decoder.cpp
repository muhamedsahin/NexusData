#include "nexusdata/image/nvjpeg.hpp"

#include <mutex>
#include <string>

#include "nexusdata/core/error.hpp"

#if defined(NEXUSDATA_WITH_NVJPEG)
#include <cuda_runtime_api.h>
#include <nvjpeg.h>

#include "nexusdata/backend/gpu_executor.hpp"
#endif

namespace nexusdata {

#if defined(NEXUSDATA_WITH_NVJPEG)

namespace {

void check_nvjpeg(nvjpegStatus_t s, const char* what) {
    if (s != NVJPEG_STATUS_SUCCESS) {
        throw Error(std::string("nvJPEG ") + what + " failed (status " +
                    std::to_string(static_cast<int>(s)) + ")");
    }
}

} // namespace

struct NvJpegDecoder::Impl {
    nvjpegHandle_t handle = nullptr;
    nvjpegJpegState_t state = nullptr;
    CudaStreamHandle stream = nullptr;
    std::mutex mu;
};

bool NvJpegDecoder::available() noexcept { return GpuExecutor::usable(0); }

NvJpegDecoder::NvJpegDecoder(int device) : impl_(std::make_unique<Impl>()), device_(device) {
    if (!GpuExecutor::usable(device)) {
        throw InvalidArgumentError("NvJpegDecoder: CUDA device " + std::to_string(device) +
                                   " unavailable");
    }
    cuda_api::set_device(device);
    check_nvjpeg(nvjpegCreateSimple(&impl_->handle), "nvjpegCreateSimple");
    check_nvjpeg(nvjpegJpegStateCreate(impl_->handle, &impl_->state), "nvjpegJpegStateCreate");
    impl_->stream = cuda_api::stream_create();
}

NvJpegDecoder::~NvJpegDecoder() {
    if (!impl_) {
        return;
    }
    try {
        cuda_api::stream_synchronize(impl_->stream);
    } catch (...) {
    }
    if (impl_->state) nvjpegJpegStateDestroy(impl_->state);
    if (impl_->handle) nvjpegDestroy(impl_->handle);
    cuda_api::stream_destroy(impl_->stream);
}

namespace {

NDArray decode_on(nvjpegHandle_t handle, nvjpegJpegState_t state, CudaStreamHandle stream,
                  GpuExecutor& ex, const std::uint8_t* data, std::size_t size) {
    if (data == nullptr || size == 0) {
        throw InvalidArgumentError("NvJpegDecoder: empty input");
    }
    int components = 0;
    nvjpegChromaSubsampling_t subsampling{};
    int widths[NVJPEG_MAX_COMPONENT] = {};
    int heights[NVJPEG_MAX_COMPONENT] = {};
    check_nvjpeg(nvjpegGetImageInfo(handle, data, size, &components, &subsampling, widths, heights),
                 "nvjpegGetImageInfo");
    const auto w = static_cast<std::size_t>(widths[0]);
    const auto h = static_cast<std::size_t>(heights[0]);
    if (w == 0 || h == 0) {
        throw Error("nvJPEG: invalid image dimensions");
    }
    NDArray out = ex.device_array(Shape{h, w, 3}, DType::UInt8, stream);
    nvjpegImage_t img{};
    img.channel[0] = static_cast<unsigned char*>(out.data());
    img.pitch[0] = w * 3;
    check_nvjpeg(nvjpegDecode(handle, state, data, size, NVJPEG_OUTPUT_RGBI, &img,
                              static_cast<cudaStream_t>(stream)),
                 "nvjpegDecode");
    return out;
}

bool wants_host(Device out) { return out.is_host(); }

} // namespace

NDArray NvJpegDecoder::decode(const std::uint8_t* data, std::size_t size, Device out) {
    std::lock_guard lock(impl_->mu);
    cuda_api::set_device(device_);
    GpuExecutor& ex = GpuExecutor::instance(device_);
    NDArray img = decode_on(impl_->handle, impl_->state, impl_->stream, ex, data, size);
    cuda_api::stream_synchronize(impl_->stream);
    return wants_host(out) ? img.cpu() : img;
}

std::vector<NDArray> NvJpegDecoder::decode_batch(const std::vector<std::vector<std::uint8_t>>& files,
                                                 Device out) {
    std::lock_guard lock(impl_->mu);
    cuda_api::set_device(device_);
    GpuExecutor& ex = GpuExecutor::instance(device_);
    std::vector<NDArray> imgs;
    imgs.reserve(files.size());
    for (const auto& f : files) {
        imgs.push_back(decode_on(impl_->handle, impl_->state, impl_->stream, ex, f.data(), f.size()));
        // The decode state's scratch buffers are reused by the next image.
        cuda_api::stream_synchronize(impl_->stream);
    }
    if (wants_host(out)) {
        for (auto& a : imgs) a = a.cpu();
    }
    return imgs;
}

#else // !NEXUSDATA_WITH_NVJPEG

struct NvJpegDecoder::Impl {};

bool NvJpegDecoder::available() noexcept { return false; }

NvJpegDecoder::NvJpegDecoder(int device) : device_(device) {
    throw InvalidArgumentError(
        "NvJpegDecoder: built without nvJPEG (reconfigure with -DNEXUSDATA_WITH_CUDA=ON "
        "-DNEXUSDATA_WITH_NVJPEG=ON)");
}

NvJpegDecoder::~NvJpegDecoder() = default;

NDArray NvJpegDecoder::decode(const std::uint8_t*, std::size_t, Device) {
    throw InvalidArgumentError("NvJpegDecoder: built without nvJPEG");
}

std::vector<NDArray> NvJpegDecoder::decode_batch(const std::vector<std::vector<std::uint8_t>>&,
                                                 Device) {
    throw InvalidArgumentError("NvJpegDecoder: built without nvJPEG");
}

#endif

} // namespace nexusdata
