#pragma once

#include <memory>

#include "nexusdata/backend/cuda_api.hpp"
#include "nexusdata/backend/device.hpp"
#include "nexusdata/core/ndarray.hpp"

namespace nexusdata {

/// Abstract compute/transfer backend for data prep (not training math).
class Backend {
public:
    virtual ~Backend() = default;
    [[nodiscard]] virtual Device device() const = 0;
    [[nodiscard]] virtual const char* name() const = 0;

    /// Copy NDArray to this backend's device (may be identity).
    [[nodiscard]] virtual NDArray to_device(const NDArray& src) const = 0;
};

class CpuBackend : public Backend {
public:
    [[nodiscard]] Device device() const override { return Device::host(); }
    [[nodiscard]] const char* name() const override { return "cpu"; }
    [[nodiscard]] NDArray to_device(const NDArray& src) const override;
};

#if defined(NEXUSDATA_WITH_CUDA)
class CudaBackend : public Backend {
public:
    explicit CudaBackend(int device_index = 0);
    ~CudaBackend() override;

    [[nodiscard]] Device device() const override { return Device::cuda(index_); }
    [[nodiscard]] const char* name() const override { return "cuda"; }
    [[nodiscard]] NDArray to_device(const NDArray& src) const override;

    [[nodiscard]] CudaStreamHandle stream() const noexcept { return stream_; }
    void synchronize() const;

private:
    int index_ = 0;
    CudaStreamHandle stream_ = nullptr;
};
#endif

[[nodiscard]] std::shared_ptr<Backend> make_backend(Device device);

} // namespace nexusdata
