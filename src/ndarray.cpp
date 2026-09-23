#include "nexusdata/core/ndarray.hpp"

#include <cstring>
#include <string>

#include "nexusdata/backend/cuda_api.hpp"
#include "nexusdata/backend/pinned.hpp"

namespace nexusdata {

namespace {

AlignedBuffer alloc_storage(std::size_t nbytes, Device device) {
    if (nbytes == 0) {
        return {};
    }
    if (device.is_host() || device.is_auto()) {
        auto buf = make_aligned_buffer(nbytes);
        std::memset(buf.get(), 0, nbytes);
        return buf;
    }
    if (device.is_cuda()) {
        if (!cuda_api::available()) {
            throw InvalidArgumentError("NDArray: CUDA device requested but CUDA unavailable");
        }
        cuda_api::set_device(device.index < 0 ? 0 : device.index);
        void* ptr = cuda_api::device_alloc(nbytes);
        return AlignedBuffer(ptr, [](void* p) { cuda_api::device_free(p); });
    }
    throw InvalidArgumentError("NDArray: unsupported device");
}

} // namespace

NDArray::NDArray(Shape shape, DType dtype, Device device)
    : dtype_(dtype),
      shape_(std::move(shape)),
      numel_(::nexusdata::numel(shape_)),
      device_(device.is_auto() ? Device::host() : device) {
    if (device_.is_cuda() && device_.index < 0) {
        device_.index = 0;
    }
    data_ = alloc_storage(nbytes(), device_);
}

NDArray NDArray::from_blob(void* data, Shape shape, DType dtype, Device device) {
    NDArray out;
    out.dtype_ = dtype;
    out.shape_ = std::move(shape);
    out.numel_ = ::nexusdata::numel(out.shape_);
    out.device_ = device.is_auto() ? Device::host() : device;
    if (out.numel_ > 0) {
        if (!data) {
            throw InvalidArgumentError("NDArray::from_blob: null data pointer");
        }
        out.data_ = AlignedBuffer(data, [](void*) {});
    }
    return out;
}

NDArray NDArray::from_shared(AlignedBuffer buf, Shape shape, DType dtype, Device device) {
    NDArray out;
    out.dtype_ = dtype;
    out.shape_ = std::move(shape);
    out.numel_ = ::nexusdata::numel(out.shape_);
    out.device_ = device.is_auto() ? Device::host() : device;
    out.data_ = std::move(buf);
    return out;
}

NDArray NDArray::clone() const {
    NDArray out(shape_, dtype_, device_);
    if (numel_ == 0 || !data_) {
        return out;
    }
    if (device_.is_host() && out.device_.is_host()) {
        std::memcpy(out.data_.get(), data_.get(), nbytes());
    } else if (device_.is_host() && out.device_.is_cuda()) {
        cuda_api::memcpy_h2d(out.data_.get(), data_.get(), nbytes(), nullptr);
        cuda_api::stream_synchronize(nullptr);
    } else if (device_.is_cuda() && out.device_.is_host()) {
        cuda_api::memcpy_d2h(out.data_.get(), data_.get(), nbytes(), nullptr);
        cuda_api::stream_synchronize(nullptr);
    } else {
        cuda_api::memcpy_d2d(out.data_.get(), data_.get(), nbytes(), nullptr);
        cuda_api::stream_synchronize(nullptr);
    }
    return out;
}

NDArray NDArray::pinned(Shape shape, DType dtype) {
    NDArray out;
    out.dtype_ = dtype;
    out.shape_ = std::move(shape);
    out.numel_ = ::nexusdata::numel(out.shape_);
    out.device_ = Device::host();
    if (out.numel_ > 0) {
        out.data_ = make_pinned_buffer(out.nbytes());
        std::memset(out.data_.get(), 0, out.nbytes());
    }
    return out;
}

void NDArray::reshape(Shape new_shape) {
    if (::nexusdata::numel(new_shape) != numel_) {
        throw ShapeError("NDArray::reshape: numel mismatch (have " +
                         std::to_string(numel_) + ", new shape " +
                         shape_to_string(new_shape) + " => " +
                         std::to_string(::nexusdata::numel(new_shape)) + ")");
    }
    shape_ = std::move(new_shape);
}

NDArray NDArray::stack(const std::vector<NDArray>& arrays) {
    if (arrays.empty()) {
        throw InvalidArgumentError("NDArray::stack: empty list");
    }
    const NDArray& first = arrays[0];
    if (!first.device().is_host()) {
        throw InvalidArgumentError("NDArray::stack: host-only in v0.5 (move to cpu() first)");
    }
    const std::size_t n = arrays.size();
    const std::size_t features = first.numel();
    const DType dt = first.dtype();

    NDArray result(Shape{n, features}, dt);
    const std::size_t row_bytes = features * size_of(dt);
    auto* dst = static_cast<std::uint8_t*>(result.data());

    for (std::size_t i = 0; i < n; ++i) {
        const NDArray& a = arrays[i];
        if (a.dtype() != dt || a.numel() != features) {
            throw ShapeError("NDArray::stack: all arrays must share dtype and numel");
        }
        if (!a.device().is_host()) {
            throw InvalidArgumentError("NDArray::stack: host-only");
        }
        std::memcpy(dst + i * row_bytes, a.data(), row_bytes);
    }
    return result;
}

NDArray NDArray::concat(const std::vector<NDArray>& arrays, int axis) {
    if (arrays.empty()) {
        throw InvalidArgumentError("NDArray::concat: empty list");
    }
    if (axis != 0) {
        throw InvalidArgumentError("NDArray::concat: only axis=0 supported");
    }
    const NDArray& first = arrays[0];
    if (first.shape().empty()) {
        throw ShapeError("NDArray::concat: cannot concat empty-rank arrays");
    }
    if (!first.device().is_host()) {
        throw InvalidArgumentError("NDArray::concat: host-only");
    }

    Shape out_shape = first.shape();
    std::size_t axis0 = 0;
    for (const NDArray& a : arrays) {
        if (a.dtype() != first.dtype()) {
            throw ShapeError("NDArray::concat: dtype mismatch");
        }
        if (a.shape().size() != first.shape().size()) {
            throw ShapeError("NDArray::concat: rank mismatch");
        }
        for (std::size_t d = 1; d < a.shape().size(); ++d) {
            if (a.shape()[d] != first.shape()[d]) {
                throw ShapeError("NDArray::concat: trailing shape mismatch");
            }
        }
        axis0 += a.shape()[0];
    }
    out_shape[0] = axis0;

    NDArray result(out_shape, first.dtype());
    std::size_t offset = 0;
    auto* dst = static_cast<std::uint8_t*>(result.data());
    for (const NDArray& a : arrays) {
        const std::size_t nb = a.nbytes();
        std::memcpy(dst + offset, a.data(), nb);
        offset += nb;
    }
    return result;
}

NDArray NDArray::cpu() const {
    if (device_.is_host()) {
        return *this;
    }
    NDArray out(shape_, dtype_, Device::host());
    if (numel_ > 0) {
        cuda_api::memcpy_d2h(out.data_.get(), data_.get(), nbytes(), nullptr);
        cuda_api::stream_synchronize(nullptr);
    }
    return out;
}

NDArray NDArray::cuda(int index) const {
    Device dev = Device::cuda(index);
    if (device_.is_cuda() && device_.index == index) {
        return *this;
    }
    NDArray out(shape_, dtype_, dev);
    if (numel_ == 0) {
        return out;
    }
    if (device_.is_host()) {
        cuda_api::memcpy_h2d(out.data_.get(), data_.get(), nbytes(), nullptr);
    } else {
        cuda_api::memcpy_d2d(out.data_.get(), data_.get(), nbytes(), nullptr);
    }
    cuda_api::stream_synchronize(nullptr);
    return out;
}

} // namespace nexusdata
