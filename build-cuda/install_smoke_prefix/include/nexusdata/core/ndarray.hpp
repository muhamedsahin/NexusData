#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "nexusdata/backend/device.hpp"
#include "nexusdata/core/allocator.hpp"
#include "nexusdata/core/dtype.hpp"
#include "nexusdata/core/error.hpp"
#include "nexusdata/core/shape.hpp"

namespace nexusdata {

class Arena;

/// Contiguous N-dimensional array used only as a data carrier (no math).
///
/// Ownership: storage is reference-counted (shared_ptr). Copy/assignment are
/// shallow. Use clone() for a deep copy.
/// Host and (when built with CUDA) device pointers are supported.
///
/// Thread-safety: concurrent reads of an immutable NDArray are safe.
class NDArray {
public:
    NDArray() = default;

    NDArray(Shape shape, DType dtype, Device device = Device::host());

    [[nodiscard]] static NDArray from_blob(void* data,
                                           Shape shape,
                                           DType dtype,
                                           Device device = Device::host());

    [[nodiscard]] const Shape& shape() const noexcept { return shape_; }
    [[nodiscard]] DType dtype() const noexcept { return dtype_; }
    [[nodiscard]] std::size_t numel() const noexcept { return numel_; }
    [[nodiscard]] std::size_t nbytes() const noexcept {
        return numel_ * size_of(dtype_);
    }
    [[nodiscard]] Device device() const noexcept { return device_; }
    [[nodiscard]] bool empty() const noexcept { return numel_ == 0 || !data_; }
    [[nodiscard]] void* data() noexcept { return data_.get(); }
    [[nodiscard]] const void* data() const noexcept { return data_.get(); }

    template <typename T>
    [[nodiscard]] T* data() {
        check_type_access<T>();
        return static_cast<T*>(data_.get());
    }

    template <typename T>
    [[nodiscard]] const T* data() const {
        check_type_access<T>();
        return static_cast<const T*>(data_.get());
    }

    [[nodiscard]] NDArray clone() const;
    void reshape(Shape new_shape);

    /// Allocate page-locked host storage (CUDA pinned when available).
    [[nodiscard]] static NDArray pinned(Shape shape, DType dtype);

    [[nodiscard]] static NDArray stack(const std::vector<NDArray>& arrays);
    [[nodiscard]] static NDArray concat(const std::vector<NDArray>& arrays, int axis = 0);

    /// Copy to host (synchronizing). Identity if already host.
    [[nodiscard]] NDArray cpu() const;
    /// Copy to CUDA device (requires CUDA build).
    [[nodiscard]] NDArray cuda(int index = 0) const;

    /// Shared storage handle (for adapters / DLPack lifetime).
    [[nodiscard]] AlignedBuffer shared_storage() const { return data_; }

    /// Internal: build NDArray that shares an existing buffer (adapters).
    static NDArray from_shared(AlignedBuffer buf, Shape shape, DType dtype, Device device);

    /// Host array carved out of @p arena (one bump-pointer step, no malloc).
    /// The array keeps its arena block alive, so it stays valid across
    /// Arena::reset(). Contents are uninitialized unless @p zero is true.
    [[nodiscard]] static NDArray from_arena(Arena& arena, Shape shape, DType dtype,
                                            bool zero = false);

private:
    template <typename T>
    void check_type_access() const {
        if (!data_ && numel_ > 0) {
            throw InvalidArgumentError("NDArray::data<T>() on empty storage");
        }
        if constexpr (std::is_same_v<std::remove_cv_t<T>, std::uint8_t>) {
            return;
        }
        if (sizeof(T) != size_of(dtype_)) {
            throw InvalidArgumentError(
                std::string("NDArray::data<T>(): sizeof(T)=") +
                std::to_string(sizeof(T)) + " does not match dtype " +
                std::string(to_string(dtype_)) + " (" +
                std::to_string(size_of(dtype_)) + " bytes)");
        }
        if (device_.is_cuda()) {
            // Typed host access of device pointer is undefined — allow void/uint8 for kernels.
            throw InvalidArgumentError(
                "NDArray::data<T>(): typed host access on CUDA tensor; use kernels or cpu()");
        }
    }

    DType dtype_ = DType::Float32;
    Shape shape_{};
    std::size_t numel_ = 0;
    Device device_ = Device::host();
    AlignedBuffer data_{};
};

} // namespace nexusdata
