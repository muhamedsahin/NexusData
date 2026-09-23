#include "nexusdata/adapter/dlpack.hpp"

#include <cstring>
#include <memory>
#include <new>
#include <vector>

namespace nexusdata {

namespace {

DLDataType dtype_to_dl(DType dt) {
    DLDataType t{};
    t.lanes = 1;
    switch (dt) {
        case DType::Bool:
            t.code = kDLBool;
            t.bits = 8;
            break;
        case DType::Int8:
            t.code = kDLInt;
            t.bits = 8;
            break;
        case DType::Int16:
            t.code = kDLInt;
            t.bits = 16;
            break;
        case DType::Int32:
            t.code = kDLInt;
            t.bits = 32;
            break;
        case DType::Int64:
            t.code = kDLInt;
            t.bits = 64;
            break;
        case DType::UInt8:
            t.code = kDLUInt;
            t.bits = 8;
            break;
        case DType::UInt16:
            t.code = kDLUInt;
            t.bits = 16;
            break;
        case DType::UInt32:
            t.code = kDLUInt;
            t.bits = 32;
            break;
        case DType::UInt64:
            t.code = kDLUInt;
            t.bits = 64;
            break;
        case DType::Float16:
            t.code = kDLFloat;
            t.bits = 16;
            break;
        case DType::BFloat16:
            t.code = kDLBfloat;
            t.bits = 16;
            break;
        case DType::Float32:
            t.code = kDLFloat;
            t.bits = 32;
            break;
        case DType::Float64:
            t.code = kDLFloat;
            t.bits = 64;
            break;
        default:
            throw InvalidArgumentError("dtype_to_dl: unsupported dtype");
    }
    return t;
}

DType dl_to_dtype(DLDataType t) {
    if (t.lanes != 1) {
        throw InvalidArgumentError("dl_to_dtype: lanes != 1 not supported");
    }
    if (t.code == kDLBool && t.bits == 8) return DType::Bool;
    if (t.code == kDLInt && t.bits == 8) return DType::Int8;
    if (t.code == kDLInt && t.bits == 16) return DType::Int16;
    if (t.code == kDLInt && t.bits == 32) return DType::Int32;
    if (t.code == kDLInt && t.bits == 64) return DType::Int64;
    if (t.code == kDLUInt && t.bits == 8) return DType::UInt8;
    if (t.code == kDLUInt && t.bits == 16) return DType::UInt16;
    if (t.code == kDLUInt && t.bits == 32) return DType::UInt32;
    if (t.code == kDLUInt && t.bits == 64) return DType::UInt64;
    if (t.code == kDLFloat && t.bits == 16) return DType::Float16;
    if (t.code == kDLBfloat && t.bits == 16) return DType::BFloat16;
    if (t.code == kDLFloat && t.bits == 32) return DType::Float32;
    if (t.code == kDLFloat && t.bits == 64) return DType::Float64;
    throw InvalidArgumentError("dl_to_dtype: unsupported DLDataType");
}

struct DlpackHolder {
    AlignedBuffer storage;
    std::vector<std::int64_t> shape;
    std::vector<std::int64_t> strides;
};

void dlpack_deleter(DLManagedTensor* self) {
    if (!self) {
        return;
    }
    auto* holder = static_cast<DlpackHolder*>(self->manager_ctx);
    delete holder;
    delete self;
}

} // namespace

DLManagedTensor* ndarray_to_dlpack(const NDArray& array) {
    if (array.empty() && array.numel() != 0) {
        throw InvalidArgumentError("ndarray_to_dlpack: empty storage");
    }
    auto* holder = new DlpackHolder();
    holder->storage = array.shared_storage();
    holder->shape.resize(array.shape().size());
    holder->strides.resize(array.shape().size());
    std::int64_t stride = 1;
    for (std::size_t i = array.shape().size(); i > 0; --i) {
        const std::size_t idx = i - 1;
        holder->shape[idx] = static_cast<std::int64_t>(array.shape()[idx]);
        holder->strides[idx] = stride;
        stride *= holder->shape[idx];
    }

    auto* managed = new DLManagedTensor();
    managed->manager_ctx = holder;
    managed->deleter = &dlpack_deleter;
    const void* raw = array.data();
    managed->dl_tensor.data = const_cast<void*>(raw);
    managed->dl_tensor.ndim = static_cast<std::int32_t>(array.shape().size());
    managed->dl_tensor.dtype = dtype_to_dl(array.dtype());
    managed->dl_tensor.shape = holder->shape.data();
    managed->dl_tensor.strides = holder->strides.data();
    managed->dl_tensor.byte_offset = 0;
    if (array.device().is_cuda()) {
        managed->dl_tensor.device.device_type = kDLCUDA;
        managed->dl_tensor.device.device_id = array.device().index < 0 ? 0 : array.device().index;
    } else {
        managed->dl_tensor.device.device_type = kDLCPU;
        managed->dl_tensor.device.device_id = 0;
    }
    return managed;
}

NDArray ndarray_from_dlpack(DLManagedTensor* tensor, bool take_ownership) {
    if (!tensor) {
        throw InvalidArgumentError("ndarray_from_dlpack: null");
    }
    const DLTensor& t = tensor->dl_tensor;
    if (t.ndim < 0) {
        throw InvalidArgumentError("ndarray_from_dlpack: negative ndim");
    }
    Shape shape;
    shape.reserve(static_cast<std::size_t>(t.ndim));
    for (int i = 0; i < t.ndim; ++i) {
        if (t.shape[i] < 0) {
            throw InvalidArgumentError("ndarray_from_dlpack: negative dim");
        }
        shape.push_back(static_cast<std::size_t>(t.shape[i]));
    }
    const DType dt = dl_to_dtype(t.dtype);
    Device device = Device::host();
    if (t.device.device_type == kDLCUDA) {
        device = Device::cuda(t.device.device_id);
    } else if (t.device.device_type != kDLCPU && t.device.device_type != kDLCUDAHost) {
        throw InvalidArgumentError("ndarray_from_dlpack: unsupported device");
    }

    // Contiguous copy (ignore exotic strides for v0.9).
    NDArray out(shape, dt, device.is_cuda() ? Device::host() : device);
    if (device.is_cuda()) {
        // Import CUDA tensors via host copy for portability in v0.9.
        throw InvalidArgumentError(
            "ndarray_from_dlpack: CUDA import requires host tensor in v0.9; export then H2D");
    }
    const std::size_t nb = out.nbytes();
    if (nb > 0) {
        if (!t.data) {
            throw InvalidArgumentError("ndarray_from_dlpack: null data");
        }
        const auto* src = static_cast<const std::uint8_t*>(t.data) + t.byte_offset;
        std::memcpy(out.data(), src, nb);
    }
    if (take_ownership && tensor->deleter) {
        tensor->deleter(tensor);
    }
    return out;
}

} // namespace nexusdata
