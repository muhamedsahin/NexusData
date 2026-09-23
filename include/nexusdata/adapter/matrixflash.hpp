#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "nexusdata/core/dtype.hpp"
#include "nexusdata/core/error.hpp"
#include "nexusdata/core/ndarray.hpp"
#include "nexusdata/loading/batch.hpp"

namespace nexusdata {

/// Zero-copy view intended for hand-off to MatrixFlash (or similar compute libs).
/// NexusData never depends on MatrixFlash headers; Flash owns any conversion.
struct FlashTensorView {
    void* data = nullptr;
    std::vector<std::int64_t> shape;
    DType dtype = DType::Float32;
    std::size_t nbytes = 0;
    bool is_cuda = false;
    int device_index = -1;
};

struct FlashBatchView {
    FlashTensorView inputs;
    FlashTensorView labels;
};

/// Unidirectional adapter: NexusData → Flash-shaped views.
/// Enable richer integration later with -DNEXUSDATA_WITH_MATRIXFLASH=ON (requires Flash SDK).
class MatrixFlashAdapter {
public:
    [[nodiscard]] static FlashTensorView view(const NDArray& a) {
        if (a.empty()) {
            throw InvalidArgumentError("MatrixFlashAdapter::view: empty array");
        }
        FlashTensorView v;
        v.data = const_cast<void*>(a.data());
        v.dtype = a.dtype();
        v.nbytes = a.nbytes();
        v.is_cuda = a.device().is_cuda();
        v.device_index = a.device().index;
        v.shape.reserve(a.shape().size());
        for (std::size_t d : a.shape()) {
            v.shape.push_back(static_cast<std::int64_t>(d));
        }
        return v;
    }

    [[nodiscard]] static FlashBatchView view_batch(const Batch& b) {
        FlashBatchView out;
        out.inputs = view(b.inputs);
        out.labels = view(b.labels);
        return out;
    }

    /// Deep-copy view into a new host NDArray (safe across Flash lifetime boundaries).
    [[nodiscard]] static NDArray copy_to_ndarray(const FlashTensorView& v) {
        if (!v.data) {
            throw InvalidArgumentError("MatrixFlashAdapter::copy_to_ndarray: null data");
        }
        Shape sh;
        sh.reserve(v.shape.size());
        for (std::int64_t d : v.shape) {
            if (d < 0) {
                throw InvalidArgumentError("MatrixFlashAdapter: negative dim");
            }
            sh.push_back(static_cast<std::size_t>(d));
        }
        NDArray out(sh, v.dtype);
        std::memcpy(out.data(), v.data, std::min(out.nbytes(), v.nbytes));
        return out;
    }

    [[nodiscard]] static bool matrixflash_linked() {
#if defined(NEXUSDATA_WITH_MATRIXFLASH)
        return true;
#else
        return false;
#endif
    }

    [[nodiscard]] static std::string status_message() {
        if (matrixflash_linked()) {
            return "MatrixFlash adapter linked (NEXUSDATA_WITH_MATRIXFLASH=ON)";
        }
        return "MatrixFlash adapter: view-only mode (build with -DNEXUSDATA_WITH_MATRIXFLASH=ON "
               "when Flash SDK is available)";
    }
};

} // namespace nexusdata
