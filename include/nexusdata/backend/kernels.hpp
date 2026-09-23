#pragma once

// Batch kernel ABI shared by the host reference implementations and the CUDA
// launchers. The same parameter block is executed by run_*_host() on CPU
// threads or launch_*_cuda() on a stream, and both produce identical bytes
// (see detail/kernel_math.hpp). Header is C++17 so nvcc can include it.

#include <cstddef>
#include <cstdint>

#include "nexusdata/backend/cuda_api.hpp"

namespace nexusdata {
namespace kernels {

inline constexpr int kMaxChannels = 16;
inline constexpr int kMaxTabularSteps = 16;

enum class ResampleKind : std::uint8_t { None = 0, Nearest = 1, Bilinear = 2 };
enum class ImageLayout : std::uint8_t { ChwF32 = 0, HwcU8 = 1 };

/// Per-sample geometry of a fused augment (crop origin + flips).
struct AugmentSample {
    std::int32_t crop_y = 0;
    std::int32_t crop_x = 0;
    std::uint8_t hflip = 0;
    std::uint8_t vflip = 0;
};

/// crop [crop_y, +crop_h) x [crop_x, +crop_w) -> resize to dh x dw -> flips ->
/// (ToTensor + Normalize to CHW float32) or HWC uint8.
///
/// Equivalent CPU chain: Crop -> Resize -> HFlip -> VFlip -> ToTensor -> NormalizeImage.
/// With resize == None, dh/dw must equal crop_h/crop_w.
struct ImageAugmentParams {
    const std::uint8_t* src = nullptr; ///< [n, sh, sw, c] uint8
    void* dst = nullptr;               ///< [n, c, dh, dw] f32 or [n, dh, dw, c] u8
    int n = 0, sh = 0, sw = 0, c = 0;
    int crop_h = 0, crop_w = 0;
    int dh = 0, dw = 0;
    ResampleKind resize = ResampleKind::None;
    ImageLayout layout = ImageLayout::ChwF32;
    /// Per-sample crops / flips (n entries; same memory space as src). nullptr = no crop offset, no flip.
    const AugmentSample* samples = nullptr;
    /// Flips applied to every sample (combined with the per-sample flags by XOR).
    std::uint8_t hflip_all = 0;
    std::uint8_t vflip_all = 0;
    bool normalize = false;
    float mean[kMaxChannels] = {};
    float stddev[kMaxChannels] = {};
};

/// Host reference / CPU path for samples [begin, end).
void run_image_augment_host(const ImageAugmentParams& p, int begin, int end);

/// Enqueue on @p stream (device pointers). Throws when CUDA is unavailable.
void launch_image_augment_cuda(const ImageAugmentParams& p, CudaStreamHandle stream);

// --- tabular fusion -------------------------------------------------------------

enum class TabularStepKind : std::uint8_t {
    Clip = 0,
    Log1p = 1,
    Standardize = 2,
    MinMax = 3,
    ScalarAdd = 4,
    ScalarSub = 5,
    ScalarMul = 6,
    ScalarDiv = 7,
};

struct TabularStep {
    TabularStepKind kind = TabularStepKind::Clip;
    double a = 0.0; ///< clip lo / minmax mul / scalar operand
    double b = 0.0; ///< clip hi / minmax add
    /// Standardize: mean / scale; MinMax: sub / div. `period` doubles each, device memory on GPU.
    const double* p0 = nullptr;
    const double* p1 = nullptr;
};

enum class TabularDType : std::uint8_t { F32 = 0, F64 = 1 };

/// Element-wise chain over rows of length row_len (column-dependent steps use
/// column = index % row_len). Values are widened to double; between steps they
/// are rounded to the storage type like the unfused chain.
struct TabularParams {
    const void* src = nullptr;
    void* dst = nullptr;
    TabularDType dtype = TabularDType::F32;
    std::size_t n = 0;
    std::size_t row_len = 1;
    int num_steps = 0;
    TabularStep steps[kMaxTabularSteps] = {};
};

/// Enqueue on @p stream. Log1p uses CUDA's log1p (<= 1 ulp from host libm);
/// every other step is bit-identical to FusedTransform on the CPU.
void launch_tabular_cuda(const TabularParams& p, CudaStreamHandle stream);

/// Empty kernel (launch-overhead probe).
void launch_noop_cuda(CudaStreamHandle stream);

} // namespace kernels
} // namespace nexusdata
