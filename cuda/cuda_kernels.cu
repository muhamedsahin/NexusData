// CUDA kernels (compiled only when NEXUSDATA_WITH_CUDA=ON, with --fmad=false so
// results are bit-identical to the CPU kernels; see backend/detail/kernel_math.hpp).

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <string>

#include "nexusdata/backend/cuda_api.hpp"
#include "nexusdata/backend/detail/kernel_math.hpp"
#include "nexusdata/backend/kernels.hpp"
#include "nexusdata/core/error.hpp"

namespace nexusdata {
namespace {

constexpr int kBlock = 256;

cudaStream_t as_stream(CudaStreamHandle s) { return static_cast<cudaStream_t>(s); }

void check_launch(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        throw Error(std::string("CUDA kernel ") + what + ": " + cudaGetErrorString(e));
    }
}

/// Grid for a grid-stride loop over @p work items: enough blocks to fill the GPU, no more.
unsigned grid_for(long long work) {
    static int sms = [] {
        int dev = 0;
        int v = 0;
        if (cudaGetDevice(&dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess) {
            return 16;
        }
        return v;
    }();
    const long long need = (work + kBlock - 1) / kBlock;
    const long long cap = static_cast<long long>(sms) * 32;
    return static_cast<unsigned>(std::max<long long>(1, std::min(need, cap)));
}

// --- HWC uint8 -> CHW float32 (shared-memory transpose) -----------------------------

struct NormArgs {
    float mean[kernels::kMaxChannels];
    float stddev[kernels::kMaxChannels];
    int normalize;
};

/// Each block stages a tile of kBlock pixels (all channels) in shared memory with
/// fully coalesced byte loads, then every thread writes its pixel into the C
/// planes (coalesced along x). mean/std come from the by-value argument block
/// (constant bank) or, for the legacy API, from device pointers.
template <int C>
__global__ void k_hwc_to_chw(const std::uint8_t* __restrict__ src, float* __restrict__ dst,
                             long long total_pix, int hw, int c_rt, NormArgs args,
                             const float* __restrict__ d_mean, const float* __restrict__ d_std) {
    extern __shared__ std::uint8_t tile[];
    __shared__ float s_mean[kernels::kMaxChannels];
    __shared__ float s_std[kernels::kMaxChannels];
    const int c = C > 0 ? C : c_rt;
    const bool norm = d_mean != nullptr || args.normalize != 0;
    if (threadIdx.x < static_cast<unsigned>(c)) {
        s_mean[threadIdx.x] = d_mean != nullptr ? d_mean[threadIdx.x] : args.mean[threadIdx.x];
        s_std[threadIdx.x] = d_std != nullptr ? d_std[threadIdx.x] : args.stddev[threadIdx.x];
    }
    __syncthreads();
    for (long long t0 = static_cast<long long>(blockIdx.x) * kBlock; t0 < total_pix;
         t0 += static_cast<long long>(gridDim.x) * kBlock) {
        const int npix = static_cast<int>(total_pix - t0 < kBlock ? total_pix - t0 : kBlock);
        const std::uint8_t* tsrc = src + t0 * c;
        for (int i = threadIdx.x; i < npix * c; i += blockDim.x) {
            tile[i] = tsrc[i];
        }
        __syncthreads();
        if (static_cast<int>(threadIdx.x) < npix) {
            const long long p = t0 + threadIdx.x;
            const long long img = p / hw;
            const int pix = static_cast<int>(p - img * hw);
            float* out = dst + img * c * hw + pix;
            const std::uint8_t* px = tile + threadIdx.x * c;
            for (int ch = 0; ch < c; ++ch) {
                out[static_cast<long long>(ch) * hw] =
                    norm ? kmath::u8_to_f32_norm(px[ch], s_mean[ch], s_std[ch])
                         : kmath::u8_to_f32(px[ch]);
            }
        }
        __syncthreads();
    }
}

void launch_hwc_to_chw(const std::uint8_t* src, float* dst, int n, int h, int w, int c,
                       const NormArgs& args, const float* d_mean, const float* d_std,
                       CudaStreamHandle stream) {
    if (c <= 0 || c > kernels::kMaxChannels) {
        throw InvalidArgumentError("hwc_u8_to_chw_f32 (CUDA): channels must be in [1, 16]");
    }
    const long long total = static_cast<long long>(n) * h * w;
    if (total == 0) {
        return;
    }
    const unsigned grid = grid_for(total);
    const std::size_t smem = static_cast<std::size_t>(kBlock) * c;
    const int hw = h * w;
    auto s = as_stream(stream);
    switch (c) {
        case 1: k_hwc_to_chw<1><<<grid, kBlock, smem, s>>>(src, dst, total, hw, c, args, d_mean, d_std); break;
        case 3: k_hwc_to_chw<3><<<grid, kBlock, smem, s>>>(src, dst, total, hw, c, args, d_mean, d_std); break;
        case 4: k_hwc_to_chw<4><<<grid, kBlock, smem, s>>>(src, dst, total, hw, c, args, d_mean, d_std); break;
        default: k_hwc_to_chw<0><<<grid, kBlock, smem, s>>>(src, dst, total, hw, c, args, d_mean, d_std); break;
    }
    check_launch("hwc_u8_to_chw_f32");
}

// --- fused crop -> resize -> flip -> (normalize) ------------------------------------

template <kernels::ResampleKind R, kernels::ImageLayout L>
__global__ void k_augment(kernels::ImageAugmentParams p) {
    const long long total = static_cast<long long>(p.n) * p.dh * p.dw;
    const std::size_t c = static_cast<std::size_t>(p.c);
    const std::size_t stride = static_cast<std::size_t>(p.sw) * c;
    const std::size_t src_img = static_cast<std::size_t>(p.sh) * stride;
    for (long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
         idx += static_cast<long long>(gridDim.x) * blockDim.x) {
        const int x = static_cast<int>(idx % p.dw);
        const long long t = idx / p.dw;
        const int y = static_cast<int>(t % p.dh);
        const int s = static_cast<int>(t / p.dh);
        int cy = 0, cx = 0;
        bool hf = p.hflip_all != 0;
        bool vf = p.vflip_all != 0;
        if (p.samples != nullptr) {
            const kernels::AugmentSample a = p.samples[s];
            cy = a.crop_y;
            cx = a.crop_x;
            hf = hf != (a.hflip != 0);
            vf = vf != (a.vflip != 0);
        }
        const int ry = vf ? p.dh - 1 - y : y;
        const int rx = hf ? p.dw - 1 - x : x;
        const std::uint8_t* base = p.src + static_cast<std::size_t>(s) * src_img +
                                   static_cast<std::size_t>(cy) * stride +
                                   static_cast<std::size_t>(cx) * c;

        std::size_t o00 = 0, o01 = 0, o10 = 0, o11 = 0;
        float wx = 0.0f, wy = 0.0f;
        if constexpr (R == kernels::ResampleKind::None) {
            o00 = static_cast<std::size_t>(ry) * stride + static_cast<std::size_t>(rx) * c;
        } else if constexpr (R == kernels::ResampleKind::Nearest) {
            o00 = static_cast<std::size_t>(kmath::nearest_tap(ry, p.crop_h, p.dh)) * stride +
                  static_cast<std::size_t>(kmath::nearest_tap(rx, p.crop_w, p.dw)) * c;
        } else {
            const kmath::BilinearTap ty = kmath::bilinear_tap(ry, p.crop_h, p.dh);
            const kmath::BilinearTap tx = kmath::bilinear_tap(rx, p.crop_w, p.dw);
            const std::size_t r0 = static_cast<std::size_t>(ty.i0) * stride;
            const std::size_t r1 = static_cast<std::size_t>(ty.i1) * stride;
            const std::size_t c0 = static_cast<std::size_t>(tx.i0) * c;
            const std::size_t c1 = static_cast<std::size_t>(tx.i1) * c;
            o00 = r0 + c0;
            o01 = r0 + c1;
            o10 = r1 + c0;
            o11 = r1 + c1;
            wx = tx.w;
            wy = ty.w;
        }

        for (std::size_t ch = 0; ch < c; ++ch) {
            std::uint8_t v;
            if constexpr (R == kernels::ResampleKind::Bilinear) {
                v = kmath::bilinear_u8(base[o00 + ch], base[o01 + ch], base[o10 + ch],
                                       base[o11 + ch], wx, wy);
            } else {
                v = base[o00 + ch];
            }
            if constexpr (L == kernels::ImageLayout::HwcU8) {
                static_cast<std::uint8_t*>(p.dst)[idx * static_cast<long long>(c) + ch] = v;
            } else {
                const long long plane = static_cast<long long>(p.dh) * p.dw;
                const long long o = (static_cast<long long>(s) * p.c + static_cast<long long>(ch)) *
                                        plane +
                                    static_cast<long long>(y) * p.dw + x;
                static_cast<float*>(p.dst)[o] =
                    p.normalize ? kmath::u8_to_f32_norm(v, p.mean[ch], p.stddev[ch])
                                : kmath::u8_to_f32(v);
            }
        }
    }
}

template <kernels::ResampleKind R>
void launch_augment_layout(const kernels::ImageAugmentParams& p, unsigned grid, cudaStream_t s) {
    if (p.layout == kernels::ImageLayout::HwcU8) {
        k_augment<R, kernels::ImageLayout::HwcU8><<<grid, kBlock, 0, s>>>(p);
    } else {
        k_augment<R, kernels::ImageLayout::ChwF32><<<grid, kBlock, 0, s>>>(p);
    }
}

// --- fused tabular chain --------------------------------------------------------------

template <typename T>
__global__ void k_tabular(kernels::TabularParams p) {
    const T* __restrict__ src = static_cast<const T*>(p.src);
    T* __restrict__ dst = static_cast<T*>(p.dst);
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < p.n;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const std::size_t col = i % p.row_len;
        double v = static_cast<double>(src[i]);
        for (int k = 0; k < p.num_steps; ++k) {
            const kernels::TabularStep& st = p.steps[k];
            switch (st.kind) {
                case kernels::TabularStepKind::Clip: v = kmath::clampd(v, st.a, st.b); break;
                case kernels::TabularStepKind::Log1p: v = log1p(v); break;
                case kernels::TabularStepKind::Standardize: v = (v - st.p0[col]) / st.p1[col]; break;
                case kernels::TabularStepKind::MinMax: v = st.b + st.a * ((v - st.p0[col]) / st.p1[col]); break;
                case kernels::TabularStepKind::ScalarAdd: v = v + st.a; break;
                case kernels::TabularStepKind::ScalarSub: v = v - st.a; break;
                case kernels::TabularStepKind::ScalarMul: v = v * st.a; break;
                case kernels::TabularStepKind::ScalarDiv: v = v / st.a; break;
            }
            if (k + 1 < p.num_steps) {
                v = static_cast<double>(static_cast<T>(v));
            }
        }
        dst[i] = static_cast<T>(v);
    }
}

__global__ void k_noop() {}

} // namespace

namespace kernels {

void launch_image_augment_cuda(const ImageAugmentParams& p, CudaStreamHandle stream) {
    if (p.c <= 0 || p.c > kMaxChannels) {
        throw InvalidArgumentError("image augment (CUDA): channels must be in [1, 16]");
    }
    const long long total = static_cast<long long>(p.n) * p.dh * p.dw;
    if (total == 0) {
        return;
    }
    const unsigned grid = grid_for(total);
    auto s = as_stream(stream);
    switch (p.resize) {
        case ResampleKind::None: launch_augment_layout<ResampleKind::None>(p, grid, s); break;
        case ResampleKind::Nearest: launch_augment_layout<ResampleKind::Nearest>(p, grid, s); break;
        case ResampleKind::Bilinear: launch_augment_layout<ResampleKind::Bilinear>(p, grid, s); break;
    }
    check_launch("image_augment");
}

void launch_tabular_cuda(const TabularParams& p, CudaStreamHandle stream) {
    if (p.num_steps <= 0 || p.num_steps > kMaxTabularSteps || p.row_len == 0) {
        throw InvalidArgumentError("tabular (CUDA): invalid step count / row length");
    }
    if (p.n == 0) {
        return;
    }
    const unsigned grid = grid_for(static_cast<long long>(p.n));
    auto s = as_stream(stream);
    if (p.dtype == TabularDType::F32) {
        k_tabular<float><<<grid, kBlock, 0, s>>>(p);
    } else {
        k_tabular<double><<<grid, kBlock, 0, s>>>(p);
    }
    check_launch("tabular");
}

void launch_noop_cuda(CudaStreamHandle stream) {
    k_noop<<<1, 1, 0, as_stream(stream)>>>();
    check_launch("noop");
}

} // namespace kernels

namespace cuda_api {

void launch_hwc_u8_to_chw_f32_normalize(const std::uint8_t* d_hwc, float* d_chw, int n, int h,
                                        int w, int c, const float* d_mean, const float* d_std,
                                        CudaStreamHandle stream) {
    NormArgs args{};
    args.normalize = 1;
    launch_hwc_to_chw(d_hwc, d_chw, n, h, w, c, args, d_mean, d_std, stream);
}

void launch_hwc_u8_to_chw_f32(const std::uint8_t* d_hwc, float* d_chw, int n, int h, int w, int c,
                              const float* host_mean, const float* host_std,
                              CudaStreamHandle stream) {
    NormArgs args{};
    if (host_mean != nullptr && c <= kernels::kMaxChannels) {
        args.normalize = 1;
        for (int i = 0; i < c; ++i) {
            args.mean[i] = host_mean[i];
            args.stddev[i] = host_std[i];
        }
    }
    launch_hwc_to_chw(d_hwc, d_chw, n, h, w, c, args, nullptr, nullptr, stream);
}

namespace {
kernels::ImageAugmentParams geometry(const std::uint8_t* src, std::uint8_t* dst, int n, int sh,
                                     int sw, int dh, int dw, int c) {
    kernels::ImageAugmentParams p;
    p.src = src;
    p.dst = dst;
    p.n = n;
    p.sh = sh;
    p.sw = sw;
    p.c = c;
    p.crop_h = sh;
    p.crop_w = sw;
    p.dh = dh;
    p.dw = dw;
    p.layout = kernels::ImageLayout::HwcU8;
    return p;
}
} // namespace

void launch_resize_nearest_u8(const std::uint8_t* d_src, std::uint8_t* d_dst, int sh, int sw,
                              int dh, int dw, int c, CudaStreamHandle stream) {
    auto p = geometry(d_src, d_dst, 1, sh, sw, dh, dw, c);
    p.resize = kernels::ResampleKind::Nearest;
    kernels::launch_image_augment_cuda(p, stream);
}

void launch_resize_bilinear_u8(const std::uint8_t* d_src, std::uint8_t* d_dst, int n, int sh,
                               int sw, int dh, int dw, int c, CudaStreamHandle stream) {
    auto p = geometry(d_src, d_dst, n, sh, sw, dh, dw, c);
    p.resize = kernels::ResampleKind::Bilinear;
    kernels::launch_image_augment_cuda(p, stream);
}

void launch_hflip_u8(const std::uint8_t* d_src, std::uint8_t* d_dst, int h, int w, int c,
                     CudaStreamHandle stream) {
    auto p = geometry(d_src, d_dst, 1, h, w, h, w, c);
    p.hflip_all = 1;
    kernels::launch_image_augment_cuda(p, stream);
}

void launch_vflip_u8(const std::uint8_t* d_src, std::uint8_t* d_dst, int h, int w, int c,
                     CudaStreamHandle stream) {
    auto p = geometry(d_src, d_dst, 1, h, w, h, w, c);
    p.vflip_all = 1;
    kernels::launch_image_augment_cuda(p, stream);
}

} // namespace cuda_api
} // namespace nexusdata
