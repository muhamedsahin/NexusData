// Phase 1 benchmarks: every row compares the v2.0 path against the v1 code path
// it replaces (the v1 loops are reproduced here verbatim for a fair baseline).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include "nexusdata/bench/harness.hpp"
#include "nexusdata/nexusdata.hpp"

using namespace nexusdata;
using nexusdata::bench::print_result;
using nexusdata::bench::time_best;

namespace {

NDArray random_f32(Shape shape, std::uint32_t seed, float lo = -3.0f, float hi = 3.0f) {
    NDArray a(std::move(shape), DType::Float32);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(lo, hi);
    for (std::size_t i = 0; i < a.numel(); ++i) a.data<float>()[i] = u(rng);
    return a;
}

void speedup(const char* what, double base, double v2) {
    std::cout << "  -> " << what << " speedup: " << std::fixed << std::setprecision(2)
              << (base / v2) << "x\n";
}

// --- 1. numeric kernels --------------------------------------------------------------

void bench_numeric() {
    std::cout << "\n[numeric] standardize float32 [262144 x 32] (32 MiB)\n";
    const std::size_t rows = 262144, cols = 32;
    const NDArray x = random_f32(Shape{rows, cols}, 1);
    std::vector<double> mean(cols, 0.25), scale(cols, 1.5);
    NDArray work = x.clone();

    const double v1 = time_best([&] {
        for (std::size_t i = 0; i < work.numel(); ++i) {
            const std::size_t j = i % cols;
            detail::set_double(work, i, (detail::as_double(work, i) - mean[j]) / scale[j]);
        }
    });
    print_result("v1 as_double loop", v1, x.nbytes());

    simd::set_isa_override(simd::Isa::Scalar);
    const double sc = time_best([&] { simd::standardize_inplace(work, mean, scale); });
    print_result("v2 scalar kernel", sc, x.nbytes());
    simd::clear_isa_override();
    const double vec = time_best([&] { simd::standardize_inplace(work, mean, scale); });
    print_result(std::string("v2 ") + simd::isa_name(simd::active_isa()), vec, x.nbytes());
    speedup("v1 -> v2 dispatch", v1, vec);

    std::cout << "\n[numeric] clip float32 (8M elements)\n";
    NDArray c = random_f32(Shape{std::size_t{1} << 23}, 2);
    const double cv1 = time_best([&] {
        for (std::size_t i = 0; i < c.numel(); ++i)
            detail::set_double(c, i, std::clamp(detail::as_double(c, i), -1.0, 1.0));
    });
    print_result("v1 as_double loop", cv1, c.nbytes());
    const double cv2 = time_best([&] { simd::clip_inplace(c, -1.0, 1.0); });
    print_result("v2 clip_inplace", cv2, c.nbytes());
    speedup("clip", cv1, cv2);
}

// --- 2. transform fusion -------------------------------------------------------------

void bench_fusion() {
    std::cout << "\n[fusion] Clip -> Log1p -> Standardize -> MinMax on batch [65536 x 64]\n";
    const std::size_t B = 65536, F = 64;
    Batch b;
    b.inputs = random_f32(Shape{B, F}, 3, 0.0f, 10.0f);
    std::vector<TransformConstPtr> ts{
        std::make_shared<ClipTransform>(0.0, 8.0), std::make_shared<Log1pTransform>(),
        std::make_shared<StandardizeTransform>(std::vector<double>(F, 1.0),
                                               std::vector<double>(F, 0.5)),
        std::make_shared<MinMaxTransform>(std::vector<double>(F, -3.0),
                                          std::vector<double>(F, 3.0))};
    Compose fused(ts);
    Compose plain(ts, false);

    // v1: every transform clones the batch and runs its own as_double loop.
    const double v1 = time_best(
        [&] {
            NDArray x = b.inputs;
            for (int t = 0; t < 4; ++t) {
                NDArray y = x.clone();
                for (std::size_t i = 0; i < y.numel(); ++i) {
                    double v = detail::as_double(y, i);
                    const std::size_t j = i % F;
                    switch (t) {
                        case 0: v = std::clamp(v, 0.0, 8.0); break;
                        case 1: v = std::log1p(v); break;
                        case 2: v = (v - 1.0) / 0.5; break;
                        default: v = 0.0 + 1.0 * ((v + 3.0) / 6.0); break;
                    }
                    (void)j;
                    detail::set_double(y, i, v);
                }
                x = y;
            }
        },
        3);
    print_result("v1 4 passes (as_double)", v1, b.inputs.nbytes());
    const double unf = time_best([&] { (void)plain.apply_batch(b); }, 3);
    print_result("v2 unfused (4 SIMD passes)", unf, b.inputs.nbytes());
    const double fu = time_best([&] { (void)fused.apply_batch(b); }, 3);
    print_result("v2 fused (1 pass, MT)", fu, b.inputs.nbytes());
    speedup("v1 -> fused", v1, fu);
    speedup("unfused -> fused", unf, fu);
}

// --- 3. image pipeline ---------------------------------------------------------------

void v1_image_pipeline(const NDArray& img, int oh, int ow, const std::vector<float>& mean,
                       const std::vector<float>& stdv) {
    const std::size_t sh = img.shape()[0], sw = img.shape()[1], c = img.shape()[2];
    NDArray r(Shape{static_cast<std::size_t>(oh), static_cast<std::size_t>(ow), c}, DType::UInt8);
    for (int y = 0; y < oh; ++y) {
        for (int x = 0; x < ow; ++x) {
            const float fy = (y + 0.5f) * static_cast<float>(sh) / static_cast<float>(oh) - 0.5f;
            const float fx = (x + 0.5f) * static_cast<float>(sw) / static_cast<float>(ow) - 0.5f;
            imgdetail::sample_bilinear(img, sh, sw, c, fy, fx,
                                       r.data<std::uint8_t>() + (static_cast<std::size_t>(y) * ow + x) * c);
        }
    }
    NDArray t(Shape{c, static_cast<std::size_t>(oh), static_cast<std::size_t>(ow)}, DType::Float32);
    const std::size_t hw = static_cast<std::size_t>(oh) * ow;
    for (std::size_t i = 0; i < hw; ++i)
        for (std::size_t ch = 0; ch < c; ++ch)
            t.data<float>()[ch * hw + i] = static_cast<float>(r.data<std::uint8_t>()[i * c + ch]) / 255.0f;
    NDArray n = t.clone();
    for (std::size_t ch = 0; ch < c; ++ch)
        for (std::size_t i = 0; i < hw; ++i)
            n.data<float>()[ch * hw + i] = (n.data<float>()[ch * hw + i] - mean[ch]) / stdv[ch];
}

void bench_image() {
    std::cout << "\n[image] 480x640x3 uint8 -> Resize(224) -> ToTensor -> Normalize, 32 images\n";
    NDArray img(Shape{480, 640, 3}, DType::UInt8);
    std::mt19937 rng(4);
    for (std::size_t i = 0; i < img.numel(); ++i)
        img.data<std::uint8_t>()[i] = static_cast<std::uint8_t>(rng() & 0xFF);
    const std::vector<float> mean{0.485f, 0.456f, 0.406f}, stdv{0.229f, 0.224f, 0.225f};
    std::vector<TransformConstPtr> ts{std::make_shared<Resize>(224, 224),
                                      std::make_shared<ToTensor>(),
                                      std::make_shared<NormalizeImage>(mean, stdv)};
    Compose fused(ts);
    Compose plain(ts, false);
    Sample s;
    s.input = img;
    const int n = 32;
    const double v1 = time_best([&] {
        for (int i = 0; i < n; ++i) v1_image_pipeline(img, 224, 224, mean, stdv);
    }, 3);
    print_result("v1 3 transforms", v1, 0, n);
    const double unf = time_best([&] {
        for (int i = 0; i < n; ++i) (void)plain.apply(s);
    }, 3);
    print_result("v2 unfused", unf, 0, n);
    const double fu = time_best([&] {
        for (int i = 0; i < n; ++i) (void)fused.apply(s);
    }, 3);
    print_result("v2 fused resize->CHW", fu, 0, n);
    speedup("v1 -> fused", v1, fu);
}

// --- 4. allocation -------------------------------------------------------------------

void bench_alloc() {
    std::cout << "\n[alloc] 256 x NDArray[1024] float32 per batch, 2000 batches\n";
    const int batches = 2000, per = 256;
    const double heap = time_best([&] {
        for (int b = 0; b < batches; ++b) {
            std::vector<NDArray> v;
            v.reserve(per);
            for (int i = 0; i < per; ++i) v.emplace_back(Shape{1024}, DType::Float32);
        }
    }, 3);
    print_result("v1 NDArray(shape) (zeroed)", heap, 0, static_cast<std::size_t>(batches) * per);
    Arena arena(std::size_t{2} << 20);
    const double ar = time_best([&] {
        for (int b = 0; b < batches; ++b) {
            {
                std::vector<NDArray> v;
                v.reserve(per);
                for (int i = 0; i < per; ++i)
                    v.push_back(NDArray::from_arena(arena, Shape{1024}, DType::Float32, true));
            }
            arena.reset();
        }
    }, 3);
    print_result("v2 from_arena (zeroed)", ar, 0, static_cast<std::size_t>(batches) * per);
    speedup("arena", heap, ar);
    BufferPool pool;
    const double bp = time_best([&] {
        for (int b = 0; b < batches; ++b) {
            std::vector<NDArray> v;
            v.reserve(per);
            for (int i = 0; i < per; ++i) v.push_back(pool.make_array(Shape{1024}, DType::Float32));
        }
    }, 3);
    print_result("v2 BufferPool (zeroed)", bp, 0, static_cast<std::size_t>(batches) * per);
    speedup("buffer pool", heap, bp);
}

// --- 5. DataLoader scaling -----------------------------------------------------------

class SlowDataset : public Dataset {
public:
    [[nodiscard]] std::size_t size() const override { return 4096; }
    [[nodiscard]] Sample get(std::size_t i) const override {
        Sample s;
        s.input = NDArray(Shape{256}, DType::Float32);
        float acc = static_cast<float>(i);
        // ~20-40 us of CPU work, standing in for decode/parse.
        for (int k = 0; k < 6000; ++k) acc = std::sqrt(acc + static_cast<float>(k));
        for (std::size_t j = 0; j < 256; ++j) s.input.data<float>()[j] = acc + static_cast<float>(j);
        s.label = NDArray(Shape{1}, DType::Int64);
        return s;
    }
};

void bench_loader() {
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    std::cout << "\n[loader] 4096 samples of synthetic CPU work, batch 64 (" << hw
              << " hardware threads)\n";
    auto ds = std::make_shared<SlowDataset>();
    double base = 0.0;
    for (int w : {0, 1, 2, 4, 8}) {
        if (w > static_cast<int>(hw)) break;
        DataLoaderOptions o;
        o.batch_size = 64;
        o.num_workers = w;
        DataLoader dl(DatasetConstPtr(ds), o);
        const double t = time_best([&] {
            for (const Batch& b : dl) (void)b;
        }, 2);
        print_result("num_workers=" + std::to_string(w), t, 0, 4096);
        if (w == 0) base = t;
        else speedup(("workers=" + std::to_string(w)).c_str(), base, t);
    }
}

} // namespace

int main() {
    std::cout << "NexusData Phase 1 benchmarks (ISA: " << simd::isa_name(simd::detected_isa())
              << ")\n";
    bench_numeric();
    bench_fusion();
    bench_image();
    bench_alloc();
    bench_loader();
    return 0;
}
