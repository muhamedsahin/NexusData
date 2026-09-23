// v2.0 Phase 2 benchmarks: CUDA kernels vs CPU paths, pipelined transfers,
// CostModel decision quality against a measured oracle, DataLoader end to end,
// nvJPEG vs stb_image. Prints a report; exits 0 without a GPU.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "nexusdata/nexusdata.hpp"
#include "nexusdata/backend/gpu_executor.hpp"
#include "nexusdata/backend/gpu_telemetry.hpp"
#include "nexusdata/image/nvjpeg.hpp"
#include "nexusdata/pipeline/augment.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include "stb_image_write.h"

using namespace nexusdata;
using clk = std::chrono::steady_clock;

namespace {

double median_ms(const std::function<void()>& fn, int reps = 7, int warmup = 2) {
    for (int i = 0; i < warmup; ++i) fn();
    std::vector<double> t;
    for (int i = 0; i < reps; ++i) {
        const auto t0 = clk::now();
        fn();
        t.push_back(std::chrono::duration<double, std::milli>(clk::now() - t0).count());
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

NDArray random_u8(Shape s, std::uint64_t seed) {
    NDArray x(std::move(s), DType::UInt8);
    PCG32 rng(seed);
    auto* p = x.data<std::uint8_t>();
    for (std::size_t i = 0; i < x.numel(); ++i) p[i] = static_cast<std::uint8_t>(rng.next_u32());
    return x;
}

NDArray random_f32(std::size_t rows, std::size_t cols, std::uint64_t seed) {
    NDArray x(Shape{rows, cols}, DType::Float32);
    PCG32 rng(seed);
    for (std::size_t i = 0; i < x.numel(); ++i) x.data<float>()[i] = rng.next_float() * 10.f - 3.f;
    return x;
}

Batch as_batch(NDArray x) {
    Batch b;
    b.inputs = std::move(x);
    return b;
}

Batch run_gpu(const Transform& t, const Batch& b, bool on_device) {
    GpuExecutor& ex = GpuExecutor::instance(0);
    auto lease = ex.acquire();
    GpuBatchContext ctx;
    ctx.executor = &ex;
    ctx.lane = &lease.lane();
    ctx.output_on_device = on_device;
    return t.apply_batch_gpu(b, ctx);
}

NDArray row(const NDArray& batch, std::size_t i) {
    Shape s(batch.shape().begin() + 1, batch.shape().end());
    NDArray r(s, batch.dtype());
    std::memcpy(r.data(), static_cast<const std::uint8_t*>(batch.data()) + i * r.nbytes(), r.nbytes());
    return r;
}

TransformConstPtr tabular_chain(std::size_t cols) {
    std::vector<double> mean(cols, 0.5), scale(cols, 2.0), lo(cols, -3.0), hi(cols, 3.0);
    return std::make_shared<Compose>(std::vector<TransformConstPtr>{
        std::make_shared<ClipTransform>(-5.0, 5.0),
        std::make_shared<StandardizeTransform>(mean, scale),
        std::make_shared<MinMaxTransform>(lo, hi, -1.0, 1.0)});
}

void image_bench(const char* title, const AugmentSpec& spec, std::size_t n, std::size_t h,
                 std::size_t w) {
    std::printf("\n== %s: %zu x %zux%zux3 uint8 ==\n", title, n, h, w);
    const NDArray host = random_u8(Shape{n, h, w, 3}, 1);
    const NDArray dev = host.cuda(0);
    const Batch hb = as_batch(host);
    const Batch db = as_batch(dev);

    // v1: per-sample unfused chain.
    std::vector<TransformConstPtr> v1;
    if (spec.crop == AugmentSpec::Crop::Random)
        v1.push_back(std::make_shared<RandomCrop>(int(spec.crop_height), int(spec.crop_width)));
    if (spec.out_height)
        v1.push_back(std::make_shared<Resize>(int(spec.out_height), int(spec.out_width)));
    if (spec.hflip_p > 0) v1.push_back(std::make_shared<RandomHorizontalFlip>(spec.hflip_p));
    v1.push_back(std::make_shared<ToTensor>());
    v1.push_back(std::make_shared<NormalizeImage>(spec.mean, spec.stddev));
    Compose v1_chain(v1, /*fuse=*/false);
    const double t_v1 = median_ms([&] {
        for (std::size_t i = 0; i < n; ++i) {
            Sample s;
            s.input = row(host, i);
            (void)v1_chain.apply(s);
        }
    }, 3, 1);

    FusedAugment aug(spec, 3);
    const double t_cpu = median_ms([&] { (void)aug.apply_batch(hb); });
    const double t_gpu_hh = median_ms([&] { (void)run_gpu(aug, hb, false); });
    const double t_gpu_hd = median_ms([&] { (void)run_gpu(aug, hb, true); });
    const double t_gpu_dd = median_ms([&] { (void)run_gpu(aug, db, true); });
    auto line = [&](const char* name, double ms) {
        std::printf("  %-38s %9.2f ms  %9.0f img/s  x%.1f vs v1\n", name, ms,
                    static_cast<double>(n) / (ms / 1000.0), t_v1 / ms);
    };
    line("v1 per-sample chain (1 thread)", t_v1);
    line("FusedAugment CPU (thread pool)", t_cpu);
    line("GPU host -> pinned host", t_gpu_hh);
    line("GPU host -> device", t_gpu_hd);
    line("GPU device -> device (kernel only)", t_gpu_dd);
}

void tabular_bench(std::size_t rows, std::size_t cols) {
    std::printf("\n== tabular chain clip -> standardize -> minmax: %zu x %zu float32 (%.1f MiB) ==\n",
                rows, cols, rows * cols * 4.0 / (1 << 20));
    const TransformConstPtr chain = tabular_chain(cols);
    const Batch hb = as_batch(random_f32(rows, cols, 2));
    const Batch db = as_batch(hb.inputs.cuda(0));
    const double t_cpu = median_ms([&] { (void)chain->apply_batch(hb); });
    const double t_hh = median_ms([&] { (void)run_gpu(*chain, hb, false); });
    const double t_dd = median_ms([&] { (void)run_gpu(*chain, db, true); });
    const double gib = rows * cols * 4.0 / (1024.0 * 1024.0 * 1024.0);
    std::printf("  %-38s %9.2f ms  %6.2f GiB/s\n", "FusedTransform CPU (thread pool)", t_cpu, gib / (t_cpu / 1e3));
    std::printf("  %-38s %9.2f ms  %6.2f GiB/s\n", "GPU host -> pinned host", t_hh, gib / (t_hh / 1e3));
    std::printf("  %-38s %9.2f ms  %6.2f GiB/s\n", "GPU device -> device", t_dd, gib / (t_dd / 1e3));
}

/// Train a CostModel online with real timings and compare its steady-state
/// choice with the measured oracle; regret = time(chosen) / time(best).
void cost_model_bench() {
    std::printf("\n== CostModel decisions vs measured oracle ==\n");
    struct Case {
        std::string name;
        GpuOpKind op;
        TransformConstPtr t;
        Batch b;
    };
    std::vector<Case> cases;
    for (std::size_t rows : {16u, 1024u, 16384u, 262144u, 2097152u}) {
        cases.push_back({"tabular " + std::to_string(rows) + "x16", GpuOpKind::TabularSmall,
                         tabular_chain(16), as_batch(random_f32(rows, 16, rows))});
    }
    AugmentSpec spec;
    spec.crop = AugmentSpec::Crop::Random;
    spec.crop_height = 224;
    spec.crop_width = 224;
    spec.hflip_p = 0.5f;
    spec.mean = {0.485f, 0.456f, 0.406f};
    spec.stddev = {0.229f, 0.224f, 0.225f};
    for (std::size_t n : {1u, 4u, 16u, 64u}) {
        cases.push_back({"image " + std::to_string(n) + "x256x256", GpuOpKind::ImageAugment,
                         std::make_shared<FusedAugment>(spec), as_batch(random_u8(Shape{n, 256, 256, 3}, n))});
    }
    double worst = 1.0;
    std::size_t correct = 0;
    for (const Case& c : cases) {
        CostModelOptions o;
        o.warmup_calls = 16;
        o.auto_telemetry = false;
        CostModel model(o);
        GpuExecutor& ex = GpuExecutor::instance(0);
        auto step = [&](bool gpu) {
            if (gpu) {
                auto lease = ex.acquire();
                GpuBatchContext ctx;
                ctx.executor = &ex;
                ctx.lane = &lease.lane();
                ScopedCostTimer t(model, c.op, ExecPath::Gpu, c.b.inputs.nbytes());
                (void)c.t->apply_batch_gpu(c.b, ctx);
            } else {
                ScopedCostTimer t(model, c.op, ExecPath::Cpu, c.b.inputs.nbytes());
                (void)c.t->apply_batch(c.b);
            }
        };
        for (int i = 0; i < 24; ++i) step(model.decide(c.op, c.b.inputs.nbytes(), 0).use_gpu);
        const CostDecision d = model.decide(c.op, c.b.inputs.nbytes(), 0);
        // Interleave the oracle runs so clock / thermal drift hits both paths equally.
        std::vector<double> cpu_t, gpu_t;
        for (int i = 0; i < 2; ++i) {
            step(false);
            step(true);
        }
        for (int i = 0; i < 15; ++i) {
            for (const bool gpu : {false, true}) {
                const auto t0 = clk::now();
                step(gpu);
                (gpu ? gpu_t : cpu_t)
                    .push_back(std::chrono::duration<double, std::milli>(clk::now() - t0).count());
            }
        }
        std::sort(cpu_t.begin(), cpu_t.end());
        std::sort(gpu_t.begin(), gpu_t.end());
        const double cpu_ms = cpu_t[cpu_t.size() / 2];
        const double gpu_ms = gpu_t[gpu_t.size() / 2];
        const bool oracle_gpu = gpu_ms < cpu_ms;
        const double chosen = d.use_gpu ? gpu_ms : cpu_ms;
        const double regret = chosen / std::min(cpu_ms, gpu_ms);
        worst = std::max(worst, regret);
        correct += (d.use_gpu == oracle_gpu) ? 1 : 0;
        std::printf("  %-22s cpu %8.3f ms  gpu %8.3f ms  predicted %8.3f / %8.3f  model:%s oracle:%s  regret %.3f\n",
                    c.name.c_str(), cpu_ms, gpu_ms, d.predicted_cpu_us / 1e3,
                    d.predicted_gpu_us / 1e3, d.use_gpu ? "GPU" : "CPU",
                    oracle_gpu ? "GPU" : "CPU", regret);
    }
    std::printf("  decisions matching oracle: %zu / %zu, worst regret %.3f (target <= 1.10)\n",
                correct, cases.size(), worst);
}

class ImageDataset : public Dataset {
public:
    explicit ImageDataset(NDArray images) : images_(std::move(images)) {}
    [[nodiscard]] std::size_t size() const override { return images_.shape()[0]; }
    [[nodiscard]] Sample get(std::size_t i) const override {
        Sample s;
        s.input = row(images_, i);
        s.label = NDArray(Shape{1}, DType::Int64);
        return s;
    }

private:
    NDArray images_;
};

void dataloader_bench() {
    std::printf("\n== DataLoader: 512 x 256x256x3, batch 64, 4 workers, FusedAugment(crop 224, flip, normalize) ==\n");
    auto ds = std::make_shared<ImageDataset>(random_u8(Shape{512, 256, 256, 3}, 5));
    AugmentSpec spec;
    spec.crop = AugmentSpec::Crop::Random;
    spec.crop_height = 224;
    spec.crop_width = 224;
    spec.hflip_p = 0.5f;
    spec.mean = {0.485f, 0.456f, 0.406f};
    spec.stddev = {0.229f, 0.224f, 0.225f};
    struct Mode {
        const char* name;
        Device transform;
        Device out;
    };
    for (const Mode m : {Mode{"transform CPU, host output", Device::host(), Device::host()},
                         Mode{"transform CUDA, host output", Device::cuda(0), Device::host()},
                         Mode{"transform CUDA, device output", Device::cuda(0), Device::cuda(0)},
                         Mode{"transform Auto (CostModel), device out", Device::auto_select(), Device::auto_select()}}) {
        DataLoaderOptions opt;
        opt.batch_size = 64;
        opt.num_workers = 4;
        opt.transform_workers = 2;
        opt.transform_device = m.transform;
        opt.device = m.out;
        opt.batch_transform = std::make_shared<FusedAugment>(spec);
        opt.cost_model = std::make_shared<CostModel>();
        DataLoader dl(DatasetConstPtr(ds), opt, collate_stack_nd);
        // Warm until the CostModel has left its A/B profiling phase.
        const std::size_t warm_batches = opt.cost_model->options().warmup_calls + 8;
        for (std::size_t seen = 0; seen < warm_batches;) {
            for (const Batch& b : dl) {
                (void)b;
                ++seen;
            }
        }
        const CostOpStats before = opt.cost_model->stats(GpuOpKind::ImageAugment);
        const auto t0 = clk::now();
        std::size_t imgs = 0;
        for (int epoch = 0; epoch < 4; ++epoch) {
            for (const Batch& b : dl) imgs += b.inputs.shape()[0];
        }
        const double s = std::chrono::duration<double>(clk::now() - t0).count();
        const CostOpStats st = opt.cost_model->stats(GpuOpKind::ImageAugment);
        std::printf("  %-40s %8.0f img/s   (timed batches cpu=%zu gpu=%zu)\n", m.name, imgs / s,
                    static_cast<std::size_t>(st.cpu_samples - before.cpu_samples),
                    static_cast<std::size_t>(st.gpu_samples - before.gpu_samples));
    }
}

void append_bytes(void* ctx, void* data, int size) {
    auto* v = static_cast<std::vector<std::uint8_t>*>(ctx);
    v->insert(v->end(), static_cast<std::uint8_t*>(data), static_cast<std::uint8_t*>(data) + size);
}

void nvjpeg_bench() {
    if (!NvJpegDecoder::available()) {
        std::printf("\n== nvJPEG: not built (NEXUSDATA_WITH_NVJPEG=OFF) ==\n");
        return;
    }
    std::printf("\n== JPEG decode: 32 x 1024x768 q90 ==\n");
    std::vector<std::vector<std::uint8_t>> files;
    for (int k = 0; k < 32; ++k) {
        std::vector<std::uint8_t> px(1024 * 768 * 3);
        for (int y = 0; y < 768; ++y)
            for (int x = 0; x < 1024; ++x)
                for (int ch = 0; ch < 3; ++ch)
                    px[(y * 1024 + x) * 3 + ch] = static_cast<std::uint8_t>(
                        127 + 120 * std::sin(0.01 * (x * (ch + 1) + y + k * 13)));
        std::vector<std::uint8_t> jpg;
        stbi_write_jpg_to_func(append_bytes, &jpg, 1024, 768, 3, px.data(), 90);
        files.push_back(std::move(jpg));
    }
    const double t_stb = median_ms([&] {
        for (const auto& f : files) (void)decode_image(f.data(), f.size());
    }, 3, 1);
    NvJpegDecoder dec(0);
    const double t_gpu_dev = median_ms([&] { (void)dec.decode_batch(files, Device::cuda(0)); }, 3, 1);
    const double t_gpu_host = median_ms([&] { (void)dec.decode_batch(files, Device::host()); }, 3, 1);
    std::printf("  %-38s %8.2f ms  %7.0f img/s\n", "stb_image (1 thread)", t_stb, 32 / (t_stb / 1e3));
    std::printf("  %-38s %8.2f ms  %7.0f img/s\n", "nvJPEG -> device", t_gpu_dev, 32 / (t_gpu_dev / 1e3));
    std::printf("  %-38s %8.2f ms  %7.0f img/s\n", "nvJPEG -> host", t_gpu_host, 32 / (t_gpu_host / 1e3));
}

} // namespace

int main() {
    if (!GpuExecutor::usable(0)) {
        std::printf("bench_gpu: no usable CUDA device (build with -DNEXUSDATA_WITH_CUDA=ON)\n");
        return 0;
    }
    const auto props = cuda_api::device_properties(0);
    std::printf("device: %s (sm_%d%d, %d SMs, %.1f GiB)  NVML: %s\n", props.name.c_str(),
                props.compute_major, props.compute_minor, props.multiprocessors,
                props.total_memory / (1024.0 * 1024.0 * 1024.0), nvml_available() ? "yes" : "no");
    const CalibrationReport rep = calibrate_auto_device_policy();
    std::printf("calibration: H2D %.2f GiB/s  D2H %.2f GiB/s  launch+sync %.1f us  host memcpy %.2f GiB/s\n",
                rep.h2d_gib_s, rep.d2h_gib_s, rep.launch_overhead_us, rep.host_memcpy_gib_s);
    if (auto t = query_gpu_telemetry(0)) {
        std::printf("telemetry: utilization %.0f%%  free VRAM %.2f GiB\n", t->utilization * 100.0,
                    t->free_vram_bytes ? *t->free_vram_bytes / (1024.0 * 1024.0 * 1024.0) : -1.0);
    }

    AugmentSpec crop;
    crop.crop = AugmentSpec::Crop::Random;
    crop.crop_height = 224;
    crop.crop_width = 224;
    crop.hflip_p = 0.5f;
    crop.mean = {0.485f, 0.456f, 0.406f};
    crop.stddev = {0.229f, 0.224f, 0.225f};
    image_bench("RandomCrop 224 + HFlip + ToTensor + Normalize", crop, 64, 256, 256);

    AugmentSpec rrc = crop;
    rrc.crop_height = 200;
    rrc.crop_width = 200;
    rrc.out_height = 224;
    rrc.out_width = 224;
    image_bench("Crop 200 + bilinear Resize 224 + HFlip + Normalize", rrc, 64, 256, 256);

    tabular_bench(1u << 20, 16);
    cost_model_bench();
    dataloader_bench();
    nvjpeg_bench();
    return 0;
}
