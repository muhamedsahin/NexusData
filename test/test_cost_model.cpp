#include <doctest/doctest.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <thread>

#include "nexusdata/backend/cost_model.hpp"
#include "nexusdata/core/error.hpp"

using namespace nexusdata;

namespace {

constexpr std::size_t MiB = std::size_t{1} << 20;

double cpu_cost(std::size_t nbytes) { return 100.0 * static_cast<double>(nbytes) / MiB; }
double gpu_cost(std::size_t nbytes) { return 40.0 + 10.0 * static_cast<double>(nbytes) / MiB; }

/// Drive @p model through warmup, feeding back synthetic timings for whichever path it picks.
void train(CostModel& m, GpuOpKind op, std::size_t decisions) {
    const std::size_t sizes[] = {1 * MiB, 2 * MiB, 4 * MiB, 8 * MiB};
    for (std::size_t i = 0; i < decisions; ++i) {
        const std::size_t n = sizes[i % 4];
        const CostDecision d = m.decide(op, n);
        m.record(op, d.use_gpu ? ExecPath::Gpu : ExecPath::Cpu, n,
                 d.use_gpu ? gpu_cost(n) : cpu_cost(n));
    }
}

CostModelOptions small_opts() {
    CostModelOptions o;
    o.warmup_calls = 16;
    o.recalibrate_every = 100;
    o.recalibrate_probes = 2;
    o.auto_telemetry = false; // deterministic gates on machines with a real GPU
    return o;
}

} // namespace

TEST_CASE("cost model: no GPU means CPU, always") {
    CostModel m;
    m.set_gpu_available(false);
    for (int i = 0; i < 100; ++i) {
        const CostDecision d = m.decide(GpuOpKind::ImageAugment, 512 * MiB);
        REQUIRE_FALSE(d.use_gpu);
        CHECK(std::string(d.reason) == "no_gpu");
    }
    CHECK(m.stats(GpuOpKind::ImageAugment).decisions == 100);
    CHECK(m.stats(GpuOpKind::ImageAugment).gpu_decisions == 0);

    if (!cuda_runtime_available()) {
        CostModel detected;
        CHECK_FALSE(detected.gpu_available());
        CHECK_FALSE(detected.should_use_gpu(GpuOpKind::NormalizeCollate, 1u << 30));
    }
}

TEST_CASE("cost model: warmup alternates both paths, then the model picks the cheaper one") {
    CostModel m(small_opts());
    m.set_gpu_available(true);
    int gpu = 0;
    for (int i = 0; i < 16; ++i) {
        const CostDecision d = m.decide(GpuOpKind::ImageAugment, 4 * MiB);
        CHECK(d.exploring);
        CHECK(std::string(d.reason) == "warmup");
        gpu += d.use_gpu ? 1 : 0;
        m.record(GpuOpKind::ImageAugment, d.use_gpu ? ExecPath::Gpu : ExecPath::Cpu,
                 (1u + static_cast<unsigned>(i % 4)) * MiB,
                 d.use_gpu ? gpu_cost((1u + i % 4) * MiB) : cpu_cost((1u + i % 4) * MiB));
    }
    CHECK(gpu == 8);

    const CostDecision big = m.decide(GpuOpKind::ImageAugment, 8 * MiB);
    CHECK(std::string(big.reason) == "model");
    CHECK_FALSE(big.exploring);
    CHECK(big.use_gpu);
    CHECK(big.predicted_cpu_us == doctest::Approx(cpu_cost(8 * MiB)).epsilon(0.05));
    CHECK(big.predicted_gpu_us == doctest::Approx(gpu_cost(8 * MiB)).epsilon(0.05));

    // Tiny batches: launch overhead dominates -> CPU.
    const CostDecision tiny = m.decide(GpuOpKind::ImageAugment, 4096);
    CHECK_FALSE(tiny.use_gpu);
    CHECK(tiny.predicted_gpu_us >= 30.0);

    // Break-even of 100x = 40 + 10x is x ~ 0.44 MiB (x1.1 hysteresis ~0.49 MiB).
    CHECK_FALSE(m.decide(GpuOpKind::ImageAugment, MiB / 4).use_gpu);
    CHECK(m.decide(GpuOpKind::ImageAugment, MiB).use_gpu);
}

TEST_CASE("cost model: a busy GPU queue flips the decision to CPU") {
    CostModel m(small_opts());
    m.set_gpu_available(true);
    train(m, GpuOpKind::NormalizeCollate, 16);
    CHECK(m.decide(GpuOpKind::NormalizeCollate, 8 * MiB, 0).use_gpu);
    // gpu 120us * (1 + 10) = 1320us > cpu 800us
    const CostDecision d = m.decide(GpuOpKind::NormalizeCollate, 8 * MiB, 10);
    CHECK_FALSE(d.use_gpu);
    CHECK(d.predicted_gpu_us == doctest::Approx(11 * gpu_cost(8 * MiB)).epsilon(0.05));
}

TEST_CASE("cost model: telemetry gates (utilization / free VRAM)") {
    CostModel m(small_opts());
    m.set_gpu_available(true);
    train(m, GpuOpKind::ImageAugment, 16);

    GpuTelemetry t;
    m.set_telemetry_provider([&t]() -> std::optional<GpuTelemetry> { return t; });
    t.utilization = 0.97;
    CostDecision d = m.decide(GpuOpKind::ImageAugment, 8 * MiB);
    CHECK_FALSE(d.use_gpu);
    CHECK(std::string(d.reason) == "gpu_busy");

    t.utilization = 0.2;
    t.free_vram_bytes = 100 * MiB;
    d = m.decide(GpuOpKind::ImageAugment, 8 * MiB);
    CHECK_FALSE(d.use_gpu);
    CHECK(std::string(d.reason) == "low_vram");

    t.free_vram_bytes = 8192 * MiB;
    CHECK(m.decide(GpuOpKind::ImageAugment, 8 * MiB).use_gpu);

    m.set_telemetry_provider([]() -> std::optional<GpuTelemetry> { return std::nullopt; });
    CHECK(m.decide(GpuOpKind::ImageAugment, 8 * MiB).use_gpu);
}

TEST_CASE("cost model: periodic recalibration probes the other path") {
    CostModel m(small_opts());
    m.set_gpu_available(true);
    train(m, GpuOpKind::ImageAugment, 16);
    std::size_t probes = 0;
    std::size_t first_probe = 0;
    for (std::size_t k = 17; k <= 16 + 250; ++k) {
        const CostDecision d = m.decide(GpuOpKind::ImageAugment, 8 * MiB);
        if (std::string(d.reason) == "recalibrate") {
            CHECK_FALSE(d.use_gpu); // model prefers GPU, probe goes to CPU
            if (probes == 0) first_probe = k;
            ++probes;
        }
    }
    CHECK(first_probe == 16 + 100);
    CHECK(probes == 4); // 2 probes at +100, 2 at +200
}

TEST_CASE("cost model: EWMA follows workload drift") {
    CostModel m(small_opts());
    m.set_gpu_available(true);
    train(m, GpuOpKind::ImageAugment, 16);
    CHECK(m.decide(GpuOpKind::ImageAugment, 8 * MiB).use_gpu);
    // The CPU path became 20x faster (e.g. images got smaller, SIMD kicked in).
    for (int i = 0; i < 200; ++i) {
        const std::size_t n = (1u + static_cast<unsigned>(i % 4)) * MiB;
        m.record(GpuOpKind::ImageAugment, ExecPath::Cpu, n, 5.0 * static_cast<double>(n) / MiB);
    }
    const CostDecision d = m.decide(GpuOpKind::ImageAugment, 8 * MiB);
    CHECK_FALSE(d.use_gpu);
    CHECK(d.predicted_cpu_us == doctest::Approx(40.0).epsilon(0.1));
}

TEST_CASE("cost model: heuristic fallback before both paths are measured") {
    CostModelOptions o;
    o.warmup_calls = 0;
    o.auto_telemetry = false;
    CostModel m(o);
    m.set_gpu_available(true);
    CostDecision d = m.decide(GpuOpKind::ImageAugment, 2 * MiB);
    CHECK(std::string(d.reason) == "heuristic");
    CHECK(d.use_gpu); // >= 1 MiB image threshold
    CHECK_FALSE(m.decide(GpuOpKind::ImageAugment, 1024).use_gpu);
    CHECK_FALSE(m.decide(GpuOpKind::TabularSmall, 64 * MiB).use_gpu);
    CHECK_FALSE(m.decide(GpuOpKind::JsonParse, 64 * MiB).use_gpu);
    CHECK(m.decide(GpuOpKind::GatherLarge, 8 * MiB).use_gpu);

    CalibrationReport r;
    r.policy.allow_cuda = false;
    r.launch_overhead_us = 12.5;
    m.apply_calibration(r);
    CHECK(m.options().launch_overhead_us == 12.5);
    CHECK_FALSE(m.decide(GpuOpKind::ImageAugment, 2 * MiB).use_gpu);
}

TEST_CASE("cost model: prediction from a single observed size keeps the launch floor") {
    CostModel m;
    m.record(GpuOpKind::ImageDecode, ExecPath::Gpu, 4 * MiB, 80.0);
    m.record(GpuOpKind::ImageDecode, ExecPath::Gpu, 4 * MiB, 80.0);
    // fixed = min(launch 40, 80) = 40; per-byte part scales: 40 + 40 * 2
    CHECK(m.predict_us(GpuOpKind::ImageDecode, ExecPath::Gpu, 8 * MiB) == doctest::Approx(120.0));
    m.record(GpuOpKind::ImageDecode, ExecPath::Cpu, 4 * MiB, 400.0);
    CHECK(m.predict_us(GpuOpKind::ImageDecode, ExecPath::Cpu, 2 * MiB) == doctest::Approx(200.0));
    CHECK(m.predict_us(GpuOpKind::ImageAugment, ExecPath::Cpu, MiB) < 0.0);

    // Invalid samples are ignored.
    m.record(GpuOpKind::ImageDecode, ExecPath::Cpu, MiB, -1.0);
    m.record(GpuOpKind::ImageDecode, ExecPath::Cpu, MiB, std::numeric_limits<double>::quiet_NaN());
    CHECK(m.stats(GpuOpKind::ImageDecode).cpu_samples == 1);

    m.reset();
    CHECK(m.stats(GpuOpKind::ImageDecode).gpu_samples == 0);

    CostModelOptions bad;
    bad.ewma_alpha = 0.0;
    CHECK_THROWS_AS(CostModel{bad}, InvalidArgumentError);
}

TEST_CASE("cost model: a cold first run and single spikes do not skew predictions") {
    CostModel m;
    const GpuOpKind op = GpuOpKind::ImageAugment;
    // First GPU batch pays allocations / module load: 10x the steady state.
    m.record(op, ExecPath::Gpu, 4 * MiB, 1000.0);
    m.record(op, ExecPath::Gpu, 4 * MiB, 100.0);
    CHECK(m.predict_us(op, ExecPath::Gpu, 4 * MiB) == doctest::Approx(100.0));
    CHECK(m.stats(op).gpu_samples == 2);

    for (int i = 0; i < 4; ++i) m.record(op, ExecPath::Gpu, 4 * MiB, 100.0);
    m.record(op, ExecPath::Gpu, 4 * MiB, 1e6); // preempted batch, clipped to 3x the prediction
    CHECK(m.predict_us(op, ExecPath::Gpu, 4 * MiB) < 150.0); // unclipped: ~2e5
    CHECK(m.stats(op).gpu_samples == 7);

    // A slower second run is real data, not a cold start.
    m.record(op, ExecPath::Cpu, 4 * MiB, 100.0);
    m.record(op, ExecPath::Cpu, 4 * MiB, 120.0);
    CHECK(m.predict_us(op, ExecPath::Cpu, 4 * MiB) > 100.0);
}

TEST_CASE("cost model: ScopedCostTimer records unless cancelled") {
    CostModel m;
    {
        ScopedCostTimer t(m, GpuOpKind::Compression, ExecPath::Cpu, MiB);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    {
        ScopedCostTimer t(m, GpuOpKind::Compression, ExecPath::Cpu, MiB);
        t.cancel();
    }
    CHECK(m.stats(GpuOpKind::Compression).cpu_samples == 1);
    CHECK(m.predict_us(GpuOpKind::Compression, ExecPath::Cpu, MiB) >= 900.0);
}

TEST_CASE("custom op kinds are registered once and tracked separately") {
    const GpuOpKind a = register_custom_op_kind("test_resample_v2");
    const GpuOpKind b = register_custom_op_kind("test_resample_v2");
    const GpuOpKind c = register_custom_op_kind("test_other_op");
    CHECK(a == b);
    CHECK(a != c);
    CHECK(is_custom_op_kind(a));
    CHECK_FALSE(is_custom_op_kind(GpuOpKind::TextTokenize));
    CHECK(op_kind_name(a) == "test_resample_v2");
    CHECK(op_kind_name(GpuOpKind::ImageAugment) == "image_augment");
    CHECK_THROWS_AS((void)register_custom_op_kind(""), InvalidArgumentError);

    CostModel m;
    m.record(a, ExecPath::Cpu, MiB, 10.0);
    CHECK(m.stats(a).cpu_samples == 1);
    CHECK(m.stats(c).cpu_samples == 0);
}

TEST_CASE("resolve_device with a cost model never returns CUDA without a real GPU") {
    CostModel m(small_opts());
    m.set_gpu_available(true); // simulated
    train(m, GpuOpKind::NormalizeCollate, 16);
    const Device d = resolve_device(Device::auto_select(), GpuOpKind::NormalizeCollate, 64 * MiB, m);
    if (!(cuda_runtime_available() && cuda_device_count() > 0)) {
        CHECK(d.is_host());
    }
    CHECK(resolve_device(Device::host(), GpuOpKind::NormalizeCollate, 64 * MiB, m).is_host());
}

TEST_CASE("calibration report includes transfer fields") {
    const CalibrationReport r = calibrate_auto_device_policy();
    CHECK(r.host_memcpy_gib_s > 0.0);
    if (!r.cuda_available) {
        CHECK(r.h2d_gib_s == 0.0);
        CHECK(r.d2h_gib_s == 0.0);
        CHECK(r.launch_overhead_us == 0.0);
    } else {
        CHECK(r.h2d_gib_s > 0.0);
        CHECK(r.launch_overhead_us > 0.0);
    }
}
