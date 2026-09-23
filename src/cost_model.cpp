#include "nexusdata/backend/cost_model.hpp"

#include <algorithm>
#include <cmath>

#include "nexusdata/backend/gpu_telemetry.hpp"
#include "nexusdata/core/error.hpp"

namespace nexusdata {

namespace {

constexpr double kMiB = 1024.0 * 1024.0;
constexpr double kColdFactor = 1.5;
constexpr double kSpikeFactor = 3.0;

double to_mib(std::size_t nbytes) { return static_cast<double>(nbytes) / kMiB; }

std::size_t idx(GpuOpKind op) { return static_cast<std::size_t>(op); }

} // namespace

void CostModel::Fit::add(double x, double y, double alpha) noexcept {
    if (n == 1 && sx > 0.0 && x > 0.0 && sy / sx > kColdFactor * (y / x)) {
        // The first run of a path pays one-time costs (allocations, page faults, module
        // load); once a warm run shows it was an outlier, restart the fit from the warm run.
        w = 1.0;
        sx = x;
        sy = y;
        sxx = x * x;
        sxy = x * y;
        n = 2;
        return;
    }
    if (n >= 2) {
        const double p = predict(x, 0.0);
        if (p > 0.0) {
            // Winsorize preemption / page-fault spikes so one bad batch cannot flip decisions.
            y = std::min(y, kSpikeFactor * p);
        }
    }
    const double d = n == 0 ? 0.0 : 1.0 - alpha;
    w = d * w + 1.0;
    sx = d * sx + x;
    sy = d * sy + y;
    sxx = d * sxx + x * x;
    sxy = d * sxy + x * y;
    ++n;
}

double CostModel::Fit::predict(double x, double intercept_floor) const noexcept {
    if (n == 0 || w <= 0.0) {
        return -1.0;
    }
    const double mx = sx / w;
    const double my = sy / w;
    const double var = sxx / w - mx * mx;
    // A single observed size cannot separate fixed from per-byte cost: scale the
    // per-byte part proportionally and keep the known fixed floor (GPU launch).
    if (n < 3 || var <= 1e-9 * (mx * mx) + 1e-18) {
        if (mx <= 0.0) {
            return my;
        }
        const double fixed = std::min(intercept_floor, my);
        return fixed + (my - fixed) * (x / mx);
    }
    double b = (sxy / w - mx * my) / var;
    double a = my - b * mx;
    if (b < 0.0) {
        return my;
    }
    if (a < 0.0) {
        a = 0.0;
        b = mx > 0.0 ? my / mx : b;
    }
    return a + b * x;
}

CostModel::CostModel(CostModelOptions options) : opt_(options) {
    if (opt_.ewma_alpha <= 0.0 || opt_.ewma_alpha > 1.0) {
        throw InvalidArgumentError("CostModel: ewma_alpha must be in (0, 1]");
    }
    if (opt_.gpu_advantage <= 0.0) {
        throw InvalidArgumentError("CostModel: gpu_advantage must be > 0");
    }
    gpu_detected_ = cuda_runtime_available() && cuda_device_count() > 0;
    if (gpu_detected_ && opt_.auto_telemetry) {
        telemetry_ = make_cuda_telemetry_provider(0);
    }
}

void CostModel::set_gpu_available(std::optional<bool> available) {
    std::lock_guard lock(mu_);
    gpu_override_ = available;
}

bool CostModel::gpu_available() const {
    std::lock_guard lock(mu_);
    return gpu_override_.value_or(gpu_detected_);
}

void CostModel::set_telemetry_provider(GpuTelemetryProvider provider) {
    std::lock_guard lock(mu_);
    telemetry_ = std::move(provider);
}

void CostModel::apply_calibration(const CalibrationReport& report) {
    std::lock_guard lock(mu_);
    if (report.h2d_gib_s > 0.0) opt_.h2d_gib_s = report.h2d_gib_s;
    if (report.d2h_gib_s > 0.0) opt_.d2h_gib_s = report.d2h_gib_s;
    if (report.launch_overhead_us > 0.0) opt_.launch_overhead_us = report.launch_overhead_us;
    opt_.fallback_policy = report.policy;
}

void CostModel::record(GpuOpKind op, ExecPath path, std::size_t nbytes, double micros) {
    if (!(micros >= 0.0) || !std::isfinite(micros)) {
        return;
    }
    std::lock_guard lock(mu_);
    ops_[idx(op)].fit[static_cast<std::size_t>(path)].add(to_mib(nbytes), micros,
                                                          opt_.ewma_alpha);
}

double CostModel::predict_locked(const OpState& s, ExecPath path, double mib) const {
    const double floor = path == ExecPath::Gpu ? opt_.launch_overhead_us : 0.0;
    return s.fit[static_cast<std::size_t>(path)].predict(mib, floor);
}

double CostModel::predict_us(GpuOpKind op, ExecPath path, std::size_t nbytes) const {
    std::lock_guard lock(mu_);
    return predict_locked(ops_[idx(op)], path, to_mib(nbytes));
}

bool CostModel::heuristic_gpu(GpuOpKind op, std::size_t nbytes) const {
    const AutoDevicePolicy& p = opt_.fallback_policy;
    if (!p.allow_cuda) {
        return false;
    }
    switch (op) {
        case GpuOpKind::ImageDecode:
        case GpuOpKind::ImageAugment:
        case GpuOpKind::NormalizeCollate:
            return nbytes >= p.image_nbytes_threshold;
        case GpuOpKind::GatherLarge:
            return nbytes >= p.tabular_nbytes_threshold;
        default:
            // TabularSmall, text/audio/json/compression and custom kinds: no static
            // evidence that the GPU helps; wait for measurements.
            return false;
    }
}

CostDecision CostModel::decide(GpuOpKind op, std::size_t nbytes, std::size_t gpu_queue_depth) {
    CostDecision d;
    GpuTelemetryProvider telemetry;
    {
        std::lock_guard lock(mu_);
        if (!gpu_override_.value_or(gpu_detected_)) {
            OpState& s = ops_[idx(op)];
            ++s.decisions;
            d.reason = "no_gpu";
            d.predicted_cpu_us = predict_locked(s, ExecPath::Cpu, to_mib(nbytes));
            return d;
        }
        telemetry = telemetry_;
    }

    // Query telemetry outside the lock: providers may block on driver calls.
    std::optional<GpuTelemetry> tele;
    if (telemetry) {
        tele = telemetry();
    }

    std::lock_guard lock(mu_);
    OpState& s = ops_[idx(op)];
    ++s.decisions;
    const double mib = to_mib(nbytes);
    d.predicted_cpu_us = predict_locked(s, ExecPath::Cpu, mib);
    const double gpu_raw = predict_locked(s, ExecPath::Gpu, mib);
    d.predicted_gpu_us =
        gpu_raw < 0.0 ? -1.0 : gpu_raw * (1.0 + static_cast<double>(gpu_queue_depth));

    auto finish = [&](bool use_gpu, bool exploring, const char* reason) {
        d.use_gpu = use_gpu;
        d.exploring = exploring;
        d.reason = reason;
        if (use_gpu) {
            ++s.gpu_decisions;
        }
        return d;
    };

    if (tele) {
        if (tele->utilization >= opt_.gpu_busy_utilization) {
            return finish(false, false, "gpu_busy");
        }
        if (tele->free_vram_bytes &&
            *tele->free_vram_bytes < std::max(opt_.min_free_vram_bytes, 2 * nbytes)) {
            return finish(false, false, "low_vram");
        }
    }

    if (s.decisions <= opt_.warmup_calls) {
        return finish(s.decisions % 2 == 0, true, "warmup");
    }
    if (opt_.recalibrate_every > 0) {
        const std::uint64_t since = s.decisions - opt_.warmup_calls;
        if (since % opt_.recalibrate_every < opt_.recalibrate_probes &&
            since >= opt_.recalibrate_every) {
            return finish(!s.prefers_gpu, true, "recalibrate");
        }
    }

    if (d.predicted_cpu_us < 0.0 || d.predicted_gpu_us < 0.0) {
        const bool g = heuristic_gpu(op, nbytes);
        s.prefers_gpu = g;
        return finish(g, false, "heuristic");
    }
    const bool g = d.predicted_gpu_us * opt_.gpu_advantage < d.predicted_cpu_us;
    s.prefers_gpu = g;
    return finish(g, false, "model");
}

CostOpStats CostModel::stats(GpuOpKind op) const {
    std::lock_guard lock(mu_);
    const OpState& s = ops_[idx(op)];
    CostOpStats out;
    out.decisions = s.decisions;
    out.gpu_decisions = s.gpu_decisions;
    const Fit& c = s.fit[0];
    const Fit& g = s.fit[1];
    out.cpu_samples = c.n;
    out.gpu_samples = g.n;
    out.cpu_us_per_mib = (c.n > 0 && c.sx > 0.0) ? c.sy / c.sx : -1.0;
    out.gpu_us_per_mib = (g.n > 0 && g.sx > 0.0) ? g.sy / g.sx : -1.0;
    return out;
}

void CostModel::reset() {
    std::lock_guard lock(mu_);
    ops_ = {};
}

CostModel& CostModel::global() {
    static CostModel model;
    return model;
}

Device resolve_device(Device requested, GpuOpKind op, std::size_t nbytes, CostModel& model,
                      std::size_t gpu_queue_depth) {
    if (!requested.is_auto()) {
        return resolve_device(requested, op, nbytes, AutoDevicePolicy{});
    }
    const bool real_gpu = cuda_runtime_available() && cuda_device_count() > 0;
    const CostDecision d = model.decide(op, nbytes, gpu_queue_depth);
    if (d.use_gpu && real_gpu) {
        return Device::cuda(0);
    }
    return Device::host();
}

} // namespace nexusdata
