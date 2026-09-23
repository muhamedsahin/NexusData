#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <random>
#include <set>
#include <vector>

#include "nexusdata/core/array_utils.hpp"
#include "nexusdata/dataset/in_memory.hpp"
#include "nexusdata/dataset/map.hpp"
#include "nexusdata/loading/dataloader.hpp"
#include "nexusdata/pipeline/fusion.hpp"
#include "nexusdata/pipeline/image.hpp"
#include "nexusdata/pipeline/tabular.hpp"

using namespace nexusdata;

namespace {

bool bit_equal(const NDArray& a, const NDArray& b) {
    if (a.dtype() != b.dtype() || a.shape() != b.shape()) return false;
    if (a.dtype() == DType::Float32) {
        for (std::size_t i = 0; i < a.numel(); ++i) {
            const float x = a.data<float>()[i], y = b.data<float>()[i];
            if (std::isnan(x) && std::isnan(y)) continue;
            if (std::memcmp(&x, &y, 4) != 0) return false;
        }
        return true;
    }
    if (a.dtype() == DType::Float64) {
        for (std::size_t i = 0; i < a.numel(); ++i) {
            const double x = a.data<double>()[i], y = b.data<double>()[i];
            if (std::isnan(x) && std::isnan(y)) continue;
            if (std::memcmp(&x, &y, 8) != 0) return false;
        }
        return true;
    }
    return a.nbytes() == 0 || std::memcmp(a.data(), b.data(), a.nbytes()) == 0;
}

NDArray random_matrix(std::size_t rows, std::size_t cols, DType dt, std::uint32_t seed,
                      double lo, double hi) {
    NDArray a(rows == 0 ? Shape{cols} : Shape{rows, cols}, dt);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(lo, hi);
    for (std::size_t i = 0; i < a.numel(); ++i) {
        double v = u(rng);
        if (dt != DType::Float32 && dt != DType::Float64) v = std::round(v);
        detail::set_double(a, i, v);
    }
    return a;
}

NDArray scalar_f32(float v) {
    NDArray a(Shape{1}, DType::Float32);
    a.data<float>()[0] = v;
    return a;
}

Sample make_sample(NDArray x) {
    Sample s;
    s.input = std::move(x);
    s.label = scalar_f32(3.0f);
    return s;
}

// v1 per-transform loops (what each tabular transform did before v2.0).
NDArray v1_chain(NDArray x, const std::vector<double>& mean, const std::vector<double>& scale,
                 double lo, double hi) {
    for (std::size_t i = 0; i < x.numel(); ++i)
        detail::set_double(x, i, std::clamp(detail::as_double(x, i), lo, hi));
    for (std::size_t i = 0; i < x.numel(); ++i)
        detail::set_double(x, i, std::log1p(detail::as_double(x, i)));
    for (std::size_t i = 0; i < x.numel(); ++i) {
        const std::size_t j = i % mean.size();
        detail::set_double(x, i, (detail::as_double(x, i) - mean[j]) / scale[j]);
    }
    return x;
}

/// User transform exposing a custom fusable kernel.
class AffineTransform : public Transform {
public:
    AffineTransform(double m, double b)
        : step_(FuseStep::make_custom("Affine", [m, b](double* v, std::size_t n, std::size_t) {
              for (std::size_t k = 0; k < n; ++k) v[k] = v[k] * m + b;
          })) {}
    [[nodiscard]] Sample apply(const Sample& s) const override { return apply_fuse_step(step_, s); }
    [[nodiscard]] bool is_fusable() const override { return true; }
    [[nodiscard]] FuseStepPtr fuse_step() const override { return step_; }

private:
    FuseStepPtr step_;
};

} // namespace

TEST_CASE("fusion: consecutive tabular transforms compile to one FusedTransform") {
    const std::size_t F = 7;
    std::vector<double> mean(F), scale(F);
    for (std::size_t j = 0; j < F; ++j) {
        mean[j] = 0.1 * static_cast<double>(j);
        scale[j] = 0.5 + static_cast<double>(j);
    }
    auto clip = std::make_shared<ClipTransform>(-1.0, 50.0);
    auto logt = std::make_shared<Log1pTransform>();
    auto stdz = std::make_shared<StandardizeTransform>(mean, scale);
    Compose fused({clip, logt, stdz});
    Compose plain({clip, logt, stdz}, /*fuse=*/false);
    REQUIRE(fused.plan().size() == 1);
    CHECK(dynamic_cast<const FusedTransform*>(fused.plan()[0].get()) != nullptr);
    CHECK(plain.plan().size() == 3);
    CHECK(fused.supports_batch());

    for (DType dt : {DType::Float32, DType::Float64}) {
        for (std::uint32_t seed = 0; seed < 5; ++seed) {
            CAPTURE(to_string(dt));
            const NDArray x = random_matrix(0, F, dt, seed, -5.0, 100.0);
            const Sample a = fused.apply(make_sample(x));
            const Sample b = plain.apply(make_sample(x));
            CHECK(bit_equal(a.input, b.input));
            CHECK(bit_equal(a.input, v1_chain(x.clone(), mean, scale, -1.0, 50.0)));
            CHECK(a.label.data<float>()[0] == 3.0f);
        }
    }

    // Integer storage: every intermediate is re-rounded like the unfused chain.
    auto mm = std::make_shared<MinMaxTransform>(std::vector<double>(F, -3.0),
                                                std::vector<double>(F, 40.0), 0.0, 100.0);
    Compose ifused({clip, mm, std::make_shared<AffineTransform>(1.5, 0.25)});
    Compose iplain({clip, mm, std::make_shared<AffineTransform>(1.5, 0.25)}, false);
    CHECK(ifused.plan().size() == 1);
    const NDArray xi = random_matrix(0, F, DType::Int32, 9, -10.0, 60.0);
    CHECK(bit_equal(ifused.apply(make_sample(xi)).input, iplain.apply(make_sample(xi)).input));
}

TEST_CASE("fusion: random / opaque transforms split fused groups") {
    auto c1 = std::make_shared<ClipTransform>(0.0, 1.0);
    auto c2 = std::make_shared<ClipTransform>(0.2, 0.8);
    auto rnd = std::make_shared<RandomApply>(c1, 0.5f, 1);
    Compose a({c1, c2, rnd, c1, std::make_shared<Log1pTransform>()});
    REQUIRE(a.plan().size() == 3);
    CHECK(dynamic_cast<const FusedTransform*>(a.plan()[0].get()) != nullptr);
    CHECK(a.plan()[1] == rnd);
    CHECK(dynamic_cast<const FusedTransform*>(a.plan()[2].get()) != nullptr);
    CHECK_FALSE(a.supports_batch());

    Compose single({c1}); // a lone step is kept as-is
    CHECK(single.plan().size() == 1);
    CHECK(single.plan()[0] == c1);
}

TEST_CASE("fusion: feature-count mismatch raises ShapeError with transform name") {
    auto stdz = std::make_shared<StandardizeTransform>(std::vector<double>{0, 0, 0},
                                                       std::vector<double>{1, 1, 1});
    Compose fused({std::make_shared<ClipTransform>(0.0, 1.0), stdz});
    const Sample s = make_sample(random_matrix(0, 4, DType::Float32, 1, 0, 1));
    CHECK_THROWS_AS((void)fused.apply(s), ShapeError);
    CHECK_THROWS_AS((void)stdz->apply(s), ShapeError);
    try {
        (void)fused.apply(s);
    } catch (const ShapeError& e) {
        CHECK(std::string(e.what()).find("StandardizeTransform") != std::string::npos);
    }
}

TEST_CASE("fusion: apply_batch equals row-wise apply") {
    const std::size_t F = 5, B = 33;
    auto chain = std::make_shared<Compose>(std::vector<TransformConstPtr>{
        std::make_shared<ClipTransform>(-2.0, 2.0),
        std::make_shared<StandardizeTransform>(std::vector<double>{1, 2, 3, 4, 5},
                                               std::vector<double>{2, 2, 2, 2, 2}),
        std::make_shared<MinMaxTransform>(std::vector<double>(F, -3.0),
                                          std::vector<double>(F, 1.0), -1.0, 1.0)});
    Batch b;
    b.inputs = random_matrix(B, F, DType::Float32, 4, -4.0, 4.0);
    b.labels = random_matrix(0, B, DType::Int64, 5, 0, 9);
    const Batch out = chain->apply_batch(b);
    CHECK(out.labels.data() == b.labels.data());
    for (std::size_t r = 0; r < B; ++r) {
        NDArray row(Shape{F}, DType::Float32);
        std::memcpy(row.data(), b.inputs.data<float>() + r * F, F * 4);
        const Sample s = chain->apply(make_sample(row));
        REQUIRE(std::memcmp(s.input.data(), out.inputs.data<float>() + r * F, F * 4) == 0);
    }

    // Big enough to take the parallel path.
    Batch big;
    big.inputs = random_matrix(1 << 14, 80, DType::Float32, 6, -4.0, 4.0);
    auto clip_log = std::make_shared<Compose>(std::vector<TransformConstPtr>{
        std::make_shared<ClipTransform>(0.0, 3.0), std::make_shared<Log1pTransform>()});
    const Batch bo = clip_log->apply_batch(big);
    NDArray ref = big.inputs.clone();
    for (std::size_t i = 0; i < ref.numel(); ++i) {
        detail::set_double(ref, i, std::clamp(detail::as_double(ref, i), 0.0, 3.0));
        detail::set_double(ref, i, std::log1p(detail::as_double(ref, i)));
    }
    CHECK(bit_equal(bo.inputs, ref));

    Compose opaque({std::make_shared<RandomApply>(std::make_shared<Log1pTransform>(), 1.0f)});
    CHECK_THROWS_AS((void)opaque.apply_batch(b), InvalidArgumentError);
}

TEST_CASE("fusion: image chains fuse into one pass, bit-exact") {
    const std::vector<float> mean{0.485f, 0.456f, 0.406f};
    const std::vector<float> stdv{0.229f, 0.224f, 0.225f};
    NDArray img(Shape{37, 53, 3}, DType::UInt8);
    std::mt19937 rng(11);
    for (std::size_t i = 0; i < img.numel(); ++i)
        img.data<std::uint8_t>()[i] = static_cast<std::uint8_t>(rng() & 0xFF);
    Sample s;
    s.input = img;
    s.label = scalar_f32(1.0f);

    for (auto interp : {ResizeInterp::Bilinear, ResizeInterp::Nearest}) {
        std::vector<TransformConstPtr> ts{std::make_shared<Resize>(24, 31, interp),
                                          std::make_shared<ToTensor>(),
                                          std::make_shared<NormalizeImage>(mean, stdv)};
        Compose fused(ts);
        Compose plain(ts, false);
        REQUIRE(fused.plan().size() == 1);
        CHECK(dynamic_cast<const FusedImageTransform*>(fused.plan()[0].get()) != nullptr);
        const Sample a = fused.apply(s);
        const Sample b = plain.apply(s);
        CHECK(a.input.shape() == Shape{3, 24, 31});
        CHECK(bit_equal(a.input, b.input));
    }

    Compose tt({std::make_shared<ToTensor>(), std::make_shared<NormalizeImage>(mean, stdv)});
    Compose tt_plain({std::make_shared<ToTensor>(), std::make_shared<NormalizeImage>(mean, stdv)},
                     false);
    CHECK(tt.plan().size() == 1);
    CHECK(bit_equal(tt.apply(s).input, tt_plain.apply(s).input));

    Compose rz({std::make_shared<Resize>(10, 10), std::make_shared<ToTensor>()});
    Compose rz_plain({std::make_shared<Resize>(10, 10), std::make_shared<ToTensor>()}, false);
    CHECK(bit_equal(rz.apply(s).input, rz_plain.apply(s).input));

    // Resize alone is not rewritten.
    Compose only_resize({std::make_shared<Resize>(10, 10)});
    CHECK(dynamic_cast<const Resize*>(only_resize.plan()[0].get()) != nullptr);

    // Non-uint8 input falls back to the original transforms (same exception).
    Sample f;
    f.input = NDArray(Shape{3, 4, 4}, DType::Float32);
    CHECK_THROWS_AS((void)rz.apply(f), ShapeError);
    CHECK_THROWS_AS((void)rz_plain.apply(f), ShapeError);

    // Channel mismatch is reported like NormalizeImage.
    Sample gray;
    gray.input = NDArray(Shape{4, 4, 1}, DType::UInt8);
    CHECK_THROWS_AS((void)tt.apply(gray), ShapeError);
    CHECK_THROWS_AS((void)tt_plain.apply(gray), ShapeError);
}

// --- DataLoader integration --------------------------------------------------------

namespace {

class FailingDataset : public Dataset {
public:
    explicit FailingDataset(std::size_t n, std::size_t bad) : n_(n), bad_(bad) {}
    [[nodiscard]] std::size_t size() const override { return n_; }
    [[nodiscard]] Sample get(std::size_t i) const override {
        if (i == bad_) throw IOError("corrupt record " + std::to_string(i));
        Sample s;
        s.input = NDArray(Shape{2}, DType::Float32);
        s.input.data<float>()[0] = static_cast<float>(i);
        s.label = NDArray(Shape{1}, DType::Int64);
        s.label.data<std::int64_t>()[0] = static_cast<std::int64_t>(i);
        return s;
    }

private:
    std::size_t n_, bad_;
};

std::shared_ptr<InMemoryDataset> tabular_ds(std::size_t n, std::size_t f) {
    NDArray labels(Shape{n}, DType::Int64);
    for (std::size_t i = 0; i < n; ++i) labels.data<std::int64_t>()[i] = static_cast<std::int64_t>(i);
    return std::make_shared<InMemoryDataset>(random_matrix(n, f, DType::Float32, 21, -3, 3), labels);
}

} // namespace

TEST_CASE("dataloader: batch_transform stage equals per-sample MapDataset") {
    const std::size_t F = 6;
    auto ds = tabular_ds(203, F);
    auto chain = std::make_shared<Compose>(std::vector<TransformConstPtr>{
        std::make_shared<ClipTransform>(-1.0, 1.0),
        std::make_shared<StandardizeTransform>(std::vector<double>(F, 0.1),
                                               std::vector<double>(F, 0.7))});

    DataLoaderOptions ref_opt;
    ref_opt.batch_size = 16;
    DataLoader ref(DatasetConstPtr(std::make_shared<MapDataset>(DatasetConstPtr(ds), chain)), ref_opt);

    for (int workers : {0, 3}) {
        DataLoaderOptions opt;
        opt.batch_size = 16;
        opt.num_workers = workers;
        opt.batch_transform = chain;
        opt.transform_workers = 2;
        opt.cost_model = std::make_shared<CostModel>();
        opt.transform_device = Device::host(); // CPU timings only (GPU paths: test_gpu)
        DataLoader dl(DatasetConstPtr(ds), opt);
        auto it_ref = ref.begin();
        std::size_t batches = 0;
        for (const Batch& b : dl) {
            const Batch r = *it_ref;
            ++it_ref;
            REQUIRE(bit_equal(b.inputs, r.inputs));
            REQUIRE(bit_equal(b.labels, r.labels));
            if (batches == 0 && workers > 0) {
                const auto st = dl.pipeline_stats(); // sampler, load, transform
                REQUIRE(st.size() == 3);
                CHECK(st[2].name == "transform");
                CHECK(st[2].num_threads == 2);
            }
            ++batches;
        }
        CHECK(batches == dl.size());
        CHECK(opt.cost_model->stats(GpuOpKind::TabularSmall).cpu_samples == batches);
        CHECK(dl.pipeline_stats().empty()); // pipeline is torn down at end of epoch
    }

    DataLoaderOptions bad;
    bad.batch_transform = std::make_shared<RandomApply>(chain, 0.5f);
    CHECK_THROWS_AS(DataLoader(DatasetConstPtr(ds), bad), InvalidArgumentError);
}

TEST_CASE("dataloader: in_order=false still yields every batch exactly once") {
    auto ds = tabular_ds(500, 3);
    DataLoaderOptions opt;
    opt.batch_size = 7;
    opt.num_workers = 4;
    opt.in_order = false;
    opt.shuffle = true;
    opt.seed = 5;
    DataLoader dl(DatasetConstPtr(ds), opt);
    std::set<std::int64_t> seen;
    std::size_t batches = 0;
    for (const Batch& b : dl) {
        for (std::size_t i = 0; i < b.labels.numel(); ++i) {
            CHECK(seen.insert(b.labels.data<std::int64_t>()[i]).second);
        }
        ++batches;
    }
    CHECK(seen.size() == 500);
    CHECK(batches == dl.size());
}

TEST_CASE("dataloader: worker errors surface in order and the loader recovers next epoch") {
    auto ds = std::make_shared<FailingDataset>(40, 17);
    DataLoaderOptions opt;
    opt.batch_size = 4;
    opt.num_workers = 3;
    DataLoader dl(DatasetConstPtr(ds), opt);
    std::size_t ok = 0;
    bool threw = false;
    try {
        for (const Batch& b : dl) {
            (void)b;
            ++ok;
        }
    } catch (const IOError&) {
        threw = true;
    }
    CHECK(threw);
    CHECK(ok == 4); // batches 0..3 precede the one containing index 17

    dl.reset_epoch(1); // no crash / hang after a failed epoch
}

TEST_CASE("dataloader: pin_memory with pooled staging buffers") {
    auto ds = tabular_ds(64, 8);
    DataLoaderOptions opt;
    opt.batch_size = 8;
    opt.num_workers = 2;
    opt.pin_memory = true;
    DataLoader dl(DatasetConstPtr(ds), opt);
    DataLoaderOptions ref_opt;
    ref_opt.batch_size = 8;
    DataLoader ref(DatasetConstPtr(ds), ref_opt);
    for (int epoch = 0; epoch < 2; ++epoch) {
        auto it = ref.begin();
        for (const Batch& b : dl) {
            const Batch r = *it;
            ++it;
            REQUIRE(bit_equal(b.inputs, r.inputs));
        }
        dl.reset_epoch(0);
    }
}
