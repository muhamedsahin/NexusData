#include <doctest/doctest.h>

#include <cmath>
#include <set>

#include "nexusdata/dataset/in_memory.hpp"
#include "nexusdata/dataset/map.hpp"
#include "nexusdata/pipeline/tabular.hpp"
#include "nexusdata/pipeline/transform.hpp"

using namespace nexusdata;

static Sample make_sample(std::initializer_list<float> feats, float label) {
    Sample s;
    s.input = NDArray(Shape{feats.size()}, DType::Float32);
    std::size_t i = 0;
    for (float v : feats) {
        s.input.data<float>()[i++] = v;
    }
    s.label = NDArray(Shape{1}, DType::Float32);
    s.label.data<float>()[0] = label;
    return s;
}

TEST_CASE("ClipTransform") {
    ClipTransform clip(0.0, 1.0);
    Sample s = make_sample({-1.0f, 0.5f, 2.0f}, 0.0f);
    Sample o = clip(s);
    CHECK(o.input.data<float>()[0] == 0.0f);
    CHECK(o.input.data<float>()[1] == 0.5f);
    CHECK(o.input.data<float>()[2] == 1.0f);
}

TEST_CASE("Compose chains transforms") {
    auto clip = std::make_shared<ClipTransform>(0.0, 10.0);
    auto log1p = std::make_shared<Log1pTransform>();
    Compose compose({clip, log1p});
    Sample s = make_sample({0.0f}, 0.0f);
    Sample o = compose(s);
    CHECK(std::abs(o.input.data<float>()[0] - std::log1p(0.0f)) < 1e-6f);
}

TEST_CASE("StandardizeTransform") {
    StandardizeTransform t({1.0, 2.0}, {2.0, 4.0});
    Sample s = make_sample({3.0f, 6.0f}, 0.0f);
    Sample o = t(s);
    CHECK(o.input.data<float>()[0] == doctest::Approx(1.0f));
    CHECK(o.input.data<float>()[1] == doctest::Approx(1.0f));
}

TEST_CASE("RandomApply determinism with seed") {
    auto clip = std::make_shared<ClipTransform>(-100.0, 0.0);
    RandomApply a(clip, 1.0f, /*seed=*/7); // always apply
    RandomApply b(clip, 1.0f, /*seed=*/7);
    Sample s = make_sample({5.0f}, 0.0f);
    CHECK(a(s).input.data<float>()[0] == b(s).input.data<float>()[0]);
    CHECK(a(s).input.data<float>()[0] == 0.0f);
}

TEST_CASE("OneOf determinism") {
    auto t0 = std::make_shared<ClipTransform>(0.0, 0.0);
    auto t1 = std::make_shared<ClipTransform>(1.0, 1.0);
    OneOf a({t0, t1}, 42);
    OneOf b({t0, t1}, 42);
    Sample s = make_sample({0.5f}, 0.0f);
    CHECK(a(s).input.data<float>()[0] == b(s).input.data<float>()[0]);
}

TEST_CASE("MapDataset applies transform lazily") {
    NDArray feats({3, 1}, DType::Float32);
    NDArray labels({3}, DType::Float32);
    feats.data<float>()[0] = 5; feats.data<float>()[1] = -2; feats.data<float>()[2] = 0.5f;
    labels.data<float>()[0] = 0; labels.data<float>()[1] = 1; labels.data<float>()[2] = 0;
    InMemoryDataset ds(std::move(feats), std::move(labels));
    auto clip = std::make_shared<ClipTransform>(0.0, 1.0);
    MapDataset mapped(ds, clip);
    CHECK(mapped.size() == 3);
    CHECK(mapped.get(0).input.data<float>()[0] == 1.0f);
    CHECK(mapped.get(1).input.data<float>()[0] == 0.0f);
}
