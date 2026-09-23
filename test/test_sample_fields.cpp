// v2.0 Phase 3: Sample::extra_tensors / metadata (plan 7.8) and their batching.

#include <doctest/doctest.h>

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "nexusdata/dataset/compose.hpp"
#include "nexusdata/dataset/dataset.hpp"
#include "nexusdata/loading/batch.hpp"
#include "nexusdata/loading/dataloader.hpp"
#include "nexusdata/pipeline/image.hpp"

using namespace nexusdata;

namespace {

NDArray f32(Shape s, float start) {
    NDArray x(std::move(s), DType::Float32);
    for (std::size_t i = 0; i < x.numel(); ++i) x.data<float>()[i] = start + static_cast<float>(i);
    return x;
}

/// Detection-style sample i: 8x8x3 image, label i, i+1 boxes, a fixed-size mask, a path.
Sample detection_sample(std::size_t i) {
    Sample s;
    s.input = NDArray(Shape{8, 8, 3}, DType::UInt8);
    std::memset(s.input.data(), static_cast<int>(i), s.input.nbytes());
    s.label = f32(Shape{1}, static_cast<float>(i));
    s.extra_tensors.set("bbox", f32(Shape{i + 1, 4}, static_cast<float>(10 * i)));
    s.extra_tensors.set("mask", f32(Shape{2, 2}, static_cast<float>(i)));
    s.metadata.set("path", "img_" + std::to_string(i) + ".jpg");
    if (i % 2 == 0) s.metadata.set("split", "even");
    return s;
}

class DetectionDataset : public Dataset {
public:
    explicit DetectionDataset(std::size_t n) : n_(n) {}
    [[nodiscard]] std::size_t size() const override { return n_; }
    [[nodiscard]] Sample get(std::size_t i) const override { return detection_sample(i); }

private:
    std::size_t n_;
};

} // namespace

TEST_CASE("FieldMap: insertion order, overwrite, lookup and erase") {
    FieldMap<std::string> m;
    CHECK(m.empty());
    m.set("b", "1");
    m.set("a", "2");
    m.set("b", "3");
    REQUIRE(m.size() == 2);
    CHECK(m.begin()->first == "b");
    CHECK(m.at("b") == "3");
    CHECK(m.contains("a"));
    CHECK(m.find("zz") == nullptr);
    CHECK_THROWS_AS((void)m.at("zz"), IndexError);
    m["c"] += "x";
    CHECK(m.at("c") == "x");
    CHECK(m.erase("a"));
    CHECK_FALSE(m.erase("a"));
    CHECK(m.size() == 2);
}

TEST_CASE("Sample: v0.1 fields keep working, new fields are opt-in") {
    Sample s{f32(Shape{3}, 0), f32(Shape{1}, 7)};
    CHECK(s.extra_tensors.empty());
    CHECK(s.metadata.empty());
    const Sample t = s.with_input(f32(Shape{2}, 5));
    CHECK(t.input.numel() == 2);
    CHECK(t.label.data<float>()[0] == 7.0f);
}

TEST_CASE("collate: equal-shape extras stack, varying shapes go ragged, metadata per sample") {
    std::vector<Sample> samples;
    for (std::size_t i = 0; i < 3; ++i) samples.push_back(detection_sample(i));
    const Batch b = collate_stack_nd(samples);

    CHECK(b.inputs.shape() == Shape{3, 8, 8, 3});
    const NDArray& mask = b.extras.at("mask");
    CHECK(mask.shape() == Shape{3, 2, 2});
    CHECK(mask.data<float>()[4] == 1.0f); // sample 1, element 0
    CHECK_FALSE(b.extras.contains("bbox"));

    const std::vector<NDArray>& boxes = b.ragged_extras.at("bbox");
    REQUIRE(boxes.size() == 3);
    CHECK(boxes[2].shape() == Shape{3, 4});
    CHECK(boxes[2].data<float>()[0] == 20.0f);

    CHECK(b.metadata.at("path") == std::vector<std::string>{"img_0.jpg", "img_1.jpg", "img_2.jpg"});
    CHECK(b.metadata.at("split") == std::vector<std::string>{"even", "", "even"});

    const RaggedBatch r = collate_ragged(samples);
    CHECK(r.extras.at("mask").shape() == Shape{3, 2, 2});
    CHECK(r.ragged_extras.at("bbox").size() == 3);

    // Plain samples: no fields, no work.
    const Batch plain = collate_stack({Sample{f32(Shape{2}, 0), f32(Shape{1}, 0)}});
    CHECK(plain.extras.empty());
    CHECK(plain.metadata.empty());
}

TEST_CASE("collate: inconsistent extra keys or dtypes are rejected") {
    std::vector<Sample> samples{detection_sample(0), detection_sample(1)};
    samples[1].extra_tensors.erase("mask");
    CHECK_THROWS_AS((void)collate_stack(samples), ShapeError);

    samples = {detection_sample(0), detection_sample(1)};
    samples[1].extra_tensors.set("mask", NDArray(Shape{2, 2}, DType::Int32));
    CHECK_THROWS_AS((void)collate_stack(samples), ShapeError);
}

TEST_CASE("transforms and ZipDataset carry extra fields along") {
    const Sample s = detection_sample(1);
    const Sample resized = Resize(4, 4).apply(s);
    CHECK(resized.input.shape() == Shape{4, 4, 3});
    CHECK(resized.extra_tensors.at("bbox").shape() == Shape{2, 4});
    CHECK(resized.metadata.at("path") == "img_1.jpg");

    const Sample norm = NormalizeImage({0.5f, 0.5f, 0.5f}, {0.5f, 0.5f, 0.5f}).apply(s);
    CHECK(norm.metadata.at("path") == "img_1.jpg");

    auto a = std::make_shared<DetectionDataset>(2);
    auto b = std::make_shared<DetectionDataset>(2);
    ZipDataset zip(a, b);
    const Sample z = zip.get(1);
    CHECK(z.metadata.at("path") == "img_1.jpg");
    CHECK(z.extra_tensors.contains("bbox"));
}

TEST_CASE("DataLoader: extra fields flow through prefetch workers") {
    auto ds = std::make_shared<DetectionDataset>(10);
    DataLoaderOptions opt;
    opt.batch_size = 4;
    opt.num_workers = 2;
    DataLoader dl(DatasetConstPtr(ds), opt, collate_stack_nd);
    std::size_t seen = 0;
    for (const Batch& b : dl) {
        const std::size_t n = b.inputs.shape()[0];
        CHECK(b.extras.at("mask").shape()[0] == n);
        REQUIRE(b.metadata.at("path").size() == n);
        CHECK(b.metadata.at("path")[0] == "img_" + std::to_string(seen) + ".jpg");
        seen += n;
    }
    CHECK(seen == 10);
}
