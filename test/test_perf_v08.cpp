#include <doctest/doctest.h>

#include <string>
#include <vector>

#include "nexusdata/backend/device.hpp"
#include "nexusdata/loading/batch.hpp"
#include "nexusdata/simd/byte_scan.hpp"

using namespace nexusdata;

TEST_CASE("find_csv_record_starts fast path matches quoted path") {
    const std::string plain = "a,b\nc,d\ne,f\n";
    auto a = find_csv_record_starts(reinterpret_cast<const std::uint8_t*>(plain.data()),
                                    plain.size());
    CHECK(a.size() == 3);
    CHECK(a[0] == 0);
    CHECK(a[1] == 4);
    CHECK(a[2] == 8);

    const std::string quoted = "\"a\n\",b\nc,d\n";
    auto b = find_csv_record_starts(reinterpret_cast<const std::uint8_t*>(quoted.data()),
                                    quoted.size());
    CHECK(b.size() >= 2);
    CHECK(b[0] == 0);
}

TEST_CASE("find_bytes scalar matches dispatch") {
    std::string s(1024, 'x');
    for (int i = 0; i < 1024; i += 17) {
        s[static_cast<std::size_t>(i)] = '\n';
    }
    auto a = find_bytes_scalar(reinterpret_cast<const std::uint8_t*>(s.data()), s.size(), '\n');
    auto b = find_bytes(reinterpret_cast<const std::uint8_t*>(s.data()), s.size(), '\n');
    CHECK(a == b);
}

TEST_CASE("calibrate_auto_device_policy sets thresholds") {
    auto rep = calibrate_auto_device_policy();
    CHECK(rep.probe_bytes > 0);
    CHECK(rep.host_memcpy_gib_s > 0.0);
    CHECK(rep.policy.image_nbytes_threshold >= 256 * 1024);
    CHECK(rep.policy.tabular_nbytes_threshold >= rep.policy.image_nbytes_threshold);
    if (!rep.cuda_available) {
        Device d = resolve_device(Device::auto_select(), GpuOpKind::ImageAugment, 1u << 20,
                                 rep.policy);
        CHECK(d.is_host());
    }
}

TEST_CASE("collate_stack direct path shape") {
    std::vector<Sample> batch;
    for (int i = 0; i < 4; ++i) {
        Sample s;
        s.input = NDArray(Shape{3}, DType::Float32);
        s.input.data<float>()[0] = static_cast<float>(i);
        s.input.data<float>()[1] = 1;
        s.input.data<float>()[2] = 2;
        s.label = NDArray(Shape{1}, DType::Int64);
        s.label.data<std::int64_t>()[0] = i;
        batch.push_back(std::move(s));
    }
    auto b = collate_stack(batch);
    CHECK(b.inputs.shape() == Shape{4, 3});
    CHECK(b.labels.shape() == Shape{4});
    CHECK(b.inputs.data<float>()[0] == doctest::Approx(0.0f));
    CHECK(b.inputs.data<float>()[3] == doctest::Approx(1.0f)); // sample 1, feature 0
    CHECK(b.labels.data<std::int64_t>()[2] == 2);
}
