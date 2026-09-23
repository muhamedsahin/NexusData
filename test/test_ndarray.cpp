#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "nexusdata/core/ndarray.hpp"

using namespace nexusdata;

TEST_CASE("NDArray allocate and write") {
    NDArray arr({10}, DType::Float32);
    float* ptr = arr.data<float>();
    for (std::size_t i = 0; i < 10; ++i) {
        ptr[i] = static_cast<float>(i) * 1.5f;
    }
    CHECK(arr.numel() == 10);
    CHECK(arr.shape() == Shape{10});
    CHECK(ptr[5] == 7.5f);
    CHECK(arr.nbytes() == 40);
}

TEST_CASE("NDArray shallow copy shares storage") {
    NDArray a({4}, DType::Int32);
    a.data<int32_t>()[0] = 7;
    NDArray b = a;
    b.data<int32_t>()[0] = 99;
    CHECK(a.data<int32_t>()[0] == 99);
}

TEST_CASE("NDArray clone is deep") {
    NDArray arr1({5}, DType::Int64);
    arr1.data<int64_t>()[0] = 42;
    NDArray arr2 = arr1.clone();
    arr2.data<int64_t>()[0] = 99;
    CHECK(arr1.data<int64_t>()[0] == 42);
    CHECK(arr2.data<int64_t>()[0] == 99);
}

TEST_CASE("NDArray reshape") {
    NDArray arr({12}, DType::Float32);
    arr.reshape({3, 4});
    CHECK(arr.shape() == Shape{3, 4});
    CHECK(arr.numel() == 12);
    CHECK_THROWS_AS(arr.reshape({5, 5}), ShapeError);
}

TEST_CASE("NDArray stack") {
    NDArray a({3}, DType::Float32);
    NDArray b({3}, DType::Float32);
    float* pa = a.data<float>();
    float* pb = b.data<float>();
    pa[0] = 1; pa[1] = 2; pa[2] = 3;
    pb[0] = 4; pb[1] = 5; pb[2] = 6;

    NDArray stacked = NDArray::stack({a, b});
    CHECK(stacked.shape() == Shape{2, 3});
    const float* ps = stacked.data<float>();
    CHECK(ps[0] == 1.0f);
    CHECK(ps[3] == 4.0f);
    CHECK(ps[5] == 6.0f);
}

TEST_CASE("NDArray concat axis 0") {
    NDArray a({2, 2}, DType::Float32);
    NDArray b({1, 2}, DType::Float32);
    a.data<float>()[0] = 1; a.data<float>()[1] = 2;
    a.data<float>()[2] = 3; a.data<float>()[3] = 4;
    b.data<float>()[0] = 5; b.data<float>()[1] = 6;
    NDArray c = NDArray::concat({a, b}, 0);
    CHECK(c.shape() == Shape{3, 2});
    CHECK(c.data<float>()[4] == 5.0f);
}

TEST_CASE("NDArray wrong typed access throws") {
    NDArray arr({5}, DType::Int64);
    CHECK_THROWS_AS(arr.data<float>(), InvalidArgumentError);
}

TEST_CASE("NDArray aligned allocation") {
    NDArray arr({8}, DType::Float32);
    auto addr = reinterpret_cast<std::uintptr_t>(arr.data());
    CHECK((addr % 64) == 0);
}
