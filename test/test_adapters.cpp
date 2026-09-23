#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "nexusdata/adapter/c_api.h"
#include "nexusdata/adapter/dlpack.hpp"
#include "nexusdata/adapter/matrixflash.hpp"
#include "nexusdata/core/ndarray.hpp"

using namespace nexusdata;

TEST_CASE("DLPack roundtrip host Float32") {
    NDArray a(Shape{2, 3}, DType::Float32);
    for (std::size_t i = 0; i < a.numel(); ++i) {
        a.data<float>()[i] = static_cast<float>(i + 1);
    }
    DLManagedTensor* m = ndarray_to_dlpack(a);
    REQUIRE(m != nullptr);
    CHECK(m->dl_tensor.ndim == 2);
    CHECK(m->dl_tensor.shape[0] == 2);
    CHECK(m->dl_tensor.shape[1] == 3);
    CHECK(m->dl_tensor.dtype.code == kDLFloat);
    CHECK(m->dl_tensor.dtype.bits == 32);

    NDArray b = ndarray_from_dlpack(m, /*take_ownership=*/false);
    CHECK(b.shape() == Shape{2, 3});
    CHECK(b.data<float>()[5] == doctest::Approx(6.0f));

    // original still valid
    CHECK(a.data<float>()[0] == doctest::Approx(1.0f));
    m->deleter(m);
}

TEST_CASE("DLPack take_ownership") {
    NDArray a(Shape{4}, DType::Int64);
    a.data<std::int64_t>()[0] = 42;
    DLManagedTensor* m = ndarray_to_dlpack(a);
    NDArray b = ndarray_from_dlpack(m, true);
    CHECK(b.data<std::int64_t>()[0] == 42);
}

TEST_CASE("C API create destroy and data") {
    const int64_t shape[2] = {3, 2};
    NexusNDArray* arr = nullptr;
    CHECK(nexus_ndarray_create(shape, 2, NEXUS_DTYPE_FLOAT32, &arr) == NEXUS_OK);
    REQUIRE(arr != nullptr);
    size_t n = 0;
    CHECK(nexus_ndarray_numel(arr, &n) == NEXUS_OK);
    CHECK(n == 6);
    void* ptr = nullptr;
    CHECK(nexus_ndarray_data(arr, &ptr) == NEXUS_OK);
    REQUIRE(ptr != nullptr);
    static_cast<float*>(ptr)[0] = 1.5f;
    NexusNDArray* clone = nullptr;
    CHECK(nexus_ndarray_clone(arr, &clone) == NEXUS_OK);
    const void* cptr = nullptr;
    CHECK(nexus_ndarray_data_const(clone, &cptr) == NEXUS_OK);
    CHECK(static_cast<const float*>(cptr)[0] == doctest::Approx(1.5f));
    nexus_ndarray_destroy(clone);
    nexus_ndarray_destroy(arr);
    CHECK(std::string(nexus_version()).find("1.0") != std::string::npos);
}

TEST_CASE("C API DLPack bridge") {
    const int64_t shape[1] = {5};
    NexusNDArray* arr = nullptr;
    REQUIRE(nexus_ndarray_create(shape, 1, NEXUS_DTYPE_FLOAT32, &arr) == NEXUS_OK);
    void* data = nullptr;
    REQUIRE(nexus_ndarray_data(arr, &data) == NEXUS_OK);
    for (int i = 0; i < 5; ++i) {
        static_cast<float*>(data)[i] = static_cast<float>(i);
    }
    void* dl = nullptr;
    CHECK(nexus_ndarray_to_dlpack(arr, &dl) == NEXUS_OK);
    NexusNDArray* back = nullptr;
    CHECK(nexus_ndarray_from_dlpack(dl, 1, &back) == NEXUS_OK);
    const void* p = nullptr;
    REQUIRE(nexus_ndarray_data_const(back, &p) == NEXUS_OK);
    CHECK(static_cast<const float*>(p)[4] == doctest::Approx(4.0f));
    nexus_ndarray_destroy(back);
    nexus_ndarray_destroy(arr);
}

TEST_CASE("MatrixFlashAdapter view and copy") {
    CHECK_FALSE(MatrixFlashAdapter::matrixflash_linked());
    NDArray a(Shape{2, 2}, DType::Float32);
    a.data<float>()[0] = 7.0f;
    auto v = MatrixFlashAdapter::view(a);
    CHECK(v.shape.size() == 2);
    CHECK(v.nbytes == 16);
    CHECK(static_cast<float*>(v.data)[0] == doctest::Approx(7.0f));
    NDArray b = MatrixFlashAdapter::copy_to_ndarray(v);
    CHECK(b.data<float>()[0] == doctest::Approx(7.0f));
    Batch batch;
    batch.inputs = a;
    batch.labels = NDArray(Shape{2}, DType::Int64);
    auto bv = MatrixFlashAdapter::view_batch(batch);
    CHECK(bv.inputs.shape[0] == 2);
}
