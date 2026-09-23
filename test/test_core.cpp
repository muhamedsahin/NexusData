#include <doctest/doctest.h>

#include <string>

#include "nexusdata/core/dtype.hpp"
#include "nexusdata/core/error.hpp"
#include "nexusdata/core/shape.hpp"
#include "nexusdata/version.hpp"

using namespace nexusdata;

TEST_CASE("DType size_of") {
    CHECK(size_of(DType::Bool) == 1);
    CHECK(size_of(DType::Float32) == 4);
    CHECK(size_of(DType::Float64) == 8);
    CHECK(size_of(DType::Int64) == 8);
    CHECK(size_of(DType::Float16) == 2);
    CHECK(size_of(DType::BFloat16) == 2);
}

TEST_CASE("Shape numel") {
    CHECK(numel({}) == 0);
    CHECK(numel({5}) == 5);
    CHECK(numel({32, 10}) == 320);
    CHECK(numel({2, 3, 4}) == 24);
}

TEST_CASE("dtype_of traits") {
    CHECK(dtype_of<float> == DType::Float32);
    CHECK(dtype_of<int64_t> == DType::Int64);
}

TEST_CASE("Error hierarchy") {
    CHECK_THROWS_AS(throw InvalidArgumentError("x"), Error);
    CHECK_THROWS_AS(throw IOError("io"), Error);
}

TEST_CASE("Error message helpers") {
    CHECK(quote_path("p") == "\"p\"");
    CHECK(format_index_error("X", 1, 1).find("[0, 1)") != std::string::npos);
    CHECK(format_loc("a.csv", 3, 0, "msg") == "a.csv:3 msg");
}

TEST_CASE("version 1.x macros") {
    CHECK(NEXUSDATA_VERSION_MAJOR == 1);
    CHECK(std::string(version_string()).find("1.0") != std::string::npos);
    CHECK(version_major() == 1);
}
