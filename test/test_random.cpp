#include <doctest/doctest.h>

#include <vector>

#include "nexusdata/core/random.hpp"

using namespace nexusdata;

TEST_CASE("PCG32 golden sequence seed=42") {
    PCG32 rng(42);
    const std::uint32_t expected[] = {
        2707161783u, 2068313097u, 3122475824u, 2211639955u, 3215226955u,
        3421331566u, 3217466285u, 2167406445u, 3860803674u, 4181216144u,
    };
    for (std::uint32_t e : expected) {
        CHECK(rng.next_u32() == e);
    }
}

TEST_CASE("PCG32 same seed same sequence") {
    PCG32 a(12345);
    PCG32 b(12345);
    for (int i = 0; i < 100; ++i) {
        CHECK(a.next_u32() == b.next_u32());
    }
}

TEST_CASE("PCG32 different seeds diverge") {
    PCG32 a(1);
    PCG32 b(2);
    CHECK(a.next_u32() != b.next_u32());
}

TEST_CASE("Fisher-Yates golden shuffle n=8 seed=42") {
    PCG32 rng(42);
    auto idx = shuffled_indices(8, rng);
    const std::vector<std::size_t> expected = {0, 1, 6, 7, 2, 4, 3, 5};
    CHECK(idx == expected);
}

TEST_CASE("Fisher-Yates is a permutation") {
    PCG32 rng(99);
    auto idx = shuffled_indices(64, rng);
    CHECK(idx.size() == 64);
    std::vector<bool> seen(64, false);
    for (std::size_t v : idx) {
        CHECK(v < 64);
        CHECK_FALSE(seen[v]);
        seen[v] = true;
    }
}

TEST_CASE("next_bounded respects range") {
    PCG32 rng(7);
    for (int i = 0; i < 1000; ++i) {
        auto v = rng.next_bounded(10);
        CHECK(v < 10);
    }
}
