#include <doctest/doctest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "nexusdata/pipeline/stage_graph.hpp"

using namespace nexusdata;

namespace {
void jitter(std::size_t v) {
    std::this_thread::sleep_for(std::chrono::microseconds((v * 7919) % 300));
}
} // namespace

TEST_CASE("stage graph: in-order delivery across parallel stages") {
    constexpr std::size_t N = 400;
    auto g = StageGraph<std::size_t>::from_range("src", N)
                 .then("square", StageOptions{4, 4},
                       [](std::size_t v) {
                           jitter(v);
                           return v * v;
                       })
                 .then("str", StageOptions{3, 2}, [](std::size_t v) { return std::to_string(v); });
    for (std::size_t i = 0; i < N; ++i) {
        auto v = g.next();
        REQUIRE(v.has_value());
        REQUIRE(*v == std::to_string(i * i));
    }
    CHECK_FALSE(g.next().has_value());
    CHECK_FALSE(g.next().has_value()); // stays exhausted

    const auto stats = g.stats();
    REQUIRE(stats.size() == 3);
    CHECK(stats[0].name == "src");
    CHECK(stats[1].num_threads == 4);
    CHECK(stats[1].items == N);
    CHECK(stats[2].items == N);
}

TEST_CASE("stage graph: unordered delivery yields every item once") {
    constexpr std::size_t N = 300;
    StageGraphOptions opt;
    opt.in_order = false;
    auto g = StageGraph<std::size_t>::from_range("src", N, 4, opt)
                 .then("work", StageOptions{6, 4}, [](std::size_t v) {
                     jitter(v);
                     return v;
                 });
    std::set<std::size_t> seen;
    while (auto v = g.next()) {
        CHECK(seen.insert(*v).second);
    }
    CHECK(seen.size() == N);
    CHECK(g.reorder_buffer_size() == 0);
}

TEST_CASE("stage graph: exceptions surface at the failing position, stream continues") {
    auto g = StageGraph<std::size_t>::from_range("src", 20)
                 .then("maybe_fail", StageOptions{3, 2}, [](std::size_t v) {
                     jitter(v);
                     if (v == 7) throw IOError("boom at 7");
                     return v;
                 });
    for (std::size_t i = 0; i < 7; ++i) {
        auto v = g.next();
        REQUIRE(v.has_value());
        CHECK(*v == i);
    }
    CHECK_THROWS_AS((void)g.next(), IOError);
    for (std::size_t i = 8; i < 20; ++i) {
        auto v = g.next();
        REQUIRE(v.has_value());
        CHECK(*v == i);
    }
    CHECK_FALSE(g.next().has_value());
}

TEST_CASE("stage graph: source exception terminates the stream after delivery") {
    auto counter = std::make_shared<int>(0);
    auto g = StageGraph<int>::from_generator("gen", [counter]() -> std::optional<int> {
                 if (*counter == 3) throw std::runtime_error("source died");
                 return (*counter)++;
             }).then("id", StageOptions{2, 2}, [](int v) { return v; });
    CHECK(*g.next() == 0);
    CHECK(*g.next() == 1);
    CHECK(*g.next() == 2);
    CHECK_THROWS_AS((void)g.next(), std::runtime_error);
    CHECK_FALSE(g.next().has_value());
}

TEST_CASE("stage graph: credit window bounds items in flight") {
    StageGraphOptions opt;
    opt.max_in_flight = 5;
    std::atomic<int> produced{0};
    auto g = StageGraph<int>::from_generator(
                 "gen",
                 [&produced]() -> std::optional<int> {
                     const int v = produced.fetch_add(1);
                     if (v >= 1000) return std::nullopt;
                     return v;
                 },
                 8, opt)
                 .then("id", StageOptions{4, 8}, [](int v) { return v; });
    for (int i = 0; i < 10; ++i) {
        REQUIRE(g.next().has_value());
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        // consumed i + 1 items; the source can be at most window ahead
        CHECK(produced.load() <= (i + 1) + 5);
    }
    CHECK(g.in_flight_window() == 5);
}

TEST_CASE("stage graph: early destruction with pending work does not hang") {
    std::atomic<int> done{0};
    {
        auto g = StageGraph<std::size_t>::from_range("src", 100000)
                     .then("slow", StageOptions{4, 2}, [&](std::size_t v) {
                         std::this_thread::sleep_for(std::chrono::microseconds(200));
                         done.fetch_add(1);
                         return v;
                     });
        for (int i = 0; i < 3; ++i) REQUIRE(g.next().has_value());
    }
    CHECK(done.load() < 1000);

    // Never started: destructor must be a no-op.
    auto idle = StageGraph<std::size_t>::from_range("src", 10).then(
        "x", StageOptions{1, 1}, [](std::size_t v) { return v; });
    (void)idle;
}

TEST_CASE("stage graph: move assignment and invalid options") {
    auto a = StageGraph<std::size_t>::from_range("a", 3).then("x", StageOptions{1, 1},
                                                             [](std::size_t v) { return v + 10; });
    auto b = StageGraph<std::size_t>::from_range("b", 2).then("y", StageOptions{1, 1},
                                                             [](std::size_t v) { return v; });
    CHECK(*b.next() == 0);
    b = std::move(a);
    CHECK(*b.next() == 10);
    CHECK(*b.next() == 11);
    CHECK(*b.next() == 12);
    CHECK_FALSE(b.next().has_value());

    auto c = StageGraph<std::size_t>::from_range("c", 1);
    CHECK_THROWS_AS((void)std::move(c).then("bad", StageOptions{0, 1},
                                            [](std::size_t v) { return v; }),
                    InvalidArgumentError);
}
