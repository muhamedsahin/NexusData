#include <doctest/doctest.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <set>
#include <thread>
#include <vector>

#include "nexusdata/core/arena_allocator.hpp"
#include "nexusdata/core/ndarray.hpp"
#include "nexusdata/core/object_pool.hpp"

using namespace nexusdata;

namespace {
bool aligned(const void* p, std::size_t a) {
    return reinterpret_cast<std::uintptr_t>(p) % a == 0;
}
} // namespace

TEST_CASE("arena: bump allocation, alignment and O(1) reuse after reset") {
    Arena arena(4096);
    std::set<const void*> seen;
    for (int i = 0; i < 20; ++i) {
        void* p = arena.allocate(37);
        CHECK(aligned(p, kDefaultAlignment));
        CHECK(seen.insert(p).second);
    }
    void* big_align = arena.allocate(10, 256);
    CHECK(aligned(big_align, 256));
    CHECK(arena.bytes_used() >= 20 * 37);
    const std::size_t upstream = arena.upstream_allocations();

    arena.reset();
    CHECK(arena.bytes_used() == 0);
    void* again = arena.allocate(37);
    CHECK(again != nullptr);
    CHECK(arena.upstream_allocations() == upstream);

    CHECK_THROWS_AS((void)arena.allocate(8, 3), InvalidArgumentError);
    CHECK_THROWS_AS(Arena(0), InvalidArgumentError);
}

TEST_CASE("arena: oversized requests get a dedicated block; reset coalesces") {
    Arena arena(1024);
    (void)arena.allocate(512);
    (void)arena.allocate(10000); // > block size
    (void)arena.allocate(800);   // does not fit block 0 remainder
    CHECK(arena.num_blocks() >= 2);
    const std::size_t reserved = arena.bytes_reserved();
    arena.reset();
    CHECK(arena.num_blocks() == 1);
    CHECK(arena.bytes_reserved() == reserved);
    const std::size_t up = arena.upstream_allocations();
    // The same batch shape now fits in the single coalesced block.
    (void)arena.allocate(512);
    (void)arena.allocate(10000);
    (void)arena.allocate(800);
    CHECK(arena.num_blocks() == 1);
    CHECK(arena.upstream_allocations() == up);
}

TEST_CASE("arena: NDArray::from_arena outlives reset") {
    Arena arena(1 << 16);
    NDArray a = NDArray::from_arena(arena, Shape{16, 4}, DType::Float32, /*zero=*/true);
    CHECK(a.shape() == Shape{16, 4});
    CHECK(aligned(a.data(), kDefaultAlignment));
    for (std::size_t i = 0; i < a.numel(); ++i) CHECK(a.data<float>()[i] == 0.0f);
    for (std::size_t i = 0; i < a.numel(); ++i) a.data<float>()[i] = static_cast<float>(i);

    arena.reset(); // block is still referenced by `a`: handed over, not reused
    for (int k = 0; k < 8; ++k) {
        NDArray b = NDArray::from_arena(arena, Shape{16, 4}, DType::Float32);
        std::memset(b.data(), 0xFF, b.nbytes());
    }
    for (std::size_t i = 0; i < a.numel(); ++i) {
        REQUIRE(a.data<float>()[i] == static_cast<float>(i));
    }

    NDArray empty = NDArray::from_arena(arena, Shape{0, 3}, DType::Int32);
    CHECK(empty.numel() == 0);
}

TEST_CASE("arena pool: leases recycle arenas; thread-local arenas are distinct") {
    ArenaPool pool(4096, 2);
    const Arena* first = nullptr;
    {
        auto lease = pool.acquire();
        first = &lease.arena();
        (void)lease->allocate(100);
    }
    CHECK(pool.cached() == 1);
    {
        auto lease = pool.acquire();
        CHECK(&lease.arena() == first);
        CHECK(lease->bytes_used() == 0); // reset on return
        auto l2 = pool.acquire();
        auto l3 = pool.acquire();
        CHECK(&l2.arena() != &l3.arena());
    }
    CHECK(pool.cached() == 2); // max_cached bound

    Arena* main_arena = &ArenaPool::thread_local_arena();
    Arena* other = nullptr;
    std::thread t([&] { other = &ArenaPool::thread_local_arena(); });
    t.join();
    CHECK(main_arena == &ArenaPool::thread_local_arena());
    CHECK(other != main_arena);
}

TEST_CASE("object pool: reuse, reset hook, bounded cache, pool outlived by handles") {
    int resets = 0;
    auto pool = std::make_unique<ObjectPool<std::vector<int>>>(
        [] { return std::make_unique<std::vector<int>>(); },
        [&](std::vector<int>& v) {
            v.clear();
            ++resets;
        },
        /*max_cached=*/1);
    std::vector<int>* addr = nullptr;
    {
        auto h = pool->acquire();
        h->push_back(42);
        addr = h.get();
    }
    CHECK(resets == 1);
    CHECK(pool->cached() == 1);
    {
        auto h = pool->acquire();
        CHECK(h.get() == addr);
        CHECK(h->empty());
        auto h2 = pool->acquire();
        CHECK(pool->created() == 2);
        CHECK(pool->reused() == 1);
    }
    CHECK(pool->cached() == 1);

    auto survivor = pool->acquire();
    pool.reset();
    survivor->push_back(1); // still valid; deleted when the handle dies
}

TEST_CASE("buffer pool: size classes") {
    std::size_t prev = 0;
    for (std::size_t b = 1; b < (1u << 22); b = b * 3 / 2 + 1) {
        const std::size_t c = BufferPool::size_class(b);
        CHECK(c >= b);
        CHECK(c >= prev);
        if (b > 256) CHECK(static_cast<double>(c) <= 1.25 * static_cast<double>(b) + 64);
        prev = c;
    }
}

TEST_CASE("buffer pool: hits, zeroing, limits, outstanding buffers after destruction") {
    BufferPoolOptions opt;
    opt.max_cached_bytes = 1 << 20;
    auto pool = std::make_unique<BufferPool>(opt);
    void* p1 = nullptr;
    {
        AlignedBuffer b = pool->acquire(1000);
        p1 = b.get();
        CHECK(aligned(p1, kDefaultAlignment));
        std::memset(p1, 0xAB, 1000);
    }
    CHECK(pool->stats().misses == 1);
    CHECK(pool->stats().cached_bytes >= 1000);
    {
        NDArray a = pool->make_array(Shape{250}, DType::Float32); // same size class
        CHECK(a.data() == p1);
        for (std::size_t i = 0; i < 250; ++i) REQUIRE(a.data<float>()[i] == 0.0f);
        CHECK(pool->stats().hits == 1);
        CHECK(pool->stats().outstanding == 1);
    }
    {
        AlignedBuffer huge = pool->acquire(2 << 20); // larger than max_cached_bytes
        (void)huge;
    }
    CHECK(pool->stats().cached_bytes <= opt.max_cached_bytes);

    pool->trim();
    CHECK(pool->stats().cached_bytes == 0);

    AlignedBuffer late = pool->acquire(4096);
    pool.reset();
    std::memset(late.get(), 1, 4096); // freed directly once released
}

TEST_CASE("buffer pool: concurrent acquire/release") {
    BufferPool pool;
    std::atomic<int> errors{0};
    std::vector<std::thread> ts;
    for (int t = 0; t < 8; ++t) {
        ts.emplace_back([&, t] {
            for (int i = 0; i < 2000; ++i) {
                const std::size_t n = 64 + static_cast<std::size_t>((i * 37 + t * 11) % 5000);
                AlignedBuffer b = pool.acquire(n);
                auto* p = static_cast<std::uint8_t*>(b.get());
                p[0] = static_cast<std::uint8_t>(t);
                p[n - 1] = static_cast<std::uint8_t>(t);
                if (p[0] != t || p[n - 1] != t) errors.fetch_add(1);
            }
        });
    }
    for (auto& th : ts) th.join();
    CHECK(errors.load() == 0);
    CHECK(pool.stats().outstanding == 0);
    CHECK(pool.stats().hits > 0);
}

TEST_CASE("buffer pool: pinned factory works without CUDA") {
    auto pinned = BufferPool::pinned(1 << 20);
    NDArray a = pinned->make_array(Shape{8, 8}, DType::Float64);
    CHECK(a.numel() == 64);
    CHECK(a.device().is_host());
}
