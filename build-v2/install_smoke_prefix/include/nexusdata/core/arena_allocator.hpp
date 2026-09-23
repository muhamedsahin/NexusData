#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include <vector>

#include "nexusdata/core/allocator.hpp"

namespace nexusdata {

/// Bump-pointer arena: one large aligned block serves many small allocations,
/// reset() recycles everything in O(#blocks) with no per-allocation free.
///
/// Lifetime safety: allocate_shared() returns buffers that alias the owning
/// block. On reset(), blocks that are still referenced (e.g. by NDArrays built
/// with NDArray::from_arena) are handed over to those references and never
/// reused, so outstanding arrays stay valid. Raw pointers from allocate() are
/// only valid until the next reset().
///
/// After a reset where more than one block was in use, the free blocks are
/// coalesced into a single block of the combined size, so a steady-state batch
/// is served by exactly one upstream allocation.
///
/// Thread-safety: not thread-safe; use one arena per thread (see ArenaPool).
class Arena {
public:
    explicit Arena(std::size_t block_bytes = std::size_t{4} << 20);

    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;
    Arena(Arena&&) noexcept = default;
    Arena& operator=(Arena&&) noexcept = default;

    /// Raw allocation, valid until reset(). @p alignment must be a power of two.
    [[nodiscard]] void* allocate(std::size_t bytes, std::size_t alignment = kDefaultAlignment);

    /// Allocation that keeps its block alive for as long as the returned buffer lives.
    [[nodiscard]] AlignedBuffer allocate_shared(std::size_t bytes,
                                                std::size_t alignment = kDefaultAlignment);

    /// Recycle all memory not referenced by outstanding allocate_shared() buffers.
    void reset();

    /// Bytes handed out since the last reset (including alignment padding).
    [[nodiscard]] std::size_t bytes_used() const noexcept { return used_; }
    /// Bytes held by the arena's blocks.
    [[nodiscard]] std::size_t bytes_reserved() const noexcept;
    [[nodiscard]] std::size_t num_blocks() const noexcept { return blocks_.size(); }
    /// Upstream (aligned_alloc) calls made over the arena's lifetime.
    [[nodiscard]] std::size_t upstream_allocations() const noexcept { return upstream_; }
    [[nodiscard]] std::size_t block_bytes() const noexcept { return block_bytes_; }

private:
    struct Block {
        AlignedBuffer mem;
        std::size_t size = 0;
        std::size_t offset = 0;
    };

    [[nodiscard]] std::size_t take(std::size_t bytes, std::size_t alignment);
    void add_block(std::size_t min_bytes);

    std::size_t block_bytes_ = 0;
    std::vector<Block> blocks_;
    std::size_t current_ = 0;
    std::size_t used_ = 0;
    std::size_t upstream_ = 0;
};

/// Pool of reusable arenas for worker threads.
///
/// Thread-safety: acquire()/release are thread-safe; each leased Arena is used by
/// a single thread at a time.
class ArenaPool {
public:
    explicit ArenaPool(std::size_t block_bytes = std::size_t{4} << 20, std::size_t max_cached = 64);

    ArenaPool(const ArenaPool&) = delete;
    ArenaPool& operator=(const ArenaPool&) = delete;

    /// RAII lease; the arena is reset and returned to the pool on destruction.
    class Lease {
    public:
        Lease() = default;
        Lease(ArenaPool* pool, std::unique_ptr<Arena> arena)
            : pool_(pool), arena_(std::move(arena)) {}
        Lease(Lease&&) noexcept = default;
        Lease& operator=(Lease&& other) noexcept;
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        ~Lease();

        [[nodiscard]] Arena& arena() const { return *arena_; }
        Arena* operator->() const { return arena_.get(); }

    private:
        void release() noexcept;
        ArenaPool* pool_ = nullptr;
        std::unique_ptr<Arena> arena_;
    };

    [[nodiscard]] Lease acquire();
    [[nodiscard]] std::size_t cached() const;

    /// Per-thread arena (process-wide, created lazily, never shared across threads).
    [[nodiscard]] static Arena& thread_local_arena();

private:
    void give_back(std::unique_ptr<Arena> arena) noexcept;

    std::size_t block_bytes_ = 0;
    std::size_t max_cached_ = 0;
    mutable std::mutex mu_;
    std::vector<std::unique_ptr<Arena>> free_;
};

} // namespace nexusdata
