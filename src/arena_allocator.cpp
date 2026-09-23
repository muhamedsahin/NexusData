#include "nexusdata/core/arena_allocator.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>

#include "nexusdata/core/error.hpp"
#include "nexusdata/core/ndarray.hpp"

namespace nexusdata {

namespace {

bool is_pow2(std::size_t v) noexcept { return v != 0 && (v & (v - 1)) == 0; }

std::size_t align_up(std::size_t v, std::size_t a) noexcept { return (v + a - 1) & ~(a - 1); }

} // namespace

Arena::Arena(std::size_t block_bytes) : block_bytes_(block_bytes) {
    if (block_bytes_ == 0) {
        throw InvalidArgumentError("Arena: block_bytes must be > 0");
    }
}

std::size_t Arena::bytes_reserved() const noexcept {
    std::size_t total = 0;
    for (const auto& b : blocks_) {
        total += b.size;
    }
    return total;
}

void Arena::add_block(std::size_t min_bytes) {
    Block b;
    b.size = std::max(block_bytes_, min_bytes);
    b.mem = make_aligned_buffer(b.size, kDefaultAlignment);
    ++upstream_;
    blocks_.push_back(std::move(b));
}

std::size_t Arena::take(std::size_t bytes, std::size_t alignment) {
    if (!is_pow2(alignment)) {
        throw InvalidArgumentError("Arena: alignment must be a power of two, got " +
                                   std::to_string(alignment));
    }
    if (bytes == 0) {
        bytes = 1;
    }
    // Blocks are 64-byte aligned; larger alignments need slack.
    const std::size_t slack = alignment > kDefaultAlignment ? alignment : 0;
    for (; current_ < blocks_.size(); ++current_) {
        Block& b = blocks_[current_];
        const auto base = reinterpret_cast<std::uintptr_t>(b.mem.get());
        const std::size_t start = align_up(base + b.offset, alignment) - base;
        if (start + bytes <= b.size) {
            used_ += (start - b.offset) + bytes;
            b.offset = start + bytes;
            return current_;
        }
    }
    add_block(bytes + slack);
    current_ = blocks_.size() - 1;
    return take(bytes, alignment);
}

void* Arena::allocate(std::size_t bytes, std::size_t alignment) {
    const std::size_t idx = take(bytes, alignment);
    const Block& b = blocks_[idx];
    return static_cast<std::uint8_t*>(b.mem.get()) + (b.offset - std::max<std::size_t>(bytes, 1));
}

AlignedBuffer Arena::allocate_shared(std::size_t bytes, std::size_t alignment) {
    const std::size_t idx = take(bytes, alignment);
    const Block& b = blocks_[idx];
    void* p = static_cast<std::uint8_t*>(b.mem.get()) + (b.offset - std::max<std::size_t>(bytes, 1));
    return AlignedBuffer(b.mem, p); // aliasing constructor: shares the block's refcount
}

void Arena::reset() {
    std::size_t live_free = 0;
    std::size_t used_blocks = 0;
    std::vector<Block> kept;
    kept.reserve(blocks_.size());
    for (auto& b : blocks_) {
        if (b.offset > 0) {
            ++used_blocks;
        }
        // use_count() == 1: only the arena references the block. Other owners
        // can only appear through this (single-threaded) arena, so the check is
        // not racy; a stale count > 1 merely drops the block conservatively.
        if (b.mem.use_count() == 1) {
            b.offset = 0;
            live_free += b.size;
            kept.push_back(std::move(b));
        }
    }
    blocks_ = std::move(kept);
    if (used_blocks > 1 && blocks_.size() > 1) {
        blocks_.clear();
        add_block(live_free);
    }
    current_ = 0;
    used_ = 0;
}

ArenaPool::ArenaPool(std::size_t block_bytes, std::size_t max_cached)
    : block_bytes_(block_bytes), max_cached_(max_cached) {
    if (block_bytes_ == 0) {
        throw InvalidArgumentError("ArenaPool: block_bytes must be > 0");
    }
}

ArenaPool::Lease& ArenaPool::Lease::operator=(Lease&& other) noexcept {
    if (this != &other) {
        release();
        pool_ = other.pool_;
        arena_ = std::move(other.arena_);
        other.pool_ = nullptr;
    }
    return *this;
}

ArenaPool::Lease::~Lease() { release(); }

void ArenaPool::Lease::release() noexcept {
    if (pool_ != nullptr && arena_) {
        pool_->give_back(std::move(arena_));
    }
    pool_ = nullptr;
}

ArenaPool::Lease ArenaPool::acquire() {
    {
        std::lock_guard lock(mu_);
        if (!free_.empty()) {
            auto a = std::move(free_.back());
            free_.pop_back();
            return Lease(this, std::move(a));
        }
    }
    return Lease(this, std::make_unique<Arena>(block_bytes_));
}

void ArenaPool::give_back(std::unique_ptr<Arena> arena) noexcept {
    try {
        arena->reset();
        std::lock_guard lock(mu_);
        if (free_.size() < max_cached_) {
            free_.push_back(std::move(arena));
        }
    } catch (...) {
        // Dropping the arena is always safe.
    }
}

std::size_t ArenaPool::cached() const {
    std::lock_guard lock(mu_);
    return free_.size();
}

Arena& ArenaPool::thread_local_arena() {
    thread_local Arena arena;
    return arena;
}

// --- NDArray integration --------------------------------------------------------

NDArray NDArray::from_arena(Arena& arena, Shape shape, DType dtype, bool zero) {
    const std::size_t bytes = ::nexusdata::numel(shape) * size_of(dtype);
    AlignedBuffer buf;
    if (bytes > 0) {
        buf = arena.allocate_shared(bytes);
        if (zero) {
            std::memset(buf.get(), 0, bytes);
        }
    }
    return NDArray::from_shared(std::move(buf), std::move(shape), dtype, Device::host());
}

} // namespace nexusdata
