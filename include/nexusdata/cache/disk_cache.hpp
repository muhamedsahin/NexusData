#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "nexusdata/cache/lru_cache.hpp"
#include "nexusdata/core/dtype.hpp"
#include "nexusdata/core/error.hpp"
#include "nexusdata/core/ndarray.hpp"
#include "nexusdata/core/shape.hpp"

namespace nexusdata {

/// Disk cache for preprocessed NDArray blobs under a directory.
/// Key = hash(content_key + pipeline_key). Files named <hex64>.bin
/// Format: magic, dtype, rank, shape..., nbytes, payload.
class DiskCache {
public:
    explicit DiskCache(std::filesystem::path root) : root_(std::move(root)) {
        std::filesystem::create_directories(root_);
    }

    [[nodiscard]] std::filesystem::path path_for(std::uint64_t key) const {
        std::ostringstream oss;
        oss << std::hex << key << ".bin";
        return root_ / oss.str();
    }

    [[nodiscard]] bool contains(std::uint64_t key) const {
        return std::filesystem::exists(path_for(key));
    }

    void put(std::uint64_t key, const NDArray& arr) {
        const auto path = path_for(key);
        const auto tmp = path.string() + ".tmp";
        std::ofstream out(tmp, std::ios::binary);
        if (!out) {
            throw IOError("DiskCache::put: cannot write temp file");
        }
        const char magic[8] = {'N', 'D', 'C', 'A', 'C', 'H', 'E', '1'};
        out.write(magic, 8);
        const auto dt = static_cast<std::uint8_t>(arr.dtype());
        out.write(reinterpret_cast<const char*>(&dt), 1);
        const auto rank = static_cast<std::uint32_t>(arr.shape().size());
        out.write(reinterpret_cast<const char*>(&rank), 4);
        for (std::size_t d : arr.shape()) {
            const auto dd = static_cast<std::uint64_t>(d);
            out.write(reinterpret_cast<const char*>(&dd), 8);
        }
        const auto nbytes = static_cast<std::uint64_t>(arr.nbytes());
        out.write(reinterpret_cast<const char*>(&nbytes), 8);
        if (nbytes > 0) {
            out.write(static_cast<const char*>(arr.data()), static_cast<std::streamsize>(nbytes));
        }
        out.close();
        std::filesystem::rename(tmp, path);
    }

    [[nodiscard]] NDArray get(std::uint64_t key) const {
        const auto path = path_for(key);
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            throw IOError("DiskCache::get: missing key");
        }
        char magic[8];
        in.read(magic, 8);
        if (std::string(magic, 8) != "NDCACHE1") {
            throw IOError("DiskCache::get: bad magic");
        }
        std::uint8_t dt = 0;
        in.read(reinterpret_cast<char*>(&dt), 1);
        std::uint32_t rank = 0;
        in.read(reinterpret_cast<char*>(&rank), 4);
        Shape shape(rank);
        for (std::uint32_t i = 0; i < rank; ++i) {
            std::uint64_t d = 0;
            in.read(reinterpret_cast<char*>(&d), 8);
            shape[i] = static_cast<std::size_t>(d);
        }
        std::uint64_t nbytes = 0;
        in.read(reinterpret_cast<char*>(&nbytes), 8);
        NDArray out(shape, static_cast<DType>(dt));
        if (nbytes != out.nbytes()) {
            throw IOError("DiskCache::get: size mismatch");
        }
        if (nbytes > 0) {
            in.read(static_cast<char*>(out.data()), static_cast<std::streamsize>(nbytes));
        }
        return out;
    }

    [[nodiscard]] static std::uint64_t make_key(const std::string& content_key,
                                                const std::string& pipeline_key) {
        const auto a = fnv1a64_str(content_key);
        const auto b = fnv1a64_str(pipeline_key);
        return a ^ (b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2));
    }

private:
    std::filesystem::path root_;
};

} // namespace nexusdata
