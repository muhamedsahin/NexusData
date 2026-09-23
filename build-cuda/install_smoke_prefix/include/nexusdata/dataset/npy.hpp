#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "nexusdata/core/ndarray.hpp"
#include "nexusdata/dataset/dataset.hpp"
#include "nexusdata/io/mapped_file.hpp"

namespace nexusdata {

struct NpyArray {
    NDArray view;           // may be non-owning (mmap) or owned clone
    bool owns_storage = true;
    std::string descr;      // e.g. "<f4"
    bool fortran_order = false;
};

/// Load a .npy file. When @p copy is false, returns a non-owning view over mmap
/// (MappedFile must outlive the view — use NpyFile for that).
[[nodiscard]] NpyArray load_npy(const std::string& path, bool copy = true);

/// Parse .npy bytes in memory (for fuzzing / embedded payloads). Always copies.
[[nodiscard]] NpyArray load_npy_from_buffer(const std::uint8_t* data, std::size_t size);

/// Keeps mmap alive for zero-copy NDArray views.
class NpyFile {
public:
    explicit NpyFile(std::string path);

    [[nodiscard]] const NDArray& array() const noexcept { return array_; }
    [[nodiscard]] const std::string& descr() const noexcept { return descr_; }

private:
    MappedFile map_;
    NDArray array_;
    std::string descr_;
};

/// Map-style dataset wrapping a single 2D feature matrix (+ optional 1D labels) from NPY.
class NpyDataset : public Dataset {
public:
    NpyDataset(std::string features_path, std::string labels_path = {});

    [[nodiscard]] std::size_t size() const override;
    [[nodiscard]] Sample get(std::size_t index) const override;

private:
    NpyFile features_;
    std::unique_ptr<NpyFile> labels_;
    std::size_t n_ = 0;
    std::size_t n_features_ = 0;
};

/// Minimal NPZ reader for *stored* (uncompressed) members only.
/// Deflated members throw with a clear message (inflate deferred / optional zlib).
class NpzFile {
public:
    explicit NpzFile(std::string path);

    [[nodiscard]] std::vector<std::string> keys() const;
    [[nodiscard]] bool contains(const std::string& name) const;
    /// Load array by key (copy into owned NDArray).
    [[nodiscard]] NDArray get(const std::string& name) const;

private:
    struct Member {
        std::string name;
        std::uint64_t offset = 0;
        std::uint64_t comp_size = 0;
        std::uint64_t uncomp_size = 0;
        std::uint16_t method = 0;
    };

    MappedFile map_;
    std::unordered_map<std::string, Member> members_;
};

} // namespace nexusdata
