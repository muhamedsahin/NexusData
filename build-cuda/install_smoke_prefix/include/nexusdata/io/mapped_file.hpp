#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace nexusdata {

/// Read-only memory-mapped file view.
/// Thread-safety: immutable after construction; concurrent reads OK.
class MappedFile {
public:
    MappedFile() = default;
    explicit MappedFile(const std::string& path);
    ~MappedFile();

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    MappedFile(MappedFile&& other) noexcept;
    MappedFile& operator=(MappedFile&& other) noexcept;

    [[nodiscard]] const std::uint8_t* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] bool empty() const noexcept { return data_ == nullptr || size_ == 0; }

private:
    void close() noexcept;

    const std::uint8_t* data_ = nullptr;
    std::size_t size_ = 0;
#if defined(_WIN32)
    void* file_handle_ = nullptr;
    void* map_handle_ = nullptr;
#else
    int fd_ = -1;
#endif
};

} // namespace nexusdata
