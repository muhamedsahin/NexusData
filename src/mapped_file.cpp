#include "nexusdata/io/mapped_file.hpp"

#include <string>
#include <utility>

#include "nexusdata/core/error.hpp"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace nexusdata {

MappedFile::MappedFile(const std::string& path) {
#if defined(_WIN32)
    file_handle_ = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                               OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file_handle_ == INVALID_HANDLE_VALUE) {
        file_handle_ = nullptr;
        throw IOError("MappedFile: cannot open \"" + path + "\"");
    }
    LARGE_INTEGER li{};
    if (!GetFileSizeEx(static_cast<HANDLE>(file_handle_), &li)) {
        close();
        throw IOError("MappedFile: GetFileSizeEx failed for \"" + path + "\"");
    }
    size_ = static_cast<std::size_t>(li.QuadPart);
    if (size_ == 0) {
        // Empty file: leave data_ null.
        return;
    }
    map_handle_ = CreateFileMappingA(static_cast<HANDLE>(file_handle_), nullptr,
                                     PAGE_READONLY, 0, 0, nullptr);
    if (!map_handle_) {
        close();
        throw IOError("MappedFile: CreateFileMapping failed for \"" + path + "\"");
    }
    void* view = MapViewOfFile(static_cast<HANDLE>(map_handle_), FILE_MAP_READ, 0, 0, 0);
    if (!view) {
        close();
        throw IOError("MappedFile: MapViewOfFile failed for \"" + path + "\"");
    }
    data_ = static_cast<const std::uint8_t*>(view);
#else
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        throw IOError("MappedFile: cannot open \"" + path + "\"");
    }
    struct stat st {};
    if (fstat(fd_, &st) != 0) {
        close();
        throw IOError("MappedFile: fstat failed for \"" + path + "\"");
    }
    size_ = static_cast<std::size_t>(st.st_size);
    if (size_ == 0) {
        return;
    }
    void* view = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (view == MAP_FAILED) {
        close();
        throw IOError("MappedFile: mmap failed for \"" + path + "\"");
    }
    data_ = static_cast<const std::uint8_t*>(view);
#if defined(MADV_SEQUENTIAL)
    ::madvise(const_cast<std::uint8_t*>(data_), size_, MADV_SEQUENTIAL);
#endif
#endif
}

MappedFile::~MappedFile() {
    close();
}

MappedFile::MappedFile(MappedFile&& other) noexcept {
    *this = std::move(other);
}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    close();
    data_ = other.data_;
    size_ = other.size_;
#if defined(_WIN32)
    file_handle_ = other.file_handle_;
    map_handle_ = other.map_handle_;
    other.file_handle_ = nullptr;
    other.map_handle_ = nullptr;
#else
    fd_ = other.fd_;
    other.fd_ = -1;
#endif
    other.data_ = nullptr;
    other.size_ = 0;
    return *this;
}

void MappedFile::close() noexcept {
#if defined(_WIN32)
    if (data_) {
        UnmapViewOfFile(data_);
        data_ = nullptr;
    }
    if (map_handle_) {
        CloseHandle(static_cast<HANDLE>(map_handle_));
        map_handle_ = nullptr;
    }
    if (file_handle_) {
        CloseHandle(static_cast<HANDLE>(file_handle_));
        file_handle_ = nullptr;
    }
#else
    if (data_ && size_ > 0) {
        ::munmap(const_cast<std::uint8_t*>(data_), size_);
    }
    data_ = nullptr;
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
#endif
    size_ = 0;
}

} // namespace nexusdata
