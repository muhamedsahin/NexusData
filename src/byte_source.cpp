#include "nexusdata/io/byte_source.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <system_error>

#include "nexusdata/core/error.hpp"

namespace nexusdata {

std::size_t ByteSource::read_full(void* dst, std::size_t n) {
    auto* p = static_cast<std::uint8_t*>(dst);
    std::size_t got = 0;
    while (got < n) {
        const std::size_t k = read(p + got, n - got);
        if (k == 0) {
            break;
        }
        got += k;
    }
    return got;
}

FileSource::FileSource(const std::string& path) : path_(path) {
    // Narrow (ANSI on Windows) paths, like MappedFile.
#if defined(_MSC_VER)
    if (fopen_s(&f_, path.c_str(), "rb") != 0) {
        f_ = nullptr;
    }
#else
    f_ = std::fopen(path.c_str(), "rb");
#endif
    if (f_ == nullptr) {
        throw IOError("FileSource: cannot open \"" + path + "\"");
    }
    std::error_code ec;
    const auto sz = std::filesystem::file_size(std::filesystem::path(path), ec);
    size_ = ec ? 0 : static_cast<std::uint64_t>(sz);
    // Callers read in large chunks; stdio's own buffer would only add a copy.
    std::setvbuf(f_, nullptr, _IONBF, 0);
}

FileSource::~FileSource() {
    if (f_ != nullptr) {
        std::fclose(f_);
    }
}

std::size_t FileSource::read(void* dst, std::size_t n) {
    if (n == 0) {
        return 0;
    }
    const std::size_t k = std::fread(dst, 1, n, f_);
    if (k == 0 && std::ferror(f_) != 0) {
        throw IOError("FileSource: read failed for \"" + path_ + "\"");
    }
    return k;
}

std::size_t MemorySource::read(void* dst, std::size_t n) {
    const std::size_t k = std::min(n, n_ - off_);
    if (k > 0) {
        std::memcpy(dst, p_ + off_, k);
        off_ += k;
    }
    return k;
}

LineReader::LineReader(ByteSourcePtr source, std::size_t chunk_bytes)
    : src_(std::move(source)), chunk_(std::max<std::size_t>(chunk_bytes, 4096)) {
    if (!src_) {
        throw InvalidArgumentError("LineReader: null source");
    }
}

bool LineReader::fill() {
    if (eof_) {
        return false;
    }
    // Keep the unconsumed tail (a partial line) at the front of the buffer.
    if (begin_ > 0) {
        std::memmove(buf_.data(), buf_.data() + begin_, end_ - begin_);
        end_ -= begin_;
        begin_ = 0;
    }
    if (buf_.size() - end_ < chunk_) {
        buf_.resize(end_ + chunk_);
    }
    const std::size_t k = src_->read(buf_.data() + end_, buf_.size() - end_);
    if (k == 0) {
        eof_ = true;
        return false;
    }
    end_ += k;
    return true;
}

bool LineReader::next(std::string_view& line) {
    std::size_t scan = begin_;
    for (;;) {
        const char* base = buf_.data();
        const void* nl = scan < end_ ? std::memchr(base + scan, '\n', end_ - scan) : nullptr;
        if (nl != nullptr) {
            const std::size_t pos = static_cast<std::size_t>(static_cast<const char*>(nl) - base);
            std::size_t stop = pos;
            if (stop > begin_ && base[stop - 1] == '\r') {
                --stop;
            }
            line = std::string_view(base + begin_, stop - begin_);
            begin_ = pos + 1;
            ++line_no_;
            return true;
        }
        const std::size_t scanned = end_ - begin_;
        if (!fill()) {
            if (begin_ == end_) {
                return false;
            }
            std::size_t stop = end_;
            if (buf_[stop - 1] == '\r') {
                --stop;
            }
            line = std::string_view(buf_.data() + begin_, stop - begin_);
            begin_ = end_;
            ++line_no_;
            return true;
        }
        scan = begin_ + scanned;
    }
}

} // namespace nexusdata
