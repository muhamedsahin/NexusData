#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace nexusdata {

/// Pull-based sequential byte stream (files, decompressors, remote objects).
/// Thread-safety: none; use one source per reader thread.
class ByteSource {
public:
    virtual ~ByteSource() = default;

    /// Read up to @p n bytes into @p dst. Returns 0 only at end of stream; may
    /// return fewer than @p n bytes before that. Throws IOError on failure.
    [[nodiscard]] virtual std::size_t read(void* dst, std::size_t n) = 0;

    /// Total stream length when known up front (plain files); nullopt otherwise.
    [[nodiscard]] virtual std::optional<std::uint64_t> size_hint() const { return std::nullopt; }

    /// Read until @p n bytes or end of stream; returns the byte count read.
    std::size_t read_full(void* dst, std::size_t n);
};

using ByteSourcePtr = std::unique_ptr<ByteSource>;

/// Buffered sequential file reader.
class FileSource final : public ByteSource {
public:
    explicit FileSource(const std::string& path);
    ~FileSource() override;
    FileSource(const FileSource&) = delete;
    FileSource& operator=(const FileSource&) = delete;

    [[nodiscard]] std::size_t read(void* dst, std::size_t n) override;
    [[nodiscard]] std::optional<std::uint64_t> size_hint() const override { return size_; }

private:
    std::FILE* f_ = nullptr;
    std::string path_;
    std::uint64_t size_ = 0;
};

/// Non-owning view over caller memory (must outlive the source).
class MemorySource final : public ByteSource {
public:
    MemorySource(const void* data, std::size_t size) noexcept
        : p_(static_cast<const std::uint8_t*>(data)), n_(size) {}
    [[nodiscard]] std::size_t read(void* dst, std::size_t n) override;
    [[nodiscard]] std::optional<std::uint64_t> size_hint() const override { return n_; }

private:
    const std::uint8_t* p_;
    std::size_t n_;
    std::size_t off_ = 0;
};

/// Newline-delimited records over any ByteSource without loading it whole
/// (e.g. a multi-GB `.jsonl.zst`). Handles "\n" and "\r\n"; the last line may
/// lack a terminator.
class LineReader {
public:
    explicit LineReader(ByteSourcePtr source, std::size_t chunk_bytes = std::size_t{1} << 20);

    /// Next line without its terminator; the view stays valid until the next call.
    /// Returns false at end of stream.
    [[nodiscard]] bool next(std::string_view& line);

    /// 1-based number of the line last returned by next().
    [[nodiscard]] std::size_t line_number() const noexcept { return line_no_; }

private:
    bool fill();

    ByteSourcePtr src_;
    std::vector<char> buf_;
    std::size_t begin_ = 0;
    std::size_t end_ = 0;
    std::size_t chunk_;
    std::size_t line_no_ = 0;
    bool eof_ = false;
};

} // namespace nexusdata
