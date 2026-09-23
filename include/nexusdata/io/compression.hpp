#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "nexusdata/io/byte_source.hpp"
#include "nexusdata/io/mapped_file.hpp"

namespace nexusdata {

/// Transparent compression layer (v2.0, plan 7.5).
///
/// Every codec is an optional build dependency (NEXUSDATA_WITH_ZLIB / _ZSTD / _LZ4 /
/// _BZIP2 / _LZMA). Using a codec that was not compiled in throws an IOError that
/// names the CMake flag to turn on.
enum class Codec : std::uint8_t {
    None = 0,
    Gzip,  ///< RFC 1952, including multi-member files (pigz, bgzip).
    Zlib,  ///< RFC 1950 stream.
    Zstd,  ///< Zstandard frames, including multi-frame files (pzstd, zstd -T).
    Lz4,   ///< LZ4 frame format (not the legacy block format).
    Bzip2, ///< bzip2, including multi-stream files (pbzip2).
    Xz,    ///< .xz container, including concatenated streams.
};

[[nodiscard]] std::string_view codec_name(Codec codec) noexcept;

/// True when support for @p codec is compiled in (Codec::None is always available).
[[nodiscard]] bool codec_available(Codec codec) noexcept;

/// Codec implied by the file suffix (".gz", ".tgz", ".zst", ".lz4", ".bz2", ".xz", ...);
/// Codec::None when the suffix is not a compression suffix. Case-insensitive.
[[nodiscard]] Codec codec_from_path(std::string_view path) noexcept;

/// Codec identified by magic bytes; Codec::None when unrecognised. Zlib streams have
/// no reliable magic and are only recognised by suffix.
[[nodiscard]] Codec sniff_codec(const void* data, std::size_t size) noexcept;

/// Path with its compression suffix removed ("a.csv.gz" -> "a.csv", "b.tgz" -> "b.tar"),
/// used to find the inner format.
[[nodiscard]] std::string strip_codec_suffix(std::string_view path);

/// Streaming decoder. Concatenated members / frames / streams decode as one stream.
class Decompressor {
public:
    struct Step {
        std::size_t consumed = 0; ///< input bytes used
        std::size_t produced = 0; ///< output bytes written
        bool done = false;        ///< stream complete (only after input_end)
    };
    virtual ~Decompressor() = default;

    /// Decode from @p in into @p out. Pass @p input_end once the final input bytes
    /// are included. Throws IOError on corrupt or truncated data.
    [[nodiscard]] virtual Step decode(const std::uint8_t* in, std::size_t in_size,
                                      std::uint8_t* out, std::size_t out_capacity,
                                      bool input_end) = 0;
};

[[nodiscard]] std::unique_ptr<Decompressor> make_decompressor(Codec codec);

struct CompressOptions {
    /// Codec level; < 0 selects the codec default.
    int level = -1;
    /// Zstd / LZ4 / Gzip: split the input into independent frames (members) of this
    /// many uncompressed bytes (0 = one frame). Independent zstd frames with recorded
    /// sizes are decoded in parallel by read_input().
    std::size_t frame_bytes = 0;
};

/// One-shot compression (writers, tests, benchmarks).
[[nodiscard]] std::vector<std::uint8_t> compress(Codec codec, const void* data, std::size_t size,
                                                 CompressOptions options = {});

/// One-shot decompression into a new buffer.
[[nodiscard]] std::vector<std::uint8_t> decompress(Codec codec, const void* data, std::size_t size);

/// Raw DEFLATE (RFC 1951, no zlib/gzip header), e.g. ZIP method-8 members such as
/// np.savez_compressed entries. @p expected_size is the recorded uncompressed size;
/// a mismatch throws IOError. Requires NEXUSDATA_WITH_ZLIB.
[[nodiscard]] std::vector<std::uint8_t> inflate_raw(const void* data, std::size_t size,
                                                    std::size_t expected_size);

/// ByteSource that decodes @p inner on the fly, one chunk at a time.
class DecompressingSource final : public ByteSource {
public:
    DecompressingSource(ByteSourcePtr inner, Codec codec,
                        std::size_t chunk_bytes = std::size_t{256} << 10);
    [[nodiscard]] std::size_t read(void* dst, std::size_t n) override;

private:
    ByteSourcePtr inner_;
    std::unique_ptr<Decompressor> dec_;
    std::vector<std::uint8_t> in_;
    std::size_t in_begin_ = 0;
    std::size_t in_end_ = 0;
    bool inner_eof_ = false;
    bool done_ = false;
};

/// Open @p path as a stream, decompressing on the fly. With @p codec unset the codec
/// is sniffed from the magic bytes, then from the suffix.
[[nodiscard]] ByteSourcePtr open_source(const std::string& path,
                                        std::optional<Codec> codec = std::nullopt);

/// Wrap an arbitrary stream (e.g. a remote object) in a decoder for @p codec
/// (Codec::None returns @p source unchanged).
[[nodiscard]] ByteSourcePtr decompress_source(ByteSourcePtr source, Codec codec);

class InputBytes;

/// Map or decode @p path (codec as in open_source()). @p num_threads bounds the
/// parallel decode of multi-frame zstd files (0 = hardware concurrency).
[[nodiscard]] InputBytes read_input(const std::string& path,
                                    std::optional<Codec> codec = std::nullopt,
                                    unsigned num_threads = 0);

/// Whole-file contents as one contiguous buffer, for parsers that need random
/// access: an uncompressed file is memory-mapped (zero copy); a compressed file is
/// decoded straight into memory (no temporary file).
class InputBytes {
public:
    InputBytes() = default;
    InputBytes(InputBytes&&) noexcept = default;
    InputBytes& operator=(InputBytes&&) noexcept = default;
    InputBytes(const InputBytes&) = delete;
    InputBytes& operator=(const InputBytes&) = delete;

    [[nodiscard]] const std::uint8_t* data() const noexcept {
        return owned_ ? owned_.get() : map_.data();
    }
    [[nodiscard]] std::size_t size() const noexcept { return owned_ ? owned_size_ : map_.size(); }
    [[nodiscard]] bool empty() const noexcept { return size() == 0; }
    /// Codec that was decoded (Codec::None: the bytes are the mapped file).
    [[nodiscard]] Codec codec() const noexcept { return codec_; }

private:
    friend InputBytes read_input(const std::string& path, std::optional<Codec> codec,
                                 unsigned num_threads);
    MappedFile map_;
    std::unique_ptr<std::uint8_t[]> owned_;
    std::size_t owned_size_ = 0;
    Codec codec_ = Codec::None;
};

} // namespace nexusdata
