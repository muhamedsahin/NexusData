// bzlib.h includes windows.h, whose min/max macros break std::min / std::max.
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "nexusdata/io/compression.hpp"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cstring>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>

#include "nexusdata/backend/thread_pool.hpp"
#include "nexusdata/core/error.hpp"

#if defined(NEXUSDATA_WITH_ZLIB)
#include <zlib.h>
#endif
#if defined(NEXUSDATA_WITH_ZSTD)
#include <zstd.h>
#endif
#if defined(NEXUSDATA_WITH_LZ4)
#include <lz4frame.h>
#endif
#if defined(NEXUSDATA_WITH_BZIP2)
#include <bzlib.h>
#endif
#if defined(NEXUSDATA_WITH_LZMA)
#include <lzma.h>
#endif

namespace nexusdata {

namespace {

// zlib / bzip2 counters are 32-bit: feed them bounded slices.
constexpr std::size_t kMaxSlice = std::size_t{1} << 30;

const char* codec_flag(Codec c) noexcept {
    switch (c) {
        case Codec::Gzip:
        case Codec::Zlib: return "NEXUSDATA_WITH_ZLIB";
        case Codec::Zstd: return "NEXUSDATA_WITH_ZSTD";
        case Codec::Lz4: return "NEXUSDATA_WITH_LZ4";
        case Codec::Bzip2: return "NEXUSDATA_WITH_BZIP2";
        case Codec::Xz: return "NEXUSDATA_WITH_LZMA";
        case Codec::None: break;
    }
    return "";
}

void require_codec(Codec c) {
    if (!codec_available(c)) {
        throw IOError(std::string("compression: ") + std::string(codec_name(c)) +
                      " support is not compiled in; reconfigure with -D" + codec_flag(c) + "=ON");
    }
}

std::string lower(std::string_view s) {
    std::string out(s);
    for (char& ch : out) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return out;
}

std::string_view suffix_of(std::string_view path) noexcept {
    const auto slash = path.find_last_of("/\\");
    const auto dot = path.find_last_of('.');
    if (dot == std::string_view::npos || (slash != std::string_view::npos && dot < slash)) {
        return {};
    }
    return path.substr(dot);
}

/// Shared decode loop: codecs implement one library call per step().
class StepDecoder : public Decompressor {
public:
    explicit StepDecoder(const char* name) : name_(name) {}

    Step decode(const std::uint8_t* in, std::size_t in_size, std::uint8_t* out,
                std::size_t out_capacity, bool input_end) final {
        Step st;
        for (;;) {
            const std::size_t c0 = st.consumed;
            const std::size_t p0 = st.produced;
            if (step(in, in_size, st.consumed, out, out_capacity, st.produced, input_end)) {
                st.done = true;
                return st;
            }
            if (st.produced == out_capacity) {
                return st;
            }
            if (st.consumed == c0 && st.produced == p0) {
                if (input_end) {
                    throw IOError(std::string(name_) + ": truncated stream");
                }
                return st;
            }
        }
    }

protected:
    /// Advance once; return true when the whole stream is complete (requires
    /// input_end and all input consumed).
    virtual bool step(const std::uint8_t* in, std::size_t n, std::size_t& consumed,
                      std::uint8_t* out, std::size_t cap, std::size_t& produced, bool end) = 0;

    /// Between members: skip zero padding (tar.gz writers pad), then require @p magic.
    /// Returns false when no further member starts in the available input.
    bool next_member(const std::uint8_t* in, std::size_t n, std::size_t& consumed,
                     std::uint8_t magic) const {
        while (consumed < n && in[consumed] == 0) {
            ++consumed;
        }
        if (consumed == n) {
            return false;
        }
        if (in[consumed] != magic) {
            throw IOError(std::string(name_) + ": trailing garbage after end of stream");
        }
        return true;
    }

    const char* name_;
};

#if defined(NEXUSDATA_WITH_ZLIB)
class ZlibDecoder final : public StepDecoder {
public:
    explicit ZlibDecoder(bool gzip) : StepDecoder(gzip ? "gzip" : "zlib"), gzip_(gzip) {
        if (inflateInit2(&z_, gzip ? 16 + MAX_WBITS : MAX_WBITS) != Z_OK) {
            throw IOError("zlib: inflateInit2 failed");
        }
    }
    ~ZlibDecoder() override { inflateEnd(&z_); }

private:
    bool step(const std::uint8_t* in, std::size_t n, std::size_t& consumed, std::uint8_t* out,
              std::size_t cap, std::size_t& produced, bool end) override {
        if (member_done_) {
            if (!gzip_) {
                if (consumed < n) throw IOError("zlib: trailing data after end of stream");
                return end;
            }
            if (!next_member(in, n, consumed, 0x1f)) {
                return end;
            }
            inflateReset(&z_);
            member_done_ = false;
        }
        const std::size_t in_n = std::min(n - consumed, kMaxSlice);
        const std::size_t out_n = std::min(cap - produced, kMaxSlice);
        z_.next_in = const_cast<Bytef*>(in + consumed);
        z_.avail_in = static_cast<uInt>(in_n);
        z_.next_out = out + produced;
        z_.avail_out = static_cast<uInt>(out_n);
        const int r = inflate(&z_, Z_NO_FLUSH);
        consumed += in_n - z_.avail_in;
        produced += out_n - z_.avail_out;
        if (r == Z_STREAM_END) {
            member_done_ = true;
            return false; // the next step checks for padding / another member
        }
        if (r == Z_OK || r == Z_BUF_ERROR) {
            return false;
        }
        throw IOError(std::string(name_) + ": corrupt data (" + (z_.msg ? z_.msg : "error") + ")");
    }

    z_stream z_{};
    bool gzip_;
    bool member_done_ = false;
};
#endif

#if defined(NEXUSDATA_WITH_ZSTD)
class ZstdDecoder final : public StepDecoder {
public:
    ZstdDecoder() : StepDecoder("zstd"), d_(ZSTD_createDCtx()) {
        if (d_ == nullptr) throw IOError("zstd: out of memory");
    }
    ~ZstdDecoder() override { ZSTD_freeDCtx(d_); }

private:
    bool step(const std::uint8_t* in, std::size_t n, std::size_t& consumed, std::uint8_t* out,
              std::size_t cap, std::size_t& produced, bool end) override {
        // After a complete frame the decoder asks for the next header, so an empty
        // call must not be mistaken for a truncated frame.
        if (consumed == n && boundary_) {
            return end;
        }
        ZSTD_inBuffer ib{in, n, consumed};
        ZSTD_outBuffer ob{out, cap, produced};
        const std::size_t r = ZSTD_decompressStream(d_, &ob, &ib);
        if (ZSTD_isError(r)) {
            throw IOError(std::string("zstd: ") + ZSTD_getErrorName(r));
        }
        consumed = ib.pos;
        produced = ob.pos;
        boundary_ = r == 0;
        return boundary_ && end && consumed == n;
    }

    ZSTD_DCtx* d_;
    bool boundary_ = false;
};
#endif

#if defined(NEXUSDATA_WITH_LZ4)
class Lz4Decoder final : public StepDecoder {
public:
    Lz4Decoder() : StepDecoder("lz4") {
        if (LZ4F_isError(LZ4F_createDecompressionContext(&d_, LZ4F_VERSION))) {
            throw IOError("lz4: cannot create decompression context");
        }
    }
    ~Lz4Decoder() override { LZ4F_freeDecompressionContext(d_); }

private:
    bool step(const std::uint8_t* in, std::size_t n, std::size_t& consumed, std::uint8_t* out,
              std::size_t cap, std::size_t& produced, bool end) override {
        if (consumed == n && boundary_) {
            return end; // see ZstdDecoder::step
        }
        std::size_t src = n - consumed;
        std::size_t dst = cap - produced;
        const std::size_t r =
            LZ4F_decompress(d_, out + produced, &dst, in + consumed, &src, nullptr);
        if (LZ4F_isError(r)) {
            throw IOError(std::string("lz4: ") + LZ4F_getErrorName(r));
        }
        consumed += src;
        produced += dst;
        boundary_ = r == 0;
        return boundary_ && end && consumed == n;
    }

    LZ4F_dctx* d_ = nullptr;
    bool boundary_ = false;
};
#endif

#if defined(NEXUSDATA_WITH_BZIP2)
class Bzip2Decoder final : public StepDecoder {
public:
    Bzip2Decoder() : StepDecoder("bzip2") { init(); }
    ~Bzip2Decoder() override { BZ2_bzDecompressEnd(&b_); }

private:
    void init() {
        b_ = bz_stream{};
        if (BZ2_bzDecompressInit(&b_, 0, 0) != BZ_OK) {
            throw IOError("bzip2: BZ2_bzDecompressInit failed");
        }
    }

    bool step(const std::uint8_t* in, std::size_t n, std::size_t& consumed, std::uint8_t* out,
              std::size_t cap, std::size_t& produced, bool end) override {
        if (stream_done_) {
            if (!next_member(in, n, consumed, 'B')) {
                return end;
            }
            BZ2_bzDecompressEnd(&b_);
            init();
            stream_done_ = false;
        }
        const std::size_t in_n = std::min(n - consumed, kMaxSlice);
        const std::size_t out_n = std::min(cap - produced, kMaxSlice);
        b_.next_in = const_cast<char*>(reinterpret_cast<const char*>(in + consumed));
        b_.avail_in = static_cast<unsigned>(in_n);
        b_.next_out = reinterpret_cast<char*>(out + produced);
        b_.avail_out = static_cast<unsigned>(out_n);
        const int r = BZ2_bzDecompress(&b_);
        consumed += in_n - b_.avail_in;
        produced += out_n - b_.avail_out;
        if (r == BZ_STREAM_END) {
            stream_done_ = true;
            return false;
        }
        if (r == BZ_OK) {
            return false;
        }
        throw IOError("bzip2: corrupt data (code " + std::to_string(r) + ")");
    }

    bz_stream b_{};
    bool stream_done_ = false;
};
#endif

#if defined(NEXUSDATA_WITH_LZMA)
class XzDecoder final : public StepDecoder {
public:
    XzDecoder() : StepDecoder("xz") {
        if (lzma_stream_decoder(&s_, UINT64_MAX, LZMA_CONCATENATED) != LZMA_OK) {
            throw IOError("xz: lzma_stream_decoder failed");
        }
    }
    ~XzDecoder() override { lzma_end(&s_); }

private:
    bool step(const std::uint8_t* in, std::size_t n, std::size_t& consumed, std::uint8_t* out,
              std::size_t cap, std::size_t& produced, bool end) override {
        s_.next_in = in + consumed;
        s_.avail_in = n - consumed;
        s_.next_out = out + produced;
        s_.avail_out = cap - produced;
        const lzma_ret r = lzma_code(&s_, end ? LZMA_FINISH : LZMA_RUN);
        consumed = n - s_.avail_in;
        produced = cap - s_.avail_out;
        if (r == LZMA_STREAM_END) {
            return true;
        }
        if (r == LZMA_OK || r == LZMA_BUF_ERROR) {
            return false;
        }
        throw IOError("xz: corrupt data (lzma_ret " + std::to_string(static_cast<int>(r)) + ")");
    }

    lzma_stream s_ = LZMA_STREAM_INIT;
};
#endif

// --- one-shot compression --------------------------------------------------------------

void append_frames(std::vector<std::uint8_t>& out, const std::uint8_t* p, std::size_t n,
                   std::size_t frame_bytes,
                   const std::function<void(std::vector<std::uint8_t>&, const std::uint8_t*,
                                            std::size_t)>& one) {
    if (frame_bytes == 0 || n <= frame_bytes) {
        one(out, p, n);
        return;
    }
    for (std::size_t off = 0; off < n; off += frame_bytes) {
        one(out, p + off, std::min(frame_bytes, n - off));
    }
}

#if defined(NEXUSDATA_WITH_ZLIB)
void deflate_one(std::vector<std::uint8_t>& out, const std::uint8_t* p, std::size_t n, int level,
                 bool gzip) {
    z_stream z{};
    if (deflateInit2(&z, level < 0 ? Z_DEFAULT_COMPRESSION : level, Z_DEFLATED,
                     gzip ? 16 + MAX_WBITS : MAX_WBITS, 8, Z_DEFAULT_STRATEGY) != Z_OK) {
        throw IOError("zlib: deflateInit2 failed");
    }
    std::size_t off = 0;
    int r = Z_OK;
    do {
        const std::size_t in_n = std::min(n - off, kMaxSlice);
        z.next_in = const_cast<Bytef*>(p + off);
        z.avail_in = static_cast<uInt>(in_n);
        const bool last = off + in_n == n;
        do {
            const std::size_t base = out.size();
            out.resize(base + (std::size_t{1} << 20));
            z.next_out = out.data() + base;
            z.avail_out = static_cast<uInt>(out.size() - base);
            r = deflate(&z, last ? Z_FINISH : Z_NO_FLUSH);
            out.resize(out.size() - z.avail_out);
            if (r == Z_STREAM_ERROR) {
                deflateEnd(&z);
                throw IOError("zlib: deflate failed");
            }
        } while (z.avail_out == 0 || (last && r != Z_STREAM_END));
        off += in_n;
    } while (off < n);
    deflateEnd(&z);
}
#endif

/// Multi-frame zstd with recorded sizes decodes frame-parallel into one buffer.
bool parallel_zstd(const std::uint8_t* src, std::size_t n, unsigned threads,
                   std::unique_ptr<std::uint8_t[]>& out, std::size_t& out_size) {
#if defined(NEXUSDATA_WITH_ZSTD)
    struct Frame {
        std::size_t src_off, src_len, dst_off, dst_len;
    };
    std::vector<Frame> frames;
    std::size_t total = 0;
    for (std::size_t off = 0; off < n;) {
        const std::size_t len = ZSTD_findFrameCompressedSize(src + off, n - off);
        if (ZSTD_isError(len)) return false;
        const unsigned long long dsz = ZSTD_getFrameContentSize(src + off, len);
        if (dsz == ZSTD_CONTENTSIZE_UNKNOWN || dsz == ZSTD_CONTENTSIZE_ERROR) return false;
        if (dsz > SIZE_MAX - total) return false;
        frames.push_back({off, len, total, static_cast<std::size_t>(dsz)});
        total += static_cast<std::size_t>(dsz);
        off += len;
    }
    if (frames.size() < 2) return false;
    std::unique_ptr<std::uint8_t[]> buf(new std::uint8_t[std::max<std::size_t>(total, 1)]);
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    ThreadPool pool(std::min<std::size_t>(threads == 0 ? hw : threads, frames.size()));
    std::mutex mu;
    std::exception_ptr err;
    pool.parallel_for(frames.size(), [&](std::size_t i) {
        try {
            thread_local struct Ctx {
                ZSTD_DCtx* d = ZSTD_createDCtx();
                ~Ctx() { ZSTD_freeDCtx(d); }
            } ctx;
            const Frame& f = frames[i];
            const std::size_t r = ZSTD_decompressDCtx(ctx.d, buf.get() + f.dst_off, f.dst_len,
                                                      src + f.src_off, f.src_len);
            if (ZSTD_isError(r) || r != f.dst_len) {
                throw IOError(std::string("zstd: frame ") + std::to_string(i) + ": " +
                              (ZSTD_isError(r) ? ZSTD_getErrorName(r) : "size mismatch"));
            }
        } catch (...) {
            std::lock_guard lock(mu);
            if (!err) err = std::current_exception();
        }
    });
    if (err) std::rethrow_exception(err);
    out = std::move(buf);
    out_size = total;
    return true;
#else
    (void)src; (void)n; (void)threads; (void)out; (void)out_size;
    return false;
#endif
}

/// Decoded-size estimate used to size the output buffer up front.
std::size_t size_hint(Codec c, const std::uint8_t* p, std::size_t n) {
    const std::size_t guess = n > SIZE_MAX / 4 ? n : 4 * n + (std::size_t{64} << 10);
#if defined(NEXUSDATA_WITH_ZSTD)
    if (c == Codec::Zstd) {
        // Sum of recorded frame sizes (stable API only; the experimental
        // ZSTD_findDecompressedSize is not exported by shared system builds).
        std::size_t total = 0;
        for (std::size_t off = 0; off < n;) {
            const std::size_t len = ZSTD_findFrameCompressedSize(p + off, n - off);
            if (ZSTD_isError(len)) return guess;
            const unsigned long long d = ZSTD_getFrameContentSize(p + off, len);
            if (d == ZSTD_CONTENTSIZE_UNKNOWN || d == ZSTD_CONTENTSIZE_ERROR || d > SIZE_MAX - total) {
                return guess;
            }
            total += static_cast<std::size_t>(d);
            off += len;
        }
        return total;
    }
#endif
    if (c == Codec::Gzip && n >= 18) {
        // ISIZE trailer (mod 2^32) of the last member: exact for the common single member.
        const std::uint32_t isize = static_cast<std::uint32_t>(p[n - 4]) |
                                    (static_cast<std::uint32_t>(p[n - 3]) << 8) |
                                    (static_cast<std::uint32_t>(p[n - 2]) << 16) |
                                    (static_cast<std::uint32_t>(p[n - 1]) << 24);
        if (isize >= n / 2) return isize;
    }
    return guess;
}

} // namespace

std::string_view codec_name(Codec codec) noexcept {
    switch (codec) {
        case Codec::None: return "none";
        case Codec::Gzip: return "gzip";
        case Codec::Zlib: return "zlib";
        case Codec::Zstd: return "zstd";
        case Codec::Lz4: return "lz4";
        case Codec::Bzip2: return "bzip2";
        case Codec::Xz: return "xz";
    }
    return "unknown";
}

bool codec_available(Codec codec) noexcept {
    switch (codec) {
        case Codec::None: return true;
        case Codec::Gzip:
        case Codec::Zlib:
#if defined(NEXUSDATA_WITH_ZLIB)
            return true;
#else
            return false;
#endif
        case Codec::Zstd:
#if defined(NEXUSDATA_WITH_ZSTD)
            return true;
#else
            return false;
#endif
        case Codec::Lz4:
#if defined(NEXUSDATA_WITH_LZ4)
            return true;
#else
            return false;
#endif
        case Codec::Bzip2:
#if defined(NEXUSDATA_WITH_BZIP2)
            return true;
#else
            return false;
#endif
        case Codec::Xz:
#if defined(NEXUSDATA_WITH_LZMA)
            return true;
#else
            return false;
#endif
    }
    return false;
}

Codec codec_from_path(std::string_view path) noexcept {
    const std::string ext = lower(suffix_of(path));
    if (ext == ".gz" || ext == ".gzip" || ext == ".tgz") return Codec::Gzip;
    if (ext == ".zz" || ext == ".zlib") return Codec::Zlib;
    if (ext == ".zst" || ext == ".zstd" || ext == ".tzst") return Codec::Zstd;
    if (ext == ".lz4") return Codec::Lz4;
    if (ext == ".bz2" || ext == ".bzip2" || ext == ".tbz" || ext == ".tbz2") return Codec::Bzip2;
    if (ext == ".xz" || ext == ".txz") return Codec::Xz;
    return Codec::None;
}

Codec sniff_codec(const void* data, std::size_t size) noexcept {
    const auto* p = static_cast<const std::uint8_t*>(data);
    if (p == nullptr) return Codec::None;
    if (size >= 2 && p[0] == 0x1f && p[1] == 0x8b) return Codec::Gzip;
    if (size >= 4 && p[0] == 0x28 && p[1] == 0xb5 && p[2] == 0x2f && p[3] == 0xfd) return Codec::Zstd;
    // Zstd skippable frame: magic 0x184D2A5?.
    if (size >= 4 && (p[0] & 0xf0) == 0x50 && p[1] == 0x2a && p[2] == 0x4d && p[3] == 0x18) {
        return Codec::Zstd;
    }
    if (size >= 4 && p[0] == 0x04 && p[1] == 0x22 && p[2] == 0x4d && p[3] == 0x18) return Codec::Lz4;
    if (size >= 4 && p[0] == 'B' && p[1] == 'Z' && p[2] == 'h' && p[3] >= '1' && p[3] <= '9') {
        return Codec::Bzip2;
    }
    if (size >= 6 && p[0] == 0xfd && p[1] == '7' && p[2] == 'z' && p[3] == 'X' && p[4] == 'Z' &&
        p[5] == 0) {
        return Codec::Xz;
    }
    return Codec::None;
}

std::string strip_codec_suffix(std::string_view path) {
    if (codec_from_path(path) == Codec::None) {
        return std::string(path);
    }
    const std::string_view ext = suffix_of(path);
    const std::string e = lower(ext);
    std::string out(path.substr(0, path.size() - ext.size()));
    if (e == ".tgz" || e == ".tzst" || e == ".tbz" || e == ".tbz2" || e == ".txz") {
        out += ".tar";
    }
    return out;
}

std::unique_ptr<Decompressor> make_decompressor(Codec codec) {
    require_codec(codec);
    switch (codec) {
#if defined(NEXUSDATA_WITH_ZLIB)
        case Codec::Gzip: return std::make_unique<ZlibDecoder>(true);
        case Codec::Zlib: return std::make_unique<ZlibDecoder>(false);
#endif
#if defined(NEXUSDATA_WITH_ZSTD)
        case Codec::Zstd: return std::make_unique<ZstdDecoder>();
#endif
#if defined(NEXUSDATA_WITH_LZ4)
        case Codec::Lz4: return std::make_unique<Lz4Decoder>();
#endif
#if defined(NEXUSDATA_WITH_BZIP2)
        case Codec::Bzip2: return std::make_unique<Bzip2Decoder>();
#endif
#if defined(NEXUSDATA_WITH_LZMA)
        case Codec::Xz: return std::make_unique<XzDecoder>();
#endif
        default: break;
    }
    throw InvalidArgumentError("make_decompressor: Codec::None has no decoder");
}

std::vector<std::uint8_t> compress(Codec codec, const void* data, std::size_t size,
                                   CompressOptions options) {
    require_codec(codec);
    if (data == nullptr && size > 0) {
        throw InvalidArgumentError("compress: null data");
    }
    const auto* p = static_cast<const std::uint8_t*>(data);
    const int level = options.level;
    std::vector<std::uint8_t> out;
    using Sink = std::vector<std::uint8_t>;
    switch (codec) {
#if defined(NEXUSDATA_WITH_ZLIB)
        case Codec::Gzip:
            append_frames(out, p, size, options.frame_bytes,
                          [&](Sink& o, const std::uint8_t* s, std::size_t k) { deflate_one(o, s, k, level, true); });
            return out;
        case Codec::Zlib:
            deflate_one(out, p, size, level, false);
            return out;
#endif
#if defined(NEXUSDATA_WITH_ZSTD)
        case Codec::Zstd:
            append_frames(out, p, size, options.frame_bytes, [&](Sink& o, const std::uint8_t* s, std::size_t k) {
                const std::size_t base = o.size();
                o.resize(base + ZSTD_compressBound(k));
                const std::size_t r = ZSTD_compress(o.data() + base, o.size() - base, s, k,
                                                    level < 0 ? ZSTD_CLEVEL_DEFAULT : level);
                if (ZSTD_isError(r)) throw IOError(std::string("zstd: ") + ZSTD_getErrorName(r));
                o.resize(base + r);
            });
            return out;
#endif
#if defined(NEXUSDATA_WITH_LZ4)
        case Codec::Lz4:
            append_frames(out, p, size, options.frame_bytes, [&](Sink& o, const std::uint8_t* s, std::size_t k) {
                LZ4F_preferences_t prefs{};
                prefs.frameInfo.contentSize = k;
                prefs.compressionLevel = level < 0 ? 0 : level;
                const std::size_t base = o.size();
                o.resize(base + LZ4F_compressFrameBound(k, &prefs));
                const std::size_t r = LZ4F_compressFrame(o.data() + base, o.size() - base, s, k, &prefs);
                if (LZ4F_isError(r)) throw IOError(std::string("lz4: ") + LZ4F_getErrorName(r));
                o.resize(base + r);
            });
            return out;
#endif
#if defined(NEXUSDATA_WITH_BZIP2)
        case Codec::Bzip2:
            // Streams are capped at 512 MiB (32-bit buffer API); pbzip2-style multi-stream output.
            append_frames(out, p, size,
                          options.frame_bytes == 0 ? (std::size_t{512} << 20)
                                                   : std::min(options.frame_bytes, std::size_t{512} << 20),
                          [&](Sink& o, const std::uint8_t* s, std::size_t k) {
                              unsigned cap = static_cast<unsigned>(k + k / 100 + 601);
                              const std::size_t base = o.size();
                              o.resize(base + cap);
                              const int r = BZ2_bzBuffToBuffCompress(
                                  reinterpret_cast<char*>(o.data() + base), &cap,
                                  const_cast<char*>(reinterpret_cast<const char*>(s)),
                                  static_cast<unsigned>(k), level < 1 ? 9 : std::min(level, 9), 0, 0);
                              if (r != BZ_OK) throw IOError("bzip2: compression failed (" + std::to_string(r) + ")");
                              o.resize(base + cap);
                          });
            return out;
#endif
#if defined(NEXUSDATA_WITH_LZMA)
        case Codec::Xz:
            append_frames(out, p, size, options.frame_bytes, [&](Sink& o, const std::uint8_t* s, std::size_t k) {
                const std::size_t base = o.size();
                o.resize(base + lzma_stream_buffer_bound(k));
                std::size_t pos = base;
                const lzma_ret r = lzma_easy_buffer_encode(
                    static_cast<std::uint32_t>(level < 0 ? 6 : std::min(level, 9)), LZMA_CHECK_CRC64,
                    nullptr, s, k, o.data(), &pos, o.size());
                if (r != LZMA_OK) throw IOError("xz: compression failed");
                o.resize(pos);
            });
            return out;
#endif
        default: break;
    }
    throw InvalidArgumentError("compress: Codec::None");
}

std::vector<std::uint8_t> decompress(Codec codec, const void* data, std::size_t size) {
    auto dec = make_decompressor(codec);
    const auto* p = static_cast<const std::uint8_t*>(data);
    std::vector<std::uint8_t> out(size_hint(codec, p, size));
    std::size_t in_off = 0;
    std::size_t out_off = 0;
    for (;;) {
        if (out_off == out.size()) {
            out.resize(out.size() + std::max(out.size() / 2, std::size_t{64} << 10));
        }
        const auto st = dec->decode(p + in_off, size - in_off, out.data() + out_off,
                                    out.size() - out_off, true);
        in_off += st.consumed;
        out_off += st.produced;
        if (st.done) break;
    }
    out.resize(out_off);
    return out;
}

std::vector<std::uint8_t> inflate_raw(const void* data, std::size_t size, std::size_t expected_size) {
    require_codec(Codec::Zlib);
#if defined(NEXUSDATA_WITH_ZLIB)
    z_stream z{};
    if (inflateInit2(&z, -MAX_WBITS) != Z_OK) {
        throw IOError("inflate_raw: inflateInit2 failed");
    }
    std::vector<std::uint8_t> out(expected_size);
    const auto* p = static_cast<const std::uint8_t*>(data);
    std::size_t in_off = 0;
    std::size_t out_off = 0;
    int r = Z_OK;
    while (r == Z_OK) {
        const std::size_t in_n = std::min(size - in_off, kMaxSlice);
        const std::size_t out_n = std::min(out.size() - out_off, kMaxSlice);
        z.next_in = const_cast<Bytef*>(p + in_off);
        z.avail_in = static_cast<uInt>(in_n);
        z.next_out = out.data() + out_off;
        z.avail_out = static_cast<uInt>(out_n);
        r = inflate(&z, Z_NO_FLUSH);
        const std::size_t used = in_n - z.avail_in;
        const std::size_t made = out_n - z.avail_out;
        in_off += used;
        out_off += made;
        if (r == Z_BUF_ERROR && used == 0 && made == 0) break;
    }
    inflateEnd(&z);
    if (r != Z_STREAM_END || out_off != expected_size) {
        throw IOError("inflate_raw: corrupt or truncated deflate data");
    }
    return out;
#else
    (void)data; (void)size; (void)expected_size;
    return {};
#endif
}

DecompressingSource::DecompressingSource(ByteSourcePtr inner, Codec codec, std::size_t chunk_bytes)
    : inner_(std::move(inner)), dec_(make_decompressor(codec)),
      in_(std::max<std::size_t>(chunk_bytes, 4096)) {
    if (!inner_) {
        throw InvalidArgumentError("DecompressingSource: null source");
    }
}

std::size_t DecompressingSource::read(void* dst, std::size_t n) {
    if (done_ || n == 0) {
        return 0;
    }
    auto* out = static_cast<std::uint8_t*>(dst);
    for (;;) {
        if (in_begin_ == in_end_ && !inner_eof_) {
            in_begin_ = 0;
            in_end_ = inner_->read(in_.data(), in_.size());
            inner_eof_ = in_end_ == 0;
        }
        const auto st = dec_->decode(in_.data() + in_begin_, in_end_ - in_begin_, out, n, inner_eof_);
        in_begin_ += st.consumed;
        if (st.done) {
            done_ = true;
            return st.produced;
        }
        if (st.produced > 0) {
            return st.produced;
        }
        if (st.consumed == 0 && in_begin_ < in_end_) {
            throw IOError("DecompressingSource: decoder made no progress");
        }
    }
}

ByteSourcePtr decompress_source(ByteSourcePtr source, Codec codec) {
    if (codec == Codec::None) {
        return source;
    }
    return std::make_unique<DecompressingSource>(std::move(source), codec);
}

namespace {

Codec resolve_codec(const std::string& path, std::optional<Codec> codec, const std::uint8_t* head,
                    std::size_t head_n) {
    if (codec) {
        return *codec;
    }
    const Codec sniffed = sniff_codec(head, head_n);
    return sniffed != Codec::None ? sniffed : codec_from_path(path);
}

} // namespace

ByteSourcePtr open_source(const std::string& path, std::optional<Codec> codec) {
    std::uint8_t head[8] = {};
    std::size_t head_n = 0;
    if (!codec) {
        FileSource probe(path);
        head_n = probe.read_full(head, sizeof(head));
    }
    const Codec c = resolve_codec(path, codec, head, head_n);
    return decompress_source(std::make_unique<FileSource>(path), c);
}

InputBytes read_input(const std::string& path, std::optional<Codec> codec, unsigned num_threads) {
    InputBytes out;
    MappedFile map(path);
    const Codec c = resolve_codec(path, codec, map.data(), map.size());
    if (c == Codec::None) {
        out.map_ = std::move(map);
        return out;
    }
    require_codec(c);
    out.codec_ = c;
    const std::uint8_t* src = map.data();
    const std::size_t n = map.size();
    if (c == Codec::Zstd && parallel_zstd(src, n, num_threads, out.owned_, out.owned_size_)) {
        return out;
    }
    auto dec = make_decompressor(c);
    // Exact hints still get slack so the final "stream complete" step has room.
    std::size_t cap = size_hint(c, src, n) + (std::size_t{64} << 10);
    std::unique_ptr<std::uint8_t[]> buf(new std::uint8_t[cap]);
    std::size_t in_off = 0;
    std::size_t len = 0;
    for (;;) {
        if (len == cap) {
            const std::size_t grown = cap + std::max(cap / 2, std::size_t{1} << 20);
            std::unique_ptr<std::uint8_t[]> bigger(new std::uint8_t[grown]);
            std::memcpy(bigger.get(), buf.get(), len);
            buf = std::move(bigger);
            cap = grown;
        }
        const auto st = dec->decode(src + in_off, n - in_off, buf.get() + len, cap - len, true);
        in_off += st.consumed;
        len += st.produced;
        if (st.done) break;
    }
    out.owned_ = std::move(buf);
    out.owned_size_ = len;
    return out;
}

} // namespace nexusdata
