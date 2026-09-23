#include "nexusdata/dataset/npy.hpp"

#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "nexusdata/core/error.hpp"

namespace nexusdata {

namespace {

bool little_endian_host() {
    const std::uint16_t x = 1;
    return *reinterpret_cast<const std::uint8_t*>(&x) == 1;
}

DType descr_to_dtype(const std::string& descr) {
    // Formats like "<f4", "|u1", "<i8", ">f8"
    if (descr.size() < 3) {
        throw IOError("npy: bad descr " + descr);
    }
    const char code = descr[descr.size() - 2];
    const char bytes = descr.back();
    if (code == 'f' && bytes == '4') return DType::Float32;
    if (code == 'f' && bytes == '8') return DType::Float64;
    if (code == 'i' && bytes == '1') return DType::Int8;
    if (code == 'i' && bytes == '2') return DType::Int16;
    if (code == 'i' && bytes == '4') return DType::Int32;
    if (code == 'i' && bytes == '8') return DType::Int64;
    if (code == 'u' && bytes == '1') return DType::UInt8;
    if (code == 'u' && bytes == '2') return DType::UInt16;
    if (code == 'u' && bytes == '4') return DType::UInt32;
    if (code == 'u' && bytes == '8') return DType::UInt64;
    if (code == 'b' && bytes == '1') return DType::Bool;
    throw IOError("npy: unsupported descr " + descr);
}

bool descr_needs_swap(const std::string& descr) {
    if (descr.empty()) return false;
    if (descr[0] == '|') return false;
    if (descr[0] == '<') return !little_endian_host();
    if (descr[0] == '>') return little_endian_host();
    return false;
}

Shape parse_shape(const std::string& header) {
    const auto pos = header.find("'shape':");
    if (pos == std::string::npos) {
        throw IOError("npy: shape not found in header");
    }
    const auto l = header.find('(', pos);
    const auto r = header.find(')', l);
    if (l == std::string::npos || r == std::string::npos) {
        throw IOError("npy: malformed shape");
    }
    Shape shape;
    std::stringstream ss(header.substr(l + 1, r - l - 1));
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        // trim
        std::size_t b = 0;
        while (b < tok.size() && (tok[b] == ' ' || tok[b] == '\t')) ++b;
        std::size_t e = tok.size();
        while (e > b && (tok[e - 1] == ' ' || tok[e - 1] == '\t')) --e;
        if (e <= b) continue;
        try {
            shape.push_back(static_cast<std::size_t>(std::stoull(tok.substr(b, e - b))));
        } catch (const std::exception&) {
            throw IOError("npy: malformed shape dimension \"" + tok.substr(b, e - b) + "\"");
        }
    }
    return shape;
}

std::string parse_descr(const std::string& header) {
    const auto pos = header.find("'descr':");
    if (pos == std::string::npos) {
        throw IOError("npy: descr not found");
    }
    const auto q1 = header.find('\'', pos + 8);
    const auto q2 = header.find('\'', q1 + 1);
    if (q1 == std::string::npos || q2 == std::string::npos) {
        throw IOError("npy: malformed descr");
    }
    return header.substr(q1 + 1, q2 - q1 - 1);
}

bool parse_fortran(const std::string& header) {
    const auto pos = header.find("'fortran_order':");
    if (pos == std::string::npos) return false;
    return header.find("True", pos) != std::string::npos &&
           header.find("True", pos) < header.find(',', pos);
}

struct ParsedNpy {
    std::string descr;
    Shape shape;
    bool fortran = false;
    std::size_t data_offset = 0;
};

ParsedNpy parse_npy_buffer(const std::uint8_t* data, std::size_t size) {
    if (size < 10) {
        throw IOError("npy: file too small");
    }
    if (!(data[0] == 0x93 && data[1] == 'N' && data[2] == 'U' && data[3] == 'M' &&
          data[4] == 'P' && data[5] == 'Y')) {
        throw IOError("npy: bad magic");
    }
    const int major = data[6];
    const int minor = data[7];
    (void)minor;
    std::size_t header_len = 0;
    std::size_t offset = 8;
    if (major == 1) {
        header_len = data[8] | (static_cast<std::size_t>(data[9]) << 8);
        offset = 10;
    } else if (major == 2 || major == 3) {
        header_len = data[8] | (static_cast<std::size_t>(data[9]) << 8) |
                     (static_cast<std::size_t>(data[10]) << 16) |
                     (static_cast<std::size_t>(data[11]) << 24);
        offset = 12;
    } else {
        throw IOError("npy: unsupported version");
    }
    if (offset + header_len > size) {
        throw IOError("npy: truncated header");
    }
    const std::string header(reinterpret_cast<const char*>(data + offset), header_len);
    ParsedNpy p;
    p.descr = parse_descr(header);
    p.shape = parse_shape(header);
    p.fortran = parse_fortran(header);
    p.data_offset = offset + header_len;
    return p;
}

NDArray make_array_from_bytes(const std::uint8_t* bytes,
                              std::size_t nbytes,
                              const ParsedNpy& p,
                              bool copy) {
    if (p.fortran) {
        throw IOError("npy: fortran_order=True not supported in v0.3 (C-order only)");
    }
    if (descr_needs_swap(p.descr)) {
        // Always copy + byte-swap when endianness differs.
        copy = true;
    }
    const DType dt = descr_to_dtype(p.descr);
    const std::size_t need = numel(p.shape) * size_of(dt);
    if (nbytes < need) {
        throw IOError("npy: truncated data payload");
    }
    if (!copy && !descr_needs_swap(p.descr)) {
        return NDArray::from_blob(const_cast<std::uint8_t*>(bytes), p.shape, dt);
    }
    NDArray out(p.shape, dt);
    std::memcpy(out.data(), bytes, need);
    if (descr_needs_swap(p.descr)) {
        const std::size_t esize = size_of(dt);
        auto* pbytes = static_cast<std::uint8_t*>(out.data());
        for (std::size_t i = 0; i < need; i += esize) {
            for (std::size_t a = 0, b = esize - 1; a < b; ++a, --b) {
                std::swap(pbytes[i + a], pbytes[i + b]);
            }
        }
    }
    return out;
}

std::uint32_t read_u32_le(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}

std::uint16_t read_u16_le(const std::uint8_t* p) {
    return static_cast<std::uint16_t>(p[0] | (p[1] << 8));
}

} // namespace

NpyArray load_npy(const std::string& path, bool copy) {
    const InputBytes map = read_input(path);
    const auto parsed = parse_npy_buffer(map.data(), map.size());
    NpyArray out;
    out.descr = parsed.descr;
    out.fortran_order = parsed.fortran;
    if (copy) {
        out.view = make_array_from_bytes(map.data() + parsed.data_offset,
                                         map.size() - parsed.data_offset, parsed, true);
        out.owns_storage = true;
    } else {
        // Caller must keep file alive — NpyFile is preferred.
        throw InvalidArgumentError("load_npy(copy=false): use NpyFile for zero-copy lifetime");
    }
    return out;
}

NpyArray load_npy_from_buffer(const std::uint8_t* data, std::size_t size) {
    if (data == nullptr) {
        throw InvalidArgumentError("load_npy_from_buffer: null data");
    }
    const auto parsed = parse_npy_buffer(data, size);
    NpyArray out;
    out.descr = parsed.descr;
    out.fortran_order = parsed.fortran;
    out.view = make_array_from_bytes(data + parsed.data_offset, size - parsed.data_offset, parsed,
                                     true);
    out.owns_storage = true;
    return out;
}

NpyFile::NpyFile(std::string path) : map_(read_input(path)) {
    const auto parsed = parse_npy_buffer(map_.data(), map_.size());
    descr_ = parsed.descr;
    array_ = make_array_from_bytes(map_.data() + parsed.data_offset,
                                   map_.size() - parsed.data_offset, parsed,
                                   descr_needs_swap(parsed.descr));
}

NpyDataset::NpyDataset(std::string features_path, std::string labels_path)
    : features_(std::move(features_path)) {
    const auto& f = features_.array();
    if (f.shape().size() == 1) {
        n_ = f.shape()[0];
        n_features_ = 1;
    } else if (f.shape().size() == 2) {
        n_ = f.shape()[0];
        n_features_ = f.shape()[1];
    } else {
        throw ShapeError("NpyDataset: features must be rank 1 or 2");
    }
    if (!labels_path.empty()) {
        labels_ = std::make_unique<NpyFile>(std::move(labels_path));
        if (labels_->array().shape().empty() || labels_->array().shape()[0] != n_) {
            throw ShapeError("NpyDataset: labels length mismatch");
        }
    }
}

std::size_t NpyDataset::size() const {
    return n_;
}

Sample NpyDataset::get(std::size_t index) const {
    if (index >= n_) {
        throw IndexError(format_index_error("NpyDataset::get", index, n_));
    }
    Sample s;
    s.input = NDArray(Shape{n_features_}, features_.array().dtype());
    const std::size_t fes = size_of(features_.array().dtype());
    const auto* src = static_cast<const std::uint8_t*>(features_.array().data()) +
                      index * n_features_ * fes;
    std::memcpy(s.input.data(), src, n_features_ * fes);
    if (labels_) {
        s.label = NDArray(Shape{1}, labels_->array().dtype());
        const std::size_t les = size_of(labels_->array().dtype());
        std::memcpy(s.label.data(),
                    static_cast<const std::uint8_t*>(labels_->array().data()) + index * les, les);
    } else {
        s.label = NDArray(Shape{1}, DType::Int64);
        s.label.data<std::int64_t>()[0] = 0;
    }
    return s;
}

NpzFile::NpzFile(std::string path) : map_(std::move(path)) {
    // Scan local file headers sequentially.
    std::size_t off = 0;
    const auto* data = map_.data();
    const std::size_t n = map_.size();
    while (off + 30 <= n) {
        if (read_u32_le(data + off) != 0x04034b50u) {
            break; // central directory or end
        }
        const std::uint16_t method = read_u16_le(data + off + 8);
        const std::uint32_t comp = read_u32_le(data + off + 18);
        const std::uint32_t uncomp = read_u32_le(data + off + 22);
        const std::uint16_t name_len = read_u16_le(data + off + 26);
        const std::uint16_t extra_len = read_u16_le(data + off + 28);
        if (off + 30 + name_len + extra_len > n) {
            throw IOError("npz: truncated local header");
        }
        std::string name(reinterpret_cast<const char*>(data + off + 30), name_len);
        const std::uint64_t data_off = off + 30 + name_len + extra_len;
        std::uint64_t comp64 = comp;
        std::uint64_t uncomp64 = uncomp;
        if (comp == 0xffffffffu || uncomp == 0xffffffffu) {
            // ZIP64 (numpy always writes it): real sizes live in extra field 0x0001,
            // uncompressed first, each present only when its 32-bit field is saturated.
            const std::uint8_t* e = data + off + 30 + name_len;
            const std::uint8_t* e_end = e + extra_len;
            bool found = false;
            while (e + 4 <= e_end) {
                const std::uint16_t id = read_u16_le(e);
                const std::uint16_t len = read_u16_le(e + 2);
                if (e + 4 + len > e_end) break;
                if (id == 0x0001) {
                    const std::uint8_t* q = e + 4;
                    const std::uint8_t* q_end = q + len;
                    auto u64 = [&](std::uint64_t& v) {
                        if (q + 8 > q_end) throw IOError("npz: short ZIP64 extra field");
                        v = static_cast<std::uint64_t>(read_u32_le(q)) |
                            (static_cast<std::uint64_t>(read_u32_le(q + 4)) << 32);
                        q += 8;
                    };
                    if (uncomp == 0xffffffffu) u64(uncomp64);
                    if (comp == 0xffffffffu) u64(comp64);
                    found = true;
                    break;
                }
                e += 4 + len;
            }
            if (!found) {
                throw IOError("npz: ZIP64 sizes missing for member \"" + name + "\"");
            }
        }
        if (comp64 > n - data_off) {
            throw IOError("npz: truncated member \"" + name + "\"");
        }
        Member m;
        m.name = name;
        m.offset = data_off;
        m.comp_size = comp64;
        m.uncomp_size = uncomp64;
        m.method = method;
        members_[name] = m;
        off = static_cast<std::size_t>(data_off + comp64);
    }
    if (members_.empty()) {
        throw IOError("npz: no members found in \"" + path + "\"");
    }
}

std::vector<std::string> NpzFile::keys() const {
    std::vector<std::string> k;
    k.reserve(members_.size());
    for (const auto& [name, _] : members_) {
        k.push_back(name);
    }
    return k;
}

bool NpzFile::contains(const std::string& name) const {
    return members_.count(name) != 0;
}

NDArray NpzFile::get(const std::string& name) const {
    auto it = members_.find(name);
    if (it == members_.end()) {
        throw IOError("npz: key not found: \"" + name + "\"");
    }
    const Member& m = it->second;
    if (m.offset + m.comp_size > map_.size()) {
        throw IOError("npz: truncated member data");
    }
    if (m.method == 8) {
        // np.savez_compressed: raw DEFLATE members.
        const std::vector<std::uint8_t> raw =
            inflate_raw(map_.data() + m.offset, static_cast<std::size_t>(m.comp_size),
                        static_cast<std::size_t>(m.uncomp_size));
        const auto parsed = parse_npy_buffer(raw.data(), raw.size());
        return make_array_from_bytes(raw.data() + parsed.data_offset,
                                     raw.size() - parsed.data_offset, parsed, true);
    }
    if (m.method != 0) {
        throw IOError("npz: member \"" + name + "\" uses ZIP method " + std::to_string(m.method) +
                      "; only stored (0) and deflate (8) are supported");
    }
    const auto* bytes = map_.data() + m.offset;
    const auto parsed = parse_npy_buffer(bytes, static_cast<std::size_t>(m.comp_size));
    return make_array_from_bytes(bytes + parsed.data_offset,
                                 static_cast<std::size_t>(m.comp_size) - parsed.data_offset,
                                 parsed, true);
}

} // namespace nexusdata
