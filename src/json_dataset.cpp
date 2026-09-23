#include "nexusdata/dataset/json.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <thread>

#include "json_detail.hpp"
#include "nexusdata/backend/thread_pool.hpp"
#include "nexusdata/core/error.hpp"
#include "nexusdata/io/compression.hpp"

namespace nexusdata {

using json::detail::Cursor;

/// Per-field columnar storage. Also used for a single record by the streaming reader.
struct JsonDataset::Column {
    DType dtype = DType::Float32;
    std::size_t esize = 4;
    std::vector<std::uint8_t> data;       ///< numeric values, all records back to back
    std::vector<std::uint64_t> elem_off{0}; ///< record i: elements [elem_off[i], elem_off[i+1])
    std::vector<std::uint32_t> dim_off{0};  ///< record i: dims [dim_off[i], dim_off[i+1])
    std::vector<std::size_t> dims;
    std::vector<std::string> strings;       ///< Metadata fields

    [[nodiscard]] std::size_t records() const noexcept { return elem_off.size() - 1; }

    void clear() {
        data.clear();
        elem_off.assign(1, 0);
        dim_off.assign(1, 0);
        dims.clear();
        strings.clear();
    }

    void append(const Column& o) {
        const std::uint64_t e0 = elem_off.back();
        const std::uint32_t d0 = dim_off.back();
        data.insert(data.end(), o.data.begin(), o.data.end());
        for (std::size_t i = 1; i < o.elem_off.size(); ++i) elem_off.push_back(e0 + o.elem_off[i]);
        for (std::size_t i = 1; i < o.dim_off.size(); ++i) dim_off.push_back(d0 + o.dim_off[i]);
        dims.insert(dims.end(), o.dims.begin(), o.dims.end());
        strings.insert(strings.end(), o.strings.begin(), o.strings.end());
    }
};

namespace {

using Column = JsonDataset::Column;

struct Step {
    enum Kind : std::uint8_t { Key, Index, Wild } kind;
    std::string key;
    std::size_t index = 0;
};

std::vector<Step> compile_path(const std::string& path) {
    std::vector<Step> steps;
    std::size_t i = 0;
    auto bad = [&](const char* why) {
        throw InvalidArgumentError("JsonField path \"" + path + "\": " + why);
    };
    while (i < path.size()) {
        if (path[i] == '.') {
            if (i == 0 || i + 1 == path.size() || path[i + 1] == '.') bad("empty key");
            ++i;
            continue;
        }
        if (path[i] == '[') {
            const std::size_t close = path.find(']', i);
            if (close == std::string::npos) bad("missing ']'");
            const std::string_view body(path.data() + i + 1, close - i - 1);
            if (body == "*") {
                steps.push_back({Step::Wild, {}, 0});
            } else {
                std::size_t idx = 0;
                const auto r = std::from_chars(body.data(), body.data() + body.size(), idx);
                if (body.empty() || r.ec != std::errc() || r.ptr != body.data() + body.size()) {
                    bad("array step must be [<index>] or [*]");
                }
                steps.push_back({Step::Index, {}, idx});
            }
            i = close + 1;
            continue;
        }
        const std::size_t stop = std::min(path.find_first_of(".[", i), path.size());
        steps.push_back({Step::Key, path.substr(i, stop - i), 0});
        i = stop;
    }
    if (steps.empty()) bad("empty path");
    return steps;
}

/// Field value scratch for the record being parsed.
struct Pending {
    std::size_t chunks = 0;
    std::size_t elem_start = 0;   ///< Column::data element count at record start
    std::size_t chunk_rank = 0;
    std::size_t chunk_dims[8] = {};
    bool has_chunk_shape = false;
};

void put_value(Column& col, double v, std::int64_t iv, bool is_int) {
    const std::size_t off = col.data.size();
    col.data.resize(off + col.esize);
    std::uint8_t* p = col.data.data() + off;
    auto put = [&](auto x) { std::memcpy(p, &x, sizeof(x)); };
    switch (col.dtype) {
        case DType::Float32: put(static_cast<float>(is_int ? static_cast<double>(iv) : v)); break;
        case DType::Float64: put(is_int ? static_cast<double>(iv) : v); break;
        case DType::Bool: put(static_cast<std::uint8_t>((is_int ? iv != 0 : v != 0.0) ? 1 : 0)); break;
        default: {
            if (!is_int) {
                if (!std::isfinite(v) || std::trunc(v) != v || std::abs(v) > 9.2e18) {
                    throw IOError("non-integer value for integer field");
                }
                iv = static_cast<std::int64_t>(v);
            }
            switch (col.dtype) {
                case DType::Int8: put(static_cast<std::int8_t>(iv)); break;
                case DType::Int16: put(static_cast<std::int16_t>(iv)); break;
                case DType::Int32: put(static_cast<std::int32_t>(iv)); break;
                case DType::Int64: put(iv); break;
                case DType::UInt8: put(static_cast<std::uint8_t>(iv)); break;
                case DType::UInt16: put(static_cast<std::uint16_t>(iv)); break;
                case DType::UInt32: put(static_cast<std::uint32_t>(iv)); break;
                case DType::UInt64: put(static_cast<std::uint64_t>(iv)); break;
                default: throw IOError("unsupported dtype for JSON field");
            }
        }
    }
}

class Extractor {
public:
    explicit Extractor(const std::vector<JsonField>& fields) : fields_(fields) {
        nodes_.emplace_back();
        for (std::size_t f = 0; f < fields.size(); ++f) {
            const JsonField& fd = fields[f];
            if (fd.dtype == DType::Float16 || fd.dtype == DType::BFloat16) {
                throw InvalidArgumentError("JsonField \"" + fd.path + "\": float16 / bfloat16 unsupported");
            }
            const auto steps = compile_path(fd.path);
            bool wild = false;
            int cur = 0;
            for (const Step& s : steps) {
                wild = wild || s.kind == Step::Wild;
                cur = child(cur, s);
            }
            nodes_[static_cast<std::size_t>(cur)].fields.push_back(f);
            wildcard_.push_back(wild);
        }
    }

    [[nodiscard]] std::vector<Column> make_columns() const {
        std::vector<Column> cols(fields_.size());
        for (std::size_t f = 0; f < fields_.size(); ++f) {
            cols[f].dtype = fields_[f].role == JsonRole::Metadata ? DType::UInt8 : fields_[f].dtype;
            cols[f].esize = size_of(cols[f].dtype);
        }
        return cols;
    }

    /// Parse one record into @p cols (one new record per column). On failure the
    /// columns are rolled back and the exception propagates.
    void extract(std::string_view record, std::vector<Column>& cols) {
        const std::size_t n_records = cols.empty() ? 0 : cols[0].records();
        pending_.assign(fields_.size(), Pending{});
        marks_.resize(cols.size());
        for (std::size_t f = 0; f < cols.size(); ++f) {
            marks_[f] = {cols[f].data.size(), cols[f].dims.size(), cols[f].strings.size()};
            pending_[f].elem_start = cols[f].data.size() / cols[f].esize;
        }
        try {
            Cursor c(record.data(), record.data() + record.size());
            walk(c, 0, cols, 0);
            if (!c.at_end()) c.fail("trailing characters after record");
            for (std::size_t f = 0; f < cols.size(); ++f) finish_field(f, cols[f]);
        } catch (...) {
            for (std::size_t f = 0; f < cols.size(); ++f) {
                cols[f].data.resize(marks_[f].data);
                cols[f].dims.resize(marks_[f].dims);
                cols[f].strings.resize(marks_[f].strings);
                cols[f].elem_off.resize(n_records + 1);
                cols[f].dim_off.resize(n_records + 1);
            }
            throw;
        }
    }

private:
    struct Node {
        std::vector<std::pair<std::string, int>> keys;
        std::vector<std::pair<std::size_t, int>> indices;
        int wildcard = -1;
        std::vector<std::size_t> fields;
        [[nodiscard]] bool has_children() const noexcept {
            return !keys.empty() || !indices.empty() || wildcard >= 0;
        }
    };

    int child(int parent, const Step& s) {
        auto make = [&]() {
            nodes_.emplace_back();
            return static_cast<int>(nodes_.size() - 1);
        };
        Node* n = &nodes_[static_cast<std::size_t>(parent)];
        if (s.kind == Step::Wild) {
            if (n->wildcard < 0) {
                const int id = make();
                nodes_[static_cast<std::size_t>(parent)].wildcard = id;
            }
            return nodes_[static_cast<std::size_t>(parent)].wildcard;
        }
        if (s.kind == Step::Index) {
            for (const auto& [i, id] : n->indices) if (i == s.index) return id;
            const int id = make();
            nodes_[static_cast<std::size_t>(parent)].indices.emplace_back(s.index, id);
            return id;
        }
        for (const auto& [k, id] : n->keys) if (k == s.key) return id;
        const int id = make();
        nodes_[static_cast<std::size_t>(parent)].keys.emplace_back(s.key, id);
        return id;
    }

    void walk(Cursor& c, int node_id, std::vector<Column>& cols, int depth) {
        if (depth > 256) c.fail("nesting too deep");
        const Node& nd = nodes_[static_cast<std::size_t>(node_id)];
        c.ws();
        const char* start = c.p;
        if (!nd.fields.empty()) {
            for (const std::size_t f : nd.fields) {
                c.p = start;
                take(c, f, cols[f]);
            }
            if (!nd.has_children()) return;
            c.p = start;
        }
        if (!nd.has_children()) {
            c.skip_value();
            return;
        }
        const char ch = c.peek();
        if (ch == '{' && !nd.keys.empty()) {
            ++c.p;
            if (c.eat('}')) return;
            do {
                if (c.peek() != '"') c.fail("expected object key");
                ++c.p;
                const char* k0 = c.p;
                const bool escaped = c.skip_string_body();
                std::string_view key(k0, static_cast<std::size_t>(c.p - 1 - k0));
                if (escaped) {
                    c.p = k0 - 1;
                    c.string(key_buf_);
                    key = key_buf_;
                }
                c.expect(':');
                int next = -1;
                for (const auto& [k, id] : nd.keys) {
                    if (k == key) {
                        next = id;
                        break;
                    }
                }
                if (next >= 0) walk(c, next, cols, depth + 1);
                else c.skip_value();
            } while (c.eat(','));
            c.expect('}');
            return;
        }
        if (ch == '[' && (!nd.indices.empty() || nd.wildcard >= 0)) {
            ++c.p;
            if (c.eat(']')) return;
            std::size_t idx = 0;
            do {
                c.ws();
                const char* elem = c.p;
                int by_index = -1;
                for (const auto& [i, id] : nd.indices) {
                    if (i == idx) {
                        by_index = id;
                        break;
                    }
                }
                if (nd.wildcard >= 0) {
                    walk(c, nd.wildcard, cols, depth + 1);
                    if (by_index >= 0) {
                        c.p = elem;
                        walk(c, by_index, cols, depth + 1);
                    }
                } else if (by_index >= 0) {
                    walk(c, by_index, cols, depth + 1);
                } else {
                    c.skip_value();
                }
                ++idx;
            } while (c.eat(','));
            c.expect(']');
            return;
        }
        c.skip_value();
    }

    /// Consume the value at a field's path.
    void take(Cursor& c, std::size_t f, Column& col) {
        const JsonField& fd = fields_[f];
        Pending& pd = pending_[f];
        if (!wildcard_[f] && pd.chunks > 0) c.fail("field matched twice");
        ++pd.chunks;
        if (fd.role == JsonRole::Metadata) {
            if (c.peek() == '"') {
                col.strings.emplace_back();
                c.string(col.strings.back());
            } else {
                const char* s = c.p;
                c.skip_value();
                col.strings.emplace_back(s, static_cast<std::size_t>(c.p - s));
            }
            return;
        }
        std::size_t dims[8];
        std::size_t leaf = SIZE_MAX;
        for (std::size_t& d : dims) d = SIZE_MAX;
        numeric(c, fd, col, dims, leaf, 0);
        // Scalars sit at depth `leaf`; arrays without scalars ([], [[], []]) end
        // at their deepest recorded dimension.
        std::size_t rank = leaf;
        if (leaf == SIZE_MAX) {
            rank = 0;
            while (rank < 8 && dims[rank] != SIZE_MAX) ++rank;
        }
        if (!pd.has_chunk_shape) {
            pd.has_chunk_shape = true;
            pd.chunk_rank = rank;
            for (std::size_t i = 0; i < rank; ++i) pd.chunk_dims[i] = dims[i];
        } else {
            bool same = pd.chunk_rank == rank;
            for (std::size_t i = 0; same && i < rank; ++i) same = pd.chunk_dims[i] == dims[i];
            if (!same) c.fail("[*] matches have different shapes");
        }
    }

    void numeric(Cursor& c, const JsonField& fd, Column& col, std::size_t* dims, std::size_t& leaf,
                 std::size_t level) {
        const char ch = c.peek();
        if (ch == '[') {
            if (level >= 8) c.fail("numeric array nested deeper than 8 levels");
            ++c.p;
            std::size_t count = 0;
            if (!c.eat(']')) {
                do {
                    numeric(c, fd, col, dims, leaf, level + 1);
                    ++count;
                } while (c.eat(','));
                c.expect(']');
            }
            if (dims[level] == SIZE_MAX) dims[level] = count;
            else if (dims[level] != count) c.fail("ragged nested array");
            return;
        }
        if (leaf == SIZE_MAX) leaf = level;
        else if (leaf != level) c.fail("mixed scalar / array nesting");
        switch (ch) {
            case 't': c.literal("true", 4); put_value(col, 1, 1, true); return;
            case 'f': c.literal("false", 5); put_value(col, 0, 0, true); return;
            case 'n':
                c.literal("null", 4);
                if (col.dtype == DType::Float32 || col.dtype == DType::Float64) {
                    put_value(col, std::numeric_limits<double>::quiet_NaN(), 0, false);
                } else if (fd.default_value) {
                    put_value(col, *fd.default_value, 0, false);
                } else {
                    c.fail("null in integer field");
                }
                return;
            case '"': {
                c.string(key_buf_);
                const auto it = std::find(fd.categories.begin(), fd.categories.end(), key_buf_);
                if (it == fd.categories.end()) {
                    if (fd.categories.empty()) c.fail("string value in numeric field (set categories)");
                    c.fail("string not in categories");
                }
                const auto id = static_cast<std::int64_t>(it - fd.categories.begin());
                put_value(col, static_cast<double>(id), id, true);
                return;
            }
            default: {
                const std::string_view tok = c.number_token();
                const bool int_dtype = col.dtype != DType::Float32 && col.dtype != DType::Float64;
                if (int_dtype && Cursor::is_integral(tok)) {
                    std::int64_t iv = 0;
                    const auto r = std::from_chars(tok.data(), tok.data() + tok.size(), iv);
                    if (r.ec == std::errc()) {
                        put_value(col, 0, iv, true);
                        return;
                    }
                }
                try {
                    put_value(col, Cursor::to_double(tok), 0, false);
                } catch (const IOError& e) {
                    c.fail(e.what());
                }
            }
        }
    }

    void finish_field(std::size_t f, Column& col) {
        const JsonField& fd = fields_[f];
        Pending& pd = pending_[f];
        if (fd.role == JsonRole::Metadata) {
            if (pd.chunks == 0) {
                if (!fd.default_value) throw IOError("missing field \"" + fd.path + "\"");
                char buf[32];
                const auto r = std::to_chars(buf, buf + sizeof(buf), *fd.default_value);
                col.strings.emplace_back(buf, static_cast<std::size_t>(r.ptr - buf));
            } else if (pd.chunks > 1) {
                throw IOError("metadata field \"" + fd.path + "\" matched more than once");
            }
            col.elem_off.push_back(col.strings.size());
            col.dim_off.push_back(static_cast<std::uint32_t>(col.dims.size()));
            return;
        }
        if (pd.chunks == 0 && !wildcard_[f]) {
            if (!fd.default_value) throw IOError("missing field \"" + fd.path + "\"");
            std::size_t n = 1;
            for (std::size_t d : fd.shape) n *= d;
            for (std::size_t i = 0; i < n; ++i) put_value(col, *fd.default_value, 0, false);
            pd.has_chunk_shape = true;
            pd.chunk_rank = 0;
        }
        const std::size_t elems = col.data.size() / col.esize - pd.elem_start;
        if (wildcard_[f]) col.dims.push_back(pd.chunks);
        for (std::size_t i = 0; i < pd.chunk_rank; ++i) col.dims.push_back(pd.chunk_dims[i]);
        const std::size_t d0 = col.dim_off.back();
        if (col.dims.size() == d0) col.dims.push_back(elems); // scalar -> [1]
        if (!fd.shape.empty()) {
            std::size_t want = 1;
            for (std::size_t d : fd.shape) want *= d;
            std::size_t lead = wildcard_[f] ? pd.chunks : 1;
            if (elems != want * lead) {
                throw IOError("field \"" + fd.path + "\": " + std::to_string(elems) +
                              " values, shape expects " + std::to_string(want * lead));
            }
            col.dims.resize(d0);
            if (wildcard_[f]) col.dims.push_back(pd.chunks);
            col.dims.insert(col.dims.end(), fd.shape.begin(), fd.shape.end());
        }
        col.elem_off.push_back(col.data.size() / col.esize);
        col.dim_off.push_back(static_cast<std::uint32_t>(col.dims.size()));
    }

    struct Mark {
        std::size_t data, dims, strings;
    };

    const std::vector<JsonField>& fields_;
    std::vector<Node> nodes_;
    std::vector<bool> wildcard_;
    std::vector<Pending> pending_;
    std::vector<Mark> marks_;
    std::string key_buf_;
};

NDArray column_array(const Column& col, std::size_t i) {
    Shape shape(col.dims.begin() + col.dim_off[i], col.dims.begin() + col.dim_off[i + 1]);
    NDArray out(std::move(shape), col.dtype);
    const std::size_t n = static_cast<std::size_t>(col.elem_off[i + 1] - col.elem_off[i]);
    if (n > 0) {
        std::memcpy(out.data(), col.data.data() + col.elem_off[i] * col.esize, n * col.esize);
    }
    return out;
}

/// Inputs / labels: one field keeps its shape, several are flattened and concatenated.
NDArray concat_role(const std::vector<JsonField>& fields, const std::vector<const Column*>& cols,
                    JsonRole role, std::size_t i) {
    std::vector<std::size_t> ids;
    for (std::size_t f = 0; f < fields.size(); ++f) {
        if (fields[f].role == role) ids.push_back(f);
    }
    if (ids.size() == 1) return column_array(*cols[ids[0]], i);
    std::size_t total = 0;
    for (std::size_t f : ids) total += static_cast<std::size_t>(cols[f]->elem_off[i + 1] - cols[f]->elem_off[i]);
    NDArray out(Shape{total}, fields[ids[0]].dtype);
    auto* dst = static_cast<std::uint8_t*>(out.data());
    for (std::size_t f : ids) {
        const Column& c = *cols[f];
        const std::size_t n = static_cast<std::size_t>(c.elem_off[i + 1] - c.elem_off[i]) * c.esize;
        if (n > 0) std::memcpy(dst, c.data.data() + c.elem_off[i] * c.esize, n);
        dst += n;
    }
    return out;
}

Sample make_sample(const std::vector<JsonField>& fields, const std::vector<const Column*>& cols,
                   std::size_t i, std::size_t record_index) {
    Sample s;
    bool has_input = false;
    bool has_label = false;
    for (const JsonField& f : fields) {
        has_input = has_input || f.role == JsonRole::Input;
        has_label = has_label || f.role == JsonRole::Label;
    }
    if (has_input) s.input = concat_role(fields, cols, JsonRole::Input, i);
    if (has_label) {
        s.label = concat_role(fields, cols, JsonRole::Label, i);
    } else {
        s.label = NDArray(Shape{1}, DType::Int64);
        s.label.data<std::int64_t>()[0] = static_cast<std::int64_t>(record_index);
    }
    for (std::size_t f = 0; f < fields.size(); ++f) {
        const JsonField& fd = fields[f];
        const std::string& key = fd.name.empty() ? fd.path : fd.name;
        if (fd.role == JsonRole::Extra) {
            s.extra_tensors.set(key, column_array(*cols[f], i));
        } else if (fd.role == JsonRole::Metadata) {
            s.metadata.set(key, cols[f]->strings[static_cast<std::size_t>(cols[f]->elem_off[i])]);
        }
    }
    return s;
}

void validate_fields(const std::vector<JsonField>& fields) {
    if (fields.empty()) throw InvalidArgumentError("JsonDataset: no fields to extract");
    for (JsonRole role : {JsonRole::Input, JsonRole::Label}) {
        std::optional<DType> dt;
        for (const JsonField& f : fields) {
            if (f.role != role) continue;
            if (dt && *dt != f.dtype) {
                throw InvalidArgumentError("JsonDataset: all " +
                                           std::string(role == JsonRole::Input ? "input" : "label") +
                                           " fields must share one dtype");
            }
            dt = f.dtype;
        }
    }
}

bool blank(std::string_view s) noexcept {
    return std::all_of(s.begin(), s.end(), [](char ch) {
        return ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n';
    });
}

std::vector<std::string_view> split_lines(std::string_view text) {
    std::vector<std::string_view> out;
    const char* p = text.data();
    const char* end = p + text.size();
    while (p < end) {
        const char* nl = static_cast<const char*>(std::memchr(p, '\n', static_cast<std::size_t>(end - p)));
        const char* stop = nl ? nl : end;
        const std::string_view line(p, static_cast<std::size_t>(stop - p));
        if (!blank(line)) out.push_back(line);
        p = nl ? nl + 1 : end;
    }
    return out;
}

/// Element spans of the record array at @p records_path.
std::vector<std::string_view> split_array(std::string_view text, const std::string& records_path) {
    Cursor c(text.data(), text.data() + text.size());
    if (!records_path.empty()) {
        for (const Step& s : compile_path(records_path)) {
            if (s.kind == Step::Wild) throw InvalidArgumentError("records_path: [*] not allowed");
            bool found = false;
            if (s.kind == Step::Key) {
                c.expect('{');
                if (!c.eat('}')) {
                    do {
                        std::string key;
                        if (c.peek() != '"') c.fail("expected object key");
                        c.string(key);
                        c.expect(':');
                        if (key == s.key) {
                            found = true;
                            break;
                        }
                        c.skip_value();
                    } while (c.eat(','));
                }
            } else {
                c.expect('[');
                for (std::size_t i = 0; !found && c.peek() != ']'; ++i) {
                    if (i == s.index) {
                        found = true;
                        break;
                    }
                    c.skip_value();
                    if (!c.eat(',')) break;
                }
            }
            if (!found) throw IOError("JsonDataset: records_path \"" + records_path + "\" not found");
        }
    }
    std::vector<std::string_view> out;
    c.expect('[');
    if (c.eat(']')) return out;
    do {
        c.ws();
        const char* s = c.p;
        c.skip_value();
        out.emplace_back(s, static_cast<std::size_t>(c.p - s));
    } while (c.eat(','));
    c.expect(']');
    return out;
}

JsonFormat detect_format(std::string_view text, const JsonOptions& opt) {
    if (opt.format != JsonFormat::Auto) return opt.format;
    if (!opt.records_path.empty()) return JsonFormat::Array;
    Cursor c(text.data(), text.data() + text.size());
    if (c.at_end() || *c.p != '[') return JsonFormat::Lines;
    // '[' could open a document or be the first line of array-per-line JSONL.
    try {
        c.skip_value();
        return c.at_end() ? JsonFormat::Array : JsonFormat::Lines;
    } catch (const IOError&) {
        return JsonFormat::Lines;
    }
}

} // namespace

std::vector<JsonField> infer_json_fields(const json::Value& record) {
    std::vector<JsonField> out;
    if (!record.is_object()) {
        throw InvalidArgumentError("infer_json_fields: record is not an object");
    }
    auto numeric_tree = [](const json::Value& v, auto&& self) -> bool {
        if (v.is_number() || v.is_bool() || v.is_null()) return true;
        if (!v.is_array()) return false;
        for (const json::Value& e : v.as_array()) {
            if (!self(e, self)) return false;
        }
        return true;
    };
    for (const auto& [key, v] : record.as_object()) {
        JsonField f;
        f.path = key;
        if (numeric_tree(v, numeric_tree)) {
            f.role = JsonRole::Input;
        } else if (v.is_string()) {
            f.role = JsonRole::Metadata;
        } else {
            continue; // nested objects need explicit paths
        }
        out.push_back(std::move(f));
    }
    return out;
}

namespace {

void schema_fields(const json::Value& schema, const std::string& prefix, std::vector<JsonField>& out) {
    const json::Value* props = schema.find("properties");
    if (props == nullptr || !props->is_object()) return;
    for (const auto& [key, prop] : props->as_object()) {
        if (!prop.is_object()) continue;
        const std::string path = prefix.empty() ? key : prefix + "." + key;
        const std::string type = prop.find("type") && prop.find("type")->is_string()
                                     ? prop.find("type")->as_string()
                                     : std::string();
        if (type == "object") {
            schema_fields(prop, path, out);
            continue;
        }
        JsonField f;
        f.path = path;
        const json::Value* leaf = &prop;
        Shape shape;
        while (leaf->find("type") && leaf->find("type")->is_string() &&
               leaf->find("type")->as_string() == "array") {
            const json::Value* items = leaf->find("items");
            if (items == nullptr || !items->is_object()) break;
            const std::int64_t lo = leaf->get_int("minItems", -1);
            const std::int64_t hi = leaf->get_int("maxItems", -2);
            if (lo != hi) {
                shape.clear();
                leaf = items;
                while (leaf->find("items") && leaf->find("items")->is_object()) leaf = leaf->find("items");
                break;
            }
            shape.push_back(static_cast<std::size_t>(lo));
            leaf = items;
        }
        const std::string leaf_type = leaf->get_string("type", "number");
        if (leaf_type == "integer") f.dtype = DType::Int64;
        else if (leaf_type == "boolean") f.dtype = DType::Bool;
        else if (leaf_type == "string") {
            const json::Value* en = leaf->find("enum");
            if (en != nullptr && en->is_array()) {
                f.dtype = DType::Int64;
                for (const json::Value& e : en->as_array()) f.categories.push_back(e.as_string());
            } else {
                f.role = JsonRole::Metadata;
            }
        } else if (leaf_type != "number") {
            continue;
        }
        f.shape = std::move(shape);
        const std::string role = prop.get_string("x-nexusdata-role", "");
        if (role == "label") f.role = JsonRole::Label;
        else if (role == "extra") f.role = JsonRole::Extra;
        else if (role == "metadata") f.role = JsonRole::Metadata;
        else if (role == "input") f.role = JsonRole::Input;
        else if (!role.empty()) {
            throw InvalidArgumentError("json schema: unknown x-nexusdata-role \"" + role + "\"");
        }
        const std::string dt = prop.get_string("x-nexusdata-dtype", "");
        if (!dt.empty()) f.dtype = dtype_from_string(dt);
        if (const json::Value* d = prop.find("default"); d != nullptr && d->is_number()) {
            f.default_value = d->as_double();
        }
        out.push_back(std::move(f));
    }
}

} // namespace

std::vector<JsonField> json_fields_from_schema(const json::Value& schema) {
    if (!schema.is_object()) throw InvalidArgumentError("json_fields_from_schema: schema is not an object");
    std::vector<JsonField> out;
    schema_fields(schema, "", out);
    if (out.empty()) throw InvalidArgumentError("json_fields_from_schema: no usable properties");
    return out;
}

JsonDataset::JsonDataset(const std::string& path, JsonOptions options) {
    const InputBytes in = read_input(path, std::nullopt, static_cast<unsigned>(std::max(0, options.num_threads)));
    try {
        load(std::string_view(reinterpret_cast<const char*>(in.data()), in.size()), std::move(options));
    } catch (const IOError& e) {
        throw IOError(path + ": " + e.what());
    }
}

std::shared_ptr<JsonDataset> JsonDataset::from_string(std::string_view text, JsonOptions options) {
    std::shared_ptr<JsonDataset> ds(new JsonDataset());
    ds->load(text, std::move(options));
    return ds;
}

void JsonDataset::load(std::string_view text, JsonOptions options) {
    const JsonFormat fmt = detect_format(text, options);
    std::vector<std::string_view> records =
        fmt == JsonFormat::Lines ? split_lines(text) : split_array(text, options.records_path);
    if (options.fields.empty()) {
        if (records.empty()) throw IOError("JsonDataset: no records");
        options.fields = infer_json_fields(json::parse(records[0]));
    }
    validate_fields(options.fields);
    fields_ = options.fields;

    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    const std::size_t threads = options.num_threads > 0 ? static_cast<std::size_t>(options.num_threads) : hw;
    const std::size_t blocks = std::max<std::size_t>(1, std::min(records.size() / 256 + 1, threads * 4));
    std::vector<std::vector<Column>> parts(blocks);
    std::vector<std::size_t> dropped(blocks, 0);
    std::mutex mu;
    std::exception_ptr err;
    {
        ThreadPool pool(std::min(threads, blocks));
        pool.parallel_for(blocks, [&](std::size_t b) {
            try {
                Extractor ex(fields_);
                std::vector<Column> cols = ex.make_columns();
                const std::size_t lo = records.size() * b / blocks;
                const std::size_t hi = records.size() * (b + 1) / blocks;
                for (std::size_t r = lo; r < hi; ++r) {
                    try {
                        ex.extract(records[r], cols);
                    } catch (const IOError& e) {
                        if (!options.skip_invalid) {
                            throw IOError("record " + std::to_string(r) + ": " + e.what());
                        }
                        ++dropped[b];
                    }
                }
                parts[b] = std::move(cols);
            } catch (...) {
                std::lock_guard lock(mu);
                if (!err) err = std::current_exception();
            }
        });
    }
    if (err) std::rethrow_exception(err);

    Extractor proto(fields_);
    std::vector<Column> merged = proto.make_columns();
    for (std::size_t f = 0; f < merged.size(); ++f) {
        std::size_t bytes = 0;
        for (const auto& p : parts) bytes += p[f].data.size();
        merged[f].data.reserve(bytes);
        for (const auto& p : parts) merged[f].append(p[f]);
    }
    columns_.clear();
    for (Column& c : merged) columns_.push_back(std::make_shared<Column>(std::move(c)));
    n_ = columns_.empty() ? 0 : columns_[0]->records();
    skipped_ = 0;
    for (std::size_t d : dropped) skipped_ += d;
}

std::size_t JsonDataset::size() const {
    return n_;
}

Sample JsonDataset::get(std::size_t index) const {
    if (index >= n_) {
        throw IndexError(format_index_error("JsonDataset::get", index, n_));
    }
    std::vector<const Column*> cols;
    cols.reserve(columns_.size());
    for (const auto& c : columns_) cols.push_back(c.get());
    return make_sample(fields_, cols, index, index);
}

namespace {

class JsonLinesIterator final : public IterableDataset::Iterator {
public:
    JsonLinesIterator(const std::string& path, const JsonOptions& opt)
        : reader_(open_source(path)), path_(path), skip_invalid_(opt.skip_invalid), fields_(opt.fields) {
        advance_first(opt);
    }

    [[nodiscard]] bool has_next() const override { return ready_; }

    [[nodiscard]] Sample next() override {
        if (!ready_) throw IndexError("JsonLinesIterable: exhausted");
        std::vector<const Column*> ptrs;
        for (const Column& c : cols_) ptrs.push_back(&c);
        Sample s = make_sample(fields_, ptrs, 0, emitted_++);
        advance();
        return s;
    }

private:
    void advance_first(const JsonOptions& opt) {
        if (fields_.empty()) {
            std::string_view line;
            while (reader_.next(line) && blank(line)) {}
            if (reader_.line_number() == 0 || blank(line)) return;
            fields_ = infer_json_fields(json::parse(line));
            validate_fields(fields_);
            ex_ = std::make_unique<Extractor>(fields_);
            cols_ = ex_->make_columns();
            if (try_line(line)) {
                ready_ = true;
                return;
            }
        } else {
            validate_fields(fields_);
            ex_ = std::make_unique<Extractor>(fields_);
            cols_ = ex_->make_columns();
        }
        (void)opt;
        advance();
    }

    bool try_line(std::string_view line) {
        for (Column& c : cols_) c.clear();
        try {
            ex_->extract(line, cols_);
            return true;
        } catch (const IOError& e) {
            if (!skip_invalid_) {
                throw IOError(path_ + ":" + std::to_string(reader_.line_number()) + ": " + e.what());
            }
            return false;
        }
    }

    void advance() {
        ready_ = false;
        std::string_view line;
        while (reader_.next(line)) {
            if (blank(line)) continue;
            if (try_line(line)) {
                ready_ = true;
                return;
            }
        }
    }

    LineReader reader_;
    std::string path_;
    bool skip_invalid_;
    std::vector<JsonField> fields_;
    std::unique_ptr<Extractor> ex_;
    std::vector<Column> cols_;
    std::size_t emitted_ = 0;
    bool ready_ = false;
};

} // namespace

JsonLinesIterable::JsonLinesIterable(std::string path, JsonOptions options)
    : path_(std::move(path)), options_(std::move(options)) {
    if (!options_.fields.empty()) validate_fields(options_.fields);
}

std::unique_ptr<IterableDataset::Iterator> JsonLinesIterable::make_iterator() const {
    return std::make_unique<JsonLinesIterator>(path_, options_);
}

} // namespace nexusdata
