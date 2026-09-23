#include "nexusdata/dataset/postgres.hpp"

#include <cstring>
#include <fstream>
#include <limits>
#include <utility>

#include "nexusdata/core/error.hpp"
#include "sql_rows.hpp"

#if defined(NEXUSDATA_WITH_POSTGRES)
#include <libpq-fe.h>
#endif

namespace nexusdata {
namespace {

constexpr std::size_t kMaxField = 16u * 1024u * 1024u;

const char* kNoLibpq =
    "PostgresDataset: built without libpq. Reconfigure with -DNEXUSDATA_WITH_POSTGRES=ON";

struct Cursor {
    const std::uint8_t* p = nullptr;
    const std::uint8_t* end = nullptr;

    [[nodiscard]] std::size_t left() const { return static_cast<std::size_t>(end - p); }

    void need(std::size_t n, const char* what) const {
        if (left() < n) {
            throw IOError(std::string("COPY BINARY: truncated ") + what);
        }
    }

    std::uint16_t u16() {
        need(2, "int16");
        const std::uint16_t v = static_cast<std::uint16_t>((p[0] << 8) | p[1]);
        p += 2;
        return v;
    }
    std::int16_t i16() { return static_cast<std::int16_t>(u16()); }

    std::uint32_t u32() {
        need(4, "int32");
        const std::uint32_t v = (static_cast<std::uint32_t>(p[0]) << 24) |
                                (static_cast<std::uint32_t>(p[1]) << 16) |
                                (static_cast<std::uint32_t>(p[2]) << 8) |
                                static_cast<std::uint32_t>(p[3]);
        p += 4;
        return v;
    }
    std::int32_t i32() { return static_cast<std::int32_t>(u32()); }

    std::uint64_t u64() {
        need(8, "int64");
        std::uint64_t v = 0;
        for (int i = 0; i < 8; ++i) {
            v = (v << 8) | p[i];
        }
        p += 8;
        return v;
    }

    void skip(std::size_t n, const char* what) {
        need(n, what);
        p += n;
    }

    void bytes(std::size_t n, void* dst) {
        need(n, "field");
        std::memcpy(dst, p, n);
        p += n;
    }
};

void read_header(Cursor& c) {
    static const std::uint8_t kSig[] = {'P', 'G', 'C', 'O', 'P', 'Y', '\n', 0xFF, '\r', '\n', 0};
    c.need(sizeof(kSig), "signature");
    if (std::memcmp(c.p, kSig, sizeof(kSig)) != 0) {
        throw IOError("COPY BINARY: bad signature");
    }
    c.p += sizeof(kSig);
    (void)c.u32(); // flags; OID presence is carried in the extension
    const std::uint32_t ext = c.u32();
    if (ext > kMaxField) {
        throw IOError("COPY BINARY: header extension is too large");
    }
    c.skip(ext, "header extension");
}

sql_detail::Cell read_field(Cursor& c, const SqlColumn& col) {
    const std::int32_t len = c.i32();
    sql_detail::Cell cell;
    if (len < 0) {
        if (len != -1) {
            throw IOError("COPY BINARY: invalid field length");
        }
        if (col.type == SqlType::Text) {
            cell.text = true;
            return cell;
        }
        if (col.type == SqlType::Float32 || col.type == SqlType::Float64) {
            cell.number = std::numeric_limits<double>::quiet_NaN();
            return cell;
        }
        throw IOError("COPY BINARY: null in non-float column '" + col.name + "'");
    }
    if (static_cast<std::size_t>(len) > kMaxField) {
        throw IOError("COPY BINARY: field '" + col.name + "' is too large");
    }
    auto width = [&](std::size_t n) {
        if (static_cast<std::size_t>(len) != n) {
            throw IOError("COPY BINARY: column '" + col.name + "' has width " +
                          std::to_string(len) + ", expected " + std::to_string(n));
        }
    };
    switch (col.type) {
        case SqlType::Bool: {
            width(1);
            std::uint8_t b = 0;
            c.bytes(1, &b);
            cell.is_int = true;
            cell.integer = b ? 1 : 0;
            break;
        }
        case SqlType::Int16: {
            width(2);
            cell.is_int = true;
            cell.integer = static_cast<std::int16_t>(c.u16());
            break;
        }
        case SqlType::Int32: {
            width(4);
            cell.is_int = true;
            cell.integer = static_cast<std::int32_t>(c.u32());
            break;
        }
        case SqlType::Int64: {
            width(8);
            cell.is_int = true;
            cell.integer = static_cast<std::int64_t>(c.u64());
            break;
        }
        case SqlType::Float32: {
            width(4);
            const std::uint32_t bits = c.u32();
            float v = 0;
            std::memcpy(&v, &bits, 4);
            cell.number = static_cast<double>(v);
            break;
        }
        case SqlType::Float64: {
            width(8);
            const std::uint64_t bits = c.u64();
            double v = 0;
            std::memcpy(&v, &bits, 8);
            cell.number = v;
            break;
        }
        case SqlType::Text:
            cell.text = true;
            cell.bytes.resize(static_cast<std::size_t>(len));
            if (len > 0) {
                c.bytes(static_cast<std::size_t>(len), cell.bytes.data());
            }
            break;
    }
    return cell;
}

/// false = trailer (no more tuples).
bool read_tuple(Cursor& c, const std::vector<SqlColumn>& cols, std::vector<sql_detail::Cell>& cells) {
    if (c.left() == 0) {
        throw IOError("COPY BINARY: missing trailer");
    }
    const std::int16_t n = c.i16();
    if (n < 0) {
        if (n != -1) {
            throw IOError("COPY BINARY: bad trailer");
        }
        return false;
    }
    if (static_cast<std::size_t>(n) != cols.size()) {
        throw IOError("COPY BINARY: tuple has " + std::to_string(n) + " fields, schema has " +
                      std::to_string(cols.size()));
    }
    cells.clear();
    cells.reserve(cols.size());
    for (const SqlColumn& col : cols) {
        cells.push_back(read_field(c, col));
    }
    return true;
}

void require_columns(const std::vector<SqlColumn>& cols) {
    if (cols.empty()) {
        throw InvalidArgumentError("COPY BINARY: at least one column is required");
    }
    for (const SqlColumn& c : cols) {
        if (c.name.empty()) {
            throw InvalidArgumentError("COPY BINARY: column name is empty");
        }
    }
}

std::vector<Sample> decode_all(const std::uint8_t* data, std::size_t size,
                               const std::vector<SqlColumn>& cols) {
    require_columns(cols);
    if (!data && size > 0) {
        throw InvalidArgumentError("COPY BINARY: null buffer");
    }
    Cursor c{data, data + size};
    read_header(c);
    std::vector<Sample> rows;
    std::vector<sql_detail::Cell> cells;
    while (read_tuple(c, cols, cells)) {
        rows.push_back(sql_detail::make_row(cols, cells, static_cast<std::int64_t>(rows.size())));
    }
    return rows;
}

std::vector<std::string> names_of(const std::vector<SqlColumn>& cols) {
    std::vector<std::string> names;
    names.reserve(cols.size());
    for (const SqlColumn& c : cols) {
        names.push_back(c.name);
    }
    return names;
}

std::vector<std::uint8_t> read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw IOError("COPY BINARY: cannot open " + quote_path(path));
    }
    in.seekg(0, std::ios::end);
    const auto n = in.tellg();
    if (n < 0) {
        throw IOError("COPY BINARY: cannot size " + quote_path(path));
    }
    in.seekg(0);
    std::vector<std::uint8_t> buf(static_cast<std::size_t>(n));
    if (n > 0) {
        in.read(reinterpret_cast<char*>(buf.data()), n);
        if (!in) {
            throw IOError("COPY BINARY: short read " + quote_path(path));
        }
    }
    return buf;
}

#if defined(NEXUSDATA_WITH_POSTGRES)

struct PgConn {
    PGconn* c = nullptr;
    explicit PgConn(const std::string& conninfo) : c(PQconnectdb(conninfo.c_str())) {
        if (!c || PQstatus(c) != CONNECTION_OK) {
            const std::string msg = c ? PQerrorMessage(c) : "connect failed";
            if (c) PQfinish(c);
            c = nullptr;
            throw IOError("PostgresDataset: connect failed: " + msg);
        }
    }
    ~PgConn() {
        if (c) PQfinish(c);
    }
    PgConn(const PgConn&) = delete;
    PgConn& operator=(const PgConn&) = delete;
};

std::vector<std::uint8_t> copy_out(PGconn* conn, const std::string& query) {
    const std::string sql = "COPY (" + query + ") TO STDOUT WITH (FORMAT binary)";
    PGresult* res = PQexec(conn, sql.c_str());
    if (!res || PQresultStatus(res) != PGRES_COPY_OUT) {
        const std::string msg = PQerrorMessage(conn);
        if (res) PQclear(res);
        throw IOError("PostgresDataset: COPY failed: " + msg);
    }
    PQclear(res);
    std::vector<std::uint8_t> buf;
    for (;;) {
        char* chunk = nullptr;
        const int n = PQgetCopyData(conn, &chunk, 0);
        if (n == -1) break;
        if (n < 0) {
            if (chunk) PQfreemem(chunk);
            throw IOError(std::string("PostgresDataset: COPY read failed: ") + PQerrorMessage(conn));
        }
        buf.insert(buf.end(), reinterpret_cast<std::uint8_t*>(chunk),
                   reinterpret_cast<std::uint8_t*>(chunk) + n);
        PQfreemem(chunk);
    }
    res = PQgetResult(conn);
    if (res) {
        const auto st = PQresultStatus(res);
        const std::string msg = PQresultErrorMessage(res);
        PQclear(res);
        if (st != PGRES_COMMAND_OK) {
            throw IOError("PostgresDataset: COPY end failed: " + msg);
        }
    }
    return buf;
}

void check_opt(const PostgresOptions& opt) {
    if (opt.conninfo.empty() || opt.query.empty()) {
        throw InvalidArgumentError("PostgresDataset: conninfo and query are required");
    }
    require_columns(opt.columns);
}

#endif

class CopyIter : public IterableDataset::Iterator {
public:
    CopyIter(std::shared_ptr<std::vector<std::uint8_t>> bytes, std::vector<SqlColumn> cols)
        : bytes_(std::move(bytes)), cols_(std::move(cols)) {
        cur_.p = bytes_->data();
        cur_.end = bytes_->data() + bytes_->size();
        read_header(cur_);
        pull();
    }
    [[nodiscard]] bool has_next() const override { return has_; }
    [[nodiscard]] Sample next() override {
        if (!has_) {
            throw IndexError("PostgresCopyIterable: exhausted");
        }
        Sample s = std::move(pending_);
        pull();
        return s;
    }

private:
    void pull() {
        std::vector<sql_detail::Cell> cells;
        has_ = read_tuple(cur_, cols_, cells);
        if (has_) {
            pending_ = sql_detail::make_row(cols_, cells, index_++);
        }
    }
    std::shared_ptr<std::vector<std::uint8_t>> bytes_;
    std::vector<SqlColumn> cols_;
    Cursor cur_{};
    Sample pending_{};
    std::int64_t index_ = 0;
    bool has_ = false;
};

} // namespace

bool postgres_support_enabled() {
#if defined(NEXUSDATA_WITH_POSTGRES)
    return true;
#else
    return false;
#endif
}

struct PostgresDataset::Impl {
    std::vector<Sample> rows;
};

PostgresCopyDataset::PostgresCopyDataset(const std::string& path, std::vector<SqlColumn> columns) {
    const auto buf = read_file(path);
    names_ = names_of(columns);
    rows_ = decode_all(buf.data(), buf.size(), columns);
}

PostgresCopyDataset::PostgresCopyDataset(const std::uint8_t* data, std::size_t size,
                                         std::vector<SqlColumn> columns) {
    names_ = names_of(columns);
    rows_ = decode_all(data, size, std::move(columns));
}

std::size_t PostgresCopyDataset::size() const { return rows_.size(); }

Sample PostgresCopyDataset::get(std::size_t index) const {
    if (index >= rows_.size()) {
        throw IndexError(format_index_error("PostgresCopyDataset::get", index, rows_.size()));
    }
    return rows_[index];
}

PostgresCopyIterable::PostgresCopyIterable(const std::uint8_t* data, std::size_t size,
                                           std::vector<SqlColumn> columns)
    : bytes_(std::make_shared<std::vector<std::uint8_t>>()), columns_(std::move(columns)) {
    require_columns(columns_);
    if (size > 0) {
        if (!data) {
            throw InvalidArgumentError("COPY BINARY: null buffer");
        }
        bytes_->assign(data, data + size);
    }
}

std::unique_ptr<IterableDataset::Iterator> PostgresCopyIterable::make_iterator() const {
    return std::make_unique<CopyIter>(bytes_, columns_);
}

#if defined(NEXUSDATA_WITH_POSTGRES)

PostgresDataset::PostgresDataset(PostgresOptions opt) {
    check_opt(opt);
    PgConn conn(opt.conninfo);
    const auto buf = copy_out(conn.c, opt.query);
    impl_ = std::make_shared<Impl>();
    impl_->rows = decode_all(buf.data(), buf.size(), opt.columns);
}

std::size_t PostgresDataset::size() const { return impl_->rows.size(); }

Sample PostgresDataset::get(std::size_t index) const {
    if (index >= impl_->rows.size()) {
        throw IndexError(format_index_error("PostgresDataset::get", index, impl_->rows.size()));
    }
    return impl_->rows[index];
}

PostgresIterable::PostgresIterable(PostgresOptions opt) : opt_(std::move(opt)) { check_opt(opt_); }

std::unique_ptr<IterableDataset::Iterator> PostgresIterable::make_iterator() const {
    PgConn conn(opt_.conninfo);
    auto buf = std::make_shared<std::vector<std::uint8_t>>(copy_out(conn.c, opt_.query));
    return std::make_unique<CopyIter>(std::move(buf), opt_.columns);
}

#else

PostgresDataset::PostgresDataset(PostgresOptions) {
    throw InvalidArgumentError(kNoLibpq);
}
std::size_t PostgresDataset::size() const { return 0; }
Sample PostgresDataset::get(std::size_t) const {
    throw InvalidArgumentError(kNoLibpq);
}

PostgresIterable::PostgresIterable(PostgresOptions) {
    throw InvalidArgumentError(kNoLibpq);
}
std::unique_ptr<IterableDataset::Iterator> PostgresIterable::make_iterator() const {
    throw InvalidArgumentError(kNoLibpq);
}

#endif

} // namespace nexusdata
