#include "nexusdata/dataset/odbc.hpp"

#include <cstring>
#include <limits>

#include "nexusdata/core/error.hpp"
#include "sql_rows.hpp"

#if defined(NEXUSDATA_WITH_ODBC)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <sql.h>
#include <sqlext.h>
#include <vector>
#endif

namespace nexusdata {
namespace {

const char* kNoOdbc =
    "OdbcDataset: built without ODBC. Reconfigure with -DNEXUSDATA_WITH_ODBC=ON";

#if defined(NEXUSDATA_WITH_ODBC)

struct Env {
    SQLHENV h = nullptr;
    Env() {
        if (SQLAllocHandle(SQL_HANDLE_ENV, SQL_NULL_HANDLE, &h) != SQL_SUCCESS) {
            throw IOError("OdbcDataset: SQLAllocHandle(ENV) failed");
        }
        SQLSetEnvAttr(h, SQL_ATTR_ODBC_VERSION, reinterpret_cast<SQLPOINTER>(SQL_OV_ODBC3), 0);
    }
    ~Env() {
        if (h) SQLFreeHandle(SQL_HANDLE_ENV, h);
    }
    Env(const Env&) = delete;
    Env& operator=(const Env&) = delete;
};

struct Dbc {
    SQLHDBC h = nullptr;
    explicit Dbc(SQLHENV env) {
        if (SQLAllocHandle(SQL_HANDLE_DBC, env, &h) != SQL_SUCCESS) {
            throw IOError("OdbcDataset: SQLAllocHandle(DBC) failed");
        }
    }
    ~Dbc() {
        if (h) {
            SQLDisconnect(h);
            SQLFreeHandle(SQL_HANDLE_DBC, h);
        }
    }
    Dbc(const Dbc&) = delete;
    Dbc& operator=(const Dbc&) = delete;
};

struct Stmt {
    SQLHSTMT h = nullptr;
    explicit Stmt(SQLHDBC dbc) {
        if (SQLAllocHandle(SQL_HANDLE_STMT, dbc, &h) != SQL_SUCCESS) {
            throw IOError("OdbcDataset: SQLAllocHandle(STMT) failed");
        }
    }
    ~Stmt() {
        if (h) SQLFreeHandle(SQL_HANDLE_STMT, h);
    }
    Stmt(const Stmt&) = delete;
    Stmt& operator=(const Stmt&) = delete;
};

[[noreturn]] void odbc_fail(SQLSMALLINT kind, SQLHANDLE h, const char* what) {
    SQLCHAR state[6] = {};
    SQLCHAR msg[512] = {};
    SQLSMALLINT len = 0;
    SQLGetDiagRec(kind, h, 1, state, nullptr, msg, sizeof(msg), &len);
    throw IOError(std::string("OdbcDataset: ") + what + ": " +
                  reinterpret_cast<char*>(msg));
}

sql_detail::Cell cell_from_binding(const SqlColumn& col, const char* text, SQLLEN ind) {
    sql_detail::Cell cell;
    if (ind == SQL_NULL_DATA) {
        if (col.type == SqlType::Text) {
            cell.text = true;
            return cell;
        }
        if (col.type == SqlType::Float32 || col.type == SqlType::Float64) {
            cell.number = std::numeric_limits<double>::quiet_NaN();
            return cell;
        }
        throw IOError("OdbcDataset: null in non-float column '" + col.name + "'");
    }
    if (col.type == SqlType::Text) {
        cell.text = true;
        if (ind > 0) cell.bytes.assign(text, text + ind);
        return cell;
    }
    if (col.type == SqlType::Float32 || col.type == SqlType::Float64) {
        double v = 0;
        std::memcpy(&v, text, sizeof(double));
        cell.number = v;
        return cell;
    }
    std::int64_t v = 0;
    std::memcpy(&v, text, sizeof(std::int64_t));
    if (col.type == SqlType::Bool) v = v ? 1 : 0;
    cell.is_int = true;
    cell.integer = v;
    return cell;
}

std::vector<Sample> fetch_all(const OdbcOptions& opt) {
    if (opt.connection.empty() || opt.query.empty()) {
        throw InvalidArgumentError("OdbcDataset: connection and query are required");
    }
    if (opt.columns.empty()) {
        throw InvalidArgumentError("OdbcDataset: at least one column is required");
    }
    const SQLULEN batch = static_cast<SQLULEN>(opt.fetch_size == 0 ? 1 : opt.fetch_size);
    Env env;
    Dbc dbc(env.h);
    SQLCHAR outbuf[8];
    SQLSMALLINT outlen = 0;
    const SQLRETURN connected = SQLDriverConnect(
        dbc.h, nullptr, reinterpret_cast<SQLCHAR*>(const_cast<char*>(opt.connection.c_str())),
        SQL_NTS, outbuf, 0, &outlen, SQL_DRIVER_NOPROMPT);
    if (connected != SQL_SUCCESS && connected != SQL_SUCCESS_WITH_INFO) {
        odbc_fail(SQL_HANDLE_DBC, dbc.h, "SQLDriverConnect");
    }
    Stmt stmt(dbc.h);
    if (SQLSetStmtAttr(stmt.h, SQL_ATTR_ROW_ARRAY_SIZE, reinterpret_cast<SQLPOINTER>(batch), 0) !=
        SQL_SUCCESS) {
        odbc_fail(SQL_HANDLE_STMT, stmt.h, "SQL_ATTR_ROW_ARRAY_SIZE");
    }
    SQLULEN got = 0;
    SQLSetStmtAttr(stmt.h, SQL_ATTR_ROWS_FETCHED_PTR, &got, 0);
    std::vector<SQLUSMALLINT> status(static_cast<std::size_t>(batch), SQL_ROW_SUCCESS);
    SQLSetStmtAttr(stmt.h, SQL_ATTR_ROW_STATUS_PTR, status.data(), 0);

    const SQLRETURN exec = SQLExecDirect(
        stmt.h, reinterpret_cast<SQLCHAR*>(const_cast<char*>(opt.query.c_str())), SQL_NTS);
    if (exec != SQL_SUCCESS && exec != SQL_SUCCESS_WITH_INFO) {
        odbc_fail(SQL_HANDLE_STMT, stmt.h, "SQLExecDirect");
    }

    constexpr std::size_t kText = 4096;
    struct Bound {
        SQLSMALLINT ctype = SQL_C_CHAR;
        std::size_t stride = 0;
        std::vector<char> buf;
        std::vector<SQLLEN> ind;
    };
    std::vector<Bound> bound(opt.columns.size());
    for (std::size_t c = 0; c < opt.columns.size(); ++c) {
        Bound& b = bound[c];
        b.ind.assign(static_cast<std::size_t>(batch), 0);
        if (opt.columns[c].type == SqlType::Text) {
            b.ctype = SQL_C_CHAR;
            b.stride = kText;
        } else if (opt.columns[c].type == SqlType::Float32 || opt.columns[c].type == SqlType::Float64) {
            b.ctype = SQL_C_DOUBLE;
            b.stride = sizeof(double);
        } else {
            b.ctype = SQL_C_SBIGINT;
            b.stride = sizeof(std::int64_t);
        }
        b.buf.assign(b.stride * static_cast<std::size_t>(batch), 0);
        const SQLRETURN br = SQLBindCol(stmt.h, static_cast<SQLUSMALLINT>(c + 1), b.ctype, b.buf.data(),
                                         static_cast<SQLLEN>(b.stride), b.ind.data());
        if (br != SQL_SUCCESS && br != SQL_SUCCESS_WITH_INFO) {
            odbc_fail(SQL_HANDLE_STMT, stmt.h, "SQLBindCol");
        }
    }

    std::vector<Sample> rows;
    for (;;) {
        const SQLRETURN fr = SQLFetch(stmt.h);
        if (fr == SQL_NO_DATA) break;
        if (fr != SQL_SUCCESS && fr != SQL_SUCCESS_WITH_INFO) {
            odbc_fail(SQL_HANDLE_STMT, stmt.h, "SQLFetch");
        }
        for (SQLULEN r = 0; r < got; ++r) {
            if (status[static_cast<std::size_t>(r)] == SQL_ROW_NOROW) continue;
            std::vector<sql_detail::Cell> cells;
            cells.reserve(opt.columns.size());
            for (std::size_t c = 0; c < opt.columns.size(); ++c) {
                const char* p = bound[c].buf.data() + bound[c].stride * static_cast<std::size_t>(r);
                cells.push_back(cell_from_binding(opt.columns[c], p, bound[c].ind[static_cast<std::size_t>(r)]));
            }
            rows.push_back(sql_detail::make_row(opt.columns, cells, static_cast<std::int64_t>(rows.size())));
        }
    }
    return rows;
}

#endif

} // namespace

bool odbc_support_enabled() {
#if defined(NEXUSDATA_WITH_ODBC)
    return true;
#else
    return false;
#endif
}

#if defined(NEXUSDATA_WITH_ODBC)

OdbcDataset::OdbcDataset(OdbcOptions opt) : rows_(fetch_all(opt)) {}

#else

OdbcDataset::OdbcDataset(OdbcOptions) {
    throw InvalidArgumentError(kNoOdbc);
}

#endif

std::size_t OdbcDataset::size() const { return rows_.size(); }

Sample OdbcDataset::get(std::size_t index) const {
    if (index >= rows_.size()) {
        throw IndexError(format_index_error("OdbcDataset::get", index, rows_.size()));
    }
    return rows_[index];
}

} // namespace nexusdata
