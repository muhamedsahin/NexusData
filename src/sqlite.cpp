#include "nexusdata/dataset/sqlite.hpp"

#include <cstring>

#include "nexusdata/core/error.hpp"

#if defined(NEXUSDATA_WITH_SQLITE)
#include <sqlite3.h>
#endif

namespace nexusdata {

bool sqlite_support_enabled() {
#if defined(NEXUSDATA_WITH_SQLITE)
    return true;
#else
    return false;
#endif
}

#if defined(NEXUSDATA_WITH_SQLITE)

SqliteDataset::SqliteDataset(const std::string& db_path, SqliteOptions opt) {
    if (opt.sql.empty()) {
        throw InvalidArgumentError("SqliteDataset: sql is required");
    }
    sqlite3* db = nullptr;
    if (sqlite3_open_v2(db_path.c_str(), &db, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) {
        const char* msg = db ? sqlite3_errmsg(db) : "open failed";
        if (db) {
            sqlite3_close(db);
        }
        throw IOError(std::string("SqliteDataset: cannot open db: ") + msg);
    }
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db, opt.sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        std::string msg = sqlite3_errmsg(db);
        sqlite3_close(db);
        throw IOError("SqliteDataset: prepare failed: " + msg);
    }
    const int cols = sqlite3_column_count(stmt);
    if (cols < 2) {
        sqlite3_finalize(stmt);
        sqlite3_close(db);
        throw IOError("SqliteDataset: need >= 2 columns");
    }
    int label_col = opt.label_column;
    if (label_col < 0) {
        label_col = cols - 1;
    }
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        Row row;
        for (int c = 0; c < cols; ++c) {
            if (c == label_col) {
                continue;
            }
            row.features.push_back(static_cast<float>(sqlite3_column_double(stmt, c)));
        }
        row.label = static_cast<float>(sqlite3_column_double(stmt, label_col));
        rows_.push_back(std::move(row));
    }
    sqlite3_finalize(stmt);
    sqlite3_close(db);
    if (rows_.empty()) {
        throw IOError("SqliteDataset: query returned no rows");
    }
}

#else

SqliteDataset::SqliteDataset(const std::string&, SqliteOptions) {
    throw InvalidArgumentError(
        "SqliteDataset: built without SQLite. Reconfigure with -DNEXUSDATA_WITH_SQLITE=ON");
}

#endif

std::size_t SqliteDataset::size() const {
    return rows_.size();
}

Sample SqliteDataset::get(std::size_t index) const {
    if (index >= rows_.size()) {
        throw IndexError("SqliteDataset::get out of range");
    }
    const auto& r = rows_[index];
    Sample s;
    s.input = NDArray(Shape{r.features.size()}, DType::Float32);
    std::memcpy(s.input.data(), r.features.data(), r.features.size() * sizeof(float));
    s.label = NDArray(Shape{1}, DType::Float32);
    s.label.data<float>()[0] = r.label;
    return s;
}

} // namespace nexusdata
