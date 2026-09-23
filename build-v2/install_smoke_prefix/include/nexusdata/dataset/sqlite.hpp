#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "nexusdata/dataset/dataset.hpp"

namespace nexusdata {

struct SqliteOptions {
    std::string sql; // e.g. SELECT f1,f2,label FROM t
    /// Column index of label in the SELECT result (-1 = last column).
    int label_column = -1;
};

/// Query-based SQLite dataset. Requires NEXUSDATA_WITH_SQLITE=ON (amalgamation).
/// Without SQLite, constructor throws a clear error.
class SqliteDataset : public Dataset {
public:
    SqliteDataset(const std::string& db_path, SqliteOptions opt);

    [[nodiscard]] std::size_t size() const override;
    [[nodiscard]] Sample get(std::size_t index) const override;

private:
    struct Row {
        std::vector<float> features;
        float label = 0;
    };
    std::vector<Row> rows_;
};

[[nodiscard]] bool sqlite_support_enabled();

} // namespace nexusdata
