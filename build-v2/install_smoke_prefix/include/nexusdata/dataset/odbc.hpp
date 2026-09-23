#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "nexusdata/dataset/dataset.hpp"
#include "nexusdata/dataset/sql_column.hpp"

namespace nexusdata {

struct OdbcOptions {
    /// ODBC connection string passed to SQLDriverConnect (no prompt).
    std::string connection;
    std::string query;
    /// One entry per selected column, in order. Required.
    std::vector<SqlColumn> columns;
    /// SQLFetch row-array size. Values below 1 are treated as 1.
    std::size_t fetch_size = 256;
};

/// Generic ODBC dataset (Windows driver manager). Bulk-fetches in row arrays
/// of `fetch_size`. Requires NEXUSDATA_WITH_ODBC=ON.
/// Without the driver manager the constructor throws InvalidArgumentError.
class OdbcDataset : public Dataset {
public:
    OdbcDataset(OdbcOptions opt);

    [[nodiscard]] std::size_t size() const override;
    [[nodiscard]] Sample get(std::size_t index) const override;

private:
    std::vector<Sample> rows_;
};

[[nodiscard]] bool odbc_support_enabled();

} // namespace nexusdata
