#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "nexusdata/dataset/dataset.hpp"
#include "nexusdata/dataset/iterable.hpp"
#include "nexusdata/dataset/sql_column.hpp"

namespace nexusdata {

struct PostgresOptions {
    /// libpq connection string, e.g. "host=127.0.0.1 dbname=nd user=nd".
    std::string conninfo;
    /// SELECT (or any query) wrapped as COPY (query) TO STDOUT WITH (FORMAT binary).
    std::string query;
    /// One entry per result column, in order. Required.
    std::vector<SqlColumn> columns;
};

/// Map-style reader. Requires NEXUSDATA_WITH_POSTGRES=ON (libpq).
/// Without libpq the constructor throws InvalidArgumentError naming the flag.
/// Rows are fetched with COPY BINARY (not row-by-row SELECT).
class PostgresDataset : public Dataset {
public:
    PostgresDataset(PostgresOptions opt);

    [[nodiscard]] std::size_t size() const override;
    [[nodiscard]] Sample get(std::size_t index) const override;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

/// Streaming COPY BINARY reader (one libpq connection per iterator).
/// Same flag requirement as PostgresDataset.
class PostgresIterable : public IterableDataset {
public:
    explicit PostgresIterable(PostgresOptions opt);

    [[nodiscard]] std::unique_ptr<Iterator> make_iterator() const override;

private:
    PostgresOptions opt_;
};

/// Map-style view of an already-produced COPY BINARY payload (file or buffer).
/// Always available: the binary layout does not need libpq. PostgresDataset
/// uses this decoder after COPY TO STDOUT.
class PostgresCopyDataset : public Dataset {
public:
    PostgresCopyDataset(const std::string& path, std::vector<SqlColumn> columns);
    PostgresCopyDataset(const std::uint8_t* data, std::size_t size, std::vector<SqlColumn> columns);

    [[nodiscard]] std::size_t size() const override;
    [[nodiscard]] Sample get(std::size_t index) const override;
    [[nodiscard]] const std::vector<std::string>& column_names() const noexcept { return names_; }

private:
    std::vector<Sample> rows_;
    std::vector<std::string> names_;
};

/// Yields one Sample per COPY BINARY tuple. The buffer must outlive the iterators
/// only if it is the caller's; this class copies @p data.
class PostgresCopyIterable : public IterableDataset {
public:
    PostgresCopyIterable(const std::uint8_t* data, std::size_t size, std::vector<SqlColumn> columns);

    [[nodiscard]] std::unique_ptr<Iterator> make_iterator() const override;

private:
    std::shared_ptr<std::vector<std::uint8_t>> bytes_;
    std::vector<SqlColumn> columns_;
};

[[nodiscard]] bool postgres_support_enabled();

} // namespace nexusdata
