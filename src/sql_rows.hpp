#pragma once

// Shared by the COPY BINARY decoder and the ODBC fetcher. Not a public header.

#include <charconv>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "nexusdata/core/error.hpp"
#include "nexusdata/core/ndarray.hpp"
#include "nexusdata/dataset/sample.hpp"
#include "nexusdata/dataset/sql_column.hpp"

namespace nexusdata {
namespace sql_detail {

struct Cell {
    bool text = false;
    bool is_int = false;
    double number = 0;
    std::int64_t integer = 0;
    std::string bytes;
};

inline std::string format_num(double v) {
    char buf[64];
    const auto r = std::to_chars(buf, buf + sizeof(buf), v, std::chars_format::general, 17);
    return std::string(buf, r.ptr);
}

inline std::string format_int(std::int64_t v) {
    char buf[32];
    const auto r = std::to_chars(buf, buf + sizeof(buf), v);
    return std::string(buf, r.ptr);
}

/// Assign roles and build one Sample. Null never matches a numeric label; floats
/// use NaN, integers and bools must be present (the caller throws before this).
inline Sample make_row(const std::vector<SqlColumn>& cols, const std::vector<Cell>& cells,
                       std::int64_t row_index) {
    if (cols.size() != cells.size()) {
        throw IOError("SQL row: column count does not match the declared schema");
    }
    std::vector<std::size_t> inputs;
    int label = -1;
    for (std::size_t c = 0; c < cols.size(); ++c) {
        const SqlRole role = cols[c].role;
        const bool textual = cols[c].type == SqlType::Text || cells[c].text;
        if (textual && (role == SqlRole::Label || role == SqlRole::Input || role == SqlRole::Extra)) {
            throw InvalidArgumentError("SQL column '" + cols[c].name +
                                       "': text cannot be input, label or extra");
        }
        if (role == SqlRole::Label) {
            if (label >= 0) {
                throw InvalidArgumentError("SQL schema: more than one label column");
            }
            label = static_cast<int>(c);
        } else if (role == SqlRole::Input || (role == SqlRole::Auto && !textual)) {
            inputs.push_back(c);
        }
    }

    Sample s;
    auto put_meta = [&](std::size_t c) {
        if (cells[c].text) {
            s.metadata.set(cols[c].name, cells[c].bytes);
        } else if (cells[c].is_int) {
            s.metadata.set(cols[c].name, format_int(cells[c].integer));
        } else {
            s.metadata.set(cols[c].name, format_num(cells[c].number));
        }
    };
    auto as_array = [&](std::size_t c) {
        if (cells[c].is_int) {
            NDArray a(Shape{1}, DType::Int64);
            a.data<std::int64_t>()[0] = cells[c].integer;
            return a;
        }
        NDArray a(Shape{1}, DType::Float64);
        a.data<double>()[0] = cells[c].number;
        return a;
    };

    for (std::size_t c = 0; c < cols.size(); ++c) {
        const SqlRole role = cols[c].role;
        const bool textual = cols[c].type == SqlType::Text || cells[c].text;
        if (role == SqlRole::Metadata || (role == SqlRole::Auto && textual)) {
            put_meta(c);
        } else if (role == SqlRole::Extra) {
            s.extra_tensors.set(cols[c].name, as_array(c));
        }
    }

    if (inputs.size() == 1) {
        s.input = as_array(inputs[0]);
    } else if (inputs.size() > 1) {
        bool any_float = false;
        for (std::size_t c : inputs) {
            if (!cells[c].is_int) any_float = true;
        }
        if (any_float) {
            NDArray a(Shape{inputs.size()}, DType::Float64);
            for (std::size_t i = 0; i < inputs.size(); ++i) {
                const Cell& cell = cells[inputs[i]];
                a.data<double>()[i] = cell.is_int ? static_cast<double>(cell.integer) : cell.number;
            }
            s.input = std::move(a);
        } else {
            NDArray a(Shape{inputs.size()}, DType::Int64);
            for (std::size_t i = 0; i < inputs.size(); ++i) {
                a.data<std::int64_t>()[i] = cells[inputs[i]].integer;
            }
            s.input = std::move(a);
        }
    } else {
        s.input = NDArray(Shape{1}, DType::Float64);
        s.input.data<double>()[0] = 0;
    }

    if (label >= 0) {
        s.label = as_array(static_cast<std::size_t>(label));
    } else {
        s.label = NDArray(Shape{1}, DType::Int64);
        s.label.data<std::int64_t>()[0] = row_index;
    }
    return s;
}

} // namespace sql_detail
} // namespace nexusdata
