#pragma once

#include <cstdint>
#include <string>

namespace nexusdata {

/// Wire type of one SQL / COPY column. Integers stay exact (int64); floats are IEEE.
enum class SqlType : std::uint8_t { Int16, Int32, Int64, Float32, Float64, Bool, Text };

/// Where the column lands on a Sample. Auto: text → metadata, numeric → input.
enum class SqlRole : std::uint8_t { Auto, Input, Label, Extra, Metadata };

struct SqlColumn {
    std::string name;
    SqlType type = SqlType::Float64;
    SqlRole role = SqlRole::Auto;
};

} // namespace nexusdata
