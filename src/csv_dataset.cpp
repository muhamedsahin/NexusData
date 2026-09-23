#include "nexusdata/dataset/csv/csv_dataset.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "nexusdata/backend/thread_pool.hpp"
#include "nexusdata/core/error.hpp"
#include "nexusdata/io/compression.hpp"
#include "nexusdata/simd/byte_scan.hpp"

namespace nexusdata {

namespace {

[[nodiscard]] bool starts_with_bom(const std::string& s) {
    return s.size() >= 3 &&
           static_cast<unsigned char>(s[0]) == 0xEF &&
           static_cast<unsigned char>(s[1]) == 0xBB &&
           static_cast<unsigned char>(s[2]) == 0xBF;
}

[[nodiscard]] std::string strip_cr(std::string line) {
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }
    return line;
}

[[nodiscard]] bool is_blank(const std::string& s) {
    for (char c : s) {
        if (!std::isspace(static_cast<unsigned char>(c))) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool is_na(const std::string& field,
                         const std::vector<std::string>& na_values) {
    for (const auto& na : na_values) {
        if (field == na) {
            return true;
        }
    }
    return false;
}

/// Parse one CSV record possibly spanning multiple physical lines (quoted newlines).
[[nodiscard]] bool read_record(std::istream& in,
                               std::string& record_out,
                               std::size_t& lines_consumed) {
    record_out.clear();
    lines_consumed = 0;
    if (!in.good() && in.eof()) {
        return false;
    }

    // Track quotes only to support embedded newlines; keep bytes intact for split_fields.
    bool in_quotes = false;
    std::string line;
    while (std::getline(in, line)) {
        ++lines_consumed;
        line = strip_cr(std::move(line));
        if (lines_consumed > 1) {
            record_out.push_back('\n');
        }
        for (std::size_t i = 0; i < line.size(); ++i) {
            const char c = line[i];
            record_out.push_back(c);
            if (c == '"') {
                if (in_quotes && i + 1 < line.size() && line[i + 1] == '"') {
                    // Escaped quote: append the second quote and skip toggle.
                    record_out.push_back(line[i + 1]);
                    ++i;
                } else {
                    in_quotes = !in_quotes;
                }
            }
        }
        if (!in_quotes) {
            return true;
        }
    }
    if (!record_out.empty()) {
        // EOF while still in quotes — return what we have; parser will error.
        return true;
    }
    return false;
}

[[nodiscard]] std::vector<std::string> split_fields(const std::string& record,
                                                    char delimiter,
                                                    const std::string& path,
                                                    std::size_t line_no) {
    std::vector<std::string> fields;
    std::string cur;
    bool in_quotes = false;
    std::size_t col = 1;

    auto flush = [&]() {
        fields.push_back(cur);
        cur.clear();
    };

    for (std::size_t i = 0; i < record.size(); ++i) {
        const char c = record[i];
        if (c == '"') {
            if (in_quotes) {
                if (i + 1 < record.size() && record[i + 1] == '"') {
                    cur.push_back('"');
                    ++i;
                    ++col;
                } else {
                    in_quotes = false;
                }
            } else {
                if (!cur.empty()) {
                    throw IOError(path + ":" + std::to_string(line_no) + ":" +
                                  std::to_string(col) +
                                  " unexpected quote in unquoted field");
                }
                in_quotes = true;
            }
        } else if (c == delimiter && !in_quotes) {
            flush();
            ++col;
        } else {
            cur.push_back(c);
            ++col;
        }
    }
    if (in_quotes) {
        throw IOError(path + ":" + std::to_string(line_no) +
                      ": unclosed quoted field");
    }
    flush();
    return fields;
}

[[nodiscard]] double parse_double(const std::string& text,
                                  const std::string& path,
                                  std::size_t line_no,
                                  std::size_t col_1based) {
    if (text.empty()) {
        throw IOError(path + ":" + std::to_string(line_no) + ":" +
                      std::to_string(col_1based) + " \"\" could not be parsed as float");
    }
    char* end = nullptr;
    const double v = std::strtod(text.c_str(), &end);
    if (end == text.c_str() || *end != '\0') {
        throw IOError(path + ":" + std::to_string(line_no) + ":" +
                      std::to_string(col_1based) + " \"" + text +
                      "\" could not be parsed as float32");
    }
    return v;
}

template <typename T>
void write_scalar(NDArray& arr, std::size_t index, double value) {
    arr.data<T>()[index] = static_cast<T>(value);
}

void store_value(NDArray& arr, std::size_t index, double value, DType dt) {
    switch (dt) {
        case DType::Float32:
            write_scalar<float>(arr, index, value);
            break;
        case DType::Float64:
            write_scalar<double>(arr, index, value);
            break;
        case DType::Int32:
            write_scalar<std::int32_t>(arr, index, value);
            break;
        case DType::Int64:
            write_scalar<std::int64_t>(arr, index, value);
            break;
        case DType::UInt8:
            write_scalar<std::uint8_t>(arr, index, value);
            break;
        default:
            throw InvalidArgumentError("CSVDataset: unsupported dtype for CSV cells");
    }
}

} // namespace

CSVDataset::CSVDataset(const std::string& path, CSVOptions options) {
    std::vector<std::vector<std::string>> rows;
    std::size_t expected_cols = 0;
    std::size_t label_idx = 0;

    auto handle_bad_row = [&](const std::string& msg) {
        if (options.error_policy == CSVErrorPolicy::Throw) {
            throw IOError(msg);
        }
    };

    auto ingest_records = [&](std::vector<std::string> records) {
        bool first_record = true;
        for (std::size_t ri = 0; ri < records.size(); ++ri) {
            std::string& record = records[ri];
            const std::size_t physical_line = ri + 1;
            if (first_record && starts_with_bom(record)) {
                record.erase(0, 3);
            }
            if (!record.empty() && record.back() == '\r') {
                record.pop_back();
            }
            if (options.skip_blank_lines && is_blank(record)) {
                continue;
            }
            if (options.comment_prefix != '\0' && !record.empty() &&
                record[0] == options.comment_prefix) {
                continue;
            }
            std::vector<std::string> fields;
            try {
                fields = split_fields(record, options.delimiter, path, physical_line);
            } catch (const IOError& e) {
                handle_bad_row(e.what());
                continue;
            }
            if (first_record) {
                first_record = false;
                if (options.has_header) {
                    column_names_ = fields;
                    expected_cols = fields.size();
                    if (expected_cols < 2) {
                        throw IOError(path + ":1: need at least 2 columns (features + label)");
                    }
                    if (options.label_column_index.has_value()) {
                        label_idx = *options.label_column_index;
                    } else if (!options.label_column.empty()) {
                        bool found = false;
                        for (std::size_t i = 0; i < column_names_.size(); ++i) {
                            if (column_names_[i] == options.label_column) {
                                label_idx = i;
                                found = true;
                                break;
                            }
                        }
                        if (!found) {
                            throw IOError(path + ":1: label column \"" + options.label_column +
                                          "\" not found in header");
                        }
                    } else {
                        label_idx = expected_cols - 1;
                    }
                    continue;
                }
                expected_cols = fields.size();
                if (expected_cols < 2) {
                    throw IOError(path + ":" + std::to_string(physical_line) +
                                  ": need at least 2 columns");
                }
                column_names_.resize(expected_cols);
                for (std::size_t i = 0; i < expected_cols; ++i) {
                    column_names_[i] = "col" + std::to_string(i);
                }
                label_idx = options.label_column_index.value_or(expected_cols - 1);
            }
            if (fields.size() != expected_cols) {
                handle_bad_row(path + ":" + std::to_string(physical_line) +
                               ": expected " + std::to_string(expected_cols) +
                               " columns, got " + std::to_string(fields.size()));
                continue;
            }
            rows.push_back(std::move(fields));
        }
    };

    // Compressed input is decoded into memory, so it always takes the parallel path.
    if (options.use_mmap || codec_from_path(path) != Codec::None) {
        const InputBytes map = read_input(path, std::nullopt, static_cast<unsigned>(std::max(0, options.num_threads)));
        if (map.empty()) {
            throw IOError("CSVDataset: empty file \"" + path + "\"");
        }
        const auto starts = find_csv_record_starts(map.data(), map.size());
        std::vector<std::string> records(starts.size());
        const std::size_t nthreads =
            options.num_threads == 0
                ? std::max<std::size_t>(1, std::thread::hardware_concurrency())
                : static_cast<std::size_t>(options.num_threads);
        ThreadPool pool(nthreads);
        pool.parallel_for(starts.size(), [&](std::size_t i) {
            const std::size_t a = starts[i];
            std::size_t b = (i + 1 < starts.size()) ? starts[i + 1] - 1 : map.size();
            if (b > a && map.data()[b - 1] == '\n') {
                --b;
            }
            if (b > a && map.data()[b - 1] == '\r') {
                --b;
            }
            records[i].assign(reinterpret_cast<const char*>(map.data() + a), b > a ? b - a : 0);
        });
        ingest_records(std::move(records));
    } else {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            throw IOError("CSVDataset: cannot open file \"" + path + "\"");
        }
        std::vector<std::string> records;
        std::string record;
        std::size_t consumed = 0;
        while (read_record(in, record, consumed)) {
            records.push_back(record);
        }
        ingest_records(std::move(records));
    }

    if (rows.empty()) {
        throw IOError(path + ": no data rows found");
    }

    n_features_ = expected_cols - 1;
    const std::size_t n = rows.size();
    NDArray features(Shape{n, n_features_}, options.feature_dtype);
    NDArray labels(Shape{n}, options.label_dtype);

    // Parallel numeric conversion
    const std::size_t nthreads =
        options.num_threads == 0
            ? std::max<std::size_t>(1, std::thread::hardware_concurrency())
            : static_cast<std::size_t>(options.num_threads);
    ThreadPool pool(nthreads);
    std::vector<std::string> errors(n);
    pool.parallel_for(n, [&](std::size_t r) {
        try {
            const auto& fields = rows[r];
            const std::size_t line_no = r + 1 + (options.has_header ? 1 : 0);
            std::size_t feat_col = 0;
            for (std::size_t c = 0; c < fields.size(); ++c) {
                if (c == label_idx) {
                    continue;
                }
                if (is_na(fields[c], options.na_values)) {
                    throw IOError(path + ":" + std::to_string(line_no) + ":" +
                                  std::to_string(c + 1) + " missing value (NA)");
                }
                const double v = parse_double(fields[c], path, line_no, c + 1);
                store_value(features, r * n_features_ + feat_col, v, options.feature_dtype);
                ++feat_col;
            }
            if (is_na(fields[label_idx], options.na_values)) {
                throw IOError(path + ":" + std::to_string(line_no) + ":" +
                              std::to_string(label_idx + 1) + " missing label");
            }
            const double lv = parse_double(fields[label_idx], path, line_no, label_idx + 1);
            store_value(labels, r, lv, options.label_dtype);
        } catch (const std::exception& e) {
            errors[r] = e.what();
        }
    });
    for (const auto& e : errors) {
        if (!e.empty()) {
            if (options.error_policy == CSVErrorPolicy::Throw) {
                throw IOError(e);
            }
        }
    }

    impl_ = InMemoryDataset(std::move(features), std::move(labels));
}

std::size_t CSVDataset::size() const {
    return impl_.size();
}

Sample CSVDataset::get(std::size_t index) const {
    return impl_.get(index);
}

} // namespace nexusdata
