#include "nexusdata/dataset/jsonl.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include "nexusdata/backend/thread_pool.hpp"
#include "nexusdata/core/error.hpp"
#include "nexusdata/io/compression.hpp"
#include "nexusdata/simd/byte_scan.hpp"

namespace nexusdata {

namespace {

std::vector<double> parse_json_number_array(const char* begin, const char* end,
                                            const std::string& path, std::size_t line) {
    while (begin < end && std::isspace(static_cast<unsigned char>(*begin))) ++begin;
    while (begin < end && std::isspace(static_cast<unsigned char>(end[-1]))) --end;
    if (begin >= end || *begin != '[') {
        throw IOError(path + ":" + std::to_string(line) + ": expected JSON array");
    }
    ++begin;
    std::vector<double> vals;
    while (begin < end) {
        while (begin < end && std::isspace(static_cast<unsigned char>(*begin))) ++begin;
        if (begin < end && *begin == ']') {
            break;
        }
        char* parse_end = nullptr;
        const double v = std::strtod(begin, &parse_end);
        if (parse_end == begin) {
            throw IOError(path + ":" + std::to_string(line) + ": bad number in JSON array");
        }
        vals.push_back(v);
        begin = parse_end;
        while (begin < end && std::isspace(static_cast<unsigned char>(*begin))) ++begin;
        if (begin < end && *begin == ',') {
            ++begin;
            continue;
        }
        if (begin < end && *begin == ']') {
            break;
        }
        if (begin < end) {
            throw IOError(path + ":" + std::to_string(line) + ": unexpected token in JSON array");
        }
    }
    return vals;
}

} // namespace

JSONLDataset::JSONLDataset(const std::string& path, JSONLOptions options) {
    const InputBytes map = read_input(path, std::nullopt, static_cast<unsigned>(std::max(0, options.num_threads)));
    if (map.empty()) {
        throw IOError("JSONLDataset: empty file \"" + path + "\"");
    }
    auto starts = find_csv_record_starts(map.data(), map.size());
    // find_csv_record_starts treats quotes; for JSONL quotes in strings are rare in our subset.
    // Also add handling: lines ending without final newline already covered.

    std::vector<std::pair<const char*, const char*>> spans;
    spans.reserve(starts.size());
    for (std::size_t i = 0; i < starts.size(); ++i) {
        const std::size_t a = starts[i];
        std::size_t b = (i + 1 < starts.size()) ? starts[i + 1] - 1 : map.size();
        // trim CR
        if (b > a && map.data()[b - 1] == '\r') {
            --b;
        }
        if (b > a && map.data()[b - 1] == '\n') {
            --b;
        }
        if (b <= a) {
            continue;
        }
        spans.emplace_back(reinterpret_cast<const char*>(map.data() + a),
                           reinterpret_cast<const char*>(map.data() + b));
    }

    std::vector<std::vector<double>> rows(spans.size());
    std::size_t threads = options.num_threads == 0
                              ? std::max<std::size_t>(1, std::thread::hardware_concurrency())
                              : static_cast<std::size_t>(options.num_threads);
    ThreadPool pool(threads);
    pool.parallel_for(spans.size(), [&](std::size_t i) {
        rows[i] = parse_json_number_array(spans[i].first, spans[i].second, path, i + 1);
    });

    if (rows.empty()) {
        throw IOError("JSONLDataset: no rows");
    }
    const std::size_t width = rows[0].size();
    if (width < 2) {
        throw IOError("JSONLDataset: need at least 1 feature + label");
    }
    for (const auto& r : rows) {
        if (r.size() != width) {
            throw IOError("JSONLDataset: inconsistent row width");
        }
    }
    n_features_ = width - 1;
    const std::size_t n = rows.size();
    NDArray features(Shape{n, n_features_}, options.feature_dtype);
    NDArray labels(Shape{n}, options.label_dtype);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n_features_; ++j) {
            if (options.feature_dtype == DType::Float32) {
                features.data<float>()[i * n_features_ + j] = static_cast<float>(rows[i][j]);
            } else {
                features.data<double>()[i * n_features_ + j] = rows[i][j];
            }
        }
        if (options.label_dtype == DType::Float32) {
            labels.data<float>()[i] = static_cast<float>(rows[i].back());
        } else {
            labels.data<double>()[i] = rows[i].back();
        }
    }
    impl_ = InMemoryDataset(std::move(features), std::move(labels));
}

std::size_t JSONLDataset::size() const {
    return impl_.size();
}

Sample JSONLDataset::get(std::size_t index) const {
    return impl_.get(index);
}

} // namespace nexusdata
