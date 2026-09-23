#include <cstring>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "nexusdata/bench/harness.hpp"
#include "nexusdata/dataset/postgres.hpp"
#include "nexusdata/dataset/sql_column.hpp"

using namespace nexusdata;

namespace {

void append_u16(std::vector<std::uint8_t>& b, std::uint16_t v) {
    b.push_back(static_cast<std::uint8_t>(v >> 8));
    b.push_back(static_cast<std::uint8_t>(v));
}
void append_u32(std::vector<std::uint8_t>& b, std::uint32_t v) {
    for (int s = 24; s >= 0; s -= 8) b.push_back(static_cast<std::uint8_t>(v >> s));
}
void append_u64(std::vector<std::uint8_t>& b, std::uint64_t v) {
    for (int s = 56; s >= 0; s -= 8) b.push_back(static_cast<std::uint8_t>(v >> s));
}

std::vector<std::uint8_t> payload(int rows) {
    std::vector<std::uint8_t> b = {'P', 'G', 'C', 'O', 'P', 'Y', '\n', 0xFF, '\r', '\n', 0};
    append_u32(b, 0);
    append_u32(b, 0);
    for (int i = 0; i < rows; ++i) {
        append_u16(b, 1);
        append_u32(b, 8);
        double v = static_cast<double>(i) * 0.5;
        std::uint64_t bits = 0;
        std::memcpy(&bits, &v, 8);
        append_u64(b, bits);
    }
    append_u16(b, static_cast<std::uint16_t>(-1));
    return b;
}

} // namespace

int main() {
    SqlColumn col;
    col.name = "x";
    col.type = SqlType::Float64;
    const auto bytes = payload(4000);
    const double sec = bench::time_best([&] {
        PostgresCopyDataset ds(bytes.data(), bytes.size(), {col});
        if (ds.size() != 4000) std::cerr << "size mismatch\n";
    }, 5);
    bench::print_result("copy_binary_4k_rows", sec, bytes.size());
    return 0;
}
