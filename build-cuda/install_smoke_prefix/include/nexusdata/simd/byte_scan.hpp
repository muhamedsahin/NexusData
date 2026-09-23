#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nexusdata {

/// Find offsets of @p needle in [data, data+len). Scalar baseline.
[[nodiscard]] std::vector<std::size_t> find_bytes_scalar(const std::uint8_t* data,
                                                         std::size_t len,
                                                         std::uint8_t needle);

/// Runtime-dispatched scanner (AVX2 if available, else scalar).
[[nodiscard]] std::vector<std::size_t> find_bytes(const std::uint8_t* data,
                                                  std::size_t len,
                                                  std::uint8_t needle);

/// Find record-start offsets for CSV/JSONL: positions after newlines where
/// cumulative quote parity is even (RFC4180-ish, no escaped-quote in parity for speed —
/// escaped quotes "" keep parity even). Good enough for structural chunking.
[[nodiscard]] std::vector<std::size_t> find_csv_record_starts(const std::uint8_t* data,
                                                              std::size_t len);

} // namespace nexusdata
