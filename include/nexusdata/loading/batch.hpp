#pragma once

#include "nexusdata/core/dtype.hpp"
#include "nexusdata/core/ndarray.hpp"
#include "nexusdata/dataset/sample.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace nexusdata {

/// One mini-batch produced by DataLoader.
struct Batch {
    NDArray inputs; // [B, ...]
    NDArray labels; // [B] or [B, ...]
    /// Optional padding mask for sequences: 1=valid, 0=pad. Shape [B, T] Float32/UInt8.
    NDArray mask;
    /// v2.0: Sample::extra_tensors whose shape and dtype agree across the batch,
    /// stacked per key to [B, ...sample shape].
    FieldMap<NDArray> extras{};
    /// v2.0: Sample::extra_tensors whose shape varies per sample (e.g. a variable
    /// number of boxes), one entry per sample.
    FieldMap<std::vector<NDArray>> ragged_extras{};
    /// v2.0: Sample::metadata per key, one string per sample ("" where absent).
    FieldMap<std::vector<std::string>> metadata{};
};

namespace detail {
/// Batch the v2.0 per-sample fields into @p extras / @p ragged / @p metadata.
/// Every sample must carry the same extra_tensors keys with matching dtypes.
void collate_fields(const std::vector<Sample>& samples, FieldMap<NDArray>& extras,
                    FieldMap<std::vector<NDArray>>& ragged,
                    FieldMap<std::vector<std::string>>& metadata);
} // namespace detail

/// Default collate: stack sample inputs and labels along axis 0.
[[nodiscard]] Batch collate_stack(const std::vector<Sample>& samples);

/// v2.0: like collate_stack, but inputs keep the sample shape: [B, ...sample.input.shape()]
/// (e.g. [B, H, W, C] images for the fused image batch transforms). collate_stack
/// flattens inputs to [B, numel] for v1 compatibility.
[[nodiscard]] Batch collate_stack_nd(const std::vector<Sample>& samples);

/// Pad variable-length 1-D inputs to max length in batch.
/// Each sample.input must be rank-1 (or [T, F] with same F). Labels stacked.
struct PadCollateOptions {
    double pad_value = 0.0;
    bool pad_labels = false;
    DType mask_dtype = DType::UInt8;
};

[[nodiscard]] Batch collate_pad_sequence(const std::vector<Sample>& samples,
                                         PadCollateOptions opt = {});

/// Keep each sample's input as a separate NDArray list (no stack). Labels still stacked when possible.
struct RaggedBatch {
    std::vector<NDArray> inputs;
    NDArray labels;
    FieldMap<NDArray> extras{};                     ///< see Batch::extras
    FieldMap<std::vector<NDArray>> ragged_extras{}; ///< see Batch::ragged_extras
    FieldMap<std::vector<std::string>> metadata{};  ///< see Batch::metadata
};

[[nodiscard]] RaggedBatch collate_ragged(const std::vector<Sample>& samples);

} // namespace nexusdata
