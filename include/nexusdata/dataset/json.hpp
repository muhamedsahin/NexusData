#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "nexusdata/core/ndarray.hpp"
#include "nexusdata/dataset/dataset.hpp"
#include "nexusdata/dataset/iterable.hpp"
#include "nexusdata/io/json.hpp"

namespace nexusdata {

/// Where an extracted JSON field goes in the Sample.
enum class JsonRole : std::uint8_t {
    Input,    ///< concatenated (in field order) into Sample::input
    Label,    ///< concatenated into Sample::label
    Extra,    ///< Sample::extra_tensors[name]
    Metadata, ///< Sample::metadata[name] (strings verbatim, other values as JSON text)
};

/// One field extracted from each record.
///
/// Paths: dotted keys with array steps, e.g. "image.size[0]", "meta.id",
/// "annotations[*].bbox". `[*]` collects the value from every element, adding a
/// leading dimension (a variable one yields ragged Extra tensors).
/// Values: numbers, booleans (1/0) and arbitrarily nested rectangular numeric arrays
/// (shape inferred, row-major). `null` becomes NaN for float dtypes.
struct JsonField {
    std::string path;
    JsonRole role = JsonRole::Input;
    DType dtype = DType::Float32;
    /// Expected per-record shape (element count must match); empty = inferred,
    /// with scalars as [1].
    Shape shape{};
    /// Key for Extra / Metadata; defaults to the path.
    std::string name{};
    /// Used when the field is absent (or null, for integer dtypes); unset = error.
    std::optional<double> default_value{};
    /// When non-empty, string values map to their index here (e.g. class names).
    std::vector<std::string> categories{};
};

enum class JsonFormat : std::uint8_t {
    Auto,  ///< '[' first => Array, otherwise Lines
    Lines, ///< JSON Lines / NDJSON: one record per line, blank lines ignored
    Array, ///< one document; records are the elements of the array at records_path
};

struct JsonOptions {
    /// Fields to extract; empty => every top-level numeric / numeric-array member of
    /// the first record becomes an Input field (see infer_json_fields()).
    std::vector<JsonField> fields{};
    JsonFormat format = JsonFormat::Auto;
    /// Array format: dotted path of the record array inside the document
    /// (e.g. "data.items"); empty = the document root.
    std::string records_path{};
    /// Drop records that fail to parse or map instead of throwing.
    bool skip_invalid = false;
    int num_threads = 0; ///< 0 => hardware concurrency
};

/// Map-style dataset over JSON Lines or a JSON array document, compressed or not
/// (via read_input()). Records are split and parsed in parallel straight into
/// columnar storage; get() only copies. Parsing walks each record once, decoding
/// only the requested paths and skipping everything else without allocation.
///
/// Thread-safety: read-only after construction; concurrent get() is safe.
class JsonDataset : public Dataset {
public:
    explicit JsonDataset(const std::string& path, JsonOptions options = {});

    /// Parse from memory (no file); @p text must stay valid only during the call.
    [[nodiscard]] static std::shared_ptr<JsonDataset> from_string(std::string_view text,
                                                                   JsonOptions options = {});

    [[nodiscard]] std::size_t size() const override;
    [[nodiscard]] Sample get(std::size_t index) const override;

    [[nodiscard]] const std::vector<JsonField>& fields() const noexcept { return fields_; }
    /// Records dropped by skip_invalid.
    [[nodiscard]] std::size_t skipped() const noexcept { return skipped_; }

    struct Column; // per-field columnar storage (implementation detail)

private:
    JsonDataset() = default;
    void load(std::string_view text, JsonOptions options);

    std::vector<JsonField> fields_;
    std::vector<std::shared_ptr<Column>> columns_;
    std::size_t n_ = 0;
    std::size_t skipped_ = 0;
};

/// Streaming JSON Lines reader: constant memory regardless of file size, with
/// transparent decompression (e.g. a 100 GB `.jsonl.zst`). Fields as in JsonDataset.
class JsonLinesIterable : public IterableDataset {
public:
    explicit JsonLinesIterable(std::string path, JsonOptions options = {});
    [[nodiscard]] std::unique_ptr<Iterator> make_iterator() const override;

private:
    std::string path_;
    JsonOptions options_;
};

/// Fields for every top-level member of @p record whose value is numeric, boolean or
/// a numeric array (role Input, Float32); strings become Metadata fields.
[[nodiscard]] std::vector<JsonField> infer_json_fields(const json::Value& record);

/// Fields from a JSON Schema object ("type": "object", "properties": {...}).
/// number / integer / boolean properties map to Float32 / Int64 / Bool, arrays of
/// those to tensors ("minItems" == "maxItems" fixes the length), strings to Metadata
/// (or to category ids when "enum" is given). The role comes from the
/// "x-nexusdata-role" keyword ("input" | "label" | "extra" | "metadata"), default
/// Input; "x-nexusdata-dtype" (e.g. "float64") overrides the dtype. Nested objects
/// recurse with dotted paths.
[[nodiscard]] std::vector<JsonField> json_fields_from_schema(const json::Value& schema);

} // namespace nexusdata
