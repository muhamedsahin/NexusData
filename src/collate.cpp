#include "nexusdata/loading/batch.hpp"

#include <algorithm>
#include <cstring>

#include "nexusdata/core/array_utils.hpp"
#include "nexusdata/core/error.hpp"

namespace nexusdata {

namespace detail {

void collate_fields(const std::vector<Sample>& samples, FieldMap<NDArray>& extras,
                    FieldMap<std::vector<NDArray>>& ragged,
                    FieldMap<std::vector<std::string>>& metadata) {
    bool any = false;
    for (const Sample& s : samples) {
        if (!s.extra_tensors.empty() || !s.metadata.empty()) {
            any = true;
            break;
        }
    }
    if (!any) {
        return;
    }
    const std::size_t B = samples.size();
    const Sample& first = samples[0];
    for (std::size_t i = 1; i < B; ++i) {
        if (samples[i].extra_tensors.size() != first.extra_tensors.size()) {
            throw ShapeError("collate: extra_tensors key set differs at sample " +
                             std::to_string(i));
        }
    }
    for (const auto& [key, ref] : first.extra_tensors) {
        bool same_shape = true;
        for (std::size_t i = 0; i < B; ++i) {
            const NDArray* t = samples[i].extra_tensors.find(key);
            if (t == nullptr) {
                throw ShapeError("collate: extra tensor '" + key + "' missing at sample " +
                                 std::to_string(i));
            }
            if (t->dtype() != ref.dtype()) {
                throw ShapeError("collate: extra tensor '" + key + "' dtype mismatch at sample " +
                                 std::to_string(i));
            }
            if (!t->device().is_host()) {
                throw InvalidArgumentError("collate: extra tensor '" + key + "' is not on host");
            }
            same_shape = same_shape && t->shape() == ref.shape();
        }
        if (!same_shape) {
            std::vector<NDArray> list;
            list.reserve(B);
            for (const Sample& s : samples) list.push_back(*s.extra_tensors.find(key));
            ragged.set(key, std::move(list));
            continue;
        }
        Shape shape{B};
        shape.insert(shape.end(), ref.shape().begin(), ref.shape().end());
        NDArray out(std::move(shape), ref.dtype());
        const std::size_t row = ref.nbytes();
        auto* dst = static_cast<std::uint8_t*>(out.data());
        for (std::size_t i = 0; i < B; ++i) {
            if (row > 0) {
                std::memcpy(dst + i * row, samples[i].extra_tensors.find(key)->data(), row);
            }
        }
        extras.set(key, std::move(out));
    }
    for (std::size_t i = 0; i < B; ++i) {
        for (const auto& [key, value] : samples[i].metadata) {
            std::vector<std::string>* col = metadata.find(key);
            if (col == nullptr) {
                col = &metadata.set(key, std::vector<std::string>(B));
            }
            (*col)[i] = value;
        }
    }
}

} // namespace detail

Batch collate_stack_nd(const std::vector<Sample>& samples) {
    Batch b = collate_stack(samples);
    Shape shape{samples.size()};
    for (std::size_t d : samples[0].input.shape()) {
        shape.push_back(d);
    }
    b.inputs.reshape(std::move(shape));
    return b;
}

Batch collate_stack(const std::vector<Sample>& samples) {
    if (samples.empty()) {
        throw InvalidArgumentError("collate_stack: empty sample list");
    }
    // v0.8: direct memcpy stack (same layout as NDArray::stack: [B, numel]).
    const NDArray& fin = samples[0].input;
    const NDArray& flab = samples[0].label;
    if (!fin.device().is_host() || !flab.device().is_host()) {
        throw InvalidArgumentError("collate_stack: host-only");
    }
    const std::size_t B = samples.size();
    const std::size_t in_n = fin.numel();
    const std::size_t lab_n = flab.numel();
    const DType in_dt = fin.dtype();
    const DType lab_dt = flab.dtype();

    Batch b;
    b.inputs = NDArray(Shape{B, in_n}, in_dt);
    b.labels = NDArray(Shape{B, lab_n}, lab_dt);
    const std::size_t in_row = in_n * size_of(in_dt);
    const std::size_t lab_row = lab_n * size_of(lab_dt);
    auto* in_dst = static_cast<std::uint8_t*>(b.inputs.data());
    auto* lab_dst = static_cast<std::uint8_t*>(b.labels.data());

    for (std::size_t i = 0; i < B; ++i) {
        const auto& s = samples[i];
        if (s.input.dtype() != in_dt || s.input.numel() != in_n ||
            s.label.dtype() != lab_dt || s.label.numel() != lab_n) {
            throw ShapeError("collate_stack: sample shape/dtype mismatch at index " +
                             std::to_string(i));
        }
        if (!s.input.device().is_host() || !s.label.device().is_host()) {
            throw InvalidArgumentError("collate_stack: host-only");
        }
        std::memcpy(in_dst + i * in_row, s.input.data(), in_row);
        std::memcpy(lab_dst + i * lab_row, s.label.data(), lab_row);
    }
    if (b.labels.shape().size() == 2 && b.labels.shape()[1] == 1) {
        b.labels.reshape(Shape{b.labels.shape()[0]});
    }
    detail::collate_fields(samples, b.extras, b.ragged_extras, b.metadata);
    return b;
}

Batch collate_pad_sequence(const std::vector<Sample>& samples, PadCollateOptions opt) {
    if (samples.empty()) {
        throw InvalidArgumentError("collate_pad_sequence: empty");
    }
    const DType dt = samples[0].input.dtype();
    std::size_t max_t = 0;
    std::size_t feat = 1;
    bool matrix = false;
    for (const auto& s : samples) {
        if (s.input.dtype() != dt) {
            throw ShapeError("collate_pad_sequence: dtype mismatch");
        }
        if (s.input.shape().size() == 1) {
            max_t = std::max(max_t, s.input.shape()[0]);
        } else if (s.input.shape().size() == 2) {
            matrix = true;
            max_t = std::max(max_t, s.input.shape()[0]);
            feat = s.input.shape()[1];
        } else {
            throw ShapeError("collate_pad_sequence: expected rank 1 or 2 inputs");
        }
    }
    const std::size_t B = samples.size();
    Batch b;
    if (!matrix) {
        b.inputs = NDArray(Shape{B, max_t}, dt);
        b.mask = NDArray(Shape{B, max_t}, opt.mask_dtype);
        std::memset(b.inputs.data(), 0, b.inputs.nbytes());
        std::memset(b.mask.data(), 0, b.mask.nbytes());
        for (std::size_t i = 0; i < B; ++i) {
            const std::size_t t = samples[i].input.numel();
            const std::size_t es = size_of(dt);
            std::memcpy(static_cast<std::uint8_t*>(b.inputs.data()) + i * max_t * es,
                        samples[i].input.data(), t * es);
            if (opt.pad_value != 0.0) {
                for (std::size_t j = t; j < max_t; ++j) {
                    detail::set_double(b.inputs, i * max_t + j, opt.pad_value);
                }
            }
            for (std::size_t j = 0; j < t; ++j) {
                if (opt.mask_dtype == DType::UInt8) {
                    b.mask.data<std::uint8_t>()[i * max_t + j] = 1;
                } else {
                    b.mask.data<float>()[i * max_t + j] = 1.0f;
                }
            }
        }
    } else {
        b.inputs = NDArray(Shape{B, max_t, feat}, dt);
        b.mask = NDArray(Shape{B, max_t}, opt.mask_dtype);
        std::memset(b.inputs.data(), 0, b.inputs.nbytes());
        std::memset(b.mask.data(), 0, b.mask.nbytes());
        const std::size_t es = size_of(dt);
        for (std::size_t i = 0; i < B; ++i) {
            const std::size_t t = samples[i].input.shape()[0];
            for (std::size_t j = 0; j < t; ++j) {
                std::memcpy(static_cast<std::uint8_t*>(b.inputs.data()) +
                                ((i * max_t + j) * feat) * es,
                            static_cast<const std::uint8_t*>(samples[i].input.data()) + j * feat * es,
                            feat * es);
                if (opt.mask_dtype == DType::UInt8) {
                    b.mask.data<std::uint8_t>()[i * max_t + j] = 1;
                } else {
                    b.mask.data<float>()[i * max_t + j] = 1.0f;
                }
            }
        }
    }

    // Labels: direct stack
    const std::size_t lab_n = samples[0].label.numel();
    const DType lab_dt = samples[0].label.dtype();
    b.labels = NDArray(Shape{B, lab_n}, lab_dt);
    const std::size_t lab_row = lab_n * size_of(lab_dt);
    auto* lab_dst = static_cast<std::uint8_t*>(b.labels.data());
    for (std::size_t i = 0; i < B; ++i) {
        if (samples[i].label.numel() != lab_n || samples[i].label.dtype() != lab_dt) {
            throw ShapeError("collate_pad_sequence: label mismatch");
        }
        std::memcpy(lab_dst + i * lab_row, samples[i].label.data(), lab_row);
    }
    if (b.labels.shape().size() == 2 && b.labels.shape()[1] == 1) {
        b.labels.reshape(Shape{b.labels.shape()[0]});
    }
    detail::collate_fields(samples, b.extras, b.ragged_extras, b.metadata);
    return b;
}

RaggedBatch collate_ragged(const std::vector<Sample>& samples) {
    if (samples.empty()) {
        throw InvalidArgumentError("collate_ragged: empty");
    }
    RaggedBatch out;
    out.inputs.reserve(samples.size());
    const std::size_t B = samples.size();
    const std::size_t lab_n = samples[0].label.numel();
    const DType lab_dt = samples[0].label.dtype();
    out.labels = NDArray(Shape{B, lab_n}, lab_dt);
    const std::size_t lab_row = lab_n * size_of(lab_dt);
    auto* lab_dst = static_cast<std::uint8_t*>(out.labels.data());
    for (std::size_t i = 0; i < B; ++i) {
        out.inputs.push_back(samples[i].input);
        std::memcpy(lab_dst + i * lab_row, samples[i].label.data(), lab_row);
    }
    if (out.labels.shape().size() == 2 && out.labels.shape()[1] == 1) {
        out.labels.reshape(Shape{out.labels.shape()[0]});
    }
    detail::collate_fields(samples, out.extras, out.ragged_extras, out.metadata);
    return out;
}

} // namespace nexusdata
