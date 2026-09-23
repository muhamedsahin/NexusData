#include "nexusdata/adapter/c_api.h"

#include <cstring>
#include <memory>
#include <new>
#include <string>

#include "nexusdata/adapter/dlpack.hpp"
#include "nexusdata/core/dtype.hpp"
#include "nexusdata/core/error.hpp"
#include "nexusdata/core/ndarray.hpp"
#include "nexusdata/core/registry.hpp"
#include "nexusdata/dataset/sample.hpp"
#include "nexusdata/io/json.hpp"
#include "nexusdata/loading/dataloader.hpp"
#include "nexusdata/pipeline/pipeline_builder.hpp"

using nexusdata::DType;
using nexusdata::Error;
using nexusdata::IndexError;
using nexusdata::InvalidArgumentError;
using nexusdata::IOError;
using nexusdata::NDArray;
using nexusdata::Shape;
using nexusdata::ShapeError;

struct NexusNDArray {
    NDArray arr;
};

struct NexusDataset {
    nexusdata::DatasetPtr ds;
};

struct NexusSample {
    nexusdata::Sample sample;
};

struct NexusDataLoader {
    std::unique_ptr<nexusdata::DataLoader> loader;
    nexusdata::DataLoader::Iterator it{};
    bool started = false;
};

struct NexusBatch {
    nexusdata::Batch batch;
};

namespace {

thread_local std::string g_last_error;

void set_error(const char* msg) {
    g_last_error = msg ? msg : "";
}

NexusStatus map_exception() {
    try {
        throw;
    } catch (const InvalidArgumentError& e) {
        set_error(e.what());
        return NEXUS_ERR_INVALID_ARGUMENT;
    } catch (const IndexError& e) {
        set_error(e.what());
        return NEXUS_ERR_INDEX;
    } catch (const ShapeError& e) {
        set_error(e.what());
        return NEXUS_ERR_SHAPE;
    } catch (const IOError& e) {
        set_error(e.what());
        return NEXUS_ERR_IO;
    } catch (const Error& e) {
        set_error(e.what());
        return NEXUS_ERR_INTERNAL;
    } catch (const std::exception& e) {
        set_error(e.what());
        return NEXUS_ERR_INTERNAL;
    } catch (...) {
        set_error("unknown exception");
        return NEXUS_ERR_INTERNAL;
    }
}

DType from_c(NexusDType d) {
    return static_cast<DType>(static_cast<std::uint8_t>(d));
}

NexusDType to_c(DType d) {
    return static_cast<NexusDType>(static_cast<std::uint8_t>(d));
}

} // namespace

extern "C" {

const char* nexus_version(void) {
#if defined(NEXUSDATA_VERSION_STRING)
    return NEXUSDATA_VERSION_STRING;
#else
    return "1.0.0";
#endif
}

const char* nexus_last_error(void) {
    return g_last_error.c_str();
}

NexusStatus nexus_ndarray_create(const int64_t* shape, size_t ndim, NexusDType dtype,
                                 NexusNDArray** out) {
    if (!out) {
        set_error("nexus_ndarray_create: null out");
        return NEXUS_ERR_INVALID_ARGUMENT;
    }
    try {
        Shape sh;
        sh.reserve(ndim);
        for (size_t i = 0; i < ndim; ++i) {
            if (!shape || shape[i] < 0) {
                set_error("nexus_ndarray_create: invalid shape");
                return NEXUS_ERR_INVALID_ARGUMENT;
            }
            sh.push_back(static_cast<std::size_t>(shape[i]));
        }
        auto* h = new NexusNDArray();
        h->arr = NDArray(std::move(sh), from_c(dtype));
        *out = h;
        set_error("");
        return NEXUS_OK;
    } catch (...) {
        return map_exception();
    }
}

void nexus_ndarray_destroy(NexusNDArray* arr) {
    delete arr;
}

NexusStatus nexus_ndarray_ndim(const NexusNDArray* arr, size_t* out) {
    if (!arr || !out) {
        set_error("nexus_ndarray_ndim: null");
        return NEXUS_ERR_INVALID_ARGUMENT;
    }
    *out = arr->arr.shape().size();
    return NEXUS_OK;
}

NexusStatus nexus_ndarray_shape(const NexusNDArray* arr, int64_t* out_shape, size_t max_ndim) {
    if (!arr || !out_shape) {
        set_error("nexus_ndarray_shape: null");
        return NEXUS_ERR_INVALID_ARGUMENT;
    }
    const auto& sh = arr->arr.shape();
    if (sh.size() > max_ndim) {
        set_error("nexus_ndarray_shape: buffer too small");
        return NEXUS_ERR_INVALID_ARGUMENT;
    }
    for (size_t i = 0; i < sh.size(); ++i) {
        out_shape[i] = static_cast<int64_t>(sh[i]);
    }
    return NEXUS_OK;
}

NexusStatus nexus_ndarray_dtype(const NexusNDArray* arr, NexusDType* out) {
    if (!arr || !out) {
        set_error("nexus_ndarray_dtype: null");
        return NEXUS_ERR_INVALID_ARGUMENT;
    }
    *out = to_c(arr->arr.dtype());
    return NEXUS_OK;
}

NexusStatus nexus_ndarray_numel(const NexusNDArray* arr, size_t* out) {
    if (!arr || !out) {
        set_error("nexus_ndarray_numel: null");
        return NEXUS_ERR_INVALID_ARGUMENT;
    }
    *out = arr->arr.numel();
    return NEXUS_OK;
}

NexusStatus nexus_ndarray_data(NexusNDArray* arr, void** out) {
    if (!arr || !out) {
        set_error("nexus_ndarray_data: null");
        return NEXUS_ERR_INVALID_ARGUMENT;
    }
    *out = arr->arr.data();
    return NEXUS_OK;
}

NexusStatus nexus_ndarray_data_const(const NexusNDArray* arr, const void** out) {
    if (!arr || !out) {
        set_error("nexus_ndarray_data_const: null");
        return NEXUS_ERR_INVALID_ARGUMENT;
    }
    *out = arr->arr.data();
    return NEXUS_OK;
}

NexusStatus nexus_ndarray_clone(const NexusNDArray* arr, NexusNDArray** out) {
    if (!arr || !out) {
        set_error("nexus_ndarray_clone: null");
        return NEXUS_ERR_INVALID_ARGUMENT;
    }
    try {
        auto* h = new NexusNDArray();
        h->arr = arr->arr.clone();
        *out = h;
        return NEXUS_OK;
    } catch (...) {
        return map_exception();
    }
}

NexusStatus nexus_ndarray_to_dlpack(const NexusNDArray* arr, void** out_dlpack) {
    if (!arr || !out_dlpack) {
        set_error("nexus_ndarray_to_dlpack: null");
        return NEXUS_ERR_INVALID_ARGUMENT;
    }
    try {
        *out_dlpack = nexusdata::ndarray_to_dlpack(arr->arr);
        return NEXUS_OK;
    } catch (...) {
        return map_exception();
    }
}

NexusStatus nexus_dataset_open(const char* uri, const char* config_json, NexusDataset** out) {
    if (!uri || !out) {
        set_error("nexus_dataset_open: null");
        return NEXUS_ERR_INVALID_ARGUMENT;
    }
    try {
        auto* h = new NexusDataset();
        h->ds = nexusdata::DataSourceRegistry::instance().open(uri, config_json ? config_json : "{}");
        *out = h;
        set_error("");
        return NEXUS_OK;
    } catch (...) {
        return map_exception();
    }
}

void nexus_dataset_destroy(NexusDataset* dataset) { delete dataset; }

NexusStatus nexus_dataset_size(const NexusDataset* dataset, size_t* out) {
    if (!dataset || !out) {
        set_error("nexus_dataset_size: null");
        return NEXUS_ERR_INVALID_ARGUMENT;
    }
    try {
        *out = dataset->ds->size();
        return NEXUS_OK;
    } catch (...) {
        return map_exception();
    }
}

NexusStatus nexus_dataset_get_input(const NexusDataset* dataset, size_t index, NexusNDArray** out) {
    if (!dataset || !out) {
        set_error("nexus_dataset_get_input: null");
        return NEXUS_ERR_INVALID_ARGUMENT;
    }
    try {
        auto* h = new NexusNDArray();
        h->arr = dataset->ds->get(index).input.clone();
        *out = h;
        set_error("");
        return NEXUS_OK;
    } catch (...) {
        return map_exception();
    }
}

NexusStatus nexus_dataset_get(const NexusDataset* dataset, size_t index, NexusSample** out) {
    if (!dataset || !out) {
        set_error("nexus_dataset_get: null");
        return NEXUS_ERR_INVALID_ARGUMENT;
    }
    try {
        auto* h = new NexusSample();
        h->sample = dataset->ds->get(index);
        *out = h;
        set_error("");
        return NEXUS_OK;
    } catch (...) {
        return map_exception();
    }
}

void nexus_sample_destroy(NexusSample* sample) { delete sample; }

NexusStatus nexus_sample_input(const NexusSample* sample, NexusNDArray** out) {
    if (!sample || !out) {
        set_error("nexus_sample_input: null");
        return NEXUS_ERR_INVALID_ARGUMENT;
    }
    try {
        auto* h = new NexusNDArray();
        h->arr = sample->sample.input.clone();
        *out = h;
        set_error("");
        return NEXUS_OK;
    } catch (...) {
        return map_exception();
    }
}

NexusStatus nexus_sample_label(const NexusSample* sample, NexusNDArray** out) {
    if (!sample || !out) {
        set_error("nexus_sample_label: null");
        return NEXUS_ERR_INVALID_ARGUMENT;
    }
    try {
        auto* h = new NexusNDArray();
        h->arr = sample->sample.label.clone();
        *out = h;
        set_error("");
        return NEXUS_OK;
    } catch (...) {
        return map_exception();
    }
}

NexusStatus nexus_dataloader_create(NexusDataset* dataset, const char* options_json, NexusDataLoader** out) {
    if (!dataset || !out) {
        set_error("nexus_dataloader_create: null");
        return NEXUS_ERR_INVALID_ARGUMENT;
    }
    try {
        nexusdata::DataLoaderOptions opt;
        if (options_json && options_json[0]) {
            const auto v = nexusdata::json::parse(options_json);
            opt.batch_size = static_cast<std::size_t>(v.get_int("batch_size", 1));
            opt.shuffle = v.get_bool("shuffle", false);
            opt.num_workers = static_cast<int>(v.get_int("num_workers", 0));
            opt.seed = static_cast<std::uint64_t>(v.get_int("seed", 0));
        }
        auto* h = new NexusDataLoader();
        h->loader = std::make_unique<nexusdata::DataLoader>(nexusdata::DatasetConstPtr(dataset->ds), opt);
        *out = h;
        set_error("");
        return NEXUS_OK;
    } catch (...) {
        return map_exception();
    }
}

void nexus_dataloader_destroy(NexusDataLoader* loader) { delete loader; }

NexusStatus nexus_dataloader_next_batch(NexusDataLoader* loader, NexusBatch** out) {
    if (!loader || !out) {
        set_error("nexus_dataloader_next_batch: null");
        return NEXUS_ERR_INVALID_ARGUMENT;
    }
    try {
        if (!loader->started) {
            loader->it = loader->loader->begin();
            loader->started = true;
        }
        if (loader->it == loader->loader->end()) {
            *out = nullptr;
            set_error("");
            return NEXUS_END_OF_EPOCH;
        }
        auto* h = new NexusBatch();
        h->batch = *loader->it;
        ++loader->it;
        *out = h;
        set_error("");
        return NEXUS_OK;
    } catch (...) {
        return map_exception();
    }
}

void nexus_batch_destroy(NexusBatch* batch) { delete batch; }

NexusStatus nexus_batch_inputs(const NexusBatch* batch, NexusNDArray** out) {
    if (!batch || !out) {
        set_error("nexus_batch_inputs: null");
        return NEXUS_ERR_INVALID_ARGUMENT;
    }
    try {
        auto* h = new NexusNDArray();
        h->arr = batch->batch.inputs.clone();
        *out = h;
        return NEXUS_OK;
    } catch (...) {
        return map_exception();
    }
}

NexusStatus nexus_pipeline_from_config(const char* path, NexusDataLoader** out) {
    if (!path || !out) {
        set_error("nexus_pipeline_from_config: null");
        return NEXUS_ERR_INVALID_ARGUMENT;
    }
    try {
        nexusdata::BuiltPipeline built = nexusdata::PipelineBuilder::from_file(path);
        auto* h = new NexusDataLoader();
        h->loader = std::move(built.loader);
        *out = h;
        set_error("");
        return NEXUS_OK;
    } catch (...) {
        return map_exception();
    }
}

NexusStatus nexus_ndarray_from_dlpack(void* dlpack, int take_ownership, NexusNDArray** out) {
    if (!dlpack || !out) {
        set_error("nexus_ndarray_from_dlpack: null");
        return NEXUS_ERR_INVALID_ARGUMENT;
    }
    try {
        auto* managed = static_cast<DLManagedTensor*>(dlpack);
        auto* h = new NexusNDArray();
        h->arr = nexusdata::ndarray_from_dlpack(managed, take_ownership != 0);
        *out = h;
        return NEXUS_OK;
    } catch (...) {
        return map_exception();
    }
}

} // extern "C"
