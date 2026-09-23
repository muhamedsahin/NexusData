#pragma once

/**
 * NexusData C API (v0.9) — stable ABI surface for FFI / language bindings.
 *
 * Error model: functions return NexusStatus; detailed message via nexus_last_error().
 * Thread-safety: nexus_last_error is thread-local.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(NEXUSDATA_BUILD_SHARED)
#if defined(NEXUSDATA_EXPORTS)
#define NEXUSDATA_API __declspec(dllexport)
#else
#define NEXUSDATA_API __declspec(dllimport)
#endif
#else
#define NEXUSDATA_API
#endif

typedef enum NexusStatus {
    NEXUS_OK = 0,
    NEXUS_ERR_INVALID_ARGUMENT = 1,
    NEXUS_ERR_INDEX = 2,
    NEXUS_ERR_SHAPE = 3,
    NEXUS_ERR_IO = 4,
    NEXUS_ERR_INTERNAL = 5,
    /** nexus_dataloader_next_batch: the epoch is finished. Not an error. */
    NEXUS_END_OF_EPOCH = 6,
} NexusStatus;

typedef enum NexusDType {
    NEXUS_DTYPE_BOOL = 0,
    NEXUS_DTYPE_INT8,
    NEXUS_DTYPE_INT16,
    NEXUS_DTYPE_INT32,
    NEXUS_DTYPE_INT64,
    NEXUS_DTYPE_UINT8,
    NEXUS_DTYPE_UINT16,
    NEXUS_DTYPE_UINT32,
    NEXUS_DTYPE_UINT64,
    NEXUS_DTYPE_FLOAT16,
    NEXUS_DTYPE_BFLOAT16,
    NEXUS_DTYPE_FLOAT32,
    NEXUS_DTYPE_FLOAT64,
} NexusDType;

typedef struct NexusNDArray NexusNDArray;

NEXUSDATA_API const char* nexus_version(void);
NEXUSDATA_API const char* nexus_last_error(void);

NEXUSDATA_API NexusStatus nexus_ndarray_create(const int64_t* shape, size_t ndim, NexusDType dtype,
                                               NexusNDArray** out);
NEXUSDATA_API void nexus_ndarray_destroy(NexusNDArray* arr);

NEXUSDATA_API NexusStatus nexus_ndarray_ndim(const NexusNDArray* arr, size_t* out);
NEXUSDATA_API NexusStatus nexus_ndarray_shape(const NexusNDArray* arr, int64_t* out_shape,
                                              size_t max_ndim);
NEXUSDATA_API NexusStatus nexus_ndarray_dtype(const NexusNDArray* arr, NexusDType* out);
NEXUSDATA_API NexusStatus nexus_ndarray_numel(const NexusNDArray* arr, size_t* out);
NEXUSDATA_API NexusStatus nexus_ndarray_data(NexusNDArray* arr, void** out);
NEXUSDATA_API NexusStatus nexus_ndarray_data_const(const NexusNDArray* arr, const void** out);

NEXUSDATA_API NexusStatus nexus_ndarray_clone(const NexusNDArray* arr, NexusNDArray** out);

/** Export as DLManagedTensor* (void* to avoid requiring dlpack.h in pure C consumers). */
NEXUSDATA_API NexusStatus nexus_ndarray_to_dlpack(const NexusNDArray* arr, void** out_dlpack);
NEXUSDATA_API NexusStatus nexus_ndarray_from_dlpack(void* dlpack, int take_ownership,
                                                    NexusNDArray** out);

typedef struct NexusDataset NexusDataset;
typedef struct NexusDataLoader NexusDataLoader;
typedef struct NexusBatch NexusBatch;

/** Open `scheme:path` through DataSourceRegistry. config_json may be null. */
NEXUSDATA_API NexusStatus nexus_dataset_open(const char* uri, const char* config_json, NexusDataset** out);
NEXUSDATA_API void nexus_dataset_destroy(NexusDataset* dataset);
NEXUSDATA_API NexusStatus nexus_dataset_size(const NexusDataset* dataset, size_t* out);
/** Copies sample.input into a new NDArray handle. */
NEXUSDATA_API NexusStatus nexus_dataset_get_input(const NexusDataset* dataset, size_t index,
                                                  NexusNDArray** out);

typedef struct NexusSample NexusSample;

/** Full sample at @p index. Destroy with nexus_sample_destroy. */
NEXUSDATA_API NexusStatus nexus_dataset_get(const NexusDataset* dataset, size_t index,
                                            NexusSample** out);
NEXUSDATA_API void nexus_sample_destroy(NexusSample* sample);
/** Copies. The sample handle stays valid. */
NEXUSDATA_API NexusStatus nexus_sample_input(const NexusSample* sample, NexusNDArray** out);
NEXUSDATA_API NexusStatus nexus_sample_label(const NexusSample* sample, NexusNDArray** out);

/** options_json: {"batch_size":N,"shuffle":bool,"seed":N,"num_workers":N}. Null = defaults. */
NEXUSDATA_API NexusStatus nexus_dataloader_create(NexusDataset* dataset, const char* options_json,
                                                  NexusDataLoader** out);
NEXUSDATA_API void nexus_dataloader_destroy(NexusDataLoader* loader);
/** *out is set on NEXUS_OK. NEXUS_END_OF_EPOCH leaves *out null. */
NEXUSDATA_API NexusStatus nexus_dataloader_next_batch(NexusDataLoader* loader, NexusBatch** out);
NEXUSDATA_API void nexus_batch_destroy(NexusBatch* batch);
NEXUSDATA_API NexusStatus nexus_batch_inputs(const NexusBatch* batch, NexusNDArray** out);

/** Build a loader from a JSON or YAML pipeline file (PipelineBuilder::from_file). */
NEXUSDATA_API NexusStatus nexus_pipeline_from_config(const char* path, NexusDataLoader** out);

#ifdef __cplusplus
} // extern "C"
#endif
