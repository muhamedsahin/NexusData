/*!
 * Minimal DLPack header (vendored for NexusData adapters).
 * Spec: https://github.com/dmlc/dlpack
 * Copyright (c) 2017 by Contributors — Apache-2.0
 */
#ifndef NEXUSDATA_DLPACK_H_
#define NEXUSDATA_DLPACK_H_

#ifdef __cplusplus
#define DLPACK_EXTERN_C extern "C"
#else
#define DLPACK_EXTERN_C
#endif

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  kDLCPU = 1,
  kDLCUDA = 2,
  kDLCUDAHost = 3,
} DLDeviceType;

typedef struct {
  DLDeviceType device_type;
  int32_t device_id;
} DLDevice;

typedef enum {
  kDLInt = 0U,
  kDLUInt = 1U,
  kDLFloat = 2U,
  kDLBfloat = 4U,
  kDLComplex = 5U,
  kDLBool = 6U,
} DLDataTypeCode;

typedef struct {
  uint8_t code;
  uint8_t bits;
  uint16_t lanes;
} DLDataType;

typedef struct {
  void* data;
  DLDevice device;
  int32_t ndim;
  DLDataType dtype;
  int64_t* shape;
  int64_t* strides;
  uint64_t byte_offset;
} DLTensor;

typedef struct DLManagedTensor {
  DLTensor dl_tensor;
  void* manager_ctx;
  void (*deleter)(struct DLManagedTensor* self);
} DLManagedTensor;

#ifdef __cplusplus
} // extern "C"
#endif

#endif // NEXUSDATA_DLPACK_H_
