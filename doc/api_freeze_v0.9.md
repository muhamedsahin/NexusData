# NexusData API freeze guide (v0.9)

This release freezes the **intended stable surface** for v1.0. Breaking changes after 1.0 require a major version bump.

## Stable (v1.0 candidates)

| Layer | Symbols |
|-------|---------|
| Core | `Error` hierarchy, `DType`, `Shape`, `NDArray`, `PCG32`, `Allocator` |
| Dataset | `Dataset`, `Sample`, `InMemoryDataset`, `Subset`, CSV/JSONL/Image/MNIST/CIFAR/NPY, Iterable compose |
| Loading | `DataLoader`, `Batch`, `collate_*` |
| Pipeline | `Transform`/`Compose`, tabular transforms, preprocessor `save`/`load`, `Pipeline` |
| Sampling | Sequential/Random + advanced samplers |
| Backend | `Device`, `resolve_device`, `calibrate_auto_device_policy`, ThreadPool |
| Adapters | C API (`c_api.h`), DLPack, `MatrixFlashAdapter` views |

## Explicitly unstable until 1.0

- CUDA kernel set and Auto thresholds (calibrated defaults may change)
- Optional readers: Parquet/HDF5/SQLite full backends
- Python module method set beyond NDArray/numpy/dlpack
- Fuzz / bench harness headers

## Compatibility rules

1. **SemVer:** `0.x` may break; `1.x` patch/minor must not break stable symbols above.
2. **C ABI:** `c_api.h` functions keep signatures; new APIs are additive.
3. **DLPack:** export uses standard `DLManagedTensor`; import is host contiguous copy in v0.9.
4. **MatrixFlash:** dependency is one-way (`NEXUSDATA_WITH_MATRIXFLASH`); Flash never required to build NexusData.
