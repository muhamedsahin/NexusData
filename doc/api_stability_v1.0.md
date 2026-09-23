# NexusData API stability (v1.0)

**NexusData 1.0 is a stable release.** Breaking changes to the surface below require a new major version.

## Guaranteed stable

| Layer | Symbols |
|-------|---------|
| Version | `nexusdata/version.hpp`, `nexus_version()` (C) |
| Core | `Error` hierarchy, `DType`, `Shape`, `NDArray`, `PCG32`, allocator |
| Dataset | `Dataset`, `Sample`, `InMemoryDataset`, `Subset`, CSV/JSONL/Image/MNIST/CIFAR/NPY, iterable compose, text/WAV/WebDataset |
| Loading | `DataLoader`, `Batch`, `collate_stack` / `collate_pad_sequence` / `collate_ragged` |
| Pipeline | `Transform`/`Compose`, tabular transforms, preprocessor + `Pipeline` save/load |
| Sampling | Sequential, Random, Weighted, Bucket, Distributed, Batch |
| Backend | `Device`, `resolve_device`, `calibrate_auto_device_policy`, ThreadPool (CPU) |
| Adapters | C API (`c_api.h`), DLPack host export/import, `MatrixFlashAdapter` views |

## Allowed to evolve in 1.x (additive / behavioral tuning)

- CUDA kernel coverage and Auto threshold defaults (still Host-safe without CUDA)
- Optional Parquet/HDF5/SQLite full backends behind CMake flags
- Python bindings beyond documented NDArray/numpy/dlpack helpers
- Bench/fuzz headers under `nexusdata/bench` and `nexusdata/fuzz`

## Compatibility rules

1. **SemVer:** MINOR adds APIs; PATCH fixes bugs; MAJOR breaks stable symbols.
2. **CMake:** `find_package(NexusData 1.0)` with `SameMajorVersion` compatibility.
3. **C ABI:** existing `c_api.h` signatures stay; new functions are additive.
4. **Shared lib:** `SOVERSION 1` for the 1.x series.
5. **MatrixFlash:** optional one-way adapter; never required to build or use NexusData.
