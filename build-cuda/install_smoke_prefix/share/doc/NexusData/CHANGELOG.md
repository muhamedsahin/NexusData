# Changelog

All notable changes to NexusData are documented here.

## [1.0.0] — 2026-09-20

### Stable release
- SemVer **1.0.0** with `SOVERSION 1`; SameMajorVersion CMake package compatibility
- Public API contract documented in `doc/api_stability_v1.0.md`
- `nexusdata/version.hpp` version macros + helpers
- Apache-2.0 `LICENSE`
- pkg-config `nexusdata.pc`
- `find_package` consumer smoke test (`install_find_package_smoke`)

### Included from 0.x
- Core: `NDArray`, PCG32, errors, shapes/dtypes
- Datasets: CSV/JSONL/Image/MNIST/CIFAR/NPY, iterable compose, text/WAV/WebDataset
- Loading: DataLoader workers/prefetch, pad/ragged collate
- Pipeline/preprocessing with save/load
- Optional CUDA backend, SQLite/Arrow/HDF5 stubs
- Hardening: sanitizers, fuzz smoke
- Perf: SIMD byte scan, calibrated Auto device policy
- Adapters: C API, DLPack, MatrixFlash views, optional Python

## [0.9.0] — Adapters & packaging
C API, DLPack, MatrixFlash adapter, optional pybind11, vcpkg/Conan manifests, API freeze prep.

## [0.8.0] — Performance
Benchmark suite, SWAR/AVX2 scan opts, collate memcpy path, Auto threshold calibration.

## [0.7.0] — Hardening
Error message helpers, fuzz harness, sanitizer CMake options, edge-case tests.

## [0.6.0] — Formats & streaming
IterableDataset, advanced samplers, pad collate, text/audio/WebDataset, pipeline persist.

## [0.5.0] — CUDA backend
Device/Auto policy, Philox, pinned memory, optional CUDA kernels.

## [0.4.0] — Perf infrastructure
ThreadPool, SIMD byte_scan, JSONL, cache, prefetch DataLoader.

## [0.3.0] — Images & binary datasets
ImageFolder, MNIST/CIFAR, NPY/NPZ, mmap.

## [0.2.0] — Pipeline & preprocessing
Transforms, scalers/encoders, splits, stats.

## [0.1.0] — Core
NDArray, Dataset, DataLoader, CSV, PCG32.
