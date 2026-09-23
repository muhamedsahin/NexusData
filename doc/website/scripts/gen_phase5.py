# -*- coding: utf-8 -*-
"""Generate Phase 5 docs: preparation, performance, integration, guides, reference, about (EN+TR)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "content" / "docs"


def w(rel: str, text: str) -> None:
    p = ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text.strip() + "\n", encoding="utf-8")
    print("wrote", rel)


# ---------------------------------------------------------------------------
# ENGLISH — preparation
# ---------------------------------------------------------------------------
w(
    "en/preparation/transforms-and-map.mdx",
    r'''
---
title: Transforms & MapDataset
description: Sample-level Transform, Compose, RandomApply, and lazy MapDataset wrapping.
status: stable
order: 10
locale: en
section: preparation
since: "1.0"
---

## Transform contract

A `Transform` takes a `Sample` and returns a new `Sample`. Deterministic unless `is_random()` is true and seeded.

```cpp
#include "nexusdata/pipeline/transform.hpp"
#include "nexusdata/dataset/map.hpp"

auto compose = std::make_shared<Compose>(std::initializer_list<TransformConstPtr>{
    std::make_shared<Resize>(224, 224),
    std::make_shared<ToTensor>(),
});
MapDataset mapped(dataset, compose);
```

## Building blocks

| Type | Role |
| --- | --- |
| `Compose` | Apply transforms in order |
| `RandomApply` | Apply child with probability `p` |
| Custom `Transform` subclass | Your own `apply()` |
| `MapDataset` / `map()` | Lazy `get(i)` → `transform->apply(parent->get(i))` |

## Seeding

Call `set_seed` on random transforms (or on `Compose`, which derives per-child streams). Prefer non-const shared ownership when children must re-seed.

## Thread-safety

Immutable after construction except RNG state on random transforms — treat random transforms as **single-threaded** per instance, or give each worker its own copy.

## See also

- [Image transforms](/docs/preparation/image-transforms)
- [Tabular transforms](/docs/preparation/tabular-transforms)
- [Pipeline persist](/docs/preparation/pipeline-persist)
''',
)

w(
    "en/preparation/image-transforms.mdx",
    r'''
---
title: Image transforms
description: Resize, crop, flip, grayscale, normalize, ToTensor, and Pad on HWC UInt8 samples.
status: stable
order: 20
locale: en
section: preparation
since: "1.0"
---

## Expectation

Image transforms in `nexusdata/pipeline/image.hpp` expect **`Sample.input` as H×W×C `UInt8`**.

## Catalog

| Class | Notes |
| --- | --- |
| `Resize` | Nearest or bilinear |
| `CenterCrop` / `RandomCrop` | Fixed output H×W |
| `RandomHorizontalFlip` / `RandomVerticalFlip` | Seeded |
| `Grayscale` | Keeps channel layout policy of the class |
| `NormalizeImage` | Per-channel mean/std (float path after ToTensor in typical pipelines) |
| `ToTensor` | UInt8 HWC → Float32 CHW scaled to \[0,1\] |
| `Pad` | Border pad to target size |

```cpp
#include "nexusdata/pipeline/image.hpp"

auto t = std::make_shared<Compose>({
    std::make_shared<RandomHorizontalFlip>(0.5f, /*seed=*/7),
    std::make_shared<Resize>(256, 256),
    std::make_shared<CenterCrop>(224, 224),
    std::make_shared<ToTensor>(),
});
```

## MapDataset pattern

Wrap `ImageFolderDataset` (or similar) with `MapDataset` so augmentation runs at `get()` time — not inside the dataset reader.

<Callout tone="tip" title="GPU note">
Optional CUDA image kernels exist when built with `NEXUSDATA_WITH_CUDA`. See [CUDA & pin_memory](/docs/performance/cuda-and-pin-memory).
</Callout>
''',
)

w(
    "en/preparation/tabular-transforms.mdx",
    r'''
---
title: Tabular transforms
description: Sample-level Clip, Log1p, Standardize, and MinMax-style scaling for feature vectors.
status: stable
order: 30
locale: en
section: preparation
since: "1.0"
---

## Headers

`nexusdata/pipeline/tabular.hpp` — operate on `Sample.input` (feature vector or row).

| Class | Behavior |
| --- | --- |
| `ClipTransform` | Clamp each element to \[min, max\] |
| `Log1pTransform` | `log1p(x)` element-wise |
| `StandardizeTransform` | `(x - mean) / scale` with fitted vectors |
| `MinMaxTransform` | Scale using fitted data min/max to a target range |

```cpp
#include "nexusdata/pipeline/tabular.hpp"

auto t = std::make_shared<Compose>({
    std::make_shared<ClipTransform>(-10.0, 10.0),
    std::make_shared<Log1pTransform>(),
});
```

## Fit vs apply

Sample-level tabular transforms that need statistics should be fed **pre-fitted** mean/scale (or use sklearn-style `Transformer`s under [Preprocessing](/docs/preparation/preprocessing) and then export constants into these transforms).
''',
)

w(
    "en/preparation/preprocessing.mdx",
    r'''
---
title: Preprocessing (fit / transform)
description: Sklearn-style Transformer API — scalers, encoders, imputer, normalizer, binarizer.
status: stable
order: 40
locale: en
section: preparation
since: "1.0"
---

## Contract

`Transformer` works on tabular matrices **`[N, F]`** as `NDArray`:

1. `fit(X)` — learn parameters
2. `transform(X)` — apply
3. `fit_transform(X)` — convenience
4. `save` / `load` — versioned text state

```cpp
#include "nexusdata/preprocessing/scaler.hpp"

StandardScaler scaler(/*with_mean=*/true, /*with_std=*/true);
scaler.fit(train_X);
NDArray Xn = scaler.transform(val_X);
scaler.save_file("scaler.txt");
```

## Scalers

| Class | Formula / notes |
| --- | --- |
| `StandardScaler` | `(x - mean) / std` (zero-variance → scale 1) |
| `MinMaxScaler` | Scale to feature range |
| `RobustScaler` | Median / IQR style |
| `MaxAbsScaler` | Divide by max abs |

## Encoders & imputers

| Class | Header |
| --- | --- |
| `LabelEncoder`, `OneHotEncoder`, `OrdinalEncoder` | `encoder.hpp` |
| `Imputer`, `Binarizer`, `Normalizer` | `imputer.hpp` |

## Leakage rule

Fit **only on training folds**. Persist with [Pipeline](/docs/preparation/pipeline-persist) so inference loads the same fitted state.
''',
)

w(
    "en/preparation/pipeline-persist.mdx",
    r'''
---
title: Pipeline save / load
description: Versioned NEXUSDATA_PIPELINE 1 snapshots for fitted Transformers and compose steps.
status: stable
order: 50
locale: en
section: preparation
since: "1.0"
---

## What is stored

`Pipeline` (`nexusdata/pipeline/persist.hpp`) holds:

- Fitted `Transformer` instances
- Optional sample-level `Compose`
- Serializable compose step strings (e.g. `"Clip -1 1"`)

Format tag: **`NEXUSDATA_PIPELINE 1`**.

```cpp
#include "nexusdata/pipeline/persist.hpp"

Pipeline pipe;
pipe.add_transformer(std::make_shared<StandardScaler>());
pipe.transformers()[0]->fit(train_X);
pipe.add_compose_step("Clip -1 1");
pipe.save_file("model_prep.ndpipe");

Pipeline loaded;
loaded.load_file("model_prep.ndpipe"); // default factory: make_transformer_by_type
NDArray out = loaded.transform_matrix(test_X);
```

## Custom types

Pass a factory to `load` / `load_file` that maps type name → `shared_ptr<Transformer>` when you register custom transformers.
''',
)

w(
    "en/preparation/split-and-stats.mdx",
    r'''
---
title: Split & dataset stats
description: random_split, k-fold helpers, Subset views, and DatasetStats summaries.
status: stable
order: 60
locale: en
section: preparation
since: "1.0"
---

## Splits

`nexusdata/dataset/split.hpp` returns `Subset` views — no data copy of parent storage.

```cpp
#include "nexusdata/dataset/split.hpp"

auto [train, val] = random_split(dataset, 0.8, 0.2, /*seed=*/42);
// or: random_split(ds, {0.7, 0.15, 0.15}, seed) → vector<Subset>
```

`Fold` pairs (`train` / `test`) support k-fold style helpers in the same header.

## Stats

`nexusdata/dataset/stats.hpp` summarizes label / feature distributions for diagnostics — use after split so you do not peek at held-out folds when deciding preprocessing.

## See also

- [Subset & memory](/docs/concepts/memory-ownership)
- [Determinism](/docs/concepts/determinism)
''',
)

w(
    "en/preparation/caching.mdx",
    r'''
---
title: Caching
description: In-memory LRU sample cache, DiskCache blobs, and CachedDataset wrappers.
status: stable
order: 70
locale: en
section: preparation
since: "1.0"
---

## LRU sample cache

`LruSampleCache` + `CachedDataset` memoize `get(i)` by string key (`prefix:index`).

```cpp
#include "nexusdata/dataset/cached.hpp"
#include "nexusdata/cache/lru_cache.hpp"

auto cache = std::make_shared<LruSampleCache>(/*capacity=*/1024);
CachedDataset cached(dataset, cache, "img");
```

## Disk cache

`DiskCache` stores preprocessed `NDArray` blobs under a directory (`NDCACHE1` magic). Key = hash(content + pipeline id). Use for expensive decode/augment outputs you want across process restarts.

<Callout tone="warning" title="Correctness">
Invalidate or namespace keys whenever transform parameters or source bytes change.
</Callout>
''',
)

# ---------------------------------------------------------------------------
# ENGLISH — performance
# ---------------------------------------------------------------------------
w(
    "en/performance/workers-and-prefetch.mdx",
    r'''
---
title: Workers & prefetch
description: num_workers, prefetch_factor, persistent_workers, and the determinism contract.
status: stable
order: 10
locale: en
section: performance
since: "1.0"
---

## Knobs

| Option | Effect |
| --- | --- |
| `num_workers = 0` | Synchronous fetch + collate on the caller thread |
| `num_workers > 0` | Background workers fill a bounded queue |
| `prefetch_factor` | Queue capacity ≈ `num_workers * factor` (minimum 2 when workers &gt; 0) |
| `persistent_workers` | Keep worker threads across `reset_epoch` |
| `timeout_ms` | Wait budget when pulling from the prefetch queue |

## Determinism

Batch *i* contains the **same sample indices** for a given seed **regardless of `num_workers`**. Worker count affects wall time, not index order.

## Practical tips

- Start with `num_workers = 0` when debugging transforms
- Raise workers until CPU or disk saturates; watch RAM (each worker holds live samples)
- Prefer `shared_ptr` datasets so worker lifetimes stay safe

See [DataLoader](/docs/concepts/dataloader).
''',
)

w(
    "en/performance/cuda-and-pin-memory.mdx",
    r'''
---
title: CUDA & pin_memory
description: Optional CUDA data-prep backend, pinned host staging, and Device placement.
status: stable
order: 20
locale: en
section: performance
since: "1.0"
---

## Build flag

Enable with CMake `-DNEXUSDATA_WITH_CUDA=ON`. Without it, CUDA APIs resolve to host / clear errors — CPU-only builds stay zero-dependency.

## DataLoader fields

```cpp
lo.pin_memory = true;           // page-locked host staging before H2D
lo.device = Device::cuda(0);    // or Device::host() / Device::auto_select()
```

## Scope

CUDA support is for **data preparation and transfers**, not training math (GEMM lives outside NexusData).

## Philox

GPU-side RNG uses Philox when CUDA is enabled — seed streams separately from host `PCG32`. See [Determinism](/docs/concepts/determinism).
''',
)

w(
    "en/performance/simd-and-scan.mdx",
    r'''
---
title: SIMD byte scan
description: Runtime-dispatched AVX2/scalar scan used by parsers — enable via NEXUSDATA_ENABLE_AVX2.
status: stable
order: 30
locale: en
section: performance
since: "1.0"
---

## What it is

`nexusdata/simd/byte_scan.hpp` provides fast newline / delimiter scanning used by CSV-style parsers. Kernels are **runtime dispatched** (AVX2 when available, else scalar).

## CMake

- `NEXUSDATA_ENABLE_AVX2` (default ON) — compile AVX2 kernels; still safe on non-AVX2 CPUs via dispatch

## Measuring

Build `bench_suite` (`NEXUSDATA_BUILD_BENCH=ON`) and run locally. Throughput depends on CPU and OS — **do not treat homepage or docs tables as guaranteed numbers** unless measured on your machine.
''',
)

w(
    "en/performance/auto-device.mdx",
    r'''
---
title: Auto device policy
description: Device::auto_select, calibration thresholds, and break-even Host vs CUDA decisions.
status: stable
order: 40
locale: en
section: performance
since: "1.0"
---

## API

```cpp
#include "nexusdata/backend/device.hpp"

Device d = Device::auto_select();
AutoDevicePolicy policy;
CalibrationReport report = calibrate_auto_device_policy(policy);
Device resolved = resolve_device(d, GpuOpKind::ImageAugment, nbytes, report.policy);
```

## Behavior

- Without CUDA build / no device → always Host
- Thresholds (`image_nbytes_threshold`, `tabular_nbytes_threshold`) come from a break-even model using measured host memcpy and assumed PCIe bandwidth
- Op kind (`GpuOpKind`) selects which threshold applies

Calibrate once at process start on the target machine.
''',
)

w(
    "en/performance/benchmarking.mdx",
    r'''
---
title: Benchmarking
description: How to build and run the microbenchmark harness — measure on your hardware.
status: stable
order: 50
locale: en
section: performance
since: "1.0"
---

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNEXUSDATA_BUILD_BENCH=ON
cmake --build build --target bench_suite
```

## Harness

`nexusdata/bench/harness.hpp` provides `Timer`, `time_best`, and `print_result` helpers used by `bench_suite`.

## Honesty

Published numbers in older notes are **machine-specific**. Re-run on your CPU/GPU before citing throughput in papers or dashboards. The marketing site marks illustrative charts as non-measured until verified.
''',
)

# ---------------------------------------------------------------------------
# ENGLISH — integration
# ---------------------------------------------------------------------------
w(
    "en/integration/dlpack.mdx",
    r'''
---
title: DLPack
description: Export and import NDArray via DLManagedTensor for framework interop.
status: stable
order: 10
locale: en
section: integration
since: "1.0"
---

```cpp
#include "nexusdata/adapter/dlpack.hpp"

DLManagedTensor* m = ndarray_to_dlpack(array);
// consumer must call m->deleter(m) when done (or take ownership)

NDArray back = ndarray_from_dlpack(m, /*take_ownership=*/false);
```

## Lifetime

`ndarray_to_dlpack` keeps `NDArray` storage alive until the DLPack deleter runs. `ndarray_from_dlpack` deep-copies onto host unless you carefully manage ownership flags.

C consumers: `nexus_ndarray_to_dlpack` / `nexus_ndarray_from_dlpack` in the [C API](/docs/integration/c-api).
''',
)

w(
    "en/integration/c-api.mdx",
    r'''
---
title: C API
description: Stable ABI surface for FFI — NexusNDArray create/destroy, dtype, DLPack bridges.
status: stable
order: 20
locale: en
section: integration
since: "1.0"
---

## Header

`nexusdata/adapter/c_api.h`

- Status codes: `NEXUS_OK`, `NEXUS_ERR_*`
- `nexus_last_error()` — thread-local message
- `nexus_version()` — version string

```c
NexusNDArray* a = NULL;
int64_t shape[2] = {2, 3};
nexus_ndarray_create(shape, 2, NEXUS_DTYPE_FLOAT32, &a);
/* ... */
nexus_ndarray_destroy(a);
```

Shared builds on Windows use `NEXUSDATA_API` dllexport/dllimport when `NEXUSDATA_BUILD_SHARED` is defined.
''',
)

w(
    "en/integration/matrixflash.mdx",
    r'''
---
title: MatrixFlash adapter
description: Unidirectional zero-copy FlashTensorView / FlashBatchView hand-off (no Flash dependency).
status: stable
order: 30
locale: en
section: integration
since: "1.0"
---

## Design

NexusData **never includes** MatrixFlash headers. `MatrixFlashAdapter` builds `FlashTensorView` / `FlashBatchView` from `NDArray` / `Batch` for a one-way hand-off.

```cpp
#include "nexusdata/adapter/matrixflash.hpp"

FlashTensorView v = MatrixFlashAdapter::view(array);
FlashBatchView b = MatrixFlashAdapter::view_batch(batch);
```

Optional CMake `-DNEXUSDATA_WITH_MATRIXFLASH=ON` declares a linked Flash SDK for richer integration later; the view helpers work without it.
''',
)

w(
    "en/integration/python-bindings.mdx",
    r'''
---
title: Python bindings
description: Optional pybind11 module (NEXUSDATA_WITH_PYTHON) for NDArray-centric interop.
status: beta
order: 40
locale: en
section: integration
since: "1.0"
---

## Enable

```bash
cmake -S . -B build -DNEXUSDATA_WITH_PYTHON=ON
```

Builds the `nexusdata_py` module when pybind11 is available. Treat the Python surface as **beta** relative to the C++ v1.0 headers — prefer C++ or C API for production ABI stability.

Prefer DLPack / NumPy bridges in your binding layer rather than re-implementing Dataset I/O in Python.
''',
)

# ---------------------------------------------------------------------------
# ENGLISH — guides
# ---------------------------------------------------------------------------
w(
    "en/guides/csv-to-loader.mdx",
    r'''
---
title: Guide — CSV to DataLoader
description: End-to-end map-style CSV training loop with shuffle, workers, and epoch reset.
status: stable
order: 10
locale: en
section: guides
since: "1.0"
---

```cpp
#include "nexusdata/nexusdata.hpp"
using namespace nexusdata;

CSVOptions opt;
opt.label_column = "target";
auto ds = std::make_shared<CSVDataset>("train.csv", opt);

DataLoaderOptions lo;
lo.batch_size = 64;
lo.shuffle = true;
lo.seed = 42;
lo.num_workers = 2;
lo.drop_last = true;

DataLoader loader(ds, lo);
for (int epoch = 0; epoch < 3; ++epoch) {
  loader.reset_epoch(42 + epoch);
  for (const Batch& b : loader) {
    // b.inputs, b.labels → your trainer
  }
}
```

See [CSV](/docs/sources/csv) and [DataLoader](/docs/concepts/dataloader).
''',
)

w(
    "en/guides/image-augment-pipeline.mdx",
    r'''
---
title: Guide — Image augment pipeline
description: ImageFolder + Compose + MapDataset + collate_stack for fixed-size tensors.
status: stable
order: 20
locale: en
section: guides
since: "1.0"
---

```cpp
auto raw = std::make_shared<ImageFolderDataset>("data/train");
auto tfm = std::make_shared<Compose>({
    std::make_shared<RandomHorizontalFlip>(0.5f, 7),
    std::make_shared<Resize>(256, 256),
    std::make_shared<CenterCrop>(224, 224),
    std::make_shared<ToTensor>(),
});
auto ds = std::make_shared<MapDataset>(raw, tfm);

DataLoaderOptions lo;
lo.batch_size = 32;
lo.shuffle = true;
lo.seed = 1;
DataLoader loader(ds, lo); // default collate_stack
```

All images in a batch must share the same CHW shape after transforms.
''',
)

w(
    "en/guides/streaming-iterable.mdx",
    r'''
---
title: Guide — Streaming IterableDataset
description: Use IterableDataset when size()/get(i) is impossible — Chain, shuffle buffer, text lines.
status: stable
order: 30
locale: en
section: guides
since: "1.0"
---

Map-style `Dataset` requires random access. For logs, shards, or infinite streams, use **`IterableDataset`** APIs in `dataset/iterable.hpp` and compose helpers (`Chain`, shuffle buffer, take/skip) from `dataset/compose.hpp`.

Do **not** fake `size()` / `get()` on streams — keep the contracts separate.

Pair with appropriate collate (`collate_pad_sequence` for variable-length text/audio).
''',
)

w(
    "en/guides/reproduce-batches.mdx",
    r'''
---
title: Guide — Reproducible batches
description: Fix seeds, prefer drop_last, and verify batch indices across worker counts.
status: stable
order: 40
locale: en
section: guides
since: "1.0"
---

## Checklist

1. Set `DataLoaderOptions::seed` (and transform seeds) explicitly
2. Call `reset_epoch(seed)` the same way on every machine
3. Prefer `drop_last = true` when comparing across shard sizes
4. Change `num_workers` freely — index order must stay identical
5. Fit preprocessors on train only; load the same pipeline file at eval

Golden tests in the C++ suite lock PCG32 / sampler behavior — mirror those seeds in docs examples when asserting equality.
''',
)

# ---------------------------------------------------------------------------
# ENGLISH — reference
# ---------------------------------------------------------------------------
w(
    "en/reference/api-overview.mdx",
    r'''
---
title: API overview
description: Map of public headers for NexusData v1.0 — core, dataset, loading, pipeline, adapters.
status: stable
order: 10
locale: en
section: reference
since: "1.0"
---

## Umbrella

```cpp
#include "nexusdata/nexusdata.hpp"
```

## Layers

| Area | Headers (under `include/nexusdata/`) |
| --- | --- |
| Core | `core/ndarray.hpp`, `dtype.hpp`, `shape.hpp`, `error.hpp`, `random.hpp`, `allocator.hpp` |
| Dataset | `dataset/dataset.hpp`, `sample.hpp`, readers, `map.hpp`, `compose.hpp`, `iterable.hpp` |
| Sampling | `sampling/sampler.hpp`, `sequential.hpp`, `random.hpp`, `advanced.hpp` |
| Loading | `loading/dataloader.hpp`, `batch.hpp` |
| Pipeline | `pipeline/transform.hpp`, `image.hpp`, `tabular.hpp`, `persist.hpp` |
| Preprocess | `preprocessing/*.hpp` |
| Backend | `backend/device.hpp`, `pinned.hpp`, `cuda_api.hpp`, `thread_pool.hpp` |
| Cache | `cache/lru_cache.hpp`, `disk_cache.hpp` |
| Adapters | `adapter/c_api.h`, `dlpack.hpp`, `matrixflash.hpp` |
| Version | `version.hpp` |

Stability notes: `doc/api_stability_v1.0.md` in the library repo.
''',
)

w(
    "en/reference/cmake-options.mdx",
    r'''
---
title: CMake options
description: Feature flags for CUDA, SQLite, Arrow/Parquet, HDF5, Python, AVX2, benches, and fuzz.
status: stable
order: 20
locale: en
section: reference
since: "1.0"
---

| Option | Default | Meaning |
| --- | --- | --- |
| `NEXUSDATA_BUILD_TESTS` | ON | Unit tests |
| `NEXUSDATA_BUILD_EXAMPLES` | ON | Examples |
| `NEXUSDATA_BUILD_BENCH` | ON | `bench_suite` |
| `NEXUSDATA_BUILD_FUZZ` | ON | Fuzz smoke targets |
| `NEXUSDATA_ENABLE_AVX2` | ON | Compile AVX2 scan kernels |
| `NEXUSDATA_WITH_CUDA` | OFF | CUDA data-prep backend |
| `NEXUSDATA_WITH_SQLITE` | OFF | SqliteDataset |
| `NEXUSDATA_WITH_ARROW` | OFF | Parquet/Arrow |
| `NEXUSDATA_WITH_HDF5` | OFF | HDF5 dataset |
| `NEXUSDATA_WITH_PYTHON` | OFF | pybind11 module |
| `NEXUSDATA_WITH_MATRIXFLASH` | OFF | Declare Flash SDK link |
| `NEXUSDATA_WARNINGS_AS_ERRORS` | OFF | `-Werror` style |

Default build aims for **zero required third-party deps**.
''',
)

w(
    "en/reference/errors.mdx",
    r'''
---
title: Error model
description: Error hierarchy, message helpers, and C API NexusStatus mapping.
status: stable
order: 30
locale: en
section: reference
since: "1.0"
---

## C++

All library errors derive from `nexusdata::Error` (`core/error.hpp`), including `InvalidArgumentError`, `IndexError`, `ShapeError`, `IOError`, and internal failures.

Catch `Error` at boundaries; use specific types when recovering.

## C API

Functions return `NexusStatus`. Call `nexus_last_error()` for a thread-local detail string after a non-`NEXUS_OK` result.
''',
)

w(
    "en/reference/versioning.mdx",
    r'''
---
title: Versioning
description: SemVer 1.0.0, SOVERSION, version.hpp macros, and package compatibility.
status: stable
order: 40
locale: en
section: reference
since: "1.0"
---

NexusData **1.0.0** uses SemVer with CMake `SameMajorVersion` package compatibility and shared-library `SOVERSION 1`.

```cpp
#include "nexusdata/version.hpp"
// NEXUSDATA_VERSION_MAJOR / MINOR / PATCH and string helpers
```

C API: `nexus_version()` returns the version string.

Breaking public C++ API changes require a major bump. See `CHANGELOG.md` and `doc/api_stability_v1.0.md`.
''',
)

# ---------------------------------------------------------------------------
# ENGLISH — about
# ---------------------------------------------------------------------------
w(
    "en/about/license.mdx",
    r'''
---
title: License
description: NexusData library license — Apache-2.0.
status: stable
order: 10
locale: en
section: about
since: "1.0"
---

The NexusData **library** is licensed under **Apache License 2.0**. See the `LICENSE` file at the repository root.

This documentation website may list the same license in site config. Third-party fonts and npm packages carry their own licenses.
''',
)

w(
    "en/about/changelog.mdx",
    r'''
---
title: Changelog
description: High-level release history from 0.x through stable 1.0.0.
status: stable
order: 20
locale: en
section: about
since: "1.0"
---

## 1.0.0

Stable SemVer release: public API contract, `version.hpp`, Apache-2.0, pkg-config, `find_package` smoke.

## 0.x highlights

| Version | Focus |
| --- | --- |
| 0.9 | C API, DLPack, MatrixFlash adapter, packaging |
| 0.8 | Benchmarks, SIMD scan, Auto device calibration |
| 0.7 | Fuzz / sanitizers / hardening |
| 0.6 | Iterable, advanced samplers, pipeline persist, more sources |
| 0.5 | Optional CUDA backend |
| 0.4–0.1 | Core DataLoader stack and early formats |

Full detail: repository `CHANGELOG.md`.
''',
)

w(
    "en/about/ecosystem.mdx",
    r'''
---
title: Ecosystem
description: NexusData among planned Nexus modules — data I/O only, no training math.
status: stable
order: 30
locale: en
section: about
since: "1.0"
---

NexusData is the **data ingress** piece. Planned siblings (compute, loss, model, optim, train, orchestration) are separate libraries. NexusData does **not** depend on them.

Optional one-way adapter views exist for a future MatrixFlash / NexusFlash Pro hand-off.

Status badges on the marketing homepage reflect design intent; only ship claims that match public headers and release notes.
''',
)

print("--- EN done; writing TR ---")

# ---------------------------------------------------------------------------
# TURKISH — preparation
# ---------------------------------------------------------------------------
w(
    "tr/preparation/transforms-and-map.mdx",
    r'''
---
title: Transform ve MapDataset
description: Sample düzeyinde Transform, Compose, RandomApply ve tembel MapDataset sarmalayıcı.
status: stable
order: 10
locale: tr
section: preparation
since: "1.0"
---

## Transform sözleşmesi

`Transform`, bir `Sample` alır ve yeni bir `Sample` döndürür. `is_random()` true ve seed verilmedikçe deterministiktir.

```cpp
#include "nexusdata/pipeline/transform.hpp"
#include "nexusdata/dataset/map.hpp"

auto compose = std::make_shared<Compose>(std::initializer_list<TransformConstPtr>{
    std::make_shared<Resize>(224, 224),
    std::make_shared<ToTensor>(),
});
MapDataset mapped(dataset, compose);
```

## Yapı taşları

| Tip | Rol |
| --- | --- |
| `Compose` | Transform'ları sırayla uygular |
| `RandomApply` | Alt transform'u `p` olasılığıyla uygular |
| Özel alt sınıf | Kendi `apply()` |
| `MapDataset` / `map()` | Tembel `get(i)` → `transform->apply(...)` |

## Seed

Rastgele transform'larda (veya `Compose` üzerinde) `set_seed` çağırın. Worker başına ayrı kopya tercih edin.

## Ayrıca bakın

- [Görüntü transform'ları](/docs/preparation/image-transforms)
- [Tablo transform'ları](/docs/preparation/tabular-transforms)
- [Pipeline kalıcılığı](/docs/preparation/pipeline-persist)
''',
)

w(
    "tr/preparation/image-transforms.mdx",
    r'''
---
title: Görüntü transform'ları
description: HWC UInt8 örneklerde Resize, crop, flip, grayscale, normalize, ToTensor ve Pad.
status: stable
order: 20
locale: tr
section: preparation
since: "1.0"
---

## Beklenti

`nexusdata/pipeline/image.hpp` içindeki transform'lar **`Sample.input` = H×W×C `UInt8`** bekler.

## Katalog

| Sınıf | Not |
| --- | --- |
| `Resize` | Nearest veya bilinear |
| `CenterCrop` / `RandomCrop` | Sabit çıkış H×W |
| `RandomHorizontalFlip` / `RandomVerticalFlip` | Seed'li |
| `Grayscale` | Kanal düzeni sınıf politikasına bağlı |
| `NormalizeImage` | Kanal başına mean/std |
| `ToTensor` | UInt8 HWC → Float32 CHW \[0,1\] |
| `Pad` | Hedef boyuta kenar dolgu |

Augmentation'ı reader içinde değil, `MapDataset` ile `get()` zamanında uygulayın.

<Callout tone="tip" title="GPU">
`NEXUSDATA_WITH_CUDA` ile opsiyonel CUDA görüntü çekirdekleri. Bkz. [CUDA ve pin_memory](/docs/performance/cuda-and-pin-memory).
</Callout>
''',
)

w(
    "tr/preparation/tabular-transforms.mdx",
    r'''
---
title: Tablo transform'ları
description: Öznitelik vektörleri için Clip, Log1p, Standardize ve MinMax tarzı ölçekleme.
status: stable
order: 30
locale: tr
section: preparation
since: "1.0"
---

`nexusdata/pipeline/tabular.hpp` — `Sample.input` üzerinde çalışır.

| Sınıf | Davranış |
| --- | --- |
| `ClipTransform` | \[min, max\] kırpma |
| `Log1pTransform` | Eleman bazlı `log1p` |
| `StandardizeTransform` | `(x - mean) / scale` |
| `MinMaxTransform` | Fitted min/max ile ölçekleme |

İstatistik gerektiren adımlar için önce [Ön işleme](/docs/preparation/preprocessing) ile fit edin, sabitleri bu transform'lara aktarın.
''',
)

w(
    "tr/preparation/preprocessing.mdx",
    r'''
---
title: Ön işleme (fit / transform)
description: Sklearn tarzı Transformer API — scaler, encoder, imputer, normalizer, binarizer.
status: stable
order: 40
locale: tr
section: preparation
since: "1.0"
---

`Transformer`, tabular **`[N, F]`** `NDArray` üzerinde çalışır: `fit` → `transform` → `save`/`load`.

```cpp
StandardScaler scaler(true, true);
scaler.fit(train_X);
NDArray Xn = scaler.transform(val_X);
```

| Sınıf | Not |
| --- | --- |
| `StandardScaler` / `MinMaxScaler` / `RobustScaler` / `MaxAbsScaler` | `scaler.hpp` |
| `LabelEncoder` / `OneHotEncoder` / `OrdinalEncoder` | `encoder.hpp` |
| `Imputer` / `Binarizer` / `Normalizer` | `imputer.hpp` |

**Sızıntı kuralı:** Yalnızca eğitim katlarında fit edin; çıkarımda aynı pipeline dosyasını yükleyin.
''',
)

w(
    "tr/preparation/pipeline-persist.mdx",
    r'''
---
title: Pipeline kaydet / yükle
description: Fitted Transformer ve compose adımları için NEXUSDATA_PIPELINE 1 anlık görüntüsü.
status: stable
order: 50
locale: tr
section: preparation
since: "1.0"
---

`Pipeline` (`persist.hpp`): fitted transformer'lar, isteğe bağlı `Compose`, serileştirilebilir adım dizgileri. Etiket: **`NEXUSDATA_PIPELINE 1`**.

```cpp
Pipeline pipe;
pipe.add_transformer(std::make_shared<StandardScaler>());
pipe.transformers()[0]->fit(train_X);
pipe.save_file("model_prep.ndpipe");
```

Özel tipler için `load` fabrikası verin (`make_transformer_by_type` varsayılan).
''',
)

w(
    "tr/preparation/split-and-stats.mdx",
    r'''
---
title: Bölme ve istatistik
description: random_split, k-fold yardımcıları, Subset görünümleri ve DatasetStats.
status: stable
order: 60
locale: tr
section: preparation
since: "1.0"
---

`split.hpp` `Subset` görünümleri döndürür — ebeveyn depolama kopyalanmaz.

```cpp
auto [train, val] = random_split(dataset, 0.8, 0.2, /*seed=*/42);
```

`stats.hpp` tanı için dağılım özetleri üretir; ön işleme kararlarından önce held-out katlara bakmayın.
''',
)

w(
    "tr/preparation/caching.mdx",
    r'''
---
title: Önbellekleme
description: Bellek içi LRU sample cache, DiskCache blob'ları ve CachedDataset.
status: stable
order: 70
locale: tr
section: preparation
since: "1.0"
---

`CachedDataset` + `LruSampleCache`, `get(i)` sonuçlarını `prefix:index` anahtarıyla saklar.

`DiskCache`, ön işlenmiş `NDArray` blob'larını dizine yazar (`NDCACHE1`). Transform veya kaynak değişince anahtarları geçersiz kılın.
''',
)

# ---------------------------------------------------------------------------
# TURKISH — performance
# ---------------------------------------------------------------------------
w(
    "tr/performance/workers-and-prefetch.mdx",
    r'''
---
title: Worker ve prefetch
description: num_workers, prefetch_factor, persistent_workers ve determinizm sözleşmesi.
status: stable
order: 10
locale: tr
section: performance
since: "1.0"
---

| Seçenek | Etki |
| --- | --- |
| `num_workers = 0` | Senkron fetch + collate |
| `num_workers > 0` | Arka plan kuyruğu |
| `prefetch_factor` | Kapasite ≈ workers × factor |
| `persistent_workers` | Epoch'lar arası worker tutma |

Aynı seed için batch *i* **worker sayısından bağımsız** aynı indeksleri içerir.

Hata ayıklarken `num_workers = 0` ile başlayın. Bkz. [DataLoader](/docs/concepts/dataloader).
''',
)

w(
    "tr/performance/cuda-and-pin-memory.mdx",
    r'''
---
title: CUDA ve pin_memory
description: Opsiyonel CUDA veri hazırlama, pinned host ve Device yerleşimi.
status: stable
order: 20
locale: tr
section: performance
since: "1.0"
---

CMake: `-DNEXUSDATA_WITH_CUDA=ON`. CUDA **yalnızca veri hazırlama / transfer** içindir; eğitim matematiği yoktur.

```cpp
lo.pin_memory = true;
lo.device = Device::cuda(0);
```

GPU RNG: Philox. Host: PCG32. Bkz. [Determinizm](/docs/concepts/determinism).
''',
)

w(
    "tr/performance/simd-and-scan.mdx",
    r'''
---
title: SIMD byte tarama
description: Parser'larda kullanılan AVX2/scalar runtime dispatch — NEXUSDATA_ENABLE_AVX2.
status: stable
order: 30
locale: tr
section: performance
since: "1.0"
---

`simd/byte_scan.hpp` CSV tarzı ayırıcı taramayı hızlandırır. AVX2 çekirdekleri derlenir; CPU desteklemiyorsa scalar'a düşer.

Ölçüm için `bench_suite` çalıştırın — dokümandaki rakamlar makineye özgüdür, garantili iddia değildir.
''',
)

w(
    "tr/performance/auto-device.mdx",
    r'''
---
title: Auto cihaz politikası
description: Device::auto_select, kalibrasyon eşikleri ve Host vs CUDA kararları.
status: stable
order: 40
locale: tr
section: performance
since: "1.0"
---

```cpp
CalibrationReport report = calibrate_auto_device_policy({});
Device resolved = resolve_device(Device::auto_select(), GpuOpKind::ImageAugment, nbytes, report.policy);
```

CUDA yoksa her zaman Host. Eşikler host memcpy ölçümü ve varsayılan PCIe modeliyle ayarlanır — hedef makinede bir kez kalibre edin.
''',
)

w(
    "tr/performance/benchmarking.mdx",
    r'''
---
title: Benchmark
description: Mikrobenchmark harness'ini derleyip kendi donanımınızda ölçme.
status: stable
order: 50
locale: tr
section: performance
since: "1.0"
---

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNEXUSDATA_BUILD_BENCH=ON
cmake --build build --target bench_suite
```

Yayınlanmış sayılar makineye özgüdür. Alıntılamadan önce kendi CPU/GPU'nuzda yeniden ölçün.
''',
)

# ---------------------------------------------------------------------------
# TURKISH — integration
# ---------------------------------------------------------------------------
w(
    "tr/integration/dlpack.mdx",
    r'''
---
title: DLPack
description: NDArray'i DLManagedTensor ile dışa/içe aktarma.
status: stable
order: 10
locale: tr
section: integration
since: "1.0"
---

```cpp
DLManagedTensor* m = ndarray_to_dlpack(array);
NDArray back = ndarray_from_dlpack(m, /*take_ownership=*/false);
```

Deleter çalışana kadar `NDArray` depolaması canlı kalır. C tarafı: [C API](/docs/integration/c-api).
''',
)

w(
    "tr/integration/c-api.mdx",
    r'''
---
title: C API
description: FFI için kararlı ABI — NexusNDArray, dtype, DLPack köprüleri.
status: stable
order: 20
locale: tr
section: integration
since: "1.0"
---

Başlık: `nexusdata/adapter/c_api.h`. Dönüş: `NexusStatus`; ayrıntı: `nexus_last_error()` (thread-local).

```c
NexusNDArray* a = NULL;
int64_t shape[2] = {2, 3};
nexus_ndarray_create(shape, 2, NEXUS_DTYPE_FLOAT32, &a);
nexus_ndarray_destroy(a);
```
''',
)

w(
    "tr/integration/matrixflash.mdx",
    r'''
---
title: MatrixFlash adaptörü
description: Tek yönlü sıfır kopya FlashTensorView / FlashBatchView (Flash bağımlılığı yok).
status: stable
order: 30
locale: tr
section: integration
since: "1.0"
---

NexusData MatrixFlash başlıklarını **içermez**. `MatrixFlashAdapter` yalnızca görünüm üretir.

Opsiyonel `-DNEXUSDATA_WITH_MATRIXFLASH=ON` ileride daha zengin bağlama için SDK bağlantısı bildirir.
''',
)

w(
    "tr/integration/python-bindings.mdx",
    r'''
---
title: Python bağları
description: Opsiyonel pybind11 modülü (NEXUSDATA_WITH_PYTHON).
status: beta
order: 40
locale: tr
section: integration
since: "1.0"
---

`-DNEXUSDATA_WITH_PYTHON=ON` ile `nexusdata_py` derlenir. C++ v1.0'a göre **beta** kabul edin; üretim ABI'si için C++/C API tercih edin.
''',
)

# ---------------------------------------------------------------------------
# TURKISH — guides
# ---------------------------------------------------------------------------
w(
    "tr/guides/csv-to-loader.mdx",
    r'''
---
title: Rehber — CSV'den DataLoader'a
description: Shuffle, worker ve epoch reset ile uçtan uca CSV eğitim döngüsü.
status: stable
order: 10
locale: tr
section: guides
since: "1.0"
---

```cpp
CSVOptions opt;
opt.label_column = "target";
auto ds = std::make_shared<CSVDataset>("train.csv", opt);

DataLoaderOptions lo;
lo.batch_size = 64;
lo.shuffle = true;
lo.seed = 42;
lo.num_workers = 2;
DataLoader loader(ds, lo);
for (const Batch& b : loader) { /* ... */ }
```

Bkz. [CSV](/docs/sources/csv), [DataLoader](/docs/concepts/dataloader).
''',
)

w(
    "tr/guides/image-augment-pipeline.mdx",
    r'''
---
title: Rehber — Görüntü augmentation
description: ImageFolder + Compose + MapDataset + collate_stack.
status: stable
order: 20
locale: tr
section: guides
since: "1.0"
---

```cpp
auto raw = std::make_shared<ImageFolderDataset>("data/train");
auto tfm = std::make_shared<Compose>({
    std::make_shared<RandomHorizontalFlip>(0.5f, 7),
    std::make_shared<Resize>(256, 256),
    std::make_shared<CenterCrop>(224, 224),
    std::make_shared<ToTensor>(),
});
auto ds = std::make_shared<MapDataset>(raw, tfm);
DataLoader loader(ds, DataLoaderOptions{});
```

Batch içindeki tüm tensörler aynı CHW şeklinde olmalıdır.
''',
)

w(
    "tr/guides/streaming-iterable.mdx",
    r'''
---
title: Rehber — IterableDataset akışı
description: size()/get(i) yokken IterableDataset, Chain ve shuffle buffer.
status: stable
order: 30
locale: tr
section: guides
since: "1.0"
---

Rastgele erişim yoksa map-style `Dataset` yerine **`IterableDataset`** kullanın. Akışlarda sahte `size()`/`get()` yazmayın.

Değişken uzunluk için `collate_pad_sequence` tercih edin.
''',
)

w(
    "tr/guides/reproduce-batches.mdx",
    r'''
---
title: Rehber — Tekrarlanabilir batch'ler
description: Seed sabitleme, drop_last ve worker sayısından bağımsız indeks doğrulama.
status: stable
order: 40
locale: tr
section: guides
since: "1.0"
---

1. `seed` ve transform seed'lerini açıkça verin  
2. `reset_epoch` aynı şekilde çağrılsın  
3. Karşılaştırmada `drop_last = true` tercih edin  
4. `num_workers` değişse de indeks sırası aynı kalmalı  
5. Fit yalnızca train; eval'de aynı pipeline dosyası  
''',
)

# ---------------------------------------------------------------------------
# TURKISH — reference
# ---------------------------------------------------------------------------
w(
    "tr/reference/api-overview.mdx",
    r'''
---
title: API genel bakış
description: NexusData v1.0 genel başlık haritası — core, dataset, loading, pipeline, adaptörler.
status: stable
order: 10
locale: tr
section: reference
since: "1.0"
---

```cpp
#include "nexusdata/nexusdata.hpp"
```

| Alan | Başlıklar |
| --- | --- |
| Core | `core/ndarray.hpp`, `dtype`, `shape`, `error`, `random`, `allocator` |
| Dataset | `dataset/*`, `map`, `compose`, `iterable` |
| Sampling | `sampling/*` |
| Loading | `loading/dataloader.hpp`, `batch.hpp` |
| Pipeline / preprocess | `pipeline/*`, `preprocessing/*` |
| Backend / cache | `backend/*`, `cache/*` |
| Adaptörler | `adapter/c_api.h`, `dlpack.hpp`, `matrixflash.hpp` |

Kararlılık: repodaki `doc/api_stability_v1.0.md`.
''',
)

w(
    "tr/reference/cmake-options.mdx",
    r'''
---
title: CMake seçenekleri
description: CUDA, SQLite, Arrow, HDF5, Python, AVX2, bench ve fuzz bayrakları.
status: stable
order: 20
locale: tr
section: reference
since: "1.0"
---

Varsayılan derleme **zorunlu üçüncü parti bağımlılık istemez**. Öne çıkan bayraklar: `NEXUSDATA_WITH_CUDA`, `_SQLITE`, `_ARROW`, `_HDF5`, `_PYTHON`, `_MATRIXFLASH`, `NEXUSDATA_ENABLE_AVX2`, `NEXUSDATA_BUILD_BENCH` / `TESTS` / `FUZZ`.
''',
)

w(
    "tr/reference/errors.mdx",
    r'''
---
title: Hata modeli
description: Error hiyerarşisi ve C API NexusStatus eşlemesi.
status: stable
order: 30
locale: tr
section: reference
since: "1.0"
---

C++: `nexusdata::Error` ve türevleri (`InvalidArgumentError`, `ShapeError`, `IOError`, …).

C: `NexusStatus` + `nexus_last_error()`.
''',
)

w(
    "tr/reference/versioning.mdx",
    r'''
---
title: Sürümleme
description: SemVer 1.0.0, SOVERSION, version.hpp ve paket uyumluluğu.
status: stable
order: 40
locale: tr
section: reference
since: "1.0"
---

NexusData **1.0.0** — SemVer, `SOVERSION 1`, CMake `SameMajorVersion`. Makrolar: `nexusdata/version.hpp`. C: `nexus_version()`.
''',
)

# ---------------------------------------------------------------------------
# TURKISH — about
# ---------------------------------------------------------------------------
w(
    "tr/about/license.mdx",
    r'''
---
title: Lisans
description: NexusData kütüphane lisansı — Apache-2.0.
status: stable
order: 10
locale: tr
section: about
since: "1.0"
---

NexusData **kütüphanesi** **Apache License 2.0** altındadır (`LICENSE`). Site yapılandırması aynı lisansı gösterebilir; npm/font paketlerinin kendi lisansları vardır.
''',
)

w(
    "tr/about/changelog.mdx",
    r'''
---
title: Değişiklik günlüğü
description: 0.x'ten kararlı 1.0.0'a özet sürüm geçmişi.
status: stable
order: 20
locale: tr
section: about
since: "1.0"
---

**1.0.0** — kararlı API, `version.hpp`, Apache-2.0, pkg-config, `find_package` smoke.

| Sürüm | Odak |
| --- | --- |
| 0.9 | C API, DLPack, MatrixFlash, paketleme |
| 0.8 | Bench, SIMD, Auto cihaz |
| 0.7 | Fuzz / sanitizer |
| 0.6 | Iterable, pipeline persist |
| 0.5 | Opsiyonel CUDA |

Ayrıntı: `CHANGELOG.md`.
''',
)

w(
    "tr/about/ecosystem.mdx",
    r'''
---
title: Ekosistem
description: NexusData'nın planlanan Nexus modülleri arasındaki yeri — yalnızca veri I/O.
status: stable
order: 30
locale: tr
section: about
since: "1.0"
---

NexusData **veri girişi** katmanıdır. Hesaplama / loss / model / optim / train ayrı kütüphanelerdir; NexusData onlara bağlı değildir.

Pazarlama rozetleri tasarım niyetini yansıtır — yalnızca public başlık ve sürüm notlarıyla uyumlu iddialar kullanın.
''',
)

print("Phase 5 content generation complete.")
