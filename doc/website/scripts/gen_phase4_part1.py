from pathlib import Path

ROOT = Path("content/docs")


def w(rel: str, text: str) -> None:
    p = ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text.strip() + "\n", encoding="utf-8")
    print("wrote", rel)


# ========== EN GETTING STARTED ==========
w(
    "en/getting-started/introduction.mdx",
    r'''
---
title: Introduction
description: What NexusData is, who it is for, hard boundaries, and its place in the Nexus ecosystem.
status: stable
order: 10
locale: en
section: getting-started
since: "1.0"
---

## What NexusData is

**NexusData** is a standalone **C++20** library for AI data ingress. It reads raw inputs, prepares them, shuffles indices, batches samples, and delivers `NDArray` payloads to host or (optionally) CUDA device memory.

```cpp
#include "nexusdata/nexusdata.hpp"
using namespace nexusdata;
```

## Why it exists

Training stacks often glue ad-hoc parsers and batch loops. NexusData standardizes:

- Map-style `Dataset` + `Sampler` + `DataLoader`
- Deterministic RNG (`PCG32`; GPU-side `Philox` when CUDA is on)
- Zero-copy friendly views (`Subset`, shared `NDArray` storage)
- Optional CUDA **data-prep only** — never training math

## Who it is for

- C++ teams wanting DataLoader-like ergonomics without a full ML framework
- Pipelines that must reproduce batch order across machines and worker counts
- Projects that may later attach NexusFlash Pro via a one-way adapter

## Hard boundaries

| Out of scope | Belongs elsewhere |
| --- | --- |
| GEMM / matmul | NexusFlash Pro (separate) |
| Autograd | Future compute stack |
| Models / loss / optim / train loop | Planned Nexus\* modules |

NexusData **does not depend** on those libraries.

## Ecosystem

```
Sources → Dataset → Sampler → DataLoader → Batch/NDArray → consumer
                                              ↓ optional
                                        NexusFlash Pro adapter
```

## Mental model

1. **Dataset** — `size()` / `get(i)`; never shuffled in place
2. **Sampler** — index sequence for an epoch
3. **DataLoader** — fetch, collate, prefetch/pin/device
4. **NDArray** — shape + dtype + memory carrier (no compute)

<Callout tone="tip" title="API honesty">
Samples match public headers in NexusData **v1.0**. Optional CMake features (`NEXUSDATA_WITH_SQLITE`, `ARROW`, `HDF5`, `CUDA`) are called out on their pages.
</Callout>
''',
)

w(
    "en/getting-started/installation.mdx",
    r'''
---
title: Installation
description: Requirements, CMake FetchContent, find_package, optional feature flags, and common failures.
status: stable
order: 20
locale: en
section: getting-started
since: "1.0"
---

## Requirements

| Tool | Version |
| --- | --- |
| Language | C++20 |
| CMake | 3.20+ |
| Compilers | GCC 11+, Clang 14+, MSVC 19.3+ |

Optional: CUDA, Arrow (Parquet), SQLite, HDF5, pybind11.

## Build from source

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
cmake --install build --prefix /path/to/prefix
```

## find_package

```cmake
find_package(NexusData REQUIRED)
target_link_libraries(app PRIVATE nexusdata::nexusdata)
```

Also: `pkg-config --libs nexusdata`, `vcpkg.json`, `conanfile.py`.

## FetchContent

```cmake
include(FetchContent)
FetchContent_Declare(
  nexusdata
  GIT_REPOSITORY https://github.com/PLACEHOLDER_ORG/nexusdata.git
  GIT_TAG        v1.0.0
)
FetchContent_MakeAvailable(nexusdata)
target_link_libraries(my_app PRIVATE nexusdata::nexusdata)
```

<Callout tone="warning" title="Repository URL">
Replace `PLACEHOLDER_ORG` once the real Git remote is set in `site.config.ts`.
</Callout>

## CMake options

| Option | Default | Purpose |
| --- | --- | --- |
| `NEXUSDATA_WITH_CUDA` | OFF | GPU data-prep backend |
| `NEXUSDATA_WITH_PYTHON` | OFF | Python bindings |
| `NEXUSDATA_WITH_SQLITE` | OFF | SQLite reader |
| `NEXUSDATA_WITH_ARROW` | OFF | Parquet/Arrow reader |
| `NEXUSDATA_WITH_HDF5` | OFF | HDF5 reader |
| `NEXUSDATA_SANITIZER` | `""` | ASan / UBSan / TSan |
| `NEXUSDATA_BUILD_FUZZ` | ON | Fuzz smoke |

## OS notes

**Linux/macOS:** Ninja + GCC/Clang; enable CUDA toolkit when needed.  
**Windows:** MSVC 19.3+ / Clang-CL; mmap uses Win32 mapping internally.

## Common failures

| Symptom | Fix |
| --- | --- |
| `find_package` not found | Set `CMAKE_PREFIX_PATH` or `NexusData_DIR` |
| CUDA unresolved | Rebuild with `NEXUSDATA_WITH_CUDA=ON` |
| Parquet/HDF5/SQLite missing | Enable matching `NEXUSDATA_WITH_*` + install deps |
''',
)

w(
    "en/getting-started/quickstart.mdx",
    r'''
---
title: Quick start
description: Five-minute path from CSVDataset to DataLoader batches with random_split.
status: stable
order: 30
locale: en
section: getting-started
since: "1.0"
---

## Goal

```cpp
#include "nexusdata/nexusdata.hpp"
using namespace nexusdata;

CSVOptions opt;
opt.label_column = "target";
CSVDataset data("train.csv", opt);

auto [train, val] = random_split(data, {0.8, 0.2}, /*seed=*/42);

DataLoaderOptions lo;
lo.batch_size = 32;
lo.shuffle = true;
lo.drop_last = true;
lo.seed = 42;

DataLoader loader(train, lo);
for (const Batch& b : loader) {
  // b.inputs [B, F], b.labels [B]
}
```

## Line by line

1. Configure `CSVOptions` (label, dtypes, mmap, threads)
2. Construct `CSVDataset` — read-only after load
3. `random_split` → `Subset` views (no row deep-copy)
4. Configure `DataLoaderOptions` and iterate `Batch`es

## Guarantees to expect

- Same seed → same index order, **independent of `num_workers`**
- `drop_last` drops incomplete final batch
- Prefer `DatasetPtr` when composing long-lived graphs

## Next

- [Project structure](/docs/getting-started/project-structure)
- [DataLoader](/docs/concepts/dataloader)
- [CSV source](/docs/sources/csv)
''',
)

w(
    "en/getting-started/project-structure.mdx",
    r'''
---
title: Project structure & concepts
description: Layered architecture — core, dataset, sampling, loading, backend, adapters.
status: stable
order: 35
locale: en
section: getting-started
since: "1.0"
---

## Layers

```
Adapters (DLPack, MatrixFlash, C API, Python)
Loading (DataLoader, Batch, collate)
Sampling (Sequential, Random, Weighted, …)
Dataset (CSV, Image, JSONL, … + compose)
Pipeline / Preprocessing
Core (NDArray, DType, Shape, RNG, errors)
Backend (CPU dispatch, CUDA, pinned, pools)
```

## Data flow

Source/Dataset → Sampler indices → DataLoader materialize → `Batch`/`NDArray` → **your** compute.

## Header map

| Area | Key headers |
| --- | --- |
| Core | `core/ndarray.hpp`, `dtype.hpp`, `shape.hpp`, `random.hpp` |
| Dataset | `dataset/*.hpp`, `dataset/csv/csv_dataset.hpp` |
| Sampling | `sampling/sampler.hpp`, `random.hpp`, `advanced.hpp` |
| Loading | `loading/dataloader.hpp`, `batch.hpp` |
| Backend | `backend/device.hpp`, `cuda_api.hpp` |
| Adapters | `adapter/dlpack.hpp`, `matrixflash.hpp`, `c_api.h` |

Umbrella: `#include "nexusdata/nexusdata.hpp"`.
''',
)

print("phase4 script part1 ok")
