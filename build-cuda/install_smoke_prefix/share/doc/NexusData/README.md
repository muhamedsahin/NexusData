# NexusData v1.0

Stable C++20 AI data loading library — load, prepare, shuffle, batch to CPU/GPU. No training math.

**License:** Apache-2.0 · **SemVer:** 1.x (see [API stability](doc/api_stability_v1.0.md))

## Build & test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

## Install

```bash
cmake --install build --prefix /path/to/prefix
```

Consumer:

```cmake
find_package(NexusData REQUIRED)
target_link_libraries(app PRIVATE nexusdata::nexusdata)
```

Also: `pkg-config --libs nexusdata`, `vcpkg.json`, `conanfile.py`.

## Optional features

| CMake option | Default | Purpose |
|--------------|---------|---------|
| `NEXUSDATA_WITH_CUDA` | OFF | GPU data-prep backend |
| `NEXUSDATA_WITH_PYTHON` | OFF | `nexusdata_py` (pybind11) |
| `NEXUSDATA_WITH_SQLITE` / `ARROW` / `HDF5` | OFF | Optional readers |
| `NEXUSDATA_SANITIZER` | "" | Address / Undefined / Thread |
| `NEXUSDATA_BUILD_FUZZ` | ON | Portable fuzz smoke |

## Quick start

```cpp
#include "nexusdata/nexusdata.hpp"
using namespace nexusdata;

CSVOptions opt;
opt.label_column = "target";
CSVDataset data("train.csv", opt);
auto [train, val] = random_split(data, {0.8, 0.2}, 42);

DataLoaderOptions lo;
lo.batch_size = 32; lo.shuffle = true; lo.seed = 42;
DataLoader loader(train, lo);
for (const Batch& b : loader) {
    // b.inputs [B, F], b.labels [B]
}
```

## Docs

- [API stability (1.0)](doc/api_stability_v1.0.md)
- [Architecture](doc/architecture_v1.0.md)
- [Changelog](CHANGELOG.md)
- Perf notes: [v0.8](doc/perf_v0.8.md)
