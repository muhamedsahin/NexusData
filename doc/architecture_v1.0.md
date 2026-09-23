# NexusData architecture (v1.0)

Stable layered AI data I/O library. Lower layers never depend on upper layers.

```
adapters/     C API · DLPack · MatrixFlash views · Python (opt)
loading/      DataLoader · Batch · collate · prefetch
pipeline/     transforms · preprocessing · persist
sampling/     Sequential · Random · Weighted · Bucket · Distributed
datasets/     map + iterable compose · readers
backend/      CPU SIMD · ThreadPool · Device/Auto · CUDA (opt)
core/         NDArray · DType · Shape · RNG · Error · version
```

## Release artifacts

| Artifact | Role |
|----------|------|
| `nexusdata::nexusdata` | Primary CMake imported target |
| `nexusdata.pc` | pkg-config |
| `vcpkg.json` / `conanfile.py` | Package manager manifests |
| `install_find_package_smoke` | ctest verifies install + consumer build |

## Non-goals (unchanged)

No matmul, autograd, layers, optimizers, or training loops. GPU kernels are data-prep only.
