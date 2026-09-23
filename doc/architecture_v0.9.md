# NexusData architecture (v0.9)

```
adapters/  C API · DLPack · MatrixFlash views · Python (optional)
    ▲
loading / pipeline / sampling / datasets / sources
    ▲
backend / core
```

## Packaging

| Artifact | Role |
|----------|------|
| `find_package(NexusData)` | `nexusdata::nexusdata` (+ Threads) |
| `vcpkg.json` | Manifest port skeleton |
| `conanfile.py` | Conan 2 package |

## Design choices

| Choice | Rationale |
|--------|-----------|
| Vendored `dlpack.h` | Zero deps; stable ABI handshake with PyTorch/TF/JAX |
| C API error codes + TLS message | FFI-friendly; no exceptions across language boundaries |
| MatrixFlash view-only by default | Core stays independent; Flash SDK never required |
| Optional pybind11 | Keeps default build zero-Python |

## Deferred to v1.0

Public docs site polish, full install smoke on all platforms, DLPack CUDA zero-copy import, richer Python Dataset/DataLoader wrappers.
