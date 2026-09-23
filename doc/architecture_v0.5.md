# NexusData architecture (v0.5)

## Backend layer

```
DataLoader ──pin_memory──► pinned host ──H2D async──► CUDA NDArray
     │
     └── resolve_device(Auto, op, nbytes) ──► Host | CUDA(i)

cuda_api (optional) : streams, kernels (HWC→CHW+normalize fused, resize, hflip)
Philox4x32          : (seed, sample_index, epoch) → identical CPU/GPU draws
```

## Build matrix

| Flag | Effect |
|------|--------|
| `NEXUSDATA_WITH_CUDA=OFF` (default) | `cuda_api_stub.cpp`, all tests on CPU |
| `NEXUSDATA_WITH_CUDA=ON` | compiles `cuda/cuda_api.cu`, links `cudart` |

## Deferred

nvJPEG hardware decode, GPUDirect Storage, HIP/Metal, full augment fusion chain, calibrated Auto thresholds from bench suite (v0.8).
