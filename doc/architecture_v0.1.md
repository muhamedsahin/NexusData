# NexusData architecture (v0.1)

## Layering

```
loading      DataLoader, Batch, collate
sampling     SequentialSampler, RandomSampler
datasets     Dataset, InMemory, Subset, CSV, random_split
core         NDArray, DType, Shape, Allocator, PCG32, Error
```

Dependencies flow downward only. Core has no knowledge of datasets or loaders.

## Ownership

- `NDArray` storage is `shared_ptr` to 64-byte-aligned memory (shallow copy by default).
- `Subset` / `DataLoader` / `random_split` hold `shared_ptr<const Dataset>` so views cannot dangle.
- Concrete datasets passed by value into `DataLoader` / `random_split` are moved into a `shared_ptr`.

## Determinism

- RNG: PCG32 (not `std::mt1997` / `std::shuffle`).
- Shuffle: Fisher-Yates with Lemire bounded integers.
- Same `(seed, stream)` ⇒ identical sequences on all platforms.

## Intentionally deferred

SIMD CSV, thread pool, prefetch, mmap, GPU, transforms, image/JSON/Parquet readers.
