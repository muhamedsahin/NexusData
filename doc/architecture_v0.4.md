# NexusData architecture (v0.4)

## Performance layer

```
DataLoader ──prefetch──► BoundedQueue ◄── ThreadPool workers
CSV/JSONL  ──mmap──────► find_* (SIMD) ──► parallel_for parse
CachedDataset ──► LruSampleCache / DiskCache
```

## Determinism contract

Shuffle/order is computed on the main thread from `seed`. Workers only execute
`materialize_batch(i)` for sequential i; the prefetch queue delivers in order.
Therefore `num_workers ∈ {0,1,2,...}` must not change batch contents.

## Backpressure

Prefetch queue capacity = `max(2, num_workers * prefetch_factor)`. Producers block
when full so memory stays bounded.
