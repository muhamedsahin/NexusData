# NexusData architecture (v0.2)

## Layering

```
pipeline       Transform, Compose, RandomApply, OneOf, tabular transforms
preprocessing  fit/transform scalers, encoders, imputer (save/load)
loading        DataLoader, Batch, collate
sampling       SequentialSampler, RandomSampler
datasets       Dataset, InMemory, Subset, Map, CSV, split*, stats
core           NDArray, DType, Shape, Allocator, PCG32, Error, array_utils
```

## Design notes

- **Leakage guard:** scalers/encoders expose explicit `fit` / `transform`. Callers must fit on train only.
- **Online stats:** `Welford` accumulator used in StandardScaler / compute_stats (streaming-ready for later).
- **Transforms vs preprocessing:** Transforms run per-sample in the data pipeline (`MapDataset`). Fitted preprocessors operate on full matrices and can emit a matching `StandardizeTransform` / `MinMaxTransform` for per-sample use.
- **Serialization:** versioned text format `NEXUSDATA_PREPROCESSOR 1` for inference parity.

## Deferred to later phases

Image transforms, SIMD, workers, GPU, JSON/Parquet, Weighted/Bucket/Distributed samplers.
