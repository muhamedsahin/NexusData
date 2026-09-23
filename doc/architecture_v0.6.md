# NexusData architecture (v0.6)

## New layers

```
IterableDataset ──► ShuffleBuffer / Chain
Map Dataset ──► Filter / Zip / Concat / take / skip / Repeat
Sampler ──► Weighted / Bucket / Distributed / Batch
Collate ──► stack | pad_sequence(+mask) | ragged
Sources ──► text, WAV, WebDataset(tar), SQLite*, Parquet*, HDF5*
Pipeline ──► transformers + compose steps ──► save/load
```

\* Optional CMake flags; default build exposes types that throw a clear configure error.

## Design choices

| Choice | Rationale |
|--------|-----------|
| Iterable separate from Dataset | Random access vs streaming are different contracts; avoids fake `size()`/`get()` |
| Pad mask on `Batch` | Sequence models need validity without a second return channel |
| WebDataset as ustar-only | Zero deps; zip deferred (needs inflate) |
| Pipeline text format | Matches preprocessor `NEXUSDATA_PREPROCESSOR 1`; human-diffable |
| Arrow/HDF5 stubs | Keep core zero-deps; link when the host project already has them |

## Deferred

Full Arrow/Parquet column projection, HDF5 hyperslabs, ZIP/WebP shards, production BPE tokenizer, SQLite amalgamation vendoring in-tree.
