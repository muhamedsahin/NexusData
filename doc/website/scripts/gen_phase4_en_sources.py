from pathlib import Path

ROOT = Path("content/docs/en/sources")
ROOT.mkdir(parents=True, exist_ok=True)

pages = {
    "image.mdx": (
        75,
        "Image / ImageFolder",
        "ImageFolder dataset, decode options, and folder layout.",
        "ImageFolder",
        """
## Layout

```
root/
  class_a/img1.jpg
  class_a/img2.jpg
  class_b/img3.jpg
```

```cpp
#include "nexusdata/dataset/image_folder.hpp"
ImageFolderDataset ds("data/train"); // labels from folder names
```

## Notes

- Decoding uses the bundled image helpers (`image/image.hpp`, stb)
- Pair with transform pipelines under `pipeline/image.hpp` for augmentations
- GPU decode/resize is optional via CUDA prep when enabled

## See also

- [Transforms overview](/docs/concepts/dataset)
- [MNIST / CIFAR](/docs/sources/mnist-cifar)
""",
    ),
    "json.mdx": (
        80,
        "JSON / JSONL",
        "Line-delimited and structured JSON datasets.",
        "JsonlDataset",
        """
## JSONL

```cpp
#include "nexusdata/dataset/jsonl.hpp"
JsonlDataset ds("samples.jsonl");
```

Each line is one record. Field mapping to `Sample` depends on options documented in the header.

## Tips

- Prefer JSONL for streaming-friendly large corpora
- Validate schema externally for production pipelines

## See also

- [Text](/docs/sources/text)
- [Adding a reader](/docs/sources/adding-readers)
""",
    ),
    "mnist-cifar.mdx": (
        85,
        "MNIST / CIFAR",
        "Built-in MNIST and CIFAR dataset readers.",
        "MnistDataset",
        """
```cpp
#include "nexusdata/dataset/mnist.hpp"
#include "nexusdata/dataset/cifar.hpp"

MnistDataset mnist("data/mnist", /*train=*/true);
CifarDataset cifar("data/cifar-10", /*train=*/true);
```

## Notes

- Expect standard binary layouts from the official distributions
- Returns image tensors + labels suitable for classification loaders

## See also

- [ImageFolder](/docs/sources/image)
""",
    ),
    "numpy.mdx": (
        90,
        "NumPy npy / npz",
        "NpyFile, NpyDataset, and NpzFile readers.",
        "NpyDataset",
        """
```cpp
#include "nexusdata/dataset/npy.hpp"

NpyFile f("batch.npy");
NpyDataset ds("samples.npy"); // map-style over array rows / leading axis
NpzFile z("bundle.npz");
```

## Notes

- Supports common NumPy v1/v2 headers as implemented in-tree
- Use for bridging Python-exported arrays into C++ loaders

## See also

- [NDArray](/docs/concepts/ndarray)
""",
    ),
    "mmap-binary.mdx": (
        95,
        "Raw binary / mmap",
        "MappedFile and raw binary access patterns.",
        "MappedFile",
        """
```cpp
#include "nexusdata/io/mapped_file.hpp"

MappedFile map("features.bin");
// interpret bytes with known dtype/shape into NDArray / custom Dataset
```

## Guidance

- Document endianness and record stride in your Dataset
- Prefer mmap for large immutable blobs; copy only when mutating

## See also

- [Memory & ownership](/docs/concepts/memory-ownership)
""",
    ),
    "parquet.mdx": (
        100,
        "Parquet / Arrow",
        "ParquetDataset — requires NEXUSDATA_WITH_ARROW.",
        "ParquetDataset",
        """
<Callout tone="warning" title="Optional dependency">
Enable `NEXUSDATA_WITH_ARROW=ON` and install Apache Arrow to build this reader.
</Callout>

```cpp
#include "nexusdata/dataset/parquet.hpp"
ParquetOptions opt;
ParquetDataset ds("table.parquet", opt);
```

## Notes

- Columnar batches map into `Sample`/`NDArray` per options
- Ideal for large analytics tables feeding training

## See also

- [CSV](/docs/sources/csv)
""",
    ),
    "sqlite.mdx": (
        105,
        "SQLite",
        "SqliteDataset — requires NEXUSDATA_WITH_SQLITE.",
        "SqliteDataset",
        """
<Callout tone="warning" title="Optional dependency">
Enable `NEXUSDATA_WITH_SQLITE=ON`.
</Callout>

```cpp
#include "nexusdata/dataset/sqlite.hpp"
SqliteDataset ds("data.db", /*query or table options*/);
```

## Notes

- Useful when samples live in relational stores
- Keep queries deterministic for reproducible epochs

## See also

- [Adding a reader](/docs/sources/adding-readers)
""",
    ),
    "text.mdx": (
        110,
        "Text",
        "TextDataset for line/document oriented corpora.",
        "TextDataset",
        """
```cpp
#include "nexusdata/dataset/text.hpp"
TextDataset ds("corpus.txt");
```

## Notes

- Pair with tokenizers/transforms outside NexusData (or custom Map)
- Encoding issues should be handled at the Dataset boundary

## See also

- [JSON / JSONL](/docs/sources/json)
""",
    ),
    "audio.mdx": (
        115,
        "Audio (WAV)",
        "WavDataset for PCM WAV samples.",
        "WavDataset",
        """
```cpp
#include "nexusdata/dataset/audio.hpp"
WavOptions opt;
WavDataset ds("wavs/", opt);
```

## Notes

- Inspect `WavInfo` for rate/channels
- Downstream transforms handle resampling/feature extraction

## See also

- [Dataset](/docs/concepts/dataset)
""",
    ),
    "webdataset.mdx": (
        120,
        "WebDataset / tar/zip",
        "WebDataset reader for sharded tar-like sample bundles.",
        "WebDataset",
        """
```cpp
#include "nexusdata/dataset/webdataset.hpp"
WebDatasetOptions opt;
WebDataset ds("shards/data-{000000..000099}.tar", opt);
```

## Notes

- Designed for large sharded corpora
- Sample members map to tensors/bytes per options

## See also

- [Iterable datasets](/docs/concepts/dataset)
""",
    ),
    "hdf5.mdx": (
        125,
        "HDF5",
        "Hdf5Dataset — requires NEXUSDATA_WITH_HDF5.",
        "Hdf5Dataset",
        """
<Callout tone="warning" title="Optional dependency">
Enable `NEXUSDATA_WITH_HDF5=ON` and link libhdf5.
</Callout>

```cpp
#include "nexusdata/dataset/hdf5.hpp"
Hdf5Options opt;
Hdf5Dataset ds("data.h5", opt);
```

## See also

- [NumPy npy](/docs/sources/numpy)
""",
    ),
    "adding-readers.mdx": (
        130,
        "Adding a reader",
        "How to add a new Dataset reader behind the stable map-style interface.",
        "Dataset",
        """
## Checklist

<Steps>
  <Step title="Implement Dataset">
    Provide `size()` and `get(i)` returning `Sample`.
  </Step>
  <Step title="Own options struct">
    Document defaults, errors, and thread-safety.
  </Step>
  <Step title="Register headers">
    Add to umbrella `nexusdata.hpp` when public.
  </Step>
  <Step title="Tests">
    Golden files + edge cases (empty, corrupt, large).
  </Step>
  <Step title="Optional deps">
    Gate heavy libraries behind `NEXUSDATA_WITH_*`.
  </Step>
</Steps>

```cpp
class MyFormatDataset : public Dataset {
public:
  explicit MyFormatDataset(std::string path, MyOptions opt = {});
  std::size_t size() const override;
  Sample get(std::size_t index) const override;
};
```

## Design rules

- Do not shuffle inside the reader — leave that to `Sampler`
- Prefer lazy I/O in `get` for huge corpora; or document eager load
- Surface `IOError` / `InvalidArgumentError` with context

## See also

- [Dataset](/docs/concepts/dataset)
- [Contributing](/docs/getting-started/introduction)
""",
    ),
}

for name, (order, title, desc, _api, body) in pages.items():
    text = f"""---
title: {title}
description: {desc}
status: stable
order: {order}
locale: en
section: sources
since: \"1.0\"
---
{body}
"""
    (ROOT / name).write_text(text.strip() + "\n", encoding="utf-8")
    print("wrote", name)

print("done", len(pages))
