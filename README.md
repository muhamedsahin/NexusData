# NexusData

Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.

C++20 library that reads datasets and turns samples into `NDArray` batches. It does not train models: no matmul, autograd, loss, or optimizer.

NexusData, veri kümelerini okuyup örnekleri `NDArray` batch'lerine çeviren bir C++20 kütüphanesidir. Model eğitmez; matmul, autograd, loss ve optimizer yoktur.

## License

[Apache License 2.0](LICENSE).

## Install

C++20 and CMake 3.20 or newer. From this directory:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
cmake --install build --prefix /path/to/prefix
```

```cmake
find_package(NexusData REQUIRED)
target_link_libraries(app PRIVATE nexusdata::nexusdata)
```

The default build has no required third-party packages. Optional readers are CMake flags, off unless noted: `NEXUSDATA_WITH_ARROW` (Parquet and Arrow IPC), `NEXUSDATA_WITH_POSTGRES` (libpq), `NEXUSDATA_WITH_ODBC`, `NEXUSDATA_WITH_SQLITE`, `NEXUSDATA_WITH_HDF5`, `NEXUSDATA_WITH_CUDA`, `NEXUSDATA_WITH_NVJPEG`. When a flag is off, that dataset's constructor throws `InvalidArgumentError` and names the flag.

`NEXUSDATA_WITH_PYTHON` builds a separate module, `nexusdata_py`. It is not the Nexus Python shell. See below.

## Image decode and resize (C++)

`load_image` and `decode_image` (`nexusdata/image/image.hpp`) decode JPEG, PNG, BMP, TGA, and GIF with stb_image. The result is an `Image`: HWC `UInt8`, shape `[H, W, C]`. `image_to_tensor` turns that into float CHW in `[0, 1]`.

`decode_media_image` (`nexusdata/image/media.hpp`) also decodes WebP (on by default) and uncompressed 8-bit TIFF. AVIF needs `NEXUSDATA_WITH_AVIF`. JPEG can use libjpeg-turbo instead of stb when `NEXUSDATA_WITH_JPEGTURBO` is on.

Resize lives in `nexusdata/simd/image_ops.hpp`. `resize_hwc_u8` resizes an HWC `uint8` buffer and keeps HWC `uint8`. `resize_to_chw_f32` resizes and writes CHW `float32` in one pass. With null mean and stddev the scale is `src / 255`. `ResampleMode` is `Nearest` or `Bilinear`.

```cpp
#include "nexusdata/image/image.hpp"
#include "nexusdata/simd/image_ops.hpp"

nexusdata::Image image = nexusdata::load_image("photo.jpg");
nexusdata::NDArray chw = nexusdata::image_to_tensor(image.data);

std::vector<float> resized(
    static_cast<std::size_t>(image.channels) * 224 * 224);
nexusdata::simd::resize_to_chw_f32(
    image.data.data<std::uint8_t>(),
    static_cast<std::size_t>(image.height),
    static_cast<std::size_t>(image.width),
    static_cast<std::size_t>(image.channels),
    resized.data(), 224, 224,
    nexusdata::simd::ResampleMode::Bilinear);
```

The C++ library does not write PNG, JPEG, or BMP. Saving is on the Python shell.

`ImageFolder` scans `root/class_name/*`, decodes on `get`, and stores the class index as the label.

## Python shell and C++

Two Python surfaces exist. They do not expose the same functions.

**Nexus Python shell** (`import nexus`, then `nexus.data`). Built from the parent Nexus repository, not from this directory:

```bash
cmake -S . -B build -DNEXUS_BUILD_PYTHON=ON
cmake --build build --target _core
```

Run that at the Nexus repository root. The extension is imported with `PYTHONPATH` pointing at `python/src`.

```python
import nexus

img = nexus.data.load_image("photo.jpg", channels=3, height=224, width=224)
nexus.data.save_image("out.png", img["chw"], img["height"], img["width"], img["channels"])
```

`load_image(path, channels=3, height=0, width=0)` returns a dict with `height`, `width`, `channels`, and `chw`. `chw` is a flat list of floats in CHW order, values in `[0, 1]`. `channels` is `0` (keep the file's channels), `1`, `3`, or `4`. Resize runs only when both `height` and `width` are greater than 0; that path is bilinear `resize_to_chw_f32`. Otherwise the decoded size is kept via `image_to_tensor`.

`save_image(path, chw, height, width, channels)` writes PNG, JPEG, or BMP. The extension chooses the container: `.jpg` and `.jpeg` are JPEG, `.bmp` is BMP, and any other extension, including `.png`, is PNG. `channels` must be 1, 3, or 4. `len(chw)` must equal `height * width * channels`. Values are clamped to `[0, 1]` and stored as 8-bit.

The same module also has `read_csv`, `Table`, `DataLoader`, and `make_loader` for a flat numeric table (last column is the label). That reader is the shell's own parser. It is not `CSVDataset`.

**`nexusdata_py`** (`-DNEXUSDATA_WITH_PYTHON=ON` inside this library) wraps `NDArray` (float32, float64, int64 to and from NumPy) and `to_dlpack`. It does not decode or save images, and it does not open datasets.

PostgreSQL (`PostgresDataset` / `PostgresIterable`, libpq COPY BINARY), ODBC (`OdbcDataset`), Parquet (`ParquetDataset`) and Arrow IPC (`ArrowIpcDataset`), remote files (`RemoteFile`: `http`, `https`, `s3`, `gs`, `az`), and the tokenizers (`Tokenizer`, `WhitespaceHashTokenizer`, `BpeTokenizer`, `WordPieceTokenizer`, `UnigramTokenizer`) are C++ APIs. The Python shell does not call them. The Python package does not expose them. `nexusdata_py` does not expose them either.

`PostgresCopyDataset` decodes an already produced COPY BINARY payload without libpq. Opening a live server still needs `NEXUSDATA_WITH_POSTGRES`.

Türkçe: PostgreSQL, ODBC, Parquet, uzak dosyalar ve tokenizer'lar C++ API'leridir. Python kabuğu bunları çağırmaz. Python paketi bunları sunmaz. Kabuğun görüntü yüzeyi `nexus.data.load_image` ve `nexus.data.save_image` ile sınırlıdır; `nexusdata_py` bu fonksiyonları da sunmaz.

A Turkish walkthrough is in [doc/kullanim.md](doc/kullanim.md).

## Dataset backends

Built in this tree, in addition to the image path above:

| C++ type | Role |
| --- | --- |
| `CSVDataset` | CSV / TSV |
| `JsonDataset`, `JSONLDataset`, `JsonLinesIterable` | JSON records and JSON Lines |
| `ImageFolder` | Class folders |
| `MNIST`, `CIFAR` | Those file layouts |
| `NpyDataset`, `NpzFile` | NumPy `.npy` / `.npz` |
| `TextLineDataset`, `TextLineIterable` | Text lines |
| `WavDataset`, `AudioDataset` | Audio files the audio reader supports |
| `WebDataset` | Tar shards |
| `InMemoryDataset`, `FunctionDataset` | Tables and callbacks you already hold |
| `RemoteFile` | Range reads over HTTP(S) and `s3` / `gs` / `az` |
| Tokenizer classes above | Encode and decode text to ids |

Flagged readers: `ParquetDataset`, `ArrowIpcDataset`, `PostgresDataset`, `OdbcDataset`, `SqliteDataset`, `Hdf5Dataset`. `VideoIterable` needs FFmpeg (`NEXUSDATA_WITH_FFMPEG`).

## Measurements

This file does not publish throughput numbers. The documentation site records one dated internal run (scalar loops against this tree's own AVX2 kernels) and labels other charts as illustrative. Those illustrative figures were not measured. Do not cite them.
