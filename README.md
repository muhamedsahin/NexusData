<div align="center">

# 📂 NexusData

### Veri kümelerini okuyup örnekleri `NDArray` batch'lerine çeviren C++20 kütüphanesi

*CSV'den Parquet'e, görüntüden ses ve metne — tek bir veri yüzeyi. Model eğitmez: matmul, autograd, loss ve optimizer yoktur.*

[![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C?style=for-the-badge&logo=cplusplus&logoColor=white)](https://en.cppreference.com/w/cpp/20)
[![CMake](https://img.shields.io/badge/CMake-3.20+-064F8C?style=for-the-badge&logo=cmake&logoColor=white)](CMakeLists.txt)
[![Deps](https://img.shields.io/badge/zorunlu_ba%C4%9F%C4%B1ml%C4%B1l%C4%B1k-yok-2EA44F?style=for-the-badge)](#-kurulum--install)
[![License](https://img.shields.io/badge/Lisans-Apache--2.0-D22128?style=for-the-badge&logo=apache&logoColor=white)](LICENSE)

[Kurulum](#-kurulum--install) • [Görüntü](#-görüntü-çözme-ve-yeniden-boyutlama-c) • [Python](#-python-kabuğu-ve-c) • [Arka uçlar](#-veri-kümesi-arka-uçları--dataset-backends) • [Nexus](https://github.com/muhamedsahin/Nexus)

</div>

> Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.

**EN:** C++20 library that reads datasets and turns samples into `NDArray` batches. It does not train models: no matmul, autograd, loss, or optimizer.

---

## ✨ Bir bakışta

<table>
<tr>
<td width="25%" align="center">🖼<br><b>Görüntü</b><br><sub>JPEG · PNG · BMP · TGA · GIF · WebP · TIFF · AVIF*</sub></td>
<td width="25%" align="center">📊<br><b>Tablo</b><br><sub>CSV/TSV · JSON/JSONL · NPY/NPZ · Parquet* · Arrow*</sub></td>
<td width="25%" align="center">🗄<br><b>Veritabanı</b><br><sub>PostgreSQL* · ODBC* · SQLite* · HDF5*</sub></td>
<td width="25%" align="center">🔤<br><b>Metin & ses</b><br><sub>BPE · WordPiece · Unigram · WAV · WebDataset</sub></td>
</tr>
</table>

<sub>* CMake bayrağı ile açılır. Bayrak kapalıyken o veri kümesinin kurucusu bayrağı adlandıran `InvalidArgumentError` fırlatır.</sub>

---

## 🛠 Kurulum / Install

C++20 ve CMake 3.20+. Bu klasörden:

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

Varsayılan derlemede zorunlu üçüncü taraf paket yoktur. İsteğe bağlı okuyucular (varsayılan kapalı):

| Bayrak | Açtığı |
| --- | --- |
| `NEXUSDATA_WITH_ARROW` | Parquet ve Arrow IPC |
| `NEXUSDATA_WITH_POSTGRES` | libpq |
| `NEXUSDATA_WITH_ODBC` · `NEXUSDATA_WITH_SQLITE` · `NEXUSDATA_WITH_HDF5` | ilgili okuyucular |
| `NEXUSDATA_WITH_CUDA` · `NEXUSDATA_WITH_NVJPEG` | GPU çözme |
| `NEXUSDATA_WITH_AVIF` · `NEXUSDATA_WITH_JPEGTURBO` · `NEXUSDATA_WITH_FFMPEG` | AVIF, libjpeg-turbo, `VideoIterable` |
| `NEXUSDATA_WITH_PYTHON` | ayrı `nexusdata_py` modülü (Nexus Python kabuğu değildir) |

---

## 🖼 Görüntü çözme ve yeniden boyutlama (C++)

- `load_image` / `decode_image` (`nexusdata/image/image.hpp`): stb_image ile JPEG, PNG, BMP, TGA, GIF → `Image` (HWC `UInt8`, `[H, W, C]`).
- `image_to_tensor`: float CHW, `[0, 1]`.
- `decode_media_image` (`nexusdata/image/media.hpp`): ayrıca WebP (varsayılan açık) ve sıkıştırmasız 8-bit TIFF.
- `nexusdata/simd/image_ops.hpp`: `resize_hwc_u8` (HWC→HWC) ve tek geçişte `resize_to_chw_f32`; `ResampleMode::Nearest` / `Bilinear`.

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

C++ kütüphanesi PNG/JPEG/BMP **yazmaz**; kaydetme Python kabuğundadır. `ImageFolder`, `root/class_name/*` klasörlerini tarar, `get` anında çözer ve sınıf indeksini etiket yapar.

---

## 🐍 Python kabuğu ve C++

İki Python yüzeyi vardır ve **aynı fonksiyonları sunmazlar**.

**1. Nexus Python kabuğu** (`import nexus` → `nexus.data`) — üst Nexus deposunun kökünden derlenir:

```bash
cmake -S . -B build -DNEXUS_BUILD_PYTHON=ON
cmake --build build --target _core
```

```python
import nexus

img = nexus.data.load_image("photo.jpg", channels=3, height=224, width=224)
nexus.data.save_image("out.png", img["chw"], img["height"], img["width"], img["channels"])
```

<details>
<summary><b>Ayrıntılar</b></summary>

- `load_image(path, channels=3, height=0, width=0)` → `height`, `width`, `channels`, `chw` (CHW sırasıyla düz float listesi, `[0, 1]`). `channels`: `0` (dosyanınki), `1`, `3`, `4`. Yeniden boyutlama yalnız `height` ve `width` > 0 iken (bilinear `resize_to_chw_f32`).
- `save_image(path, chw, height, width, channels)` PNG, JPEG veya BMP yazar; uzantı kabı seçer (`.jpg/.jpeg` → JPEG, `.bmp` → BMP, diğer her şey → PNG). `channels` 1, 3 veya 4; `len(chw) == height*width*channels`; değerler `[0,1]`'e kırpılıp 8-bit saklanır.
- Aynı modülde düz sayısal tablo için `read_csv`, `Table`, `DataLoader`, `make_loader` (son sütun etiket). Bu, kabuğun kendi ayrıştırıcısıdır; `CSVDataset` değildir.

</details>

**2. `nexusdata_py`** (`-DNEXUSDATA_WITH_PYTHON=ON`) — `NDArray` (float32, float64, int64 ↔ NumPy) ve `to_dlpack`. Görüntü çözmez/kaydetmez, veri kümesi açmaz.

> PostgreSQL, ODBC, Parquet, uzak dosyalar ve tokenizer'lar **yalnız C++ API'sidir**. Python kabuğu ve `nexusdata_py` bunları sunmaz.
> `PostgresCopyDataset` hazır COPY BINARY yükünü libpq olmadan çözer; canlı sunucu yine `NEXUSDATA_WITH_POSTGRES` ister.

Türkçe anlatım: [doc/kullanim.md](doc/kullanim.md).

---

## 🗂 Veri kümesi arka uçları / Dataset backends

| C++ türü | Görev |
| --- | --- |
| `CSVDataset` | CSV / TSV |
| `JsonDataset`, `JSONLDataset`, `JsonLinesIterable` | JSON kayıtları ve JSON Lines |
| `ImageFolder` | Sınıf klasörleri |
| `MNIST`, `CIFAR` | Bu dosya düzenleri |
| `NpyDataset`, `NpzFile` | NumPy `.npy` / `.npz` |
| `TextLineDataset`, `TextLineIterable` | Metin satırları |
| `WavDataset`, `AudioDataset` | Ses okuyucunun desteklediği dosyalar |
| `WebDataset` | Tar parçaları |
| `InMemoryDataset`, `FunctionDataset` | Elinizdeki tablolar ve geri çağrılar |
| `RemoteFile` | HTTP(S) ve `s3` / `gs` / `az` aralık okuma |
| `Tokenizer`, `WhitespaceHashTokenizer`, `BpeTokenizer`, `WordPieceTokenizer`, `UnigramTokenizer` | Metin ↔ kimlik |

Bayraklı: `ParquetDataset`, `ArrowIpcDataset`, `PostgresDataset`, `OdbcDataset`, `SqliteDataset`, `Hdf5Dataset`, `VideoIterable` (FFmpeg).

---

## 📏 Ölçümler / Measurements

Bu dosya verim sayısı yayımlamaz. Belge sitesi tarihli tek bir iç koşuyu (skaler döngüler vs bu ağacın AVX2 çekirdekleri) kaydeder; diğer grafikleri "örnek" olarak işaretler. **O örnek değerler ölçülmemiştir, alıntılanmamalıdır.**

## 📄 Lisans / License

[Apache License 2.0](LICENSE).

---

<div align="center">

**🎓 Bu proje BTÜ (Bursa Teknik Üniversitesi) Bilgisayar Mühendisliği 1. sınıf öğrencisi tarafından yapay zeka destekli olarak yapıldı.**

<sub>Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.</sub>

<sub>⚡ [Nexus](https://github.com/muhamedsahin/Nexus) ekosisteminin bir parçasıdır.</sub>

</div>
