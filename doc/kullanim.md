# NexusData kullanım kılavuzu

Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.

Bu dosya kütüphaneyi kaynak ağaçtan kurmak, görüntü okuyup yazmak ve hangi API'nin C++'ta kalıp hangisinin Python kabuğundan çağrıldığını anlatır. Eğitim matematiği (matmul, autograd, loss, optimizer) NexusData'nın işi değildir.

İki dilli site `doc/website` içindedir. Bu kılavuz bir PDF değildir.

## Lisans

Kütüphane **Apache License 2.0** altındadır. Metin depo kökündeki `LICENSE` dosyasındadır.

## Kurulum

C++20 ve CMake 3.20 veya daha yenisi gerekir. Komutlar `lib/NexusData` dizininde çalışır:

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

Varsayılan derleme zorunlu üçüncü parti paket istemez. Aşağıdaki okuyucular bayrak kapalıyken kurulmaz; constructor `InvalidArgumentError` fırlatır ve bayrağın adını söyler.

| Bayrak | Ne açar |
| --- | --- |
| `NEXUSDATA_WITH_ARROW` | `ParquetDataset`, `ArrowIpcDataset` |
| `NEXUSDATA_WITH_POSTGRES` | Canlı sunucuya `PostgresDataset` / `PostgresIterable` (libpq, COPY BINARY) |
| `NEXUSDATA_WITH_ODBC` | `OdbcDataset` |
| `NEXUSDATA_WITH_SQLITE` | `SqliteDataset` |
| `NEXUSDATA_WITH_HDF5` | `Hdf5Dataset` |
| `NEXUSDATA_WITH_CUDA` | Veri hazırlama için CUDA. Eğitim matematiği değil |
| `NEXUSDATA_WITH_NVJPEG` | nvJPEG. `NEXUSDATA_WITH_CUDA` gerekir |
| `NEXUSDATA_WITH_FFMPEG` | `VideoIterable` |
| `NEXUSDATA_WITH_PYTHON` | Yalnızca `nexusdata_py`. Aşağıdaki Python kabuğu değil |

`PostgresCopyDataset`, libpq olmadan hazır bir COPY BINARY yükünü çözer. Canlı bağlantı yine `NEXUSDATA_WITH_POSTGRES` ister.

## Görüntü çözme ve yeniden boyutlandırma (C++)

`load_image` ve `decode_image` (`nexusdata/image/image.hpp`) JPEG, PNG, BMP, TGA ve GIF dosyalarını stb_image ile çözer. Sonuç `Image` olur: HWC `UInt8`, şekil `[H, W, C]`. `image_to_tensor` bunu `[0, 1]` aralığında float CHW yapar.

`decode_media_image` WebP'yi (varsayılan derlemede açık) ve sıkıştırılmamış 8-bit TIFF'i de çözer. AVIF için `NEXUSDATA_WITH_AVIF` gerekir. `NEXUSDATA_WITH_JPEGTURBO` açıksa JPEG, stb yerine libjpeg-turbo kullanır.

Yeniden boyutlandırma `nexusdata/simd/image_ops.hpp` içindedir.

- `resize_hwc_u8`: HWC `uint8` tamponunu yeniden boyutlandırır, düzen HWC `uint8` kalır.
- `resize_to_chw_f32`: tek geçişte CHW `float32` yazar. Ortalama ve standart sapma `nullptr` ise ölçek `src / 255` olur.
- `ResampleMode`: `Nearest` veya `Bilinear`.

```cpp
#include "nexusdata/image/image.hpp"
#include "nexusdata/simd/image_ops.hpp"

nexusdata::Image image = nexusdata::load_image("foto.jpg");
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

C++ kütüphanesi PNG, JPEG veya BMP yazmaz. Yazma Python kabuğundadır.

`ImageFolder`, `kök/sınıf_adı/*` düzenini tarar. Çözme `get` içindedir. Etiket, sıralanmış sınıf indeksidir.

## Python kabuğu: yükleme ve kaydetme

Nexus Python kabuğu üst depodan derlenir. `NEXUSDATA_WITH_PYTHON` bunu üretmez. Nexus deposunun kökünde:

```bash
cmake -S . -B build -DNEXUS_BUILD_PYTHON=ON
cmake --build build --target _core
```

Modül `python/src` `PYTHONPATH` üzerindeyken `import nexus` olur. Görüntü fonksiyonları `nexus.data` altındadır.

```python
import nexus

goruntu = nexus.data.load_image("foto.jpg", channels=3, height=224, width=224)
nexus.data.save_image(
    "cikti.png",
    goruntu["chw"],
    goruntu["height"],
    goruntu["width"],
    goruntu["channels"],
)
```

`load_image(path, channels=3, height=0, width=0)` bir sözlük döndürür: `height`, `width`, `channels`, `chw`. `chw` düz bir float listesidir, düzen CHW'dir, değerler `[0, 1]` aralığındadır. `channels` için `0` dosyadaki kanal sayısını korur; diğer geçerli değerler `1`, `3` ve `4`'tür. Yeniden boyutlandırma ancak `height` ve `width` ikisi birden 0'dan büyükse çalışır; o yol bilinear `resize_to_chw_f32` kullanır. Aksi halde çözülen boyut `image_to_tensor` ile korunur.

`save_image(path, chw, height, width, channels)` PNG, JPEG veya BMP yazar. Uzantı kabı seçer: `.jpg` ve `.jpeg` JPEG, `.bmp` BMP, `.png` dahil diğer her uzantı PNG'dir. `channels` 1, 3 veya 4 olmalıdır. `chw` uzunluğu `height * width * channels` olmalıdır. Değerler `[0, 1]` aralığına kırpılır ve 8 bit olarak yazılır.

Aynı modülde `read_csv`, `Table`, `DataLoader` ve `make_loader` vardır. Bu, son sütunu etiket olan düz sayısal bir tablo içindir. `CSVDataset` değildir.

`nexusdata_py` yalnızca `NDArray` (float32, float64, int64 ile NumPy) ve `to_dlpack` sarar. Görüntü çözmez, görüntü yazmaz, veri kümesi açmaz.

## Python kabuğunun çağırmadığı C++ API'leri

PostgreSQL (`PostgresDataset`, `PostgresIterable`), ODBC (`OdbcDataset`), Parquet (`ParquetDataset`) ve Arrow IPC (`ArrowIpcDataset`), uzak dosyalar (`RemoteFile`: `http`, `https`, `s3`, `gs`, `az`) ve tokenizer'lar (`Tokenizer`, `WhitespaceHashTokenizer`, `BpeTokenizer`, `WordPieceTokenizer`, `UnigramTokenizer`) C++ API'leridir.

Python kabuğu bunları çağırmaz. Python paketi bunları sunmaz. `nexusdata_py` de sunmaz.

Tokenizer örneği yalnızca C++'tır:

```cpp
#include "nexusdata/dataset/text.hpp"

nexusdata::BpeTokenizer tok("vocab.json", "merges.txt");
std::vector<std::int64_t> ids = tok.encode("merhaba dunya");
```

Uzak dosya da yalnızca C++'tır: `RemoteFile uri("https://ornek.example/veri.bin")`, ardından `size()` ve `read_at`.

## Diğer veri kaynakları

Bayraksız derlemede ayrıca şunlar vardır: `CSVDataset`, `JsonDataset`, `JSONLDataset`, `JsonLinesIterable`, `MNIST`, `CIFAR`, `NpyDataset`, `NpzFile`, `TextLineDataset`, `TextLineIterable`, `WavDataset`, `AudioDataset`, `WebDataset`, `InMemoryDataset`, `FunctionDataset`.

## Ölçüm

Bu kılavuz hız sayısı vermez. Sitedeki performans sayfası, bir tarihte bu ağacın kendi skaler döngüleri ile AVX2 çekirdeklerini karşılaştıran bir koşuyu yazar. `illustrative` işaretli grafikler ölçülmemiştir; kaynak olarak kullanılmaz.
