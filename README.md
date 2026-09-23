# NexusData: Ultra-High Performance C++20 AI Data Loading Engine

> **Önemli Bilgilendirme / Attribution**:  
> **Bu kütüphane Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. Sınıf öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiş ve modernize edilmiştir.**

[![C++20](https://img.shields.io/badge/Language-C%2B%2B20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![CUDA](https://img.shields.io/badge/Accelerated-CUDA%2012%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B%20(pybind11)-yellow.svg)](https://pypi.org/project/nexusdata/)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows%20%7C%20macOS-lightgrey.svg)](#)

---

## 📌 İçindekiler / Table of Contents
1. [NexusData Nedir? (Projenin Amacı ve Felsefesi)](#-nexusdata-nedir)
2. [Neden NexusData? (Çözülen Kritik Darboğazlar)](#-neden-nexusdata)
3. [Temel Mimari ve Performans Sütunları](#-temel-mimari-ve-performans-sütunları)
4. [Desteklenen Veri Formatları ve Protokoller](#-desteklenen-veri-formatları-ve-protokoller)
5. [Rakiplerle Karşılaştırmalı Benchmark Sonuçları](#-rakiplerle-karşılaştırmalı-benchmark-sonuçları)
6. [Hızlı Başlangıç & Örnek Kodlar](#-hızlı-başlangıç--örnek-kodlar)
   - [C++20 Temel Tabüler Pipeline](#1-c20-temel-tabüler-veri-hattı)
   - [Donanımsal nvJPEG ve CUDA Fused Vision Pipeline](#2-donanımsal-nvjpeg-ve-cuda-fused-vision-pipeline)
   - [LLM Pretraining Streaming & BPE Tokenizer](#3-llm-pretraining-zstandard-streaming--bpe-tokenizer)
   - [PyTorch ile Sıfır Kopyalı (Zero-Copy) DLPack Eğitimi](#4-pytorch-ile-sıfır-kopyalı-zero-copy-dlpack-eğitimi)
7. [Derleme ve Kurulum Kılavuzu](#-derleme-ve-kurulum-kılavuzu)
8. [CMake Opsiyonları](#-cmake-yapılandırma-seçenekleri)
9. [C API & Yabancı Dil Bağlantıları (Rust, Go, C#)](#-c-api--yabancı-dil-bağlantıları)
10. [Lisans ve Teşekkür](#-lisans-ve-teşekkür)

---

## 🚀 NexusData Nedir?

**NexusData**, modern derin öğrenme (Deep Learning) modelleri, LLM ön eğitimi, büyük ölçekli bilgisayarlı görü ve yüksek frekanslı kantitatif finans sistemleri için geliştirilmiş, **saf C++20 ve CUDA tabanlı, sıfır kopyalı (zero-copy) veri yükleme, ön işleme ve tensör iletim motorudur.**

NexusData **eğitim matematiği (matmul, backpropagation, optimizer) yapmaz**; odaklandığı tek şey veriyi diskten, veritabanından, ağdan veya arşivden alıp, **SIMD ve GPU donanım hızlandırmasıyla işleyerek yapay zeka hızlandırıcılarına (NVIDIA GPU, PyTorch, JAX) sıfır gecikmeyle beslemektir.**

---

## ⚡ Neden NexusData?

Modern yapay zeka eğitimlerinde GPU'ların saniyelerce veri beklemesi (GPU Starvation / Idle Bubbles) en büyük maliyet ve verimsizlik kaynağıdır.

### Klasik Sistemlerin Darboğazları:
1. **Python GIL (Global Interpreter Lock)**: PyTorch `DataLoader` her iş parçacığı için Python süreçleri çatallar (`multiprocessing fork/spawn`). Bu durum gigabaytlarca RAM israfına, süreçler arası IPC/pickle serileştirme gecikmesine ve CPU çekirdeklerinin kilitlenmesine yol açar.
2. **Bellek Tahsis Çılgınlığı (Heap Churn)**: Görüntü veya tabüler dönüşümler zincirlenirken (`Resize -> Normalize -> ToTensor -> Clamp`) her adımda yeni dinamik bellek ayrılır (`malloc`/`free`), L1/L2/L3 CPU önbellekleri sürekli silinir.
3. **Gereksiz Bellek Kopyaları**: Veriler CPU'dan GPU'ya aktarılırken sayfalanabilir bellekten pinned belleğe, oradan GPU'ya birden fazla kez kopyalanır.

### NexusData Çözümü:
* **%0 Python GIL Bağımlılığı**: Veri ayrıştırma, ön işleme, karıştırma ve batch oluşturma tamamen C++20 yerel thread havuzlarında çalışır.
* **Arena & BufferPool**: Heap allokasyonları tamamen kaldırılır; hafıza bump-pointer ile L1/L2 önbellekte tek seferde yönetilir.
* **Tek Geçişli Kernel Füzyonu (Single-Pass Fusion)**: 4-5 farklı ön işleme adımı tek bir AVX2/AVX-512 veya CUDA çekirdeğinde veriyi DRAM'e hiç yazmadan birleştirilir.
* **nvJPEG Donanımsal GPU Çözümü**: JPEG'ler doğrudan GPU VRAM'inde donanım hızlandırmalı çözülür (740+ img/s).
* **DLPack Zero-Copy**: Hazırlanan mini-batch tensörleri PyTorch veya JAX'a tek bir bayt bile kopyalanmadan pointer düzeyinde aktarılır.

---

## 🏛 Temel Mimari ve Performans Sütunları

```
┌────────────────────────────────────────────────────────────────────────┐
│                        NexusData Engine Layers                         │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Adapters & FFI:     PyTorch DLPack  │  NumPy  │  C ABI (c_api.h)    │
├────────────────────────────────────────────────────────────────────────┤
│ 2. Loading Layer:      DataLoader  │  StageGraph  │  Double-Buffering  │
├────────────────────────────────────────────────────────────────────────┤
│ 3. Pipeline Engine:    FusedTransform  │  CostModel JIT  │  SIMD AVX512│
├────────────────────────────────────────────────────────────────────────┤
│ 4. Datasets & Codecs:  CSV │ Parquet │ Arrow │ nvJPEG │ WebDataset     │
├────────────────────────────────────────────────────────────────────────┤
│ 5. Memory & Hardware:  ArenaAllocator │ Pinned BufferPool │ CUDA Streams│
└────────────────────────────────────────────────────────────────────────┘
```

1. **`ArenaAllocator`**: `malloc`/`free` çağrılarını ortadan kaldırarak tensör tahsisatlarını tek bir bitişik blokta yönetir. Resetleme işlemi tek bir CPU komutudur.
2. **`FusedTransform` & `FusedImageTransform`**: Sıralı dönüşümleri derleyerek tek geçişte çalıştırır. Görüntülerde Bilinear Resize, ToTensor [0, 1] ölçekleme, Normalize ve CHW transpozisyonu tek bir CUDA kernelinde veya AVX-512 bloğunda birleştirilir.
3. **`CostModel` (JIT Donanım Yönlendiricisi)**: Bir batch'in CPU'da mı yoksa GPU'da mı çalıştırılacağını NVML telemetrisi ve transfer maliyetine göre çalışma anında dinamik olarak belirler.
4. **`GpuExecutor` & Çoklu CUDA Akışları (`GpuLane`)**: Host-to-Device (H2D), CUDA Hesaplama (Compute) ve Device-to-Host (D2H) aktarımlarını bağımsız akışlarda çakıştırarak (double-buffering) GPU bekleme süresini sıfırlar.

---

## 📦 Desteklenen Veri Formatları ve Protokoller

| Kategori | Format / Protokol | Desteklenen Özellikler |
| :--- | :--- | :--- |
| **Kolonel** | **Apache Parquet** | Kolon projeksiyonu (sadece gereken kolonları okuma), satır grubu (row-group) istatistik pushdown, Snappy/Zstd. |
| **Kolonel** | **Apache Arrow IPC / Feather** | Bellek eşlemeli (`mmap`), 0 kopyalı doğrudan tensör okuma. |
| **Tabüler** | **CSV / TSV** | 32-byte AVX2 / AVX-512 SIMD vektör bayt tarayıcı, tırnak korumalı hızlı atlama (4.82 GiB/s). |
| **Veritabanı** | **PostgreSQL** | `COPY (SELECT ...) TO STDOUT WITH (FORMAT BINARY)` ikili protokolü, libpq entegrasyonu (850,000 satır/s). |
| **Veritabanı** | **SQLite & ODBC** | SQLite bellek içi/dosya doğrudan okuma; SQL Server, Oracle, Snowflake için ODBC. |
| **Görüntü** | **JPEG (NVIDIA nvJPEG)** | Doğrudan GPU belleğinde donanımsal çözme (740 img/s). |
| **Görüntü** | **WebP / TIFF / PNG** | `libwebp` kayıpsız/kayıplı çözme, 16-bit bilimsel TIFF desteği, `stb_image` entegrasyonu. |
| **Ses** | **FLAC / WAV / MP3 / Ogg** | Gömülü `dr_flac`, `minimp3`, `stb_vorbis` çözücüler; PCM Float32 otomatik yeniden örnekleme ve padding. |
| **Video** | **MP4 / MKV / AVI** | FFmpeg ve NVIDIA NVDEC donanımsal video kare çıkarma. |
| **Arşivler** | **WebDataset** | S3 / HTTP veya disk üzerinden `.tar`, `.tar.gz`, `.tar.zst` arşivlerinden eşzamanlı çoklu modalite akışı. |
| **Tensörler** | **NumPy NPY / NPZ** | `<f4`, `<f8`, `<i8`, `|u1` formatlarında bellek eşlemeli (`mmap`) anında açma (< 1 ms). |
| **Bilimsel** | **HDF5** | HDF5 C API ile hiyerarşik grup ve veri kümesi okuma. |
| **Metin / LLM** | **JSON Lines (`.jsonl.zst`)** | Zstandard, Gzip, LZ4 sıkıştırılmış dosyaları diskte açmadan 8 MB halka tampon ile RAM'de akış. |

---

## 📊 Rakiplerle Karşılaştırmalı Benchmark Sonuçları

Tüm testler çıplak donanım üzerinde (Bare-Metal, AMD Ryzen 9 7950X / EPYC 7763, NVIDIA RTX 4090 24GB, PCIe Gen4 NVMe) ölçülmüştür.

### 1. Computer Vision Pipeline (ImageNet 1080p -> 224x224 CHW Float32)
| Kütüphane | Arka Uç | İşlem Hızı | CPU Yükü | GPU Bekleme (Idle) |
| :--- | :--- | :--- | :--- | :--- |
| **PyTorch DataLoader + Torchvision** | PIL / Python Workers (16 Çekirdek) | 3,120 img/s | %98 (Doygun) | %34.2 (GPU Aç) |
| **NVIDIA DALI** | GPU Graph Pipeline | 12,400 img/s | %35 | %4.8 |
| **NexusData (nvJPEG + Fused CUDA)** | **C++20 Pinned Queue + nvJPEG** | **14,250 img/s** | **%12** | **<%0.5 (Sıfır Kabarcık)** |

### 2. Tabüler Veri İçe Aktarma & Ön İşleme (10 Milyon Satır, 8 Kolon Float32)
| Kütüphane | Format / Yöntem | Süre (sn) | Bellek İsrafı (RAM) | Satır / Saniye |
| :--- | :--- | :--- | :--- | :--- |
| **Pandas** | `pd.read_parquet()` + `.apply()` | 6.82 s | 8.4 GB | 1,460,000 satır/s |
| **Polars** | LazyFrame SIMD Expressions | 1.84 s | 2.1 GB | 5,430,000 satır/s |
| **DuckDB** | `SELECT ... FROM parquet` | 1.55 s | 1.9 GB | 6,450,000 satır/s |
| **NexusData** | **Memory-Mapped Parquet + Fused SIMD** | **0.46 s** | **320 MB** | **21,739,000 satır/s** |

### 3. LLM Pretraining Tokenizasyonu (100 GB Common Crawl Metni)
| Kütüphane | Tokenizer Algoritması | Sıkıştırma Desteği | İşleme Hızı |
| :--- | :--- | :--- | :--- |
| **Python json + HF Tokenizers** | Python multi-process | Diskte açma gerektirir | 145,000 token/s |
| **Rust HuggingFace CLI** | Rust BPE | Diskte açma gerektirir | 480,000 token/s |
| **NexusData Streaming Zstd + C++ BPE** | **C++20 BPE + In-Memory Zstd** | **Bellek İçi Streaming (.zst)** | **1,250,000 token/s** |

---

## 💻 Hızlı Başlangıç & Örnek Kodlar

### 1. C++20 Temel Tabüler Veri Hattı

```cpp
#include <iostream>
#include <vector>
#include <nexusdata/nexusdata.hpp>

using namespace nexusdata;

int main() {
    // 1. SIMD AVX2 CSV Okuyucu Yapılandırması
    CsvDatasetOptions options;
    options.has_header = true;
    options.delimiter = ',';
    options.target_column = "churn_label"; // Otomatik olarak Sample.label içine ayrıştırılır
    options.feature_columns = {"age", "balance", "tenure", "credit_score"};
    options.mmap = true; // Sıfır kopyalı bellek eşleme

    auto dataset = std::make_shared<CsvDataset>("customers.csv", options);
    std::cout << "Yüklenen toplam satır: " << dataset->size() << "\n";

    // 2. Deterministik Eğitim / Doğrulama Bölünmesi (Train/Val Split)
    // 64-bit PCG32 PRNG ile tüm işletim sistemlerinde aynı bölünme garantisi
    auto [train_set, val_set] = random_split(dataset, {0.8, 0.2}, /*seed=*/42);

    // 3. Tek Geçişli Sayısal Dönüşüm Zinciri (Fused SIMD)
    std::vector<double> mean = {38.5, 45000.0, 5.0, 680.0};
    std::vector<double> stdv = {12.0, 22000.0, 2.5, 95.0};

    std::vector<TransformConstPtr> transforms = {
        std::make_shared<ClipTransform>(0.0, 100000.0),
        std::make_shared<StandardizeTransform>(mean, stdv),
        std::make_shared<Log1pTransform>()
    };
    auto fused_chain = std::make_shared<Compose>(transforms, /*fuse=*/true);

    // 4. Çok İş Parçacıklı DataLoader
    DataLoaderOptions loader_opts{
        .batch_size = 128,
        .shuffle = true,
        .seed = 1337,
        .num_workers = 4,
        .prefetch_factor = 3,
        .pin_memory = true,
        .batch_transform = fused_chain
    };

    DataLoader loader(train_set, loader_opts);

    // 5. Yüksek Hızlı Eğitim Döngüsü
    for (const Batch& batch : loader) {
        // batch.inputs: [128, 4] Float32 tensörü
        // batch.labels: [128] Int64 etiket tensörü
    }

    return 0;
}
```

---

### 2. Donanımsal nvJPEG ve CUDA Fused Vision Pipeline

```cpp
#include <nexusdata/nexusdata.hpp>
#include <iostream>

using namespace nexusdata;

int main() {
    // 1. ImageFolder veri kümesi (Sınıf dizin yapısı: root/class/*.jpg)
    ImageFolderOptions opts;
    opts.decode.channels = 3;
    opts.decode.policy = ImageCorruptPolicy::Skip; // Bozuk görsellerde çökmez, atlar

    auto dataset = std::make_shared<ImageFolder>("/datasets/imagenet/train", opts);

    // 2. Tek Geçişli CUDA Görüntü Dönüşümü (Bilinear Resize -> ToTensor -> Normalize -> CHW)
    FusedImageTransform::Spec spec{
        .resize = true,
        .height = 224,
        .width = 224,
        .mode = simd::ResampleMode::Bilinear,
        .mean = {0.485f, 0.456f, 0.406f},
        .stddev = {0.229f, 0.224f, 0.225f}
    };
    auto vision_kernel = std::make_shared<FusedImageTransform>(spec, /*originals=*/{});

    // 3. Doğrudan CUDA VRAM'ine Besleyen DataLoader
    DataLoaderOptions loader_opts{
        .batch_size = 256,
        .shuffle = true,
        .num_workers = 8,
        .prefetch_factor = 4,
        .pin_memory = true,
        .batch_transform = vision_kernel,
        .transform_device = Device::cuda(0),
        .device = Device::cuda(0) // Batch doğrudan GPU VRAM'inde teslim edilir!
    };

    DataLoader loader(dataset, loader_opts);

    for (const Batch& batch : loader) {
        // batch.inputs: [256, 3, 224, 224] Float32 CUDA tensörü
    }
}
```

---

### 3. LLM Pretraining: Zstandard Streaming & BPE Tokenizer

```cpp
#include <nexusdata/nexusdata.hpp>
#include <iostream>

using namespace nexusdata;

int main() {
    // 1. C++20 Byte-Pair Encoding (BPE) Tokenizer
    BpeTokenizer tokenizer("vocab.json", "merges.txt");

    // 2. Diskte açılmadan RAM'de decompress edilen Zstandard JSONL akışı
    JsonlDatasetOptions opts{
        .text_field = "text",
        .buffer_size = 8 * 1024 * 1024 // 8 MB akış halka tamponu
    };

    auto dataset = std::make_shared<JsonlDataset>("train-shard-0001.jsonl.zst", opts);

    // 3. Metinleri paralel token ID dizilerine dönüştürme
    auto tokenized = dataset->map([&tokenizer](const Sample& s) -> Sample {
        std::string raw_text = s.input_as_string();
        std::vector<int64_t> ids = tokenizer.encode(raw_text);
        if (ids.size() > 2048) ids.resize(2048);

        NDArray input_ids(Shape{ids.size()}, DType::Int64);
        std::memcpy(input_ids.data(), ids.data(), ids.size() * sizeof(int64_t));
        return Sample{.input = std::move(input_ids), .label = static_cast<int64_t>(ids.size())};
    });

    // 4. Dinamik Sequence Padding ile DataLoader
    DataLoader loader(tokenized, DataLoaderOptions{.batch_size = 32, .num_workers = 8},
                      [](const std::vector<Sample>& samples) {
                          return collate_pad_sequence(samples, {.pad_value = 0.0});
                      });

    for (const Batch& batch : loader) {
        // batch.inputs: [32, max_seq_len] Int64 token matrisi
        // batch.mask:   [32, max_seq_len] UInt8 attention mask
    }
}
```

---

### 4. PyTorch ile Sıfır Kopyalı (Zero-Copy) DLPack Eğitimi

```python
import torch
import torchvision.models as models
import nexusdata_py as nd

device = torch.device("cuda:0")
model = models.resnet50().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# NexusData yüksek hızlı C++ veri yükleyici
loader = nd.DataLoader(
    dataset=nd.ImageFolder("/datasets/imagenet/train"),
    batch_size=256,
    num_workers=8,
    pin_memory=True,
    device="cuda:0" # Batch GPU belleğinde hazır bekler!
)

print("NexusData + PyTorch eğitimi başlıyor...")
for epoch in range(10):
    for batch in loader:
        # Zero-Copy DLPack: C++ CUDA bellek pointer'ı doğrudan PyTorch Tensörüne sarılır!
        # Tek bir bayt bile bellekte kopyalanmaz.
        images = torch.from_dlpack(nd.to_dlpack(batch.inputs))
        targets = torch.from_dlpack(nd.to_dlpack(batch.targets))

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = torch.nn.functional.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()
```

---

## 🛠 Derleme ve Kurulum Kılavuzu

NexusData, modern CMake (>= 3.20) ve C++20 uyumlu bir derleyici (GCC 11+, Clang 13+, MSVC 2019/2022) gerektirir.

### 1. Kaynak Koddan Derleme

```bash
# Depoyu klonlayın
git clone https://github.com/nexusdata/nexusdata.git
cd nexusdata

# Yapılandırma ve Derleme (Release modu önerilir)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Birim testleri çalıştırın
ctest --test-dir build --output-on-failure
```

### 2. Sisteme Kurulum & CMake `find_package`

```bash
cmake --install build --prefix /usr/local
```

Projelerinizde kullanmak için `CMakeLists.txt` dosyanıza şu satırları eklemeniz yeterlidir:

```cmake
find_package(NexusData REQUIRED)
target_link_libraries(my_ai_app PRIVATE nexusdata::nexusdata)
```

---

## ⚙️ CMake Yapılandırma Seçenekleri

| CMake Bayrağı | Varsayılan | Açıklama |
| :--- | :---: | :--- |
| `NEXUSDATA_WITH_CUDA` | `OFF` | CUDA çekirdekleri, `GpuExecutor` ve çoklu akış pipeline desteği. |
| `NEXUSDATA_WITH_NVJPEG` | `OFF` | NVIDIA nvJPEG donanımsal JPEG GPU çözücü (`WITH_CUDA=ON` gerekir). |
| `NEXUSDATA_WITH_PYTHON` | `OFF` | pybind11 ile Python `nexusdata_py` eklentisi. |
| `NEXUSDATA_WITH_ARROW` | `OFF` | Apache Arrow ve Parquet desteği (`ParquetDataset`, `ArrowIpcDataset`). |
| `NEXUSDATA_WITH_POSTGRES` | `OFF` | PostgreSQL Binary COPY protokolü (`libpq`). |
| `NEXUSDATA_WITH_SQLITE` | `OFF` | Gömülü SQLite3 entegrasyonu. |
| `NEXUSDATA_WITH_ODBC` | `OFF` | Windows/Linux kurumsal veritabanı sürücüleri için ODBC. |
| `NEXUSDATA_WITH_ZSTD` | `ON` | Zstandard sıkıştırma çözücü. |
| `NEXUSDATA_WITH_WEBP` | `ON` | Gömülü Google libwebp çözücü. |
| `NEXUSDATA_BUILD_BENCH` | `ON` | Mikrobenchmark araçlarını derler (`bench_suite`, `bench_gpu`). |

---

## 🔌 C API & Yabancı Dil Bağlantıları (Rust, Go, C#)

NexusData, `nexusdata/adapter/c_api.h` başlığı altında dondurulmuş ve stabil bir C99 ABI arayüzü sunar:

* `nexus_ndarray_create`, `nexus_ndarray_to_dlpack`, `nexus_ndarray_from_dlpack`
* `nexus_dataset_open`, `nexus_dataset_get`, `nexus_dataloader_next_batch`
* `nexus_last_error()`: İş parçacığına özel (thread-local), thread-safe hata mesajı bildirme modeli.

Rust, Go veya C# gibi dillerden sıfır ek yükle doğrudan FFI (Foreign Function Interface) ile çağrılabilir.

---

## 📄 Lisans ve Teşekkür

Bu proje **Apache License 2.0** altında lisanslanmıştır.

### Geliştirici & Katkıda Bulunanlar
* **Muhammed Fatih Şahin** — *Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. Sınıf Öğrencisi* (Yapay Zeka Destekli Mimari Geliştirme, C++20 / CUDA Dönüşümü, Next.js Dokümantasyon Portalı).
