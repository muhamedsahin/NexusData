# ART DIRECTION & ARCHITECTURAL MANIFESTO: NEXUSDATA
**Doküman Sürümü:** 1.0.0 (R0 Fazı)  
**Tarih:** 21 Eylül 2026  
**Kapsam:** NexusData Ana Sayfası & 3D WebGL Katmanı Görsel Yeniden Tasarımı  

---

## 1. MEVCUT DURUM GÖRSEL TEŞHİSİ (Playwright SwiftShader Baseline)

`scripts/shots.ts` (Chromium + SwiftShader WebGL2) ile alınan ekran görüntüleri (`shots/round-0/`) piksel piksel incelenmiş ve aşağıdaki somut problemler tespit edilmiştir:

| Kare | Dosya | Gözlemlenen Mevcut Durum | Neden Kötü? | Yapılacak Değişiklik |
| :--- | :--- | :--- | :--- | :--- |
| **Hero** (p=0.00) | `desktop_p000.png` | Sol taraf metin, butonlar ve 15 satırlık dev `CMakeLists.txt` koduyla tıka basa dolu; sağ taraf boş. Ekran genelinde rastgele pembe (`#e879f9`) ve camgöbeği (`#2dd4bf`) düz sprite noktalar dönüyor. Noktalar metnin üzerinden geçiyor. | Tipografik kontrast sıfıra iniyor, alt başlık ve pipeline satırı parçacık gürültüsünden okunamıyor. Odak noktası yok. Dev kod bloğu hero zarafetini yok ediyor. | - H1 ve brand **NexusData** olarak korunacak.<br>- 3D nesne `camera.setViewOffset` ile sağ 7 kolona ortalanacak.<br>- Sol metin arkasına gradyan maske konacak, metin arkasında parçacık olmayacak.<br>- Dev kod bloğu kalkacak, yerine kompakt tek satırlık "Kurulum Hapı" (pill) gelecek.<br>- Noktalar yerine 8×8×8 yuvarlak hücreli, nefes alan tekil bir kafes (lattice) gelecek. |
| **Kaynaklar** (p=0.08) | `desktop_p008.png` | Hero'daki CMake bloğunun alt kenarı yukarıda asılı kalırken, alttan dev bir cam kart giriyor. 3D noktalar rastgele eğriliyor; formatları veya akışı temsil eden hiçbir nesne yok. | İki bölüm birbirine biniyor. 3D katmanının neyi anlattığı anlaşılmıyor; salt gürültü. Kart 3D alanını boğuyor. | - Büyük kart yerine sağ tarafa hizalı kompakt başlık altyazısı konacak.<br>- 6 format (CSV, JPG, JSON, PARQUET, SQLITE, NPY) 3D cam çipler ve hunide birleşen kübik veri akışları olarak modellenecek. |
| **Shuffle** (p=0.24-0.32) | `desktop_p024.png` | Ekranın %40'ını kaplayan soluk cam kartın yanında rastgele kıvrılmış pembe/mavi çift nokta dizileri var. Slider standart HTML çubuğu, thumb soluk. Ortada yeşil bir imleç dairesi asılı kalmış. Sol altta rastgele 'N' harfli bir düğme var. | Hücre veya veri sırası algısı yok. Rastgele kıvrımlar shuffle hissi vermiyor. Slider kullanılamaz derecede silik. | - 64 hücreli 8×8 düzlem.<br>- Her hücre üzerinde digit atlas ile index numarası.<br>- Seed değişiminde Bezier yaylarıyla z-ekseninde sıçrayarak yer değiştirme.<br>- Büyük thumb'lı, doldurulmuş izli, zar butonlu, klavye destekli HUD slider. |
| **Batch/GPU** (p=0.40-0.48) | `desktop_p040.png`, `desktop_p048.png` | Ekranda boşlukta süzülen 5 adet eğik neon çizgi (parçacık şeritleri) dışında hiçbir şey yok. Kartlar ekran sınırlarından dışarı taşıyor ve kesiliyor. | Ne batch anlaşılıyor, ne de GPU. Görsel bir hikâye veya donanım hissi yok. | - Hücreler levhalara (slabs) ayrılacak.<br>- Emissive torus kapısından (DataLoader ring) geçecekler.<br>- GPU sahnesinde prosedürel çip kalıbı ve animasyonlu çift buffer şeridi (GPU vs CPU) yer alacak. |
| **Mobil** (390×844) | `mobile_p000.png`, `mobile_p024.png` | Sayfa bütünüyle sola kaymış ve taşmış durumda! Başlık `exusData`, navbar `sData`, butonlar `Started`, kartlar `ataset, shuffle` şeklinde sola doğru kesilmiş. | Mobilde layout kırılmış, yatay taşma (horizontal overflow) mevcut. Mobil kullanıcı için site kullanılamaz halde. | - Mobilde yatay scroll kesin olarak engellenecek (`overflow-x: clip`, esnek tipografi `clamp`).<br>- Üst %45'te 3D nesne (dokunmatik döndürme), alt %55'te temiz dikey akış. |

---

## 2. GÖRSEL DİL & TOKEN SİSTEMİ

### 2.1 Renk Token Tablosu
Renkler CSS değişkeni ve Three.js `uniform` değerleri olarak birebir senkronize tek kaynaktan yönetilecektir.

| Token Adı | HEX Değeri | Three.js `Color` | CSS Değişkeni | Rolü ve Katı Kuralı |
| :--- | :--- | :--- | :--- | :--- |
| **Background Deep** | `#04060B` | `new THREE.Color(0x04060B)` | `--bg-deep` | Sahne zemin rengi, canvas clearColor. |
| **Background Center** | `#0A1020` | `new THREE.Color(0x0A1020)` | `--bg-center` | Merkez radial vignette gradyanı (hafif derinlik). |
| **Foreground Primary** | `#E8F1FF` | `new THREE.Color(0xE8F1FF)` | `--fg-primary` | H1, birincil metinler, hücre veri tavanı (1.0). |
| **Foreground Muted** | `rgba(232, 241, 255, 0.70)` | `new THREE.Color(0xA3B2CC)` | `--fg-muted` | Alt başlıklar, açıklamalar, gövde metni (%70 opaklık). |
| **Electric Cyan** | `#2DE2D0` | `new THREE.Color(0x2DE2D0)` | `--accent-cyan` | **Birincil vurgu:** Yalnızca emissive parlayan öğeler ve birincil CTA. |
| **Deep Violet** | `#5B4BDB` | `new THREE.Color(0x5B4BDB)` | `--accent-violet` | **İkincil vurgu:** Yalnızca derinlik gradyanlarında, ekranda en fazla %10 alan. |
| **Amber Warning** | `#FFB020` | `new THREE.Color(0xFFB020)` | `--accent-amber` | **Özel vurgu:** SADECE Bölüm 3'teki eksik batch ve `drop_last` sahnesinde. |
| **YASAK RENKLER** | `#e879f9`, pembe, magenta | - | - | **Rastgele pembe/magenta kullanımı tamamen kaldırılmıştır.** |

#### Veri Renk Rampası (Data Color Ramp)
Hücrelerin rengi rastgele atanmaz; `0.0 .. 1.0` aralığındaki simüle edilmiş tensör/tensör-öncesi veri değerinden türer:
1. `val = 0.0` → Koyu Petrol Mavisi (`#0B1B2B`)
2. `val = 0.5` → Elektrik Camgöbeği (`#2DE2D0`)
3. `val = 1.0` → Saf Beyaz Camgöbeği Parıltısı (`#FFFFFF` + cyan rim)

---

### 2.2 Materyal ve Geometri Mimarisi
- **Geometri:** Hücreler için `InstancedMesh` + `RoundedBoxGeometry(size=0.32, radius=0.04, smoothness=3)`. 512 hücre (Hero 8×8×8) tek bir draw call ile çizilir.
- **Shader:** Özel `InstancedMesh` shader'ı:
  - **Fresnel Rim:** Görüş açısına göre kenarlarda parlayan elektrik mavisi/cyan rim (`pow(1.0 - dot(N, V), 3.0)`).
  - **Emissive Core:** Hücrenin merkezinde veri değerine bağlı iç ışık.
  - **İnce Kenar Çizgisi:** Hücre konturlarını netleştiren hassas kenar vurgusu.
  - **Cam Hissi:** Ağır `MeshPhysicalMaterial(transmission=1)` yerine, fresnel + Lightformer ortam yansıması + iç emissive parlama ile ultra-performanslı cam efekti.
- **Zemin / Topraklama:** Hero nesnenin altında `Y = -2.2` seviyesinde merkeze odaklı, kenarlarda yumuşakça sönen grid çizgisi (`grid-fade plane`). Nesne uzay boşluğunda amaçsızca yüzmeyecek, sahneye oturacaktır.
- **Atmosfer Katmanı:**
  - Yıldız/toz: Tam 800 adet nokta. 3 derinlik katmanı, `size < 2px`, `opacity <= 0.3`. Metin bölgelerinde `uTextAvoidance` ile opaklık sıfıra iner.
  - Hero arkası hacimsel ışık billboard'u: Yumuşak additive radial ışık ("god ray" / ambient core).

---

### 2.3 Post-Processing Pipeline (`@react-three/postprocessing`)

| Efekt | Parametreler | Mantık / Amaç |
| :--- | :--- | :--- |
| **Tone Mapping** | `ACESFilmicToneMapping` | Sinematik renk sıkıştırması, parlak alanların çiğ beyaz patlamasını önler. |
| **Selective Bloom** | `luminanceThreshold: 0.85`, `intensity: 0.75`, `mipmapBlur: true` | Yalnızca emissive çekirdekler ve parlayan halkalar ışıldar. Beyaz leke oluşumu engellenir. |
| **Depth of Field** | `focusDistance: 0.03`, `focalLength: 0.05`, `bokehScale: 2.0` | Hover edilen hücreye veya aktif sahne odak noktasına yumuşak geçişle odaklanır. |
| **Vignette** | `darkness: 0.4`, `offset: 0.3` | Odak merkezini vurgulayan karanlık kenarlar. |
| **Dinamik Aberration**| `offset: vec2(velocity * 0.002)` | Yalnızca hızlı scroll anında mikro renk saçılması; durağan halde sıfır. |
| **Kalite Kademeleri** | High: Tümü açık<br>Medium: DOF kapalı, bloom düşük<br>Low: Post tamamen kapalı | Cihaz performansına göre kademeli düşüş. |

---

## 3. KAMERA KEYFRAME TABLOSU (`CatmullRomCurve3`)

Scroll ilerlemesi `p ∈ [0.0, 1.0]` Lenis (lerp: 0.08) ve GSAP ScrollTrigger üzerinden tek bir zustand store'unda akar. Kamera bu eğri üzerinde enterpole edilir.

| Bölüm | Scroll Aralığı `p` | Kamera Pozisyonu `(X, Y, Z)` | Hedef / LookAt `(X, Y, Z)` | FOV | Odak Kaydırma (`viewOffset`) | Sahne Uniform Modülasyonları |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0. Hero: Kafes** | `0.00 .. 0.14` | `(1.2, 0.4, 6.8)` | `(0.8, 0.0, 0.0)` | 38° | Sağa kaydırma: `X: -260, Y: 0` | `uChaos: 0.0`, `uLattice: 1.0`, `uGlow: 1.0` |
| **1. Kaynaklar** | `0.14 .. 0.30` | `(-1.0, 1.2, 7.5)` | `(-0.4, 0.0, 0.0)` | 40° | Sola kaydırma: `X: 200, Y: 0` | `uChaos: 0.8`, `uFunnel: 1.0`, `uFlow: 1.0` |
| **2. Shuffle & Seed** | `0.30 .. 0.46` | `(1.4, 0.2, 6.2)` | `(1.0, 0.0, 0.0)` | 36° | Sağa kaydırma: `X: -280, Y: 0` | `uSorted: 0.0..1.0`, `uPlane: 1.0` |
| **3. Batch/DataLoader**| `0.46 .. 0.62` | `(-1.2, 0.6, 6.5)` | `(-0.6, 0.0, 0.0)` | 38° | Sola kaydırma: `X: 240, Y: 0` | `uBatchSlabs: 1.0`, `uGateActive: 1.0` |
| **4. Prefetch & GPU** | `0.62 .. 0.76` | `(1.1, 0.8, 5.8)` | `(0.7, 0.1, 0.0)` | 35° | Sağa kaydırma: `X: -240, Y: 0` | `uChipGlow: 1.0`, `uBufferSync: 1.0` |
| **5. Tensör & Eko** | `0.76 .. 0.88` | `(0.0, 1.5, 9.2)` | `(0.0, 0.2, 0.0)` | 44° | Merkez: `X: 0, Y: 0` | `uConstellation: 1.0`, `uMatrixDataGlow: 1.5` |
| **Sakinler & CTA** | `0.88 .. 1.00` | `(0.0, 0.0, 10.5)` | `(0.0, 0.0, 0.0)` | 42° | Merkez: `X: 0, Y: 0` | `uLattice: 1.0`, `uBackgroundDim: 0.35` |

---

## 4. BÖLÜMLERİN STORYBOARD'U (ASCII KADRAJ TASLAKLARI)

### Bölüm 0 — Hero: "NDArray Kafesi"
```
+-----------------------------------------------------------------------------------+
| [NexusData [dev]]                   [Docs]  [Perf]  [GitHub]  [EN|TR] [Theme]     |
|                                                                                   |
|  [Status] [Preview]                           . . * .                             |
|  # NexusData                            .  +---+---+---+                          |
|                                          / |   |   |   | \                        |
|  The pipeline that turns raw data       +---+---+---+---+ |                       |
|  into tensors                           |   | 8x8x8 |   | +   <-- 3D Odak:        |
|                                         +---+---+---+---+/        NDArray Kafesi  |
|  [Source->Dataset->Sampler->...]        |   |Lattice|   |         (RoundedBox)    |
|                                         +---+---+---+---+         Nefes dalgası   |
|  [ Get Started ]  [ Documentation ]            \     /            Fare etkileşimi |
|                                                 +---+                             |
|  [>_ FetchContent: NexusData v1.0 [Copy]]     ----------------- (Faded Grid Plane)|
|                                                                                   |
|  v SCROLL TO EXPLORE                                                              |
+-----------------------------------------------------------------------------------+
```
- **Sol Taraf (Kolon 1–5):** `NexusData` başlığı, animasyonlu SVG pipeline darbesi, CTA butonları, tek satırlık kompakt Kurulum Hapı (tıklayınca CMake/FetchContent açılır).
- **Sağ Taraf (Kolon 6–12):** 8×8×8 yuvarlak köşeli instanced hücre kafesi. Fare yaklaşımında hücreler yükselip yaylanarak dalga üretir. Tıklamada shockwave halkası geçer.
- **Etkileşim:** Hücre hover'ında mini bilimsel HUD tooltip: `[3, 5, 1] = 0.842 · float32`.

---

### Bölüm 1 — Kaynaklar: "Format Akışları ve Huni"
```
+-----------------------------------------------------------------------------------+
|                      [CSV] ----,                                                  |
|                      [JSON] ----\   . . . .                                       |
|  01 / 06             [IMG]  -----( Huni / Funnel )                                |
|  Read every format   [PARQ] ----/   (Daralan halkalar)                            |
|  Sources feed into   [SQL]  ---'          |                                       |
|  unified streams...  [NPY]  --'           v                                       |
|                                     (Tek akış hattı)                              |
|  [Glass HUD: 6 Format Rozeti]                                                     |
+-----------------------------------------------------------------------------------+
```
- **Sol Taraf:** Kompakt bölüm altyazısı (≤ 420px, HUD tasarımı). 6 format durum rozeti.
- **Sağ Taraf:** 6 adet cam "çip"ten (CSV, JSON, IMG, PARQUET, SQLITE, NPY) çıkan küçük küpler spline eğrisi boyunca akar ve parlayan daralan halkalardan (huni) geçerek tekil bir veri akışında toplanır. Hover edilen format parlar.

---

### Bölüm 2 — Dataset, Shuffle & Seed: "8×8 Düzlem & Bezier Sıçramaları"
```
+-----------------------------------------------------------------------------------+
|  02 / 06                                  +----+----+----+----+----+----+----+----+
|  Dataset & Seed                           | 00 | 01 | 02 | 03 | 04 | 05 | 06 | 07 |
|  Deterministic index                      +----+----+----+----+----+----+----+----+
|  shuffling. Same seed,                    | 08 | 09 | 10 | 11 | 12 | 13 | 14 | 15 |
|  exact same order.                        +----+----+----+----+----+----+----+----+
|                                           | .. Bezier yay sıçramaları (z-up)  ..  |
|  [ Glass HUD:                             +----+----+----+----+----+----+----+----+
|    Seed: [ 42 ] [Zar]                     |    |    |    |    |    |    |    | 63 |
|    [===o================] 0..999          +----+----+----+----+----+----+----+----+
|    Sıra: 14, 02, 55, 31, ...              | (8x8 Hücre Düzlemi, Digit Atlas UV)   |
|    Durum: Aynı sıra dogrulandi v ]        +---------------------------------------+
+-----------------------------------------------------------------------------------+
```
- **Sağ Taraf:** Kameraya tam bakan 8×8 (64 hücre) düzlem. Her hücrenin yüzeyinde digit atlas dokusu ile index numarası (`00` - `63`) net olarak okunur.
- **Etkileşim:** HUD içindeki yeni slider (belirgin thumb, doldurulmuş iz, sayı kutusu, zar butonu) veya ok tuşlarıyla seed değiştiğinde, hücreler Bezier yaylarıyla z-ekseninde kalkarak yeni konumlarına uçar. Aynı seed girildiğinde birebir aynı pozisyonlara geri dönerler.

---

### Bölüm 3 — Batch & DataLoader: "Levhalar ve Torus Kapısı"
```
+-----------------------------------------------------------------------------------+
|                                            (  O  ) <-- Parlayan Torus Kapısı      |
|  03 / 06                                  /     \     (DataLoader Ring)           |
|  Batch & DataLoader                     [Levha 1: B=8]   --> [Geçti: Cyan]        |
|  Slicing samples into                   [Levha 2: B=8]   --> [Geçti: Cyan]        |
|  aligned batches.                       [Levha 3: B=8]   --> [Geçti: Cyan]        |
|                                         [Levha 4: B=4]   --> [Kehribar / Drop]    |
|  [ Glass HUD:                                                 (Dissolve parçalanma|
|    batch_size: [-] 8 [+]                                       veya küçük geçiş)  |
|    [x] drop_last (aktif)                                                          |
|    Cıktı: [8, features] x 3 ]                                                     |
+-----------------------------------------------------------------------------------+
```
- **Sağ Taraf:** Hücreler batch boyutuna göre yatay levhalar (slabs) halinde gruplanır. Parlayan bir torus kapısından (DataLoader ring) tek tek akarlar.
- **Drop-last Etkisi:** Son eksik batch kehribar (`#FFB020`) renklidir. `drop_last` anahtarı açıkken kapıya varmadan parlayan kenarlı gürültü (noise-threshold dissolve) ile çözülür; kapalıyken küçük batch olarak geçer.

---

### Bölüm 4 — Prefetch & GPU: "Çip Kalıbı ve Çift Buffer"
```
+-----------------------------------------------------------------------------------+
|  04 / 06                                  +------------------------------------+  |
|  Prefetch & GPU                           | [Prosedürel Çip Kalıbı]            |  |
|  Dual-buffer pipelining.                  | Nabız atan çekirdek + kesikli izler|  |
|  GPU only where it wins.                  +------------------------------------+  |
|                                           | GPU: [ Batch N   ] =====> Tüketim  |  |
|  [ Mini HUD:                              | CPU: [ Batch N+1 ] =====> Hazırlık |  |
|    GPU Durumu: Yalnızca Prep Kernelleri   +------------------------------------+  |
|    Politika: CPU yolu daima hazır ]       (Senkronize kayan zaman şeritleri)      |
+-----------------------------------------------------------------------------------+
```
- **Sağ Taraf:** Prosedürel 3D çip kalıbı (katmanlı kutu, animated dash izler, nabız atan çekirdek) ve altında çift buffer mekanizmasını temsil eden iki eşzamanlı kayan zaman şeridi.

---

### Bölüm 5 — Tensör & Ekosistem: "Takımyıldız"
```
+-----------------------------------------------------------------------------------+
|                                      (NexusLoss)                                  |
|                                       [planned]                                   |
|                        (NexusModel)       |       (NexusOptim)                    |
|                         [planned]         |        [planned]                      |
|                                 \         |        /                              |
|   05 / 06                        \        |       /                               |
|   Ecosystem                       *(NexusData)*   <-- En parlak merkez düğüm      |
|   Tensörler NexusFlash Pro'ya     /   [Active]    \                               |
|   ve ekosisteme akar.            /        |        \                              |
|                        (NexusTrain)       |     (NexusFlash Pro)                  |
|                         [planned]         |        [planned]                      |
|                                      (AI Engine)                                  |
+-----------------------------------------------------------------------------------+
```
- Kamera geri çekilir: 7 düğümlü 3D takımyıldız. Merkezde en parlak elektrik mavisi ile `NexusData` durur. Diğer 6 düğüm (`NexusFlash Pro`, `NexusLoss`, `NexusModel`, `NexusOptim`, `NexusTrain`, `AI Engine`) daha soluk, cam küreler olarak ışık darbeleriyle bağlıdır. Hover edilen düğüm parlar ve rol açıklaması belirir.

---

### Sakin Bölümler & Kapanış CTA (Normal Scroll, Canvas Sakin Arka Planda)
- **Özellikler Grid'i:** 8 kart, fare yönüne bağlı 3D tilt + spotlight gradyan sınırları, dürüst durum rozetleri.
- **Performans Bölümü:** Yalnızca gerçek `benchmarks.json` verisi; yoksa dürüstçe "Ölçülmedi" (not-measured) uyarısı.
- **Kod Sekmeleri:** `t.raw` ile düzeltilmiş, daktilo efektli, satır vurgulu C++20 örnekleri.
- **Kapanış CTA:** 3D kafes merkeze toplanır, "Hemen Başla" butonu, tek satır kurulum hapı, GitHub linki ve zarif footer.

---

## 5. KALDIRILACAKLAR VE KORUNACAKLAR LİSTESİ

### 5.1 Kesinlikle Kaldırılacaklar (Drop List)
1. **Düz Points Sprite Bulutu:** Rastgele renkli, derinliksiz 1728 noktalık `Points` materyali tamamen kaldırılacak.
2. **Pembe / Magenta Paleti:** `#e879f9` ve kontrolsüz pembe tonları tamamen çıkarılacak.
3. **Metin Üstü Parçacık Geçişi:** Metin arkasında parçacık ve parıltı bulunması yasaklanacak.
4. **Hero 15 Satırlık Dev Kod Bloğu:** Hero'yu boğan dev `CMakeLists.txt` kutusu kaldırılacak; yerine tek satırlık kompakt "Kurulum Hapı" (pill) getirilecek.
5. **3D'yi Kapatan Dev Cam Kartlar:** Ekranın %40'ını kaplayan hantal kutular kaldırılacak; karşı tarafa yerleşen kompakt HUD elemanları kullanılacak.
6. **Bozuk Range Slider:** Thumb'ı görünmeyen, erişilebilirliği olmayan HTML slider kaldırılacak.
7. **Köşede Takılan Custom Cursor:** Fare ayrıldığında kaybolmayan ve takılan imleç bug'ı düzeltilecek.
8. **Eski İsim Kalıntıları:** Kodda ve arayüzde kalan tüm eski isimler **NexusData** olarak düzeltilecek.
9. **Mobildeki Yatay Taşıntı:** Mobilde sayfanın sola kayıp taşmasına neden olan tüm CSS genişlik hataları temizlenecek.

### 5.2 Kesinlikle Korunacaklar (Preserve List)
1. **i18n ve Dil Mimarisi:** `next-intl`, `messages/en.json`, `messages/tr.json`, `navigation.ts`, `routing.ts` yapısı bozulmayacak. Tüm metinler i18n kaynaklı kalacak.
2. **Dokümantasyon Motoru:** `/docs` rotası, MDX işleme, Orama arama motoru, TOC ve doküman sayfaları. (Docs sayfalarında Three.js asla yüklenmeyecektir).
3. **İçerik Dürüstlüğü:** Sahte benchmark eklenmeyecek. "Illustrative/örnek veri" ve `not-measured` rozetleri zarif bir dille korunacak.
4. **`site.config.ts` Yapısı:** Konfigürasyon nesnesi ve placeholder mantığı korunacak.
5. **Mevcut Testler:** Unit testler (`vitest`) ve e2e testler (`playwright`) korunacak ve yeni yapıya uyumlu çalışacaktır.

---

## 6. TASARIM TERCİHLERİ VE SEÇİM GEREKÇELERİ (RATIONALE)

1. **InstancedMesh + RoundedBoxGeometry vs. Points:**  
   *Seçim:* Instanced RoundedBox.  
   *Gerekçe:* Points sprite'ları hiçbir zaman fiziksel bir hacim, kenar, köşe veya hücre hissi veremez. InstancedMesh modern GPU'larda 512–4096 nesne için tek draw-call ile 60+ FPS çalışır ve her hücreye gerçek bir "veri hücresi" kimliği kazandırır.
2. **Fresnel + Environment vs. Transmission (Glass) Material:**  
   *Seçim:* Custom Fresnel Rim + Ambient Reflection.  
   *Gerekçe:* Three.js `MeshPhysicalMaterial(transmission > 0)` her karede sahneyi arka buffer'a kopyalar. Instanced 512 hücre üzerinde transmission kullanmak performansı çökertecektir. Fresnel rim + selective bloom aynı cam/kristal hissini sıfır ek render maliyetiyle verir.
3. **CatmullRomCurve3 Tekil Kamera Yolu vs. Dağınık Rig:**  
   *Seçim:* CatmullRomCurve3 6 keyframe eğrisi.  
   *Gerekçe:* Her bölümün kamerayı rastgele itip çekmesi sahneler arası sarsıntı yaratır. Tek bir 3D spline eğrisi ve `camera.setViewOffset`, kesintisiz tek dünya (continuous world) hissini ve metin-nesne kadraj dengesini matematiksel olarak garanti eder.

---

## 7. FAZ PLANI VE ONAY KRİTERLERİ

- [x] **R0 — Art Direction & Teşhis:** Mevcut ekran görüntüleri incelendi, `ART_DIRECTION.md` tamamlandı. (Şu anki aşama — Onay bekleniyor).
- [ ] **R1 — Hero:** Yükleme→Hero geçişi, NexusData NDArray kafesi, fare/touch etkileşimi, tipografi, kompakt kurulum hapı, animated SVG pipeline, nav rayı. (3 tur görsel kontrol).
- [ ] **R2 — Bölüm 1–2:** Kaynaklar (Format çipleri & huni) + Shuffle&Seed (İndeksli 8×8 düzlem, Bezier sıçramaları, yeni HUD slider).
- [ ] **R3 — Bölüm 3–4:** Batch&DataLoader (Levhalar, torus kapı, drop_last kehribar dissolve) + Prefetch&GPU (Çip kalıbı, çift buffer şeritleri).
- [ ] **R4 — Bölüm 5 + Sakin Bölümler + CTA:** Takımyıldız, 3D tilt yetenek kartları, performans önizlemesi, kod sekmeleri, toparlanan kafes CTA.
- [ ] **R5 — Cila & Rapor:** Mobil uyum, kalite kademeleri, WebGL fallback, erişilebilirlik, bundle raporu.

**R0 Teslim Edildi. Devam etmek için onayınızı bekliyorum.**

