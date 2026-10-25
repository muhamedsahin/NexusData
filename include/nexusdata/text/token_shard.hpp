#pragma once

/**
 * @file token_shard.hpp
 * @brief FAZ 8.C-8.E: NXTK parca bicimi, manifest, bellek eslemli okuyucu, Feistel karistirmali
 *        devam edebilen ornekleyici, paketleme (cu_seqlens + kayip maskesi), karisim, rank dilimi.
 *
 * NXTK dosyasi (kucuk-endian):
 *   [0..64)  baslik: "NXTK", surum u32 (1), dtype baytlari u32 (2: u16, 4: u32), ayrilmis u32,
 *            token sayisi u64, belge sayisi u64, tokenizer SHA-256 (32 bayt)
 *   [64..)   token dizisi (dtype), 8'e hizali dolgu, belge ofsetleri u64 [belge sayisi + 1]
 * Belgeler EOS ile birlesik yazilir (yazici her belgenin sonuna eos ekler, eos verilmisse).
 *
 * Manifest (JSON): {"version":1, "tokenizer":{"name","digest","vocab_size","eos"},
 *   "shards":[{"path","tokens","docs","sha256"}], "total_tokens", "total_docs"}
 *
 * Ornekleyici: parca icinde pencere w, [w*T, w*T + T + 1) tokenlarini kapsar (girdi T + hedef).
 * Iki duzeyli karistirma: epoch e icin parca permutasyonu P_e ve her parcanin pencere
 * permutasyonu Q_{e,s}; ikisi de Feistel (4 tur, dongu yuruyusu) ile O(1) bellek. Durum
 * {epoch, parca imleci, pencere imleci, global ornek, tohum}: kaydedilip yuklendiginde sonraki
 * ornek kesintisiz akistakiyle birebir aynidir. Rank r / R: global ornek g % R == r olanlari alir.
 *
 * Paketleme: penceredeki belge baslangiclari d (w*T < d < w*T + T) yeni bolum acar ->
 * cu_seqlens = [0, d1 - w*T, ..., T]; hedef pozisyonu p'nin hedefi (w*T + p + 1) bir belge
 * baslangiciysa kayip maskesi 0 (belge arasi tahmin yok).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexusdata/io/mapped_file.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace nexusdata::text {

struct ShardHeader {
  char magic[4] = {'N', 'X', 'T', 'K'};
  std::uint32_t version = 1;
  std::uint32_t dtype_bytes = 2;
  std::uint32_t reserved = 0;
  std::uint64_t num_tokens = 0;
  std::uint64_t num_docs = 0;
  std::array<std::uint8_t, 32> tokenizer_digest{};
};
static_assert(sizeof(ShardHeader) == 64, "NXTK basligi 64 bayt");

/// Akisli yazici: tokenlar dogrudan diske, ofsetler bellekte (belge basina 8 bayt).
class ShardWriter {
 public:
  ShardWriter(const std::string& path, std::uint32_t vocab_size, const std::array<std::uint8_t, 32>& digest,
              std::int64_t eos = -1);
  ~ShardWriter();
  ShardWriter(const ShardWriter&) = delete;
  ShardWriter& operator=(const ShardWriter&) = delete;
  void add_document(const std::uint32_t* tokens, std::size_t count);
  /// Basligi ve ofsetleri yazar, dosyayi kapatir; SHA-256 (tum dosya) dondurur.
  std::array<std::uint8_t, 32> finish();
  [[nodiscard]] std::uint64_t tokens() const noexcept { return header_.num_tokens; }
  [[nodiscard]] std::uint64_t docs() const noexcept { return header_.num_docs; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  ShardHeader header_;
  std::int64_t eos_;
};

class ShardReader {
 public:
  explicit ShardReader(const std::string& path);
  [[nodiscard]] const ShardHeader& header() const noexcept { return header_; }
  [[nodiscard]] std::uint64_t num_tokens() const noexcept { return header_.num_tokens; }
  [[nodiscard]] std::uint64_t num_docs() const noexcept { return header_.num_docs; }
  [[nodiscard]] std::uint32_t token(std::uint64_t i) const noexcept;
  /// [begin, begin + count) tokenlarini int32 olarak kopyalar.
  void copy(std::uint64_t begin, std::size_t count, std::int32_t* out) const;
  /// Belge d'nin baslangic ofseti (d == num_docs: toplam token).
  [[nodiscard]] std::uint64_t doc_offset(std::uint64_t d) const noexcept;
  /// pos'u iceren belgenin indeksi (ikili arama).
  [[nodiscard]] std::uint64_t doc_of(std::uint64_t pos) const noexcept;

 private:
  MappedFile file_;
  ShardHeader header_;
  const std::uint8_t* tokens_ = nullptr;
  const std::uint64_t* offsets_ = nullptr;
};

struct ManifestShard {
  std::string path;  // manifest dizinine gore
  std::uint64_t tokens = 0;
  std::uint64_t docs = 0;
  std::string sha256;
};

struct Manifest {
  std::string tokenizer_name;
  std::string tokenizer_digest;
  std::uint32_t vocab_size = 0;
  std::int64_t eos = -1;
  std::vector<ManifestShard> shards;
  std::string dir;  // yuklendigi dizin (kaydedilmez)

  [[nodiscard]] std::uint64_t total_tokens() const noexcept;
  void save(const std::string& path) const;
  [[nodiscard]] static Manifest load(const std::string& path);
  /// Her parcanin SHA-256'sini yeniden hesaplar; uyusmayan ilk parca adini dondurur (bos: tamam).
  [[nodiscard]] std::string verify() const;
};

/// [0, n) uzerinde tohumlu sozde rastgele permutasyon (Feistel + dongu yuruyusu), O(1) bellek.
class FeistelPermutation {
 public:
  FeistelPermutation(std::uint64_t n, std::uint64_t seed);
  [[nodiscard]] std::uint64_t operator()(std::uint64_t i) const noexcept;
  [[nodiscard]] std::uint64_t size() const noexcept { return n_; }

 private:
  std::uint64_t n_;
  int half_bits_;
  std::uint64_t mask_;
  std::uint64_t keys_[4];
};

struct PackedSample {
  std::vector<std::int32_t> tokens;      // T + 1 (girdi = ilk T, hedef = son T)
  std::vector<std::int32_t> cu_seqlens;  // [0, ..., T] bolum sinirlari
  std::vector<std::uint8_t> loss_mask;   // T (1: hedef sayilir)
  std::uint32_t source = 0;              // karisimda kaynak indeksi
  std::uint64_t shard = 0;
  std::uint64_t window = 0;
};

struct SamplerState {
  std::uint64_t epoch = 0;
  std::uint64_t shard_cursor = 0;
  std::uint64_t window_cursor = 0;
  std::uint64_t global_index = 0;  // tum rank'lar uzerinden uretilen ornek sayisi
  std::uint64_t seed = 0;
  [[nodiscard]] std::string serialize() const;
  [[nodiscard]] static SamplerState deserialize(const std::string& s);
  bool operator==(const SamplerState&) const = default;
};

/// Manifest'in parcalarini esler, pencereleri karistirarak uretir.
class TokenStream {
 public:
  TokenStream(const Manifest& manifest, int window, std::uint64_t seed, int rank = 0, int world = 1);
  /// Bir sonraki ornegi doldurur (sonsuz: epoch biterse yeni karistirmayla devam).
  void next(PackedSample& out);
  [[nodiscard]] SamplerState state() const noexcept { return state_; }
  void restore(const SamplerState& s);
  [[nodiscard]] std::uint64_t windows_per_epoch() const noexcept { return total_windows_; }
  [[nodiscard]] int window() const noexcept { return T_; }

 private:
  void advance();  // bir global ornek ilerle (rank'tan bagimsiz)
  [[nodiscard]] std::uint64_t windows_in(std::uint64_t shard) const noexcept;
  void fill(std::uint64_t shard, std::uint64_t window, PackedSample& out) const;
  [[nodiscard]] const ShardReader& reader(std::uint64_t shard) const;

  static constexpr std::size_t kMaxOpen = 2;
  std::vector<std::string> paths_;
  std::vector<std::uint64_t> tokens_;
  mutable std::vector<std::unique_ptr<ShardReader>> shards_;  // tembel: yalniz acik parcalar
  mutable std::vector<std::uint64_t> open_;                   // acilis sirasi (en eski once)
  int T_;
  int rank_;
  int world_;
  std::uint64_t total_windows_ = 0;
  SamplerState state_;
};

/// Agirlikli kaynak karisimi: global ornek g icin kaynak = hash(seed, g) ile agirlikli secim
/// (belirlenimci; devamda ayni). Kaynak basina ayri imlec.
class MixtureStream {
 public:
  MixtureStream(std::vector<std::unique_ptr<TokenStream>> sources, std::vector<double> weights, std::uint64_t seed);
  void next(PackedSample& out);
  [[nodiscard]] std::string serialize_state() const;
  void restore_state(const std::string& s);

 private:
  std::vector<std::unique_ptr<TokenStream>> sources_;
  std::vector<double> cumulative_;
  std::uint64_t seed_;
  std::uint64_t index_ = 0;
};

/// En iyi uyum azalan (best-fit decreasing) kutu paketleme: belgeler kesilmeden T'lik kutulara.
/// Donus: kutu basina belge indeksleri. T'den uzun belge tek basina (kesilmeden) kutu olur.
[[nodiscard]] std::vector<std::vector<std::uint64_t>> best_fit_pack(const std::vector<std::uint64_t>& lengths,
                                                                    std::uint64_t capacity);

}  // namespace nexusdata::text
