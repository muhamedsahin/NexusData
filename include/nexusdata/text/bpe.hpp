#pragma once

/**
 * @file bpe.hpp
 * @brief Bayt duzeyi BPE (GPT-2 / tiktoken uyumlu): on-bolme, kodlama, cozme, egitim.
 *
 * On-bolme (regex'in elle yazilmis durum makinesi; leftmost-first, acgozlu, ayni sonuc):
 *  - Gpt2:   tiktoken gpt2: '(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s
 *  - Cl100k: tiktoken cl100k_base: '(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+
 *            | ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s
 *  - Llama3: Meta LLaMA-3 deseni = Cl100k, \s++$ kurali olmadan.
 * o200k (buyuk/kucuk harf siniflari, \p{M}) desteklenmez.
 * \p{L}/\p{N}: unicode_tables.inc (Unicode 16); \s: Unicode White_Space (Rust regex ile ayni).
 *
 * Kodlama tiktoken `byte_pair_merge` ile ayni: parca tumuyle sozlukteyse tek token; degilse
 * bitisik cift rank'lari arasinda en kucugu (esitlikte en soldaki) tekrar tekrar birlesir.
 *
 * Egitim HF tokenizers `BpeTrainer` ile ayni kurallar: ozel tokenlar once, sonra 256 bayt
 * (GPT-2 bayt->unicode eslemesinin kod noktasi sirasinda), sonra birlesimler; en yuksek sayim,
 * esitlikte en kucuk (id1, id2); oncelik kuyrugu girdileri konum kumesi tasir ve tembel guncellenir.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace nexusdata::text {

enum class Pretokenizer { Gpt2, Cl100k, Llama3 };

[[nodiscard]] const char* to_string(Pretokenizer p) noexcept;
[[nodiscard]] Pretokenizer pretokenizer_from_string(const std::string& name);
/// Unicode tablolarinin surumu (on-bolme \p{L}/\p{N}).
[[nodiscard]] const char* unicode_version() noexcept;

/// Metni on-bolme parcalarina ayirir; cikti [baslangic, bitis) bayt ofsetleri.
void pretokenize(std::string_view text, Pretokenizer p, std::vector<std::pair<std::uint32_t, std::uint32_t>>& out);

/// GPT-2 bytes_to_unicode: bayt dizisinin HF/GPT-2 "bayt-unicode" UTF-8 gosterimi.
[[nodiscard]] std::string bytes_to_unicode(std::string_view bytes);

class ByteBpe {
 public:
  ByteBpe();
  ~ByteBpe();
  ByteBpe(ByteBpe&&) noexcept;
  ByteBpe& operator=(ByteBpe&&) noexcept;
  ByteBpe(const ByteBpe&) = delete;
  ByteBpe& operator=(const ByteBpe&) = delete;

  /// tiktoken bicimi: her satir "<base64(bayt)> <rank>". Ozel tokenlar ayrica verilir.
  [[nodiscard]] static ByteBpe load_tiktoken(const std::string& path, Pretokenizer p,
                                             std::vector<std::pair<std::string, std::uint32_t>> specials = {});
  /// token_bytes[r] = rank r'nin baytlari (bos olanlar atlanir: rank delikleri).
  [[nodiscard]] static ByteBpe from_ranks(std::vector<std::string> token_bytes, Pretokenizer p,
                                          std::vector<std::pair<std::string, std::uint32_t>> specials = {});
  void save_tiktoken(const std::string& path) const;

  /// Ozel token metni sira metin olarak kodlanir (tiktoken encode_ordinary).
  [[nodiscard]] std::vector<std::uint32_t> encode(std::string_view text) const;
  void encode_append(std::string_view text, std::vector<std::uint32_t>& out) const;
  /// Belgeler bagimsiz; is parcacigi basina ayri tamponlar (paylasilan durum yok).
  [[nodiscard]] std::vector<std::vector<std::uint32_t>> encode_batch(const std::vector<std::string_view>& docs,
                                                                     int threads) const;
  [[nodiscard]] std::string decode(const std::vector<std::uint32_t>& ids) const;
  /// id'nin baytlari (ozel tokenlar dahil); bilinmeyen id hata.
  [[nodiscard]] const std::string& token_bytes(std::uint32_t id) const;

  [[nodiscard]] std::uint32_t vocab_size() const noexcept;  // en buyuk id + 1
  [[nodiscard]] std::optional<std::uint32_t> special(const std::string& name) const;
  [[nodiscard]] Pretokenizer pretokenizer() const noexcept;
  /// SHA-256(on-bolme adi, rank'lar, ozel tokenlar): parca basliginda tokenizer kimligi.
  [[nodiscard]] std::array<std::uint8_t, 32> digest() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// Akisli cozme: yalniz tamamlanmis UTF-8 dizilerini dondurur; yarim kalan baytlar bir sonraki
/// token'a kadar tamponda bekler (FAZ 16 uretim akisi icin).
class StreamDecoder {
 public:
  explicit StreamDecoder(const ByteBpe& bpe) : bpe_(&bpe) {}
  [[nodiscard]] std::string push(std::uint32_t id);
  /// Kalan (gecersiz/yarim) baytlari oldugu gibi dondurur.
  [[nodiscard]] std::string flush();

 private:
  const ByteBpe* bpe_;
  std::string pending_;
};

struct BpeTrainOptions {
  std::uint32_t vocab_size = 32000;
  std::uint64_t min_frequency = 0;
  std::vector<std::string> special_tokens;
  Pretokenizer pretokenizer = Pretokenizer::Gpt2;
  int threads = 0;  // 0: donanim
};

struct BpeTrainResult {
  ByteBpe tokenizer;  // rank = HF id (ozel tokenlar ranks disinda, specials olarak)
  std::vector<std::pair<std::string, std::string>> merges;  // HF bayt-unicode gosteriminde
  std::vector<std::string> vocab;                           // id -> HF bayt-unicode gosterimi
  std::uint64_t words = 0;                                  // farkli on-bolme parcasi
};

[[nodiscard]] BpeTrainResult train_bpe(const std::vector<std::string_view>& texts, const BpeTrainOptions& options);

// ---- SHA-256 (FIPS 180-4) -------------------------------------------------------------------
class Sha256 {
 public:
  Sha256();
  void update(const void* data, std::size_t size);
  [[nodiscard]] std::array<std::uint8_t, 32> finish();

 private:
  void block(const std::uint8_t* p);
  std::uint32_t h_[8];
  std::uint8_t buf_[64];
  std::size_t len_ = 0;
  std::uint64_t total_ = 0;
};

[[nodiscard]] std::string to_hex(const std::array<std::uint8_t, 32>& digest);
[[nodiscard]] std::array<std::uint8_t, 32> sha256_file(const std::string& path);

}  // namespace nexusdata::text
