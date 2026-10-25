/**
 * @file bpe.cpp
 * @brief Bayt duzeyi BPE: on-bolme durum makineleri, tiktoken uyumlu kodlama, HF uyumlu egitim,
 *        SHA-256.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexusdata/text/bpe.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>

namespace nexusdata::text {

namespace {
#include "nexusdata/text/unicode_tables.inc"

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error("nexusdata BPE: " + what); }

// ---- Unicode siniflari --------------------------------------------------------------------
template <std::size_t N>
bool in_ranges(const std::uint32_t (&r)[N][2], std::uint32_t cp) {
  std::size_t lo = 0;
  std::size_t hi = N;
  while (lo < hi) {
    const std::size_t mid = (lo + hi) / 2;
    if (cp < r[mid][0]) {
      hi = mid;
    } else if (cp > r[mid][1]) {
      lo = mid + 1;
    } else {
      return true;
    }
  }
  return false;
}

enum Cls : std::uint8_t { kOther = 0, kLetter = 1, kNumber = 2, kSpace = 3 };

struct AsciiTable {
  std::uint8_t c[128];
  constexpr AsciiTable() : c{} {
    for (int i = 0; i < 128; ++i) {
      c[i] = kOther;
    }
    for (int i = 'a'; i <= 'z'; ++i) {
      c[i] = kLetter;
    }
    for (int i = 'A'; i <= 'Z'; ++i) {
      c[i] = kLetter;
    }
    for (int i = '0'; i <= '9'; ++i) {
      c[i] = kNumber;
    }
    for (const int i : {9, 10, 11, 12, 13, 32}) {
      c[i] = kSpace;
    }
  }
};
constexpr AsciiTable kAscii{};

// Unicode White_Space (Rust regex \s).
bool is_space_cp(std::uint32_t cp) {
  if (cp < 128) {
    return kAscii.c[cp] == kSpace;
  }
  return cp == 0x85 || cp == 0xA0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 ||
         cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

Cls classify(std::uint32_t cp) {
  if (cp < 128) {
    return static_cast<Cls>(kAscii.c[cp]);
  }
  if (is_space_cp(cp)) {
    return kSpace;
  }
  if (in_ranges(kLetterRanges, cp)) {
    return kLetter;
  }
  if (in_ranges(kNumberRanges, cp)) {
    return kNumber;
  }
  return kOther;
}

// UTF-8 kod noktasi; gecersiz dizi 1 baytlik "diger" sayilir (0xFFFFFFFF).
inline std::uint32_t decode_cp(std::string_view s, std::size_t i, std::size_t* len) {
  const auto b0 = static_cast<unsigned char>(s[i]);
  if (b0 < 0x80) {
    *len = 1;
    return b0;
  }
  auto cont = [&](std::size_t k) {
    return i + k < s.size() && (static_cast<unsigned char>(s[i + k]) & 0xC0) == 0x80;
  };
  if ((b0 & 0xE0) == 0xC0 && cont(1)) {
    *len = 2;
    return ((b0 & 0x1Fu) << 6) | (static_cast<unsigned char>(s[i + 1]) & 0x3Fu);
  }
  if ((b0 & 0xF0) == 0xE0 && cont(1) && cont(2)) {
    *len = 3;
    return ((b0 & 0x0Fu) << 12) | ((static_cast<unsigned char>(s[i + 1]) & 0x3Fu) << 6) |
           (static_cast<unsigned char>(s[i + 2]) & 0x3Fu);
  }
  if ((b0 & 0xF8) == 0xF0 && cont(1) && cont(2) && cont(3)) {
    *len = 4;
    return ((b0 & 0x07u) << 18) | ((static_cast<unsigned char>(s[i + 1]) & 0x3Fu) << 12) |
           ((static_cast<unsigned char>(s[i + 2]) & 0x3Fu) << 6) | (static_cast<unsigned char>(s[i + 3]) & 0x3Fu);
  }
  *len = 1;
  return 0xFFFFFFFFu;
}

struct Cursor {
  std::string_view s;
  // i konumundaki karakterin sinifi ve uzunlugu
  Cls at(std::size_t i, std::size_t* len, std::uint32_t* cp = nullptr) const {
    const std::uint32_t c = decode_cp(s, i, len);
    if (cp != nullptr) {
      *cp = c;
    }
    return c == 0xFFFFFFFFu ? kOther : classify(c);
  }
  // i'den baslayan, sinifi `want` olan karakterlerin sonu
  std::size_t run(std::size_t i, Cls want) const {
    std::size_t len = 0;
    while (i < s.size()) {
      if (at(i, &len) != want) {
        break;
      }
      i += len;
    }
    return i;
  }
  // bosluk calismasi icin \s+(?!\S) | \s+
  std::size_t whitespace(std::size_t i) const {
    std::size_t len = 0;
    std::size_t last = i;
    std::size_t count = 0;
    std::size_t j = i;
    while (j < s.size() && at(j, &len) == kSpace) {
      last = j;
      j += len;
      ++count;
    }
    if (j == s.size() || count < 2) {
      return j;
    }
    return last;  // son bosluk karakteri bir sonraki parcaya kalir
  }
};

inline char ascii_lower(char c) { return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c; }

std::size_t match_gpt2(const Cursor& cur, std::size_t i) {
  const std::string_view s = cur.s;
  const std::size_t n = s.size();
  if (s[i] == '\'' && i + 1 < n) {
    const char a = s[i + 1];
    if (a == 's' || a == 't' || a == 'm' || a == 'd') {
      return i + 2;
    }
    if (i + 2 < n) {
      const char b = s[i + 2];
      if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) {
        return i + 3;
      }
    }
  }
  std::size_t len = 0;
  const Cls c = cur.at(i, &len);
  if (s[i] == ' ' && i + 1 < n) {
    std::size_t l2 = 0;
    const Cls c2 = cur.at(i + 1, &l2);
    if (c2 != kSpace) {
      return cur.run(i + 1, c2);  // ' ?\p{L}+ | ' ?\p{N}+ | ' ?[^\s\p{L}\p{N}]+
    }
  }
  if (c != kSpace) {
    return cur.run(i, c);
  }
  return cur.whitespace(i);
}

bool cl100k_contraction(const Cursor& cur, std::size_t i, std::size_t* end) {
  const std::string_view s = cur.s;
  if (s[i] != '\'' || i + 1 >= s.size()) {
    return false;
  }
  std::size_t l1 = 0;
  std::uint32_t c1 = 0;
  (void)cur.at(i + 1, &l1, &c1);
  // (?i:[sdmt]) : basit buyuk/kucuk harf katlama; 's' icin U+017F (uzun s) da eslesir.
  if (c1 == 's' || c1 == 'S' || c1 == 0x17F || c1 == 'd' || c1 == 'D' || c1 == 'm' || c1 == 'M' || c1 == 't' ||
      c1 == 'T') {
    *end = i + 1 + l1;
    return true;
  }
  if (i + 2 < s.size() && c1 < 128) {
    const char a = ascii_lower(static_cast<char>(c1));
    const char b = ascii_lower(s[i + 2]);
    if ((a == 'l' && b == 'l') || (a == 'v' && b == 'e') || (a == 'r' && b == 'e')) {
      *end = i + 3;
      return true;
    }
  }
  return false;
}

std::size_t match_cl100k(const Cursor& cur, std::size_t i, bool end_rule) {
  const std::string_view s = cur.s;
  const std::size_t n = s.size();
  std::size_t end = 0;
  if (cl100k_contraction(cur, i, &end)) {
    return end;
  }
  std::size_t len = 0;
  const Cls c = cur.at(i, &len);
  // [^\r\n\p{L}\p{N}]?+\p{L}+
  if (c == kLetter) {
    return cur.run(i, kLetter);
  }
  if (c != kNumber && s[i] != '\r' && s[i] != '\n' && i + len < n) {
    std::size_t l2 = 0;
    if (cur.at(i + len, &l2) == kLetter) {
      return cur.run(i + len, kLetter);
    }
  }
  // \p{N}{1,3}
  if (c == kNumber) {
    std::size_t j = i;
    for (int k = 0; k < 3 && j < n; ++k) {
      std::size_t l = 0;
      if (cur.at(j, &l) != kNumber) {
        break;
      }
      j += l;
    }
    return j;
  }
  // ' ?[^\s\p{L}\p{N}]++[\r\n]*'
  auto others_then_newlines = [&](std::size_t from) {
    std::size_t j = cur.run(from, kOther);
    while (j < n && (s[j] == '\r' || s[j] == '\n')) {
      ++j;
    }
    return j;
  };
  if (s[i] == ' ' && i + 1 < n) {
    std::size_t l2 = 0;
    if (cur.at(i + 1, &l2) == kOther) {
      return others_then_newlines(i + 1);
    }
  }
  if (c == kOther) {
    return others_then_newlines(i);
  }
  // tiktoken cl100k: \s++$ (metin sonuna uzanan bosluk calismasi tek parca); LLaMA-3'te yok.
  if (end_rule && cur.run(i, kSpace) == n) {
    return n;
  }
  // \s*[\r\n]  -> bosluk calismasindaki son CR/LF'ye kadar
  std::size_t j = i;
  std::size_t last_nl = std::string_view::npos;
  while (j < n) {
    std::size_t l = 0;
    if (cur.at(j, &l) != kSpace) {
      break;
    }
    if (s[j] == '\r' || s[j] == '\n') {
      last_nl = j;
    }
    j += l;
  }
  if (last_nl != std::string_view::npos) {
    return last_nl + 1;
  }
  return cur.whitespace(i);
}

// ---- Bayt <-> unicode (GPT-2) ---------------------------------------------------------------
struct ByteUnicode {
  std::uint32_t cp[256];
  ByteUnicode() {
    std::vector<int> bs;
    for (int b = '!'; b <= '~'; ++b) bs.push_back(b);
    for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
    for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);
    std::vector<bool> used(256, false);
    for (const int b : bs) {
      used[static_cast<std::size_t>(b)] = true;
      cp[b] = static_cast<std::uint32_t>(b);
    }
    int extra = 0;
    for (int b = 0; b < 256; ++b) {
      if (!used[static_cast<std::size_t>(b)]) {
        cp[b] = static_cast<std::uint32_t>(256 + extra++);
      }
    }
  }
};
const ByteUnicode& byte_unicode() {
  static const ByteUnicode t;
  return t;
}

void append_utf8(std::string& out, std::uint32_t cp) {
  if (cp < 0x80) {
    out += static_cast<char>(cp);
  } else if (cp < 0x800) {
    out += static_cast<char>(0xC0 | (cp >> 6));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  } else {
    out += static_cast<char>(0xE0 | (cp >> 12));
    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  }
}

// ---- Rank tablosu: acik adresleme, anahtar baytlari tek havuzda -----------------------------
inline std::uint64_t hash_bytes(const char* p, std::size_t n) {
  std::uint64_t h = 0xcbf29ce484222325ULL ^ (n * 0x9E3779B97F4A7C15ULL);
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    std::uint64_t w = 0;
    std::memcpy(&w, p + i, 8);
    h = (h ^ w) * 0x100000001B3ULL;
    h ^= h >> 29;
  }
  std::uint64_t w = 0;
  std::memcpy(&w, p + i, n - i);
  h = (h ^ w) * 0x100000001B3ULL;
  h ^= h >> 32;
  return h * 0x9E3779B97F4A7C15ULL;
}

class RankTable {
 public:
  void build(const std::vector<std::string>& tokens) {
    std::size_t cap = 16;
    while (cap < tokens.size() * 2) {
      cap <<= 1;
    }
    slots_.assign(cap, Slot{});
    mask_ = cap - 1;
    for (std::size_t r = 0; r < tokens.size(); ++r) {
      const std::string& t = tokens[r];
      if (t.empty()) {
        continue;
      }
      const std::uint64_t h = hash_bytes(t.data(), t.size());
      std::size_t k = h & mask_;
      while (slots_[k].len != 0) {
        k = (k + 1) & mask_;
      }
      slots_[k] = Slot{h, static_cast<std::uint32_t>(pool_.size()), static_cast<std::uint32_t>(t.size()),
                       static_cast<std::uint32_t>(r)};
      pool_ += t;
    }
  }
  [[nodiscard]] std::uint32_t find(const char* p, std::size_t n) const {
    const std::uint64_t h = hash_bytes(p, n);
    std::size_t k = h & mask_;
    while (true) {
      const Slot& s = slots_[k];
      if (s.len == 0) {
        return kNone;
      }
      if (s.hash == h && s.len == n && std::memcmp(pool_.data() + s.off, p, n) == 0) {
        return s.rank;
      }
      k = (k + 1) & mask_;
    }
  }
  static constexpr std::uint32_t kNone = std::numeric_limits<std::uint32_t>::max();

 private:
  struct Slot {
    std::uint64_t hash = 0;
    std::uint32_t off = 0;
    std::uint32_t len = 0;
    std::uint32_t rank = 0;
  };
  std::vector<Slot> slots_;
  std::string pool_;
  std::size_t mask_ = 0;
};

std::string base64_decode(std::string_view in) {
  static int table[256];
  static bool init = false;
  if (!init) {
    std::fill(std::begin(table), std::end(table), -1);
    const char* a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (int i = 0; i < 64; ++i) table[static_cast<unsigned char>(a[i])] = i;
    init = true;
  }
  std::string out;
  int val = 0;
  int bits = -8;
  for (const char ch : in) {
    if (ch == '=') break;
    const int d = table[static_cast<unsigned char>(ch)];
    if (d < 0) fail("gecersiz base64");
    val = (val << 6) | d;
    bits += 6;
    if (bits >= 0) {
      out += static_cast<char>((val >> bits) & 0xFF);
      bits -= 8;
    }
  }
  return out;
}

std::string base64_encode(std::string_view in) {
  const char* a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  std::size_t i = 0;
  for (; i + 3 <= in.size(); i += 3) {
    const std::uint32_t v = (static_cast<unsigned char>(in[i]) << 16) | (static_cast<unsigned char>(in[i + 1]) << 8) |
                            static_cast<unsigned char>(in[i + 2]);
    out += a[(v >> 18) & 63];
    out += a[(v >> 12) & 63];
    out += a[(v >> 6) & 63];
    out += a[v & 63];
  }
  if (i + 1 == in.size()) {
    const std::uint32_t v = static_cast<unsigned char>(in[i]) << 16;
    out += a[(v >> 18) & 63];
    out += a[(v >> 12) & 63];
    out += "==";
  } else if (i + 2 == in.size()) {
    const std::uint32_t v = (static_cast<unsigned char>(in[i]) << 16) | (static_cast<unsigned char>(in[i + 1]) << 8);
    out += a[(v >> 18) & 63];
    out += a[(v >> 12) & 63];
    out += a[(v >> 6) & 63];
    out += '=';
  }
  return out;
}

}  // namespace

const char* to_string(Pretokenizer p) noexcept {
  switch (p) {
    case Pretokenizer::Gpt2:
      return "gpt2";
    case Pretokenizer::Cl100k:
      return "cl100k";
    case Pretokenizer::Llama3:
      return "llama3";
  }
  return "unknown";
}

Pretokenizer pretokenizer_from_string(const std::string& name) {
  if (name == "gpt2") return Pretokenizer::Gpt2;
  if (name == "cl100k" || name == "cl100k_base") return Pretokenizer::Cl100k;
  if (name == "llama3") return Pretokenizer::Llama3;
  fail("bilinmeyen on-bolme: " + name);
}

const char* unicode_version() noexcept { return kUnicodeVersion; }

void pretokenize(std::string_view text, Pretokenizer p, std::vector<std::pair<std::uint32_t, std::uint32_t>>& out) {
  out.clear();
  if (text.size() > std::numeric_limits<std::uint32_t>::max()) {
    fail("belge 4 GiB'tan buyuk");
  }
  const Cursor cur{text};
  std::size_t i = 0;
  while (i < text.size()) {
    const std::size_t j = p == Pretokenizer::Gpt2 ? match_gpt2(cur, i) : match_cl100k(cur, i, p == Pretokenizer::Cl100k);
    out.emplace_back(static_cast<std::uint32_t>(i), static_cast<std::uint32_t>(j));
    i = j;
  }
}

std::string bytes_to_unicode(std::string_view bytes) {
  std::string out;
  out.reserve(bytes.size() * 2);
  for (const char b : bytes) {
    append_utf8(out, byte_unicode().cp[static_cast<unsigned char>(b)]);
  }
  return out;
}

// ---- ByteBpe ---------------------------------------------------------------------------------
struct ByteBpe::Impl {
  Pretokenizer pre = Pretokenizer::Gpt2;
  std::vector<std::string> tokens;  // id -> bayt (ozel tokenlar dahil)
  std::vector<char> is_special;
  RankTable table;
  std::uint32_t byte_rank[256];
  std::vector<std::pair<std::string, std::uint32_t>> specials;

  void finalize() {
    std::vector<std::string> ranked = tokens;
    for (std::size_t i = 0; i < ranked.size(); ++i) {
      if (is_special[i] != 0) {
        ranked[i].clear();
      }
    }
    table.build(ranked);
    for (int b = 0; b < 256; ++b) {
      const char c = static_cast<char>(b);
      byte_rank[b] = table.find(&c, 1);
      if (byte_rank[b] == RankTable::kNone) {
        fail("sozlukte tek bayt token'i eksik: " + std::to_string(b));
      }
    }
  }

  // tiktoken byte_pair_merge.
  void encode_piece(const char* p, std::size_t n, std::vector<std::uint32_t>& out, std::vector<std::pair<std::size_t, std::uint32_t>>& parts) const {
    if (n == 1) {
      out.push_back(byte_rank[static_cast<unsigned char>(*p)]);
      return;
    }
    const std::uint32_t whole = table.find(p, n);
    if (whole != RankTable::kNone) {
      out.push_back(whole);
      return;
    }
    if (n > kLongPiece) {
      encode_long_piece(p, n, out);
      return;
    }
    // parts[i] = (baslangic, parts[i]..parts[i+2] birlesirse rank); tiktoken ile ayni sira:
    // komsu rank'lar silmeden ONCE (parts[i+3] ile) hesaplanir.
    parts.clear();
    for (std::size_t i = 0; i + 1 < n; ++i) {
      parts.emplace_back(i, table.find(p + i, 2));
    }
    parts.emplace_back(n - 1, RankTable::kNone);
    parts.emplace_back(n, RankTable::kNone);
    auto rank_of = [&](std::size_t i) -> std::uint32_t {
      if (i + 3 < parts.size()) {
        return table.find(p + parts[i].first, parts[i + 3].first - parts[i].first);
      }
      return RankTable::kNone;
    };
    while (true) {
      std::uint32_t best = RankTable::kNone;
      std::size_t at = 0;
      for (std::size_t i = 0; i + 1 < parts.size(); ++i) {
        if (parts[i].second < best) {
          best = parts[i].second;
          at = i;
        }
      }
      if (best == RankTable::kNone) {
        break;
      }
      if (at > 0) {
        parts[at - 1].second = rank_of(at - 1);
      }
      parts[at].second = rank_of(at);
      parts.erase(parts.begin() + static_cast<std::ptrdiff_t>(at) + 1);
    }
    for (std::size_t i = 0; i + 1 < parts.size(); ++i) {
      const std::uint32_t r = table.find(p + parts[i].first, parts[i + 1].first - parts[i].first);
      if (r == RankTable::kNone) {
        fail("birlesim sonucu sozlukte yok (bozuk rank tablosu)");
      }
      out.push_back(r);
    }
  }

  // Uzun parcalarda (or. kod derlemindeki uzun bosluk / sembol dizileri) dogrusal tarama O(n^2)
  // olur. Bagli liste + tembel gecersizlestirmeli min-yigin ayni secimi yapar: en kucuk rank,
  // esitlikte en soldaki (baslangic ofseti) — tiktoken'in dongusuyle ayni birlesim sirasi.
  static constexpr std::size_t kLongPiece = 256;

  void encode_long_piece(const char* p, std::size_t n, std::vector<std::uint32_t>& out) const {
    struct Cand {
      std::uint32_t rank;
      std::uint32_t start;
      std::uint32_t len;  // birlesik aralik uzunlugu (gecerlilik kontrolu)
      bool operator>(const Cand& o) const noexcept { return rank != o.rank ? rank > o.rank : start > o.start; }
    };
    std::vector<std::uint32_t> next(n + 1);
    std::vector<std::uint32_t> prev(n + 1);
    std::vector<char> alive(n + 1, 1);
    for (std::size_t i = 0; i <= n; ++i) {
      next[i] = static_cast<std::uint32_t>(i + 1);
      prev[i] = static_cast<std::uint32_t>(i == 0 ? 0 : i - 1);
    }
    std::priority_queue<Cand, std::vector<Cand>, std::greater<Cand>> heap;
    auto push = [&](std::uint32_t i) {
      const std::uint32_t j = next[i];
      if (j >= n) return;
      const std::uint32_t end = next[j];
      const std::uint32_t r = table.find(p + i, end - i);
      if (r != RankTable::kNone) heap.push({r, i, end - i});
    };
    for (std::uint32_t i = 0; i + 1 < n; ++i) push(i);
    while (!heap.empty()) {
      const Cand c = heap.top();
      heap.pop();
      if (!alive[c.start]) continue;
      const std::uint32_t j = next[c.start];
      if (j >= n || next[j] - c.start != c.len) continue;  // komsu degismis: eski aday
      // c.start ile j birlesir.
      alive[j] = 0;
      next[c.start] = next[j];
      prev[next[j]] = c.start;
      push(c.start);
      if (c.start > 0) push(prev[c.start]);
    }
    for (std::uint32_t i = 0; i < n; i = next[i]) {
      const std::uint32_t r = table.find(p + i, next[i] - i);
      if (r == RankTable::kNone) fail("birlesim sonucu sozlukte yok (bozuk rank tablosu)");
      out.push_back(r);
    }
  }
};

ByteBpe::ByteBpe() : impl_(std::make_unique<Impl>()) {}
ByteBpe::~ByteBpe() = default;
ByteBpe::ByteBpe(ByteBpe&&) noexcept = default;
ByteBpe& ByteBpe::operator=(ByteBpe&&) noexcept = default;

ByteBpe ByteBpe::from_ranks(std::vector<std::string> token_bytes, Pretokenizer p,
                            std::vector<std::pair<std::string, std::uint32_t>> specials) {
  ByteBpe b;
  Impl& m = *b.impl_;
  m.pre = p;
  m.tokens = std::move(token_bytes);
  m.is_special.assign(m.tokens.size(), 0);
  for (const auto& [name, id] : specials) {
    if (id >= m.tokens.size()) {
      m.tokens.resize(id + 1);
      m.is_special.resize(id + 1, 0);
    }
    m.tokens[id] = name;
    m.is_special[id] = 1;
  }
  m.specials = std::move(specials);
  m.finalize();
  return b;
}

ByteBpe ByteBpe::load_tiktoken(const std::string& path, Pretokenizer p,
                               std::vector<std::pair<std::string, std::uint32_t>> specials) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    fail("acilamadi: " + path);
  }
  std::vector<std::string> tokens;
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    const std::size_t sp = line.find(' ');
    if (sp == std::string::npos) fail("gecersiz tiktoken satiri");
    const std::uint32_t rank = static_cast<std::uint32_t>(std::stoul(line.substr(sp + 1)));
    if (rank >= tokens.size()) tokens.resize(rank + 1);
    tokens[rank] = base64_decode(std::string_view(line).substr(0, sp));
  }
  return from_ranks(std::move(tokens), p, std::move(specials));
}

void ByteBpe::save_tiktoken(const std::string& path) const {
  std::ofstream out(path, std::ios::binary);
  for (std::size_t r = 0; r < impl_->tokens.size(); ++r) {
    if (impl_->is_special[r] != 0 || impl_->tokens[r].empty()) continue;
    out << base64_encode(impl_->tokens[r]) << ' ' << r << '\n';
  }
}

void ByteBpe::encode_append(std::string_view text, std::vector<std::uint32_t>& out) const {
  std::vector<std::pair<std::uint32_t, std::uint32_t>> pieces;
  std::vector<std::pair<std::size_t, std::uint32_t>> parts;
  pretokenize(text, impl_->pre, pieces);
  for (const auto& [a, e] : pieces) {
    impl_->encode_piece(text.data() + a, e - a, out, parts);
  }
}

std::vector<std::uint32_t> ByteBpe::encode(std::string_view text) const {
  std::vector<std::uint32_t> out;
  out.reserve(text.size() / 3 + 4);
  encode_append(text, out);
  return out;
}

std::vector<std::vector<std::uint32_t>> ByteBpe::encode_batch(const std::vector<std::string_view>& docs,
                                                              int threads) const {
  std::vector<std::vector<std::uint32_t>> out(docs.size());
  if (threads <= 0) {
    threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  }
  threads = std::min<int>(threads, static_cast<int>(std::max<std::size_t>(docs.size(), 1)));
  std::atomic<std::size_t> next{0};
  auto work = [&] {
    for (std::size_t i = next.fetch_add(1); i < docs.size(); i = next.fetch_add(1)) {
      out[i] = encode(docs[i]);
    }
  };
  if (threads <= 1) {
    work();
    return out;
  }
  std::vector<std::thread> pool;
  for (int t = 0; t < threads; ++t) pool.emplace_back(work);
  for (auto& t : pool) t.join();
  return out;
}

const std::string& ByteBpe::token_bytes(std::uint32_t id) const {
  if (id >= impl_->tokens.size() || (impl_->tokens[id].empty())) {
    fail("bilinmeyen token id: " + std::to_string(id));
  }
  return impl_->tokens[id];
}

std::string ByteBpe::decode(const std::vector<std::uint32_t>& ids) const {
  std::string out;
  for (const std::uint32_t id : ids) out += token_bytes(id);
  return out;
}

std::uint32_t ByteBpe::vocab_size() const noexcept { return static_cast<std::uint32_t>(impl_->tokens.size()); }

std::optional<std::uint32_t> ByteBpe::special(const std::string& name) const {
  for (const auto& [n, id] : impl_->specials) {
    if (n == name) return id;
  }
  return std::nullopt;
}

Pretokenizer ByteBpe::pretokenizer() const noexcept { return impl_->pre; }

std::array<std::uint8_t, 32> ByteBpe::digest() const {
  Sha256 h;
  const std::string pre = to_string(impl_->pre);
  h.update(pre.data(), pre.size());
  for (std::size_t r = 0; r < impl_->tokens.size(); ++r) {
    const std::uint64_t len = impl_->tokens[r].size();
    const std::uint8_t sp = impl_->is_special[r];
    h.update(&len, sizeof(len));
    h.update(&sp, 1);
    h.update(impl_->tokens[r].data(), impl_->tokens[r].size());
  }
  return h.finish();
}

// ---- Akisli cozme ------------------------------------------------------------------------------
namespace {
// s'nin sonunda tamamlanmamis bir UTF-8 dizisinin kac bayt oldugunu dondurur (0: tam).
std::size_t incomplete_tail(const std::string& s) {
  const std::size_t n = s.size();
  for (std::size_t back = 1; back <= 3 && back <= n; ++back) {
    const auto b = static_cast<unsigned char>(s[n - back]);
    if ((b & 0xC0) == 0x80) continue;  // devam bayti, geri git
    std::size_t need = 1;
    if ((b & 0xE0) == 0xC0) need = 2;
    else if ((b & 0xF0) == 0xE0) need = 3;
    else if ((b & 0xF8) == 0xF0) need = 4;
    return need > back ? back : 0;
  }
  return 0;
}
}  // namespace

std::string StreamDecoder::push(std::uint32_t id) {
  pending_ += bpe_->token_bytes(id);
  const std::size_t tail = incomplete_tail(pending_);
  std::string out = pending_.substr(0, pending_.size() - tail);
  pending_.erase(0, pending_.size() - tail);
  return out;
}

std::string StreamDecoder::flush() {
  std::string out;
  out.swap(pending_);
  return out;
}

// ---- Egitim (HF tokenizers BpeTrainer ile ayni kurallar) ---------------------------------------
namespace {

using Pair = std::uint64_t;
inline Pair make_pair_id(std::uint32_t a, std::uint32_t b) { return (static_cast<std::uint64_t>(a) << 32) | b; }
inline std::uint32_t first(Pair p) { return static_cast<std::uint32_t>(p >> 32); }
inline std::uint32_t second(Pair p) { return static_cast<std::uint32_t>(p & 0xFFFFFFFFu); }

struct Word {
  std::vector<std::uint32_t> sym;
  // HF word.merge: soldan saga, cakismayan eslesmeler; komsu cift degisimleri.
  void merge(std::uint32_t c1, std::uint32_t c2, std::uint32_t rep, std::vector<std::pair<Pair, int>>& changes) {
    std::size_t i = 0;
    while (i < sym.size()) {
      if (sym[i] == c1 && i + 1 < sym.size() && sym[i + 1] == c2) {
        if (i > 0) {
          changes.emplace_back(make_pair_id(sym[i - 1], c1), -1);
          changes.emplace_back(make_pair_id(sym[i - 1], rep), 1);
        }
        sym[i] = rep;
        sym.erase(sym.begin() + static_cast<std::ptrdiff_t>(i) + 1);
        if (i + 1 < sym.size()) {
          changes.emplace_back(make_pair_id(c2, sym[i + 1]), -1);
          changes.emplace_back(make_pair_id(rep, sym[i + 1]), 1);
        }
      }
      ++i;
    }
  }
};

struct Merge {
  Pair pair;
  std::uint64_t count;
  std::shared_ptr<std::vector<std::uint32_t>> pos;  // HF: kuyruga girdigi andaki kelime kumesi
};
struct MergeLess {
  bool operator()(const Merge& a, const Merge& b) const {
    if (a.count != b.count) return a.count < b.count;  // buyuk sayim ustte
    return a.pair > b.pair;                            // esitlikte kucuk (id1, id2) ustte
  }
};

}  // namespace

BpeTrainResult train_bpe(const std::vector<std::string_view>& texts, const BpeTrainOptions& opt) {
  const int threads = opt.threads > 0 ? opt.threads : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  // 1) Kelime sayimi: is parcacigi basina ayri harita, sonra birlestirme.
  std::vector<std::unordered_map<std::string, std::uint64_t>> local(static_cast<std::size_t>(threads));
  {
    std::atomic<std::size_t> next{0};
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t) {
      pool.emplace_back([&, t] {
        std::vector<std::pair<std::uint32_t, std::uint32_t>> pieces;
        auto& m = local[static_cast<std::size_t>(t)];
        for (std::size_t i = next.fetch_add(1); i < texts.size(); i = next.fetch_add(1)) {
          pretokenize(texts[i], opt.pretokenizer, pieces);
          for (const auto& [a, e] : pieces) {
            ++m[std::string(texts[i].substr(a, e - a))];
          }
        }
      });
    }
    for (auto& th : pool) th.join();
  }
  std::unordered_map<std::string, std::uint64_t> counts = std::move(local[0]);
  for (std::size_t t = 1; t < local.size(); ++t) {
    for (auto& [w, c] : local[t]) counts[w] += c;
    local[t].clear();
  }
  // HF: kelimeler, haritanin yineleme sirasinda indekslenir; sonuc (birlesim sirasi) indekslere
  // bagli degildir. Belirlenimcilik icin bayt sirasina gore.
  std::vector<std::pair<std::string, std::uint64_t>> wc(counts.begin(), counts.end());
  counts.clear();
  std::sort(wc.begin(), wc.end());

  // 2) Sozluk: ozel tokenlar, sonra 256 bayt (bayt-unicode kod noktasi sirasinda).
  std::vector<std::string> id_bytes;   // id -> bayt
  std::vector<std::string> id_unicode;  // id -> HF gosterimi
  std::unordered_map<std::string, std::uint32_t> unicode_to_id;
  std::vector<std::pair<std::string, std::uint32_t>> specials;
  for (const std::string& s : opt.special_tokens) {
    if (unicode_to_id.count(s) != 0) continue;
    const auto id = static_cast<std::uint32_t>(id_unicode.size());
    unicode_to_id[s] = id;
    id_unicode.push_back(s);
    id_bytes.push_back(s);
    specials.emplace_back(s, id);
  }
  std::vector<int> byte_order(256);
  for (int b = 0; b < 256; ++b) byte_order[static_cast<std::size_t>(b)] = b;
  std::sort(byte_order.begin(), byte_order.end(),
            [](int a, int b) { return byte_unicode().cp[a] < byte_unicode().cp[b]; });
  std::uint32_t byte_id[256];
  for (const int b : byte_order) {
    std::string u;
    append_utf8(u, byte_unicode().cp[b]);
    const auto id = static_cast<std::uint32_t>(id_unicode.size());
    byte_id[b] = id;
    unicode_to_id[u] = id;
    id_unicode.push_back(u);
    id_bytes.emplace_back(1, static_cast<char>(b));
  }
  // 3) Kelimeler ve cift sayimlari.
  std::vector<Word> words(wc.size());
  std::vector<std::uint64_t> wcount(wc.size());
  for (std::size_t i = 0; i < wc.size(); ++i) {
    for (const char ch : wc[i].first) words[i].sym.push_back(byte_id[static_cast<unsigned char>(ch)]);
    wcount[i] = wc[i].second;
  }
  std::unordered_map<Pair, std::int64_t> pair_counts;
  std::unordered_map<Pair, std::vector<std::uint32_t>> where;
  for (std::size_t i = 0; i < words.size(); ++i) {
    const auto& s = words[i].sym;
    for (std::size_t k = 0; k + 1 < s.size(); ++k) {
      const Pair p = make_pair_id(s[k], s[k + 1]);
      pair_counts[p] += static_cast<std::int64_t>(wcount[i]);
      auto& v = where[p];
      if (v.empty() || v.back() != i) v.push_back(static_cast<std::uint32_t>(i));
    }
  }
  std::priority_queue<Merge, std::vector<Merge>, MergeLess> queue;
  for (auto& [p, pos] : where) {
    const std::int64_t c = pair_counts[p];
    if (c > 0) queue.push(Merge{p, static_cast<std::uint64_t>(c), std::make_shared<std::vector<std::uint32_t>>(std::move(pos))});
  }
  where.clear();

  BpeTrainResult result;
  result.words = wc.size();
  std::vector<std::pair<Pair, int>> changes;
  std::vector<std::pair<std::pair<Pair, int>, std::uint32_t>> all_changes;
  while (unicode_to_id.size() < opt.vocab_size && !queue.empty()) {
    Merge top = queue.top();
    queue.pop();
    const std::int64_t current = pair_counts[top.pair];
    if (static_cast<std::int64_t>(top.count) != current) {
      top.count = static_cast<std::uint64_t>(std::max<std::int64_t>(current, 0));
      queue.push(std::move(top));
      continue;
    }
    if (top.count < 1 || opt.min_frequency > top.count) break;
    const std::uint32_t a = first(top.pair);
    const std::uint32_t b = second(top.pair);
    const std::string new_unicode = id_unicode[a] + id_unicode[b];
    std::uint32_t new_id = 0;
    const auto found = unicode_to_id.find(new_unicode);
    if (found != unicode_to_id.end()) {
      new_id = found->second;
    } else {
      new_id = static_cast<std::uint32_t>(id_unicode.size());
      unicode_to_id[new_unicode] = new_id;
      id_unicode.push_back(new_unicode);
      id_bytes.push_back(id_bytes[a] + id_bytes[b]);
    }
    result.merges.emplace_back(id_unicode[a], id_unicode[b]);
    all_changes.clear();
    for (const std::uint32_t iw : *top.pos) {
      changes.clear();
      words[iw].merge(a, b, new_id, changes);
      for (const auto& c : changes) all_changes.emplace_back(c, iw);
    }
    std::unordered_map<Pair, std::vector<std::uint32_t>> fresh;
    for (const auto& [ch, iw] : all_changes) {
      pair_counts[ch.first] += static_cast<std::int64_t>(ch.second) * static_cast<std::int64_t>(wcount[iw]);
      if (ch.second > 0) {
        auto& v = fresh[ch.first];
        if (v.empty() || v.back() != iw) v.push_back(iw);  // degisimler kelime kelime gruplu
      }
    }
    for (auto& [p, pos] : fresh) {
      const std::int64_t c = pair_counts[p];
      if (c > 0) queue.push(Merge{p, static_cast<std::uint64_t>(c), std::make_shared<std::vector<std::uint32_t>>(std::move(pos))});
    }
  }
  result.vocab = id_unicode;
  result.tokenizer = ByteBpe::from_ranks(std::move(id_bytes), opt.pretokenizer, specials);
  return result;
}

// ---- SHA-256 -------------------------------------------------------------------------------------
namespace {
constexpr std::uint32_t kK[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
    0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
    0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
    0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
    0xc67178f2};
inline std::uint32_t rotr(std::uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
}  // namespace

Sha256::Sha256() : h_{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19} {}

void Sha256::block(const std::uint8_t* p) {
  std::uint32_t w[64];
  for (int i = 0; i < 16; ++i) {
    w[i] = (static_cast<std::uint32_t>(p[4 * i]) << 24) | (static_cast<std::uint32_t>(p[4 * i + 1]) << 16) |
           (static_cast<std::uint32_t>(p[4 * i + 2]) << 8) | p[4 * i + 3];
  }
  for (int i = 16; i < 64; ++i) {
    const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
    const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }
  std::uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4], f = h_[5], g = h_[6], h = h_[7];
  for (int i = 0; i < 64; ++i) {
    const std::uint32_t t1 = h + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + kK[i] + w[i];
    const std::uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }
  h_[0] += a;
  h_[1] += b;
  h_[2] += c;
  h_[3] += d;
  h_[4] += e;
  h_[5] += f;
  h_[6] += g;
  h_[7] += h;
}

void Sha256::update(const void* data, std::size_t size) {
  const auto* p = static_cast<const std::uint8_t*>(data);
  total_ += size;
  while (size > 0) {
    if (len_ == 0 && size >= 64) {
      block(p);
      p += 64;
      size -= 64;
      continue;
    }
    const std::size_t take = std::min<std::size_t>(64 - len_, size);
    std::memcpy(buf_ + len_, p, take);
    len_ += take;
    p += take;
    size -= take;
    if (len_ == 64) {
      block(buf_);
      len_ = 0;
    }
  }
}

std::array<std::uint8_t, 32> Sha256::finish() {
  const std::uint64_t bits = total_ * 8;
  const std::uint8_t pad = 0x80;
  update(&pad, 1);
  const std::uint8_t zero = 0;
  while (len_ != 56) update(&zero, 1);
  std::uint8_t lenb[8];
  for (int i = 0; i < 8; ++i) lenb[i] = static_cast<std::uint8_t>(bits >> (56 - 8 * i));
  update(lenb, 8);
  std::array<std::uint8_t, 32> out{};
  for (int i = 0; i < 8; ++i) {
    out[static_cast<std::size_t>(4 * i)] = static_cast<std::uint8_t>(h_[i] >> 24);
    out[static_cast<std::size_t>(4 * i + 1)] = static_cast<std::uint8_t>(h_[i] >> 16);
    out[static_cast<std::size_t>(4 * i + 2)] = static_cast<std::uint8_t>(h_[i] >> 8);
    out[static_cast<std::size_t>(4 * i + 3)] = static_cast<std::uint8_t>(h_[i]);
  }
  return out;
}

std::string to_hex(const std::array<std::uint8_t, 32>& d) {
  static const char* hex = "0123456789abcdef";
  std::string s;
  for (const std::uint8_t b : d) {
    s += hex[b >> 4];
    s += hex[b & 15];
  }
  return s;
}

std::array<std::uint8_t, 32> sha256_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) fail("acilamadi: " + path);
  Sha256 h;
  std::vector<char> buf(1 << 20);
  while (in) {
    in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
    const std::streamsize n = in.gcount();
    if (n > 0) h.update(buf.data(), static_cast<std::size_t>(n));
  }
  return h.finish();
}

}  // namespace nexusdata::text
