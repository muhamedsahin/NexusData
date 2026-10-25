/**
 * @file token_shard.cpp
 * @brief FAZ 8.C-8.E: NXTK yazici/okuyucu, manifest, Feistel permutasyonu, ornekleyici, karisim,
 *        paketleme.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexusdata/text/token_shard.hpp"

#include "nexusdata/io/json.hpp"
#include "nexusdata/text/bpe.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

namespace nexusdata::text {

namespace {
[[noreturn]] void fail(const std::string& what) { throw std::runtime_error("nexusdata NXTK: " + what); }

std::uint64_t splitmix(std::uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}
std::uint64_t mix(std::uint64_t a, std::uint64_t b, std::uint64_t c) {
  return splitmix(splitmix(a ^ splitmix(b)) ^ splitmix(c + 0x632BE59BD9B4E019ULL));
}
}  // namespace

// ---- Yazici ----------------------------------------------------------------------------------
struct ShardWriter::Impl {
  std::ofstream out;
  std::string path;
  std::vector<std::uint64_t> offsets{0};
  std::vector<std::uint8_t> buf;
};

ShardWriter::ShardWriter(const std::string& path, std::uint32_t vocab_size, const std::array<std::uint8_t, 32>& digest,
                         std::int64_t eos)
    : impl_(std::make_unique<Impl>()), eos_(eos) {
  header_.dtype_bytes = vocab_size <= 65536 ? 2 : 4;
  header_.tokenizer_digest = digest;
  impl_->path = path;
  impl_->out.open(path, std::ios::binary | std::ios::trunc);
  if (!impl_->out) fail("yazilamadi: " + path);
  impl_->out.write(reinterpret_cast<const char*>(&header_), sizeof(header_));  // yer tutucu
}

ShardWriter::~ShardWriter() = default;

void ShardWriter::add_document(const std::uint32_t* tokens, std::size_t count) {
  auto& buf = impl_->buf;
  const std::size_t extra = eos_ >= 0 ? 1 : 0;
  buf.resize((count + extra) * header_.dtype_bytes);
  for (std::size_t i = 0; i < count + extra; ++i) {
    const std::uint32_t t = i < count ? tokens[i] : static_cast<std::uint32_t>(eos_);
    if (header_.dtype_bytes == 2) {
      if (t > 0xFFFF) fail("u16 parcada 65535'ten buyuk token");
      const auto v = static_cast<std::uint16_t>(t);
      std::memcpy(buf.data() + 2 * i, &v, 2);
    } else {
      std::memcpy(buf.data() + 4 * i, &t, 4);
    }
  }
  impl_->out.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
  header_.num_tokens += count + extra;
  header_.num_docs += 1;
  impl_->offsets.push_back(header_.num_tokens);
}

std::array<std::uint8_t, 32> ShardWriter::finish() {
  auto& out = impl_->out;
  const std::uint64_t body = 64 + header_.num_tokens * header_.dtype_bytes;
  const std::uint64_t pad = (8 - body % 8) % 8;
  const char zeros[8] = {};
  out.write(zeros, static_cast<std::streamsize>(pad));
  out.write(reinterpret_cast<const char*>(impl_->offsets.data()),
            static_cast<std::streamsize>(impl_->offsets.size() * sizeof(std::uint64_t)));
  out.seekp(0);
  out.write(reinterpret_cast<const char*>(&header_), sizeof(header_));
  out.close();
  if (!out) fail("yazma hatasi: " + impl_->path);
  return sha256_file(impl_->path);
}

// ---- Okuyucu ---------------------------------------------------------------------------------
ShardReader::ShardReader(const std::string& path) : file_(path) {
  if (file_.size() < sizeof(ShardHeader)) fail("cok kisa: " + path);
  std::memcpy(&header_, file_.data(), sizeof(header_));
  if (std::memcmp(header_.magic, "NXTK", 4) != 0) fail("sihir NXTK degil: " + path);
  if (header_.version != 1) fail("desteklenmeyen surum: " + path);
  if (header_.dtype_bytes != 2 && header_.dtype_bytes != 4) fail("gecersiz dtype: " + path);
  const std::uint64_t body = 64 + header_.num_tokens * header_.dtype_bytes;
  const std::uint64_t off = body + (8 - body % 8) % 8;
  if (off + (header_.num_docs + 1) * 8 != file_.size()) fail("boyut basliktakiyle uyusmuyor: " + path);
  tokens_ = file_.data() + 64;
  offsets_ = reinterpret_cast<const std::uint64_t*>(file_.data() + off);
}

std::uint32_t ShardReader::token(std::uint64_t i) const noexcept {
  if (header_.dtype_bytes == 2) {
    std::uint16_t v = 0;
    std::memcpy(&v, tokens_ + 2 * i, 2);
    return v;
  }
  std::uint32_t v = 0;
  std::memcpy(&v, tokens_ + 4 * i, 4);
  return v;
}

void ShardReader::copy(std::uint64_t begin, std::size_t count, std::int32_t* out) const {
  if (begin + count > header_.num_tokens) fail("okuma parca disinda");
  if (header_.dtype_bytes == 2) {
    const auto* p = reinterpret_cast<const std::uint16_t*>(tokens_) + begin;
    for (std::size_t i = 0; i < count; ++i) out[i] = p[i];
  } else {
    std::memcpy(out, tokens_ + 4 * begin, count * 4);
  }
}

std::uint64_t ShardReader::doc_offset(std::uint64_t d) const noexcept { return offsets_[d]; }

std::uint64_t ShardReader::doc_of(std::uint64_t pos) const noexcept {
  const std::uint64_t* end = offsets_ + header_.num_docs + 1;
  const std::uint64_t* it = std::upper_bound(offsets_, end, pos);
  return static_cast<std::uint64_t>(it - offsets_) - 1;
}

// ---- Manifest --------------------------------------------------------------------------------
std::uint64_t Manifest::total_tokens() const noexcept {
  std::uint64_t n = 0;
  for (const auto& s : shards) n += s.tokens;
  return n;
}

void Manifest::save(const std::string& path) const {
  std::ostringstream o;
  std::uint64_t docs = 0;
  for (const auto& s : shards) docs += s.docs;
  o << "{\n  \"version\": 1,\n  \"tokenizer\": {\"name\": " << json::quote(tokenizer_name)
    << ", \"digest\": " << json::quote(tokenizer_digest) << ", \"vocab_size\": " << vocab_size << ", \"eos\": " << eos
    << "},\n  \"total_tokens\": " << total_tokens() << ",\n  \"total_docs\": " << docs << ",\n  \"shards\": [\n";
  for (std::size_t i = 0; i < shards.size(); ++i) {
    const auto& s = shards[i];
    o << "    {\"path\": " << json::quote(s.path) << ", \"tokens\": " << s.tokens << ", \"docs\": " << s.docs
      << ", \"sha256\": " << json::quote(s.sha256) << "}" << (i + 1 < shards.size() ? ",\n" : "\n");
  }
  o << "  ]\n}\n";
  std::ofstream(path, std::ios::binary) << o.str();
}

Manifest Manifest::load(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) fail("manifest acilamadi: " + path);
  std::stringstream ss;
  ss << in.rdbuf();
  const json::Value v = json::parse(ss.str());
  if (v.get_int("version", 0) != 1) fail("manifest surumu 1 degil");
  Manifest m;
  m.dir = std::filesystem::path(path).parent_path().string();
  if (const json::Value* t = v.find("tokenizer")) {
    m.tokenizer_name = t->get_string("name", "");
    m.tokenizer_digest = t->get_string("digest", "");
    m.vocab_size = static_cast<std::uint32_t>(t->get_int("vocab_size", 0));
    m.eos = t->get_int("eos", -1);
  }
  const json::Value* shards = v.find("shards");
  if (shards == nullptr || !shards->is_array()) fail("manifest'te shards yok");
  for (const json::Value& s : shards->as_array()) {
    ManifestShard ms;
    ms.path = s.get_string("path", "");
    ms.tokens = static_cast<std::uint64_t>(s.get_int("tokens", 0));
    ms.docs = static_cast<std::uint64_t>(s.get_int("docs", 0));
    ms.sha256 = s.get_string("sha256", "");
    m.shards.push_back(ms);
  }
  return m;
}

std::string Manifest::verify() const {
  for (const auto& s : shards) {
    const std::string p = (std::filesystem::path(dir) / s.path).string();
    if (to_hex(sha256_file(p)) != s.sha256) return s.path;
  }
  return "";
}

// ---- Feistel -----------------------------------------------------------------------------------
FeistelPermutation::FeistelPermutation(std::uint64_t n, std::uint64_t seed) : n_(n) {
  int bits = 2;
  while (bits < 64 && (std::uint64_t{1} << bits) < n) bits += 2;
  half_bits_ = bits / 2;
  mask_ = (std::uint64_t{1} << half_bits_) - 1;
  for (int r = 0; r < 4; ++r) keys_[r] = splitmix(seed + static_cast<std::uint64_t>(r) * 0x9E3779B97F4A7C15ULL);
}

std::uint64_t FeistelPermutation::operator()(std::uint64_t i) const noexcept {
  if (n_ <= 1) return 0;
  std::uint64_t x = i;
  do {  // dongu yuruyusu: 2^bits alani uzerinde bijeksiyon, [0, n) disi degerler tekrar sifrelenir
    std::uint64_t l = x >> half_bits_;
    std::uint64_t r = x & mask_;
    for (int k = 0; k < 4; ++k) {
      const std::uint64_t f = splitmix(r ^ keys_[k]) & mask_;
      const std::uint64_t nl = r;
      r = l ^ f;
      l = nl;
    }
    x = (l << half_bits_) | r;
  } while (x >= n_);
  return x;
}

// ---- Durum ---------------------------------------------------------------------------------------
std::string SamplerState::serialize() const {
  std::ostringstream o;
  o << "v1," << epoch << ',' << shard_cursor << ',' << window_cursor << ',' << global_index << ',' << seed;
  return o.str();
}

SamplerState SamplerState::deserialize(const std::string& s) {
  SamplerState st;
  char comma = 0;
  std::istringstream in(s);
  std::string tag;
  std::getline(in, tag, ',');
  if (tag != "v1") fail("bilinmeyen ornekleyici durumu: " + s);
  in >> st.epoch >> comma >> st.shard_cursor >> comma >> st.window_cursor >> comma >> st.global_index >> comma >> st.seed;
  if (!in) fail("bozuk ornekleyici durumu: " + s);
  return st;
}

// ---- TokenStream ---------------------------------------------------------------------------------
TokenStream::TokenStream(const Manifest& manifest, int window, std::uint64_t seed, int rank, int world)
    : T_(window), rank_(rank), world_(world) {
  if (window < 1 || world < 1 || rank < 0 || rank >= world) fail("gecersiz pencere/rank");
  // Parcalar tembel eslenir: ornekleyici bir anda tek parcada gezer, en fazla kMaxOpen parca acik
  // kalir. Tumu acik tutulsaydi dokunulan eslemelerin sayfa tablolari (Windows'ta 2 MB basina
  // 4 KB ozel bellek) veri boyuyla dogrusal buyurdu.
  for (const auto& s : manifest.shards) {
    paths_.push_back((std::filesystem::path(manifest.dir) / s.path).string());
    tokens_.push_back(s.tokens);
    total_windows_ += windows_in(tokens_.size() - 1);
  }
  shards_.resize(paths_.size());
  if (total_windows_ == 0) fail("hicbir parcada T+1 token yok");
  state_.seed = seed;
  // Ilk gecerli konuma yerles (penceresiz parcalari atla).
  const FeistelPermutation ps(tokens_.size(), mix(seed, 0, 0xA11));
  while (windows_in(ps(state_.shard_cursor)) == 0) ++state_.shard_cursor;
}

const ShardReader& TokenStream::reader(std::uint64_t shard) const {
  if (!shards_[shard]) {
    while (open_.size() >= kMaxOpen) {
      shards_[open_.front()].reset();
      open_.erase(open_.begin());
    }
    shards_[shard] = std::make_unique<ShardReader>(paths_[shard]);
    if (shards_[shard]->num_tokens() != tokens_[shard]) {
      fail("parca token sayisi manifest ile uyusmuyor: " + paths_[shard]);
    }
    open_.push_back(shard);
  }
  return *shards_[shard];
}

std::uint64_t TokenStream::windows_in(std::uint64_t shard) const noexcept {
  const std::uint64_t n = tokens_[shard];
  return n >= static_cast<std::uint64_t>(T_) + 1 ? (n - 1) / static_cast<std::uint64_t>(T_) : 0;
}

void TokenStream::advance() {
  const FeistelPermutation ps(tokens_.size(), mix(state_.seed, state_.epoch, 0xA11));
  ++state_.window_cursor;
  if (state_.window_cursor >= windows_in(ps(state_.shard_cursor))) {
    state_.window_cursor = 0;
    while (true) {
      ++state_.shard_cursor;
      if (state_.shard_cursor >= tokens_.size()) {
        state_.shard_cursor = 0;
        ++state_.epoch;
      }
      const FeistelPermutation pe(tokens_.size(), mix(state_.seed, state_.epoch, 0xA11));
      if (windows_in(pe(state_.shard_cursor)) > 0) break;
    }
  }
  ++state_.global_index;
}

void TokenStream::fill(std::uint64_t shard, std::uint64_t window, PackedSample& out) const {
  const ShardReader& r = reader(shard);
  const std::uint64_t T = static_cast<std::uint64_t>(T_);
  const std::uint64_t start = window * T;
  out.tokens.resize(T + 1);
  r.copy(start, T + 1, out.tokens.data());
  out.cu_seqlens.assign(1, 0);
  out.loss_mask.assign(T, 1);
  for (std::uint64_t d = r.doc_of(start) + 1; d <= r.num_docs(); ++d) {
    const std::uint64_t o = r.doc_offset(d);
    if (o > start + T) break;
    if (o < start + T) out.cu_seqlens.push_back(static_cast<std::int32_t>(o - start));  // yeni bolum
    if (o >= start + 1) out.loss_mask[o - start - 1] = 0;  // hedef bir sonraki belgenin ilk token'i
  }
  out.cu_seqlens.push_back(static_cast<std::int32_t>(T));
  out.shard = shard;
  out.window = window;
}

void TokenStream::next(PackedSample& out) {
  while (true) {
    const bool mine = state_.global_index % static_cast<std::uint64_t>(world_) == static_cast<std::uint64_t>(rank_);
    if (mine) {
      const FeistelPermutation ps(tokens_.size(), mix(state_.seed, state_.epoch, 0xA11));
      const std::uint64_t shard = ps(state_.shard_cursor);
      const FeistelPermutation pw(windows_in(shard), mix(state_.seed, state_.epoch, shard + 1));
      fill(shard, pw(state_.window_cursor), out);
    }
    advance();
    if (mine) return;
  }
}

void TokenStream::restore(const SamplerState& s) { state_ = s; }

// ---- Karisim -----------------------------------------------------------------------------------
MixtureStream::MixtureStream(std::vector<std::unique_ptr<TokenStream>> sources, std::vector<double> weights,
                             std::uint64_t seed)
    : sources_(std::move(sources)), seed_(seed) {
  if (sources_.empty() || weights.size() != sources_.size()) fail("karisim: kaynak/agirlik sayisi");
  double total = 0.0;
  for (const double w : weights) {
    if (!(w > 0.0)) fail("karisim: agirliklar pozitif olmali");
    total += w;
  }
  double acc = 0.0;
  for (const double w : weights) {
    acc += w / total;
    cumulative_.push_back(acc);
  }
  cumulative_.back() = 1.0;
}

void MixtureStream::next(PackedSample& out) {
  const double u = static_cast<double>(splitmix(seed_ ^ splitmix(index_)) >> 11) * (1.0 / 9007199254740992.0);
  std::size_t k = 0;
  while (k + 1 < cumulative_.size() && u >= cumulative_[k]) ++k;
  sources_[k]->next(out);
  out.source = static_cast<std::uint32_t>(k);
  ++index_;
}

std::string MixtureStream::serialize_state() const {
  std::string s = "m1|" + std::to_string(index_);
  for (const auto& src : sources_) s += "|" + src->state().serialize();
  return s;
}

void MixtureStream::restore_state(const std::string& s) {
  std::vector<std::string> parts;
  std::size_t a = 0;
  while (true) {
    const std::size_t b = s.find('|', a);
    parts.push_back(s.substr(a, b == std::string::npos ? std::string::npos : b - a));
    if (b == std::string::npos) break;
    a = b + 1;
  }
  if (parts.size() != sources_.size() + 2 || parts[0] != "m1") fail("bozuk karisim durumu");
  index_ = std::stoull(parts[1]);
  for (std::size_t i = 0; i < sources_.size(); ++i) sources_[i]->restore(SamplerState::deserialize(parts[i + 2]));
}

// ---- Kutu paketleme ------------------------------------------------------------------------------
std::vector<std::vector<std::uint64_t>> best_fit_pack(const std::vector<std::uint64_t>& lengths, std::uint64_t capacity) {
  std::vector<std::uint64_t> order(lengths.size());
  for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::stable_sort(order.begin(), order.end(), [&](std::uint64_t a, std::uint64_t b) { return lengths[a] > lengths[b]; });
  std::vector<std::vector<std::uint64_t>> bins;
  std::multimap<std::uint64_t, std::size_t> free;  // kalan kapasite -> kutu
  for (const std::uint64_t d : order) {
    const std::uint64_t len = lengths[d];
    if (len >= capacity) {
      bins.push_back({d});
      continue;
    }
    auto it = free.lower_bound(len);  // en kucuk yeterli bosluk
    if (it == free.end()) {
      bins.push_back({d});
      free.emplace(capacity - len, bins.size() - 1);
    } else {
      const std::size_t b = it->second;
      const std::uint64_t left = it->first - len;
      free.erase(it);
      bins[b].push_back(d);
      if (left > 0) free.emplace(left, b);
    }
  }
  return bins;
}

}  // namespace nexusdata::text
