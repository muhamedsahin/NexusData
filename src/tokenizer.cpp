#include "nexusdata/dataset/text.hpp"

#include <cctype>
#include <fstream>
#include <limits>
#include <sstream>
#include <utility>

#include "nexusdata/core/error.hpp"
#include "nexusdata/io/json.hpp"

namespace nexusdata {
namespace {

std::string trim_copy(std::string s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
    std::size_t i = 0;
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
    return s.substr(i);
}

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw IOError("tokenizer: cannot open " + quote_path(path));
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

void load_vocab(const std::string& path,
                std::unordered_map<std::string, std::int64_t>& token_to_id,
                std::unordered_map<std::int64_t, std::string>& id_to_token) {
    const std::string text = read_file(path);
    std::size_t i = 0;
    while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i]))) ++i;
    if (i < text.size() && text[i] == '{') {
        const json::Value root = json::parse(text);
        if (!root.is_object()) throw IOError("tokenizer: vocab JSON must be an object");
        for (const auto& kv : root.as_object()) {
            const std::int64_t id = kv.second.as_int();
            token_to_id[kv.first] = id;
            id_to_token[id] = kv.first;
        }
        return;
    }
    std::int64_t id = 0;
    std::istringstream lines(text);
    std::string line;
    while (std::getline(lines, line)) {
        line = trim_copy(std::move(line));
        if (line.empty()) continue;
        token_to_id[line] = id;
        id_to_token[id] = line;
        ++id;
    }
}

std::string lookup_id(const std::unordered_map<std::int64_t, std::string>& id_to_token,
                      std::int64_t id) {
    const auto it = id_to_token.find(id);
    if (it == id_to_token.end()) {
        throw IOError("tokenizer: id " + std::to_string(id) + " is not in the vocab");
    }
    return it->second;
}

} // namespace

BpeTokenizer::BpeTokenizer(std::string vocab_path, std::string merges_path) {
    load_vocab(vocab_path, token_to_id_, id_to_token_);
    std::istringstream lines(read_file(merges_path));
    std::string line;
    int rank = 0;
    while (std::getline(lines, line)) {
        line = trim_copy(std::move(line));
        if (line.empty() || line[0] == '#') continue;
        const auto sp = line.find(' ');
        if (sp == std::string::npos || sp == 0 || sp + 1 >= line.size()) {
            throw IOError("tokenizer: merge line must be \"left right\"");
        }
        const std::string left = line.substr(0, sp);
        const std::string right = trim_copy(line.substr(sp + 1));
        merge_rank_[left + "\n" + right] = rank++;
    }
}

std::vector<std::int64_t> BpeTokenizer::encode(const std::string& text) const {
    std::vector<std::string> pieces;
    pieces.reserve(text.size());
    for (unsigned char c : text) {
        const std::string tok(1, static_cast<char>(c));
        if (!token_to_id_.count(tok)) {
            throw IOError("tokenizer: byte not in BPE vocab");
        }
        pieces.push_back(tok);
    }
    while (pieces.size() >= 2) {
        int best = std::numeric_limits<int>::max();
        std::size_t at = pieces.size();
        for (std::size_t i = 0; i + 1 < pieces.size(); ++i) {
            const auto it = merge_rank_.find(pieces[i] + "\n" + pieces[i + 1]);
            if (it != merge_rank_.end() && it->second < best) {
                best = it->second;
                at = i;
            }
        }
        if (at == pieces.size()) break;
        pieces[at] += pieces[at + 1];
        pieces.erase(pieces.begin() + static_cast<std::ptrdiff_t>(at + 1));
    }
    std::vector<std::int64_t> ids;
    ids.reserve(pieces.size());
    for (const std::string& p : pieces) {
        const auto it = token_to_id_.find(p);
        if (it == token_to_id_.end()) throw IOError("tokenizer: merged piece not in BPE vocab");
        ids.push_back(it->second);
    }
    return ids;
}

std::string BpeTokenizer::decode(const std::vector<std::int64_t>& ids) const {
    std::string out;
    for (std::int64_t id : ids) out += lookup_id(id_to_token_, id);
    return out;
}

WordPieceTokenizer::WordPieceTokenizer(std::string vocab_path, std::string unk) {
    load_vocab(vocab_path, token_to_id_, id_to_token_);
    const auto it = token_to_id_.find(unk);
    unk_id_ = it == token_to_id_.end() ? static_cast<std::int64_t>(-1) : it->second;
}

std::vector<std::int64_t> WordPieceTokenizer::encode(const std::string& text) const {
    std::vector<std::int64_t> ids;
    std::size_t i = 0;
    while (i < text.size()) {
        while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i]))) ++i;
        if (i >= text.size()) break;
        std::size_t j = i;
        while (j < text.size() && !std::isspace(static_cast<unsigned char>(text[j]))) ++j;
        const std::string word = text.substr(i, j - i);
        bool first = true;
        std::size_t start = 0;
        while (start < word.size()) {
            const std::size_t limit = std::min(word.size(), start + 100);
            bool found = false;
            for (std::size_t end = limit; end > start; --end) {
                const std::string piece = word.substr(start, end - start);
                const std::string key = first ? piece : "##" + piece;
                const auto it = token_to_id_.find(key);
                if (it == token_to_id_.end()) continue;
                ids.push_back(it->second);
                start = end;
                first = false;
                found = true;
                break;
            }
            if (!found) {
                if (unk_id_ < 0) throw IOError("tokenizer: word is not in the WordPiece vocab");
                ids.push_back(unk_id_);
                break;
            }
        }
        i = j;
    }
    return ids;
}

std::string WordPieceTokenizer::decode(const std::vector<std::int64_t>& ids) const {
    std::string out;
    for (std::size_t n = 0; n < ids.size(); ++n) {
        const std::string tok = lookup_id(id_to_token_, ids[n]);
        if (tok.size() >= 2 && tok[0] == '#' && tok[1] == '#') {
            out += tok.substr(2);
        } else {
            if (!out.empty()) out.push_back(' ');
            out += tok;
        }
    }
    return out;
}

UnigramTokenizer::UnigramTokenizer(std::string vocab_path) {
    std::istringstream lines(read_file(vocab_path));
    std::string line;
    std::int64_t id = 0;
    while (std::getline(lines, line)) {
        line = trim_copy(std::move(line));
        if (line.empty() || line[0] == '#') continue;
        const auto tab = line.find('\t');
        if (tab == std::string::npos) throw IOError("tokenizer: unigram line needs a tab score");
        const std::string tok = line.substr(0, tab);
        const std::string score_s = trim_copy(line.substr(tab + 1));
        try {
            const double score = std::stod(score_s);
            pieces_[tok] = Piece{id, score};
            id_to_token_[id] = tok;
            ++id;
        } catch (const std::exception&) {
            throw IOError("tokenizer: unigram score is not a number");
        }
    }
    if (pieces_.empty()) throw IOError("tokenizer: empty unigram vocab");
}

std::vector<std::int64_t> UnigramTokenizer::encode(const std::string& text) const {
    const std::size_t n = text.size();
    const double neg = -1.0e300;
    std::vector<double> dp(n + 1, neg);
    std::vector<int> back(n + 1, -1);
    dp[0] = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (dp[i] < neg / 2) continue;
        const std::size_t limit = std::min(n, i + 32);
        for (std::size_t end = i + 1; end <= limit; ++end) {
            const auto it = pieces_.find(text.substr(i, end - i));
            if (it == pieces_.end()) continue;
            const double sc = dp[i] + it->second.score;
            if (sc > dp[end]) {
                dp[end] = sc;
                back[end] = static_cast<int>(i);
            }
        }
    }
    if (back[n] < 0 && n != 0) throw IOError("tokenizer: unigram could not segment the text");
    std::vector<std::int64_t> rev;
    for (std::size_t i = n; i > 0;) {
        const int b = back[i];
        if (b < 0 || static_cast<std::size_t>(b) >= i) {
            throw IOError("tokenizer: unigram could not segment the text");
        }
        const auto it = pieces_.find(text.substr(static_cast<std::size_t>(b), i - static_cast<std::size_t>(b)));
        if (it == pieces_.end()) throw IOError("tokenizer: unigram piece missing");
        rev.push_back(it->second.id);
        i = static_cast<std::size_t>(b);
    }
    std::vector<std::int64_t> ids(rev.rbegin(), rev.rend());
    return ids;
}

std::string UnigramTokenizer::decode(const std::vector<std::int64_t>& ids) const {
    std::string out;
    for (std::int64_t id : ids) out += lookup_id(id_to_token_, id);
    return out;
}

} // namespace nexusdata
