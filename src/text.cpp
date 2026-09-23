#include "nexusdata/dataset/text.hpp"

#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>

#include "nexusdata/core/error.hpp"

namespace nexusdata {

TextLineDataset::TextLineDataset(const std::string& path, TextOptions opt) {
    std::ifstream in(path);
    if (!in) {
        throw IOError("TextLineDataset: cannot open " + quote_path(path));
    }
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (opt.skip_empty && line.empty()) {
            continue;
        }
        if (opt.max_length > 0 && line.size() > opt.max_length) {
            line.resize(opt.max_length);
        }
        lines_.push_back(std::move(line));
    }
}

std::size_t TextLineDataset::size() const {
    return lines_.size();
}

Sample TextLineDataset::get(std::size_t index) const {
    if (index >= lines_.size()) {
        throw IndexError(format_index_error("TextLineDataset::get", index, lines_.size()));
    }
    Sample s;
    const auto& line = lines_[index];
    s.input = NDArray(Shape{line.size()}, DType::UInt8);
    if (!line.empty()) {
        std::memcpy(s.input.data(), line.data(), line.size());
    }
    s.label = NDArray(Shape{1}, DType::Int64);
    s.label.data<std::int64_t>()[0] = static_cast<std::int64_t>(index);
    return s;
}

struct TextLineIterable::It::Impl {
    std::ifstream in;
    TextOptions opt;
    std::string pending;
    bool has_pending = false;
    std::int64_t line_id = 0;
    bool eof = false;

    bool read_next() {
        if (eof) {
            return false;
        }
        std::string line;
        while (std::getline(in, line)) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            if (opt.skip_empty && line.empty()) {
                continue;
            }
            if (opt.max_length > 0 && line.size() > opt.max_length) {
                line.resize(opt.max_length);
            }
            pending = std::move(line);
            has_pending = true;
            return true;
        }
        eof = true;
        return false;
    }
};

TextLineIterable::It::It(std::string path, TextOptions opt) : impl_(std::make_unique<Impl>()) {
    impl_->opt = opt;
    impl_->in.open(path);
    if (!impl_->in) {
        throw IOError("TextLineIterable: cannot open " + quote_path(path));
    }
    impl_->read_next();
}

TextLineIterable::It::~It() = default;

bool TextLineIterable::It::has_next() const {
    return impl_->has_pending;
}

Sample TextLineIterable::It::next() {
    if (!impl_->has_pending) {
        throw IndexError("TextLineIterable: exhausted");
    }
    Sample s;
    s.input = NDArray(Shape{impl_->pending.size()}, DType::UInt8);
    if (!impl_->pending.empty()) {
        std::memcpy(s.input.data(), impl_->pending.data(), impl_->pending.size());
    }
    s.label = NDArray(Shape{1}, DType::Int64);
    s.label.data<std::int64_t>()[0] = impl_->line_id++;
    impl_->has_pending = false;
    impl_->read_next();
    return s;
}

TextLineIterable::TextLineIterable(std::string path, TextOptions opt)
    : path_(std::move(path)), opt_(opt) {}

std::unique_ptr<IterableDataset::Iterator> TextLineIterable::make_iterator() const {
    return std::make_unique<It>(path_, opt_);
}

std::vector<std::int64_t> WhitespaceHashTokenizer::encode(const std::string& text) const {
    std::vector<std::int64_t> ids;
    std::istringstream iss(text);
    std::string tok;
    while (iss >> tok) {
        std::uint64_t h = 14695981039346656037ULL;
        for (unsigned char c : tok) {
            h ^= c;
            h *= 1099511628211ULL;
        }
        ids.push_back(static_cast<std::int64_t>(h % static_cast<std::uint64_t>(vocab_)));
    }
    return ids;
}

std::string WhitespaceHashTokenizer::decode(const std::vector<std::int64_t>& ids) const {
    std::ostringstream oss;
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (i) {
            oss << ' ';
        }
        oss << "id" << ids[i];
    }
    return oss.str();
}

} // namespace nexusdata
