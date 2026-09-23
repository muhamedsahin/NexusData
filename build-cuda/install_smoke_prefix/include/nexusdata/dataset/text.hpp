#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "nexusdata/dataset/dataset.hpp"
#include "nexusdata/dataset/iterable.hpp"

namespace nexusdata {

struct TextOptions {
    /// Max characters kept per line (0 = unlimited).
    std::size_t max_length = 0;
    bool skip_empty = true;
};

/// Line-based text file. Sample.input = UInt8 UTF-8 bytes [L], label = line index.
class TextLineDataset : public Dataset {
public:
    explicit TextLineDataset(const std::string& path, TextOptions opt = {});

    [[nodiscard]] std::size_t size() const override;
    [[nodiscard]] Sample get(std::size_t index) const override;

    [[nodiscard]] const std::vector<std::string>& lines() const noexcept { return lines_; }

private:
    std::vector<std::string> lines_;
};

/// Streaming text lines (does not load entire file into RAM).
class TextLineIterable : public IterableDataset {
public:
    explicit TextLineIterable(std::string path, TextOptions opt = {});

    class It : public Iterator {
    public:
        It(std::string path, TextOptions opt);
        ~It() override;
        [[nodiscard]] bool has_next() const override;
        [[nodiscard]] Sample next() override;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

    [[nodiscard]] std::unique_ptr<Iterator> make_iterator() const override;

private:
    std::string path_;
    TextOptions opt_;
};

/// Abstract tokenizer interface (no model weights).
class Tokenizer {
public:
    virtual ~Tokenizer() = default;
    [[nodiscard]] virtual std::vector<std::int64_t> encode(const std::string& text) const = 0;
    [[nodiscard]] virtual std::string decode(const std::vector<std::int64_t>& ids) const = 0;
};

/// Whitespace split → hash ids (demo tokenizer for tests / bucketing).
class WhitespaceHashTokenizer : public Tokenizer {
public:
    explicit WhitespaceHashTokenizer(std::int64_t vocab_size = 10000) : vocab_(vocab_size) {}
    [[nodiscard]] std::vector<std::int64_t> encode(const std::string& text) const override;
    [[nodiscard]] std::string decode(const std::vector<std::int64_t>& ids) const override;

private:
    std::int64_t vocab_ = 10000;
};

} // namespace nexusdata
