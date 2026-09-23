#pragma once

// Private JSON scanning primitives shared by the DOM parser (json_value.cpp) and the
// path-directed record extractor (json_dataset.cpp).

#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>

#include "nexusdata/core/error.hpp"

namespace nexusdata::json::detail {

struct Cursor {
    const char* p;
    const char* end;
    const char* begin;

    Cursor(const char* b, const char* e) : p(b), end(e), begin(b) {}

    [[noreturn]] void fail(const char* what) const {
        throw IOError(std::string("json: ") + what + " at byte " +
                      std::to_string(static_cast<std::size_t>(p - begin)));
    }

    void ws() noexcept {
        while (p < end && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) ++p;
    }
    [[nodiscard]] bool at_end() noexcept {
        ws();
        return p >= end;
    }
    [[nodiscard]] char peek() {
        ws();
        if (p >= end) fail("unexpected end of input");
        return *p;
    }
    void expect(char c) {
        if (peek() != c) {
            char msg[] = "expected 'x'";
            msg[10] = c;
            fail(msg);
        }
        ++p;
    }
    /// Consume @p c if it is next; returns whether it was.
    [[nodiscard]] bool eat(char c) {
        if (peek() == c) {
            ++p;
            return true;
        }
        return false;
    }
    void literal(const char* word, std::size_t n) {
        if (static_cast<std::size_t>(end - p) < n || std::memcmp(p, word, n) != 0) fail("invalid literal");
        p += n;
    }

    /// Skip a string body after the opening quote; returns true when it held escapes.
    bool skip_string_body() {
        bool escaped = false;
        for (;;) {
            // Jump to the next quote or backslash.
            const char* q = p;
            while (q < end && *q != '"' && *q != '\\') {
                if (static_cast<unsigned char>(*q) < 0x20) {
                    p = q;
                    fail("control character in string");
                }
                ++q;
            }
            if (q >= end) {
                p = q;
                fail("unterminated string");
            }
            if (*q == '"') {
                p = q + 1;
                return escaped;
            }
            escaped = true;
            p = q + 2; // backslash + escaped char (validated when decoded)
            if (p > end) fail("unterminated escape");
        }
    }

    /// Decode a string (opening quote not yet consumed) into @p out.
    void string(std::string& out) {
        expect('"');
        const char* start = p;
        if (!skip_string_body()) {
            out.assign(start, static_cast<std::size_t>(p - 1 - start));
            return;
        }
        out.clear();
        const char* s = start;
        const char* stop = p - 1;
        while (s < stop) {
            if (*s != '\\') {
                out.push_back(*s++);
                continue;
            }
            ++s;
            switch (*s++) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u': {
                    std::uint32_t cp = hex4(s, stop);
                    s += 4;
                    if (cp >= 0xd800 && cp <= 0xdbff) {
                        if (stop - s < 6 || s[0] != '\\' || s[1] != 'u') fail("unpaired surrogate");
                        const std::uint32_t lo = hex4(s + 2, stop);
                        if (lo < 0xdc00 || lo > 0xdfff) fail("invalid surrogate pair");
                        cp = 0x10000 + ((cp - 0xd800) << 10) + (lo - 0xdc00);
                        s += 6;
                    }
                    utf8(cp, out);
                    break;
                }
                default: fail("invalid escape");
            }
        }
    }

    std::uint32_t hex4(const char* s, const char* stop) const {
        if (stop - s < 4) fail("short \\u escape");
        std::uint32_t v = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = s[i];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= static_cast<std::uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f') v |= static_cast<std::uint32_t>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= static_cast<std::uint32_t>(c - 'A' + 10);
            else fail("invalid \\u escape");
        }
        return v;
    }

    static void utf8(std::uint32_t cp, std::string& out) {
        if (cp < 0x80) {
            out.push_back(static_cast<char>(cp));
        } else if (cp < 0x800) {
            out.push_back(static_cast<char>(0xc0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
        } else if (cp < 0x10000) {
            out.push_back(static_cast<char>(0xe0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
        } else {
            out.push_back(static_cast<char>(0xf0 | (cp >> 18)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3f)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
        }
    }

    /// Number token span, validated against the JSON grammar.
    std::string_view number_token() {
        ws();
        const char* s = p;
        const char* q = p;
        if (q < end && *q == '-') ++q;
        if (q >= end) fail("invalid number");
        if (*q == '0') {
            ++q;
        } else if (*q >= '1' && *q <= '9') {
            while (q < end && *q >= '0' && *q <= '9') ++q;
        } else {
            fail("invalid number");
        }
        if (q < end && *q == '.') {
            ++q;
            if (q >= end || *q < '0' || *q > '9') fail("invalid number");
            while (q < end && *q >= '0' && *q <= '9') ++q;
        }
        if (q < end && (*q == 'e' || *q == 'E')) {
            ++q;
            if (q < end && (*q == '+' || *q == '-')) ++q;
            if (q >= end || *q < '0' || *q > '9') fail("invalid number");
            while (q < end && *q >= '0' && *q <= '9') ++q;
        }
        p = q;
        return {s, static_cast<std::size_t>(q - s)};
    }

    static bool is_integral(std::string_view tok) noexcept {
        return tok.find_first_of(".eE") == std::string_view::npos;
    }

    static double to_double(std::string_view tok) {
        double v = 0;
        const auto r = std::from_chars(tok.data(), tok.data() + tok.size(), v);
        if (r.ec == std::errc::result_out_of_range) {
            // from_chars leaves v unset; strtod saturates (1e999 -> inf, 1e-999 -> 0).
            return std::strtod(std::string(tok).c_str(), nullptr);
        }
        return v;
    }

    /// Skip one value of any type without decoding it.
    void skip_value(int depth = 0) {
        if (depth > 512) fail("nesting too deep");
        switch (peek()) {
            case '"': ++p; skip_string_body(); return;
            case '{':
                ++p;
                if (eat('}')) return;
                do {
                    if (peek() != '"') fail("expected object key");
                    ++p;
                    skip_string_body();
                    expect(':');
                    skip_value(depth + 1);
                } while (eat(','));
                expect('}');
                return;
            case '[':
                ++p;
                if (eat(']')) return;
                do {
                    skip_value(depth + 1);
                } while (eat(','));
                expect(']');
                return;
            case 't': literal("true", 4); return;
            case 'f': literal("false", 5); return;
            case 'n': literal("null", 4); return;
            default: (void)number_token(); return;
        }
    }
};

} // namespace nexusdata::json::detail
