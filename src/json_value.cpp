#include "nexusdata/io/json.hpp"

#include <charconv>
#include <cmath>
#include <limits>

#include "json_detail.hpp"
#include "nexusdata/core/error.hpp"

namespace nexusdata::json {

namespace {

[[noreturn]] void type_error(const char* want) {
    throw InvalidArgumentError(std::string("json::Value: not ") + want);
}

Value parse_value(detail::Cursor& c, int depth) {
    if (depth > 512) c.fail("nesting too deep");
    switch (c.peek()) {
        case '"': {
            std::string s;
            c.string(s);
            return Value(std::move(s));
        }
        case '{': {
            ++c.p;
            Value::Object obj;
            if (c.eat('}')) return Value(std::move(obj));
            do {
                std::string key;
                if (c.peek() != '"') c.fail("expected object key");
                c.string(key);
                c.expect(':');
                obj.emplace_back(std::move(key), parse_value(c, depth + 1));
            } while (c.eat(','));
            c.expect('}');
            return Value(std::move(obj));
        }
        case '[': {
            ++c.p;
            Value::Array arr;
            if (c.eat(']')) return Value(std::move(arr));
            do {
                arr.push_back(parse_value(c, depth + 1));
            } while (c.eat(','));
            c.expect(']');
            return Value(std::move(arr));
        }
        case 't': c.literal("true", 4); return Value(true);
        case 'f': c.literal("false", 5); return Value(false);
        case 'n': c.literal("null", 4); return Value();
        default: {
            const std::string_view tok = c.number_token();
            if (detail::Cursor::is_integral(tok)) {
                std::int64_t i = 0;
                const auto r = std::from_chars(tok.data(), tok.data() + tok.size(), i);
                if (r.ec == std::errc()) return Value(i);
            }
            return Value(detail::Cursor::to_double(tok));
        }
    }
}

void dump_to(const Value& v, std::string& out) {
    switch (v.type()) {
        case Value::Type::Null: out += "null"; return;
        case Value::Type::Bool: out += v.as_bool() ? "true" : "false"; return;
        case Value::Type::Int: out += std::to_string(v.as_int()); return;
        case Value::Type::Double: {
            const double d = v.as_double();
            if (!std::isfinite(d)) {
                out += "null"; // JSON has no inf / nan
                return;
            }
            char buf[32];
            const auto r = std::to_chars(buf, buf + sizeof(buf), d);
            std::string_view s(buf, static_cast<std::size_t>(r.ptr - buf));
            out += s;
            if (s.find_first_of(".eEn") == std::string_view::npos) out += ".0"; // stay a Double
            return;
        }
        case Value::Type::String: out += quote(v.as_string()); return;
        case Value::Type::Array: {
            out += '[';
            bool first = true;
            for (const Value& e : v.as_array()) {
                if (!first) out += ',';
                first = false;
                dump_to(e, out);
            }
            out += ']';
            return;
        }
        case Value::Type::Object: {
            out += '{';
            bool first = true;
            for (const auto& [k, e] : v.as_object()) {
                if (!first) out += ',';
                first = false;
                out += quote(k);
                out += ':';
                dump_to(e, out);
            }
            out += '}';
            return;
        }
    }
}

} // namespace

bool Value::as_bool() const {
    if (const bool* b = std::get_if<bool>(&v_)) return *b;
    type_error("a bool");
}

double Value::as_double() const {
    if (const double* d = std::get_if<double>(&v_)) return *d;
    if (const std::int64_t* i = std::get_if<std::int64_t>(&v_)) return static_cast<double>(*i);
    type_error("a number");
}

std::int64_t Value::as_int() const {
    if (const std::int64_t* i = std::get_if<std::int64_t>(&v_)) return *i;
    if (const double* d = std::get_if<double>(&v_)) {
        if (std::trunc(*d) == *d && std::abs(*d) < 9.2e18) return static_cast<std::int64_t>(*d);
    }
    type_error("an integer");
}

const std::string& Value::as_string() const {
    if (const std::string* s = std::get_if<std::string>(&v_)) return *s;
    type_error("a string");
}

const Value::Array& Value::as_array() const {
    if (const Array* a = std::get_if<Array>(&v_)) return *a;
    type_error("an array");
}

Value::Array& Value::as_array() {
    if (Array* a = std::get_if<Array>(&v_)) return *a;
    type_error("an array");
}

const Value::Object& Value::as_object() const {
    if (const Object* o = std::get_if<Object>(&v_)) return *o;
    type_error("an object");
}

Value::Object& Value::as_object() {
    if (Object* o = std::get_if<Object>(&v_)) return *o;
    type_error("an object");
}

const Value* Value::find(std::string_view key) const noexcept {
    const Object* o = std::get_if<Object>(&v_);
    if (o == nullptr) return nullptr;
    for (const auto& [k, v] : *o) {
        if (k == key) return &v;
    }
    return nullptr;
}

const Value* Value::at_path(std::string_view path) const {
    const Value* cur = this;
    std::size_t i = 0;
    while (cur != nullptr && i < path.size()) {
        if (path[i] == '.') {
            ++i;
            continue;
        }
        if (path[i] == '[') {
            const std::size_t close = path.find(']', i);
            if (close == std::string_view::npos) {
                throw InvalidArgumentError("json path: missing ']' in \"" + std::string(path) + "\"");
            }
            std::size_t idx = 0;
            const auto r = std::from_chars(path.data() + i + 1, path.data() + close, idx);
            if (r.ec != std::errc() || r.ptr != path.data() + close) {
                throw InvalidArgumentError("json path: bad index in \"" + std::string(path) + "\"");
            }
            const Array* a = std::get_if<Array>(&cur->v_);
            cur = (a != nullptr && idx < a->size()) ? &(*a)[idx] : nullptr;
            i = close + 1;
            continue;
        }
        const std::size_t stop = path.find_first_of(".[", i);
        const std::string_view key = path.substr(i, stop == std::string_view::npos ? std::string_view::npos : stop - i);
        cur = cur->find(key);
        i = stop == std::string_view::npos ? path.size() : stop;
    }
    return cur;
}

std::string Value::get_string(std::string_view key, std::string fallback) const {
    const Value* v = find(key);
    return v != nullptr ? v->as_string() : std::move(fallback);
}

double Value::get_double(std::string_view key, double fallback) const {
    const Value* v = find(key);
    return v != nullptr ? v->as_double() : fallback;
}

std::int64_t Value::get_int(std::string_view key, std::int64_t fallback) const {
    const Value* v = find(key);
    return v != nullptr ? v->as_int() : fallback;
}

bool Value::get_bool(std::string_view key, bool fallback) const {
    const Value* v = find(key);
    return v != nullptr ? v->as_bool() : fallback;
}

Value& Value::set(std::string key, Value value) {
    if (is_null()) v_ = Object{};
    Object& o = as_object();
    for (auto& [k, v] : o) {
        if (k == key) {
            v = std::move(value);
            return v;
        }
    }
    o.emplace_back(std::move(key), std::move(value));
    return o.back().second;
}

Value& Value::push_back(Value value) {
    if (is_null()) v_ = Array{};
    Array& a = as_array();
    a.push_back(std::move(value));
    return a.back();
}

std::string Value::dump() const {
    std::string out;
    dump_to(*this, out);
    return out;
}

Value parse(std::string_view text) {
    detail::Cursor c(text.data(), text.data() + text.size());
    Value v = parse_value(c, 0);
    if (!c.at_end()) c.fail("trailing characters after document");
    return v;
}

std::string quote(std::string_view s) {
    std::string out;
    out.reserve(s.size() + 2);
    out += '"';
    for (const char ch : s) {
        const auto u = static_cast<unsigned char>(ch);
        switch (ch) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            default:
                if (u < 0x20) {
                    static const char* hex = "0123456789abcdef";
                    out += "\\u00";
                    out += hex[u >> 4];
                    out += hex[u & 15];
                } else {
                    out += ch;
                }
        }
    }
    out += '"';
    return out;
}

} // namespace nexusdata::json
