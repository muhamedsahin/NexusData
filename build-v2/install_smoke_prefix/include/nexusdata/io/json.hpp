#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace nexusdata::json {

/// Small JSON DOM for configuration-sized documents (schemas, dataset configs).
/// Bulk record parsing in JsonDataset does not build a DOM.
class Value {
public:
    enum class Type : std::uint8_t { Null, Bool, Int, Double, String, Array, Object };
    using Array = std::vector<Value>;
    using Object = std::vector<std::pair<std::string, Value>>; ///< insertion order kept

    Value() = default;
    Value(std::nullptr_t) {}
    Value(bool b) : v_(b) {}
    Value(int i) : v_(static_cast<std::int64_t>(i)) {}
    Value(std::int64_t i) : v_(i) {}
    Value(double d) : v_(d) {}
    Value(const char* s) : v_(std::string(s)) {}
    Value(std::string s) : v_(std::move(s)) {}
    Value(Array a) : v_(std::move(a)) {}
    Value(Object o) : v_(std::move(o)) {}

    [[nodiscard]] Type type() const noexcept { return static_cast<Type>(v_.index()); }
    [[nodiscard]] bool is_null() const noexcept { return type() == Type::Null; }
    [[nodiscard]] bool is_bool() const noexcept { return type() == Type::Bool; }
    [[nodiscard]] bool is_number() const noexcept {
        return type() == Type::Int || type() == Type::Double;
    }
    [[nodiscard]] bool is_string() const noexcept { return type() == Type::String; }
    [[nodiscard]] bool is_array() const noexcept { return type() == Type::Array; }
    [[nodiscard]] bool is_object() const noexcept { return type() == Type::Object; }

    /// Typed accessors throw InvalidArgumentError on a type mismatch.
    [[nodiscard]] bool as_bool() const;
    [[nodiscard]] double as_double() const;        ///< Int or Double
    [[nodiscard]] std::int64_t as_int() const;     ///< Int, or an integral Double
    [[nodiscard]] const std::string& as_string() const;
    [[nodiscard]] const Array& as_array() const;
    [[nodiscard]] Array& as_array();
    [[nodiscard]] const Object& as_object() const;
    [[nodiscard]] Object& as_object();

    /// Object member lookup; nullptr when absent or not an object.
    [[nodiscard]] const Value* find(std::string_view key) const noexcept;
    /// Dotted path lookup ("a.b[2].c"); nullptr when any step is absent.
    [[nodiscard]] const Value* at_path(std::string_view path) const;

    /// Convenience getters with a fallback when the key is absent.
    [[nodiscard]] std::string get_string(std::string_view key, std::string fallback) const;
    [[nodiscard]] double get_double(std::string_view key, double fallback) const;
    [[nodiscard]] std::int64_t get_int(std::string_view key, std::int64_t fallback) const;
    [[nodiscard]] bool get_bool(std::string_view key, bool fallback) const;

    /// Insert or replace an object member (a Null value becomes an empty object).
    /// Returns the member, not *this.
    Value& set(std::string key, Value value);
    /// Append to an array (a Null value becomes an empty array). Returns the element.
    Value& push_back(Value value);

    /// Compact serialisation (round-trips through parse()).
    [[nodiscard]] std::string dump() const;

    friend bool operator==(const Value& a, const Value& b) { return a.v_ == b.v_; }
    friend bool operator!=(const Value& a, const Value& b) { return !(a == b); }

private:
    std::variant<std::monostate, bool, std::int64_t, double, std::string, Array, Object> v_;
};

/// Parse one JSON document (RFC 8259). Throws IOError with the byte offset of the
/// first error. Integers that fit in int64 stay exact (Type::Int).
[[nodiscard]] Value parse(std::string_view text);

/// Escape @p s as a JSON string literal, including the quotes.
[[nodiscard]] std::string quote(std::string_view s);

} // namespace nexusdata::json
