#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace nexusdata {

/// Base exception for all NexusData errors.
/// Thread-safety: immutable after construction; safe to throw across threads.
class Error : public std::runtime_error {
public:
    explicit Error(std::string message)
        : std::runtime_error(std::move(message)) {}
};

/// Invalid argument / misuse of the public API.
class InvalidArgumentError : public Error {
public:
    explicit InvalidArgumentError(std::string message)
        : Error(std::move(message)) {}
};

/// Index out of valid range.
class IndexError : public Error {
public:
    explicit IndexError(std::string message)
        : Error(std::move(message)) {}
};

/// Shape / dtype mismatch.
class ShapeError : public Error {
public:
    explicit ShapeError(std::string message)
        : Error(std::move(message)) {}
};

/// I/O or parse failure (CSV, files, etc.).
class IOError : public Error {
public:
    explicit IOError(std::string message)
        : Error(std::move(message)) {}
};

// --- Message helpers (v0.7 hardening) ---------------------------------------

/// Quote a filesystem path for error messages.
[[nodiscard]] inline std::string quote_path(const std::string& path) {
    return "\"" + path + "\"";
}

/// "Component: index I out of range [0, N)"
[[nodiscard]] inline std::string format_index_error(const char* what,
                                                    std::size_t index,
                                                    std::size_t size) {
    return std::string(what) + ": index " + std::to_string(index) +
           " out of range [0, " + std::to_string(size) + ")";
}

/// "path:line:col message" (omit col when 0; omit both when line == 0).
[[nodiscard]] inline std::string format_loc(const std::string& path,
                                            std::size_t line,
                                            std::size_t col,
                                            const std::string& message) {
    std::string out = path;
    if (line > 0) {
        out += ":" + std::to_string(line);
        if (col > 0) {
            out += ":" + std::to_string(col);
        }
    }
    out += " " + message;
    return out;
}

} // namespace nexusdata
