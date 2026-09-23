#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace nexusdata {

struct RemoteOptions {
    /// Bytes per cached chunk. Range requests are aligned to this size.
    std::size_t chunk_size = 1 << 20;
    /// How many chunks to keep. 0 disables the cache (every read hits the network).
    std::size_t cache_chunks = 8;
    std::string access_key;
    std::string secret_key;
    std::string region = "us-east-1";
    /// Azure account used to expand az://container/blob. Ignored for other schemes.
    std::string account;
};

/// Inputs for AWS Signature Version 4 (header auth). Header names must already
/// be lowercase; values are trimmed by the signer. `payload_hash` is lowercase hex.
struct SigV4Request {
    std::string method = "GET";
    std::string canonical_uri = "/";
    std::string canonical_query;
    std::string host;
    std::string amz_date; ///< YYYYMMDDTHHMMSSZ
    std::string payload_hash;
    std::string region = "us-east-1";
    std::string service = "s3";
    std::string access_key;
    std::string secret_key;
    /// Extra signed headers besides host, x-amz-date and x-amz-content-sha256.
    std::vector<std::pair<std::string, std::string>> headers;
};

struct SigV4Result {
    std::string canonical_request;
    std::string string_to_sign;
    std::string signature;
    std::string authorization;
};

/// Pure function (no network). Used by s3:// reads and by tests against AWS vectors.
[[nodiscard]] SigV4Result sigv4_sign(const SigV4Request& request);

/// Random-access file over HTTP(S) range reads.
/// Schemes: http:// (cleartext sockets), https:// (WinHTTP on Windows),
/// s3://bucket/key, gs://bucket/key, az://container/blob (account in options).
/// Downloaded chunks sit in an LRU so a repeated range is not fetched again.
/// Not safe for concurrent read_at on the same instance.
class RemoteFile {
public:
    explicit RemoteFile(std::string uri, RemoteOptions opt = {});

    [[nodiscard]] const std::string& uri() const noexcept { return uri_; }
    [[nodiscard]] std::uint64_t size() const;
    void read_at(std::uint64_t offset, void* dst, std::size_t n) const;

    /// HTTP requests actually sent (cache hits do not increment).
    [[nodiscard]] std::uint64_t http_requests() const;

private:
    struct State;
    std::shared_ptr<State> state_;
    std::string uri_;
};

} // namespace nexusdata
