#include "nexusdata/io/remote_file.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <list>
#include <unordered_map>
#include <utility>
#include <vector>

#include "nexusdata/core/error.hpp"

#ifndef NOMINMAX
#define NOMINMAX
#endif

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
#include <winhttp.h>
using nd_socket = SOCKET;
constexpr nd_socket kBadSocket = INVALID_SOCKET;
#else
#include <netdb.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
using nd_socket = int;
constexpr nd_socket kBadSocket = -1;
#endif

namespace nexusdata {
namespace {

void sha256(const std::uint8_t* data, std::size_t len, std::uint8_t out[32]);

std::uint32_t rotr(std::uint32_t x, std::uint32_t n) { return (x >> n) | (x << (32 - n)); }

void sha256(const std::uint8_t* data, std::size_t len, std::uint8_t out[32]) {
    static const std::uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    std::uint32_t h0 = 0x6a09e667, h1 = 0xbb67ae85, h2 = 0x3c6ef372, h3 = 0xa54ff53a;
    std::uint32_t h4 = 0x510e527f, h5 = 0x9b05688c, h6 = 0x1f83d9ab, h7 = 0x5be0cd19;
    std::vector<std::uint8_t> msg(data, data + len);
    msg.push_back(0x80);
    while ((msg.size() % 64) != 56) msg.push_back(0);
    const std::uint64_t bits = static_cast<std::uint64_t>(len) * 8;
    for (int i = 7; i >= 0; --i) msg.push_back(static_cast<std::uint8_t>((bits >> (i * 8)) & 0xff));
    for (std::size_t off = 0; off < msg.size(); off += 64) {
        std::uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (static_cast<std::uint32_t>(msg[off + i * 4]) << 24) |
                   (static_cast<std::uint32_t>(msg[off + i * 4 + 1]) << 16) |
                   (static_cast<std::uint32_t>(msg[off + i * 4 + 2]) << 8) |
                   static_cast<std::uint32_t>(msg[off + i * 4 + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        std::uint32_t a = h0, b = h1, c = h2, d = h3, e = h4, f = h5, g = h6, h = h7;
        for (int i = 0; i < 64; ++i) {
            const std::uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const std::uint32_t ch = (e & f) ^ ((~e) & g);
            const std::uint32_t t1 = h + S1 + ch + K[i] + w[i];
            const std::uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t t2 = S0 + maj;
            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        h0 += a;
        h1 += b;
        h2 += c;
        h3 += d;
        h4 += e;
        h5 += f;
        h6 += g;
        h7 += h;
    }
    const std::uint32_t hs[8] = {h0, h1, h2, h3, h4, h5, h6, h7};
    for (int i = 0; i < 8; ++i) {
        out[i * 4] = static_cast<std::uint8_t>(hs[i] >> 24);
        out[i * 4 + 1] = static_cast<std::uint8_t>(hs[i] >> 16);
        out[i * 4 + 2] = static_cast<std::uint8_t>(hs[i] >> 8);
        out[i * 4 + 3] = static_cast<std::uint8_t>(hs[i]);
    }
}

void hmac_sha256(const std::uint8_t* key, std::size_t key_len, const std::uint8_t* msg, std::size_t msg_len,
                 std::uint8_t out[32]) {
    std::uint8_t k[64] = {};
    if (key_len > 64) {
        sha256(key, key_len, k);
    } else {
        std::memcpy(k, key, key_len);
    }
    std::uint8_t ipad[64], opad[64];
    for (int i = 0; i < 64; ++i) {
        ipad[i] = static_cast<std::uint8_t>(k[i] ^ 0x36);
        opad[i] = static_cast<std::uint8_t>(k[i] ^ 0x5c);
    }
    std::vector<std::uint8_t> inner(64 + msg_len);
    std::memcpy(inner.data(), ipad, 64);
    if (msg_len) std::memcpy(inner.data() + 64, msg, msg_len);
    std::uint8_t ih[32];
    sha256(inner.data(), inner.size(), ih);
    std::uint8_t outer[96];
    std::memcpy(outer, opad, 64);
    std::memcpy(outer + 64, ih, 32);
    sha256(outer, 96, out);
}

std::string hex32(const std::uint8_t b[32]) {
    static const char* kHex = "0123456789abcdef";
    std::string s(64, '0');
    for (int i = 0; i < 32; ++i) {
        s[i * 2] = kHex[b[i] >> 4];
        s[i * 2 + 1] = kHex[b[i] & 0xf];
    }
    return s;
}

std::string trim(std::string s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
    return s;
}

std::string lower_copy(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

struct Endpoint {
    bool tls = false;
    std::string host;
    std::string port = "80";
    std::string path = "/";
    std::string query;
    std::string service;
    bool sign = false;
};

Endpoint resolve_uri(const std::string& uri, const RemoteOptions& opt) {
    const auto colon = uri.find(':');
    if (colon == std::string::npos || colon == 0) {
        throw InvalidArgumentError("RemoteFile: URI needs a scheme");
    }
    const std::string scheme = lower_copy(uri.substr(0, colon));
    std::string rest = uri.substr(colon + 1);
    Endpoint ep;
    auto split_host_path = [&](std::string body) {
        if (body.rfind("//", 0) == 0) body.erase(0, 2);
        const auto slash = body.find('/');
        ep.host = slash == std::string::npos ? body : body.substr(0, slash);
        ep.path = slash == std::string::npos ? "/" : body.substr(slash);
        const auto q = ep.path.find('?');
        if (q != std::string::npos) {
            ep.query = ep.path.substr(q + 1);
            ep.path.erase(q);
        }
        if (ep.path.empty()) ep.path = "/";
    };
    if (scheme == "http" || scheme == "https") {
        ep.tls = scheme == "https";
        ep.port = ep.tls ? "443" : "80";
        if (rest.rfind("//", 0) != 0) {
            throw InvalidArgumentError("RemoteFile: expected " + scheme + "://");
        }
        split_host_path(rest);
        const auto c = ep.host.rfind(':');
        if (c != std::string::npos && ep.host.find(']') == std::string::npos) {
            ep.port = ep.host.substr(c + 1);
            ep.host.erase(c);
        }
    } else if (scheme == "s3") {
        ep.tls = true;
        ep.port = "443";
        ep.service = "s3";
        ep.sign = !opt.access_key.empty();
        split_host_path(rest);
        const std::string bucket = ep.host;
        const std::string key = ep.path;
        ep.host = bucket + ".s3." + opt.region + ".amazonaws.com";
        ep.path = key;
    } else if (scheme == "gs") {
        ep.tls = true;
        ep.port = "443";
        split_host_path(rest);
        ep.path = "/" + ep.host + (ep.path == "/" ? "" : ep.path);
        ep.host = "storage.googleapis.com";
    } else if (scheme == "az") {
        if (opt.account.empty()) {
            throw InvalidArgumentError("RemoteFile: az:// needs RemoteOptions::account");
        }
        ep.tls = true;
        ep.port = "443";
        split_host_path(rest);
        ep.host = opt.account + ".blob.core.windows.net";
    } else {
        throw InvalidArgumentError("RemoteFile: unsupported scheme '" + scheme + "'");
    }
    if (ep.host.empty()) {
        throw InvalidArgumentError("RemoteFile: missing host");
    }
    return ep;
}

struct HttpResult {
    int status = 0;
    std::vector<std::uint8_t> body;
    std::uint64_t total = 0;
    bool have_total = false;
};

std::string header_value(const std::string& headers, const std::string& key) {
    const std::string needle = lower_copy(key);
    std::size_t i = 0;
    while (i < headers.size()) {
        auto eol = headers.find("\r\n", i);
        if (eol == std::string::npos) eol = headers.size();
        std::string line = headers.substr(i, eol - i);
        i = eol + 2;
        const auto c = line.find(':');
        if (c == std::string::npos) continue;
        if (lower_copy(line.substr(0, c)) == needle) return trim(line.substr(c + 1));
    }
    return {};
}

void note_length(HttpResult& r, const std::string& headers) {
    const std::string cr = header_value(headers, "content-range");
    if (!cr.empty()) {
        const auto slash = cr.rfind('/');
        if (slash != std::string::npos && slash + 1 < cr.size() && cr[slash + 1] != '*') {
            r.total = std::strtoull(cr.c_str() + slash + 1, nullptr, 10);
            r.have_total = true;
        }
    }
    if (!r.have_total) {
        const std::string cl = header_value(headers, "content-length");
        if (!cl.empty() && r.status == 200) {
            r.total = std::strtoull(cl.c_str(), nullptr, 10);
            r.have_total = true;
        }
    }
}

#if defined(_WIN32)
void winsock_once() {
    static const int ok = [] {
        WSADATA w;
        return WSAStartup(MAKEWORD(2, 2), &w) == 0 ? 1 : 0;
    }();
    if (!ok) throw IOError("RemoteFile: WSAStartup failed");
}
#endif

void close_socket(nd_socket s) {
    if (s == kBadSocket) return;
#if defined(_WIN32)
    closesocket(s);
#else
    ::close(s);
#endif
}

nd_socket dial(const std::string& host, const std::string& port) {
#if defined(_WIN32)
    winsock_once();
#endif
    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    if (getaddrinfo(host.c_str(), port.c_str(), &hints, &res) != 0 || !res) {
        throw IOError("RemoteFile: cannot resolve " + host);
    }
    nd_socket s = kBadSocket;
    for (addrinfo* p = res; p; p = p->ai_next) {
        s = ::socket(p->ai_family, p->ai_socktype, p->ai_protocol);
        if (s == kBadSocket) continue;
        if (::connect(s, p->ai_addr, static_cast<int>(p->ai_addrlen)) == 0) break;
        close_socket(s);
        s = kBadSocket;
    }
    freeaddrinfo(res);
    if (s == kBadSocket) throw IOError("RemoteFile: connect failed: " + host + ":" + port);
    return s;
}

void send_all(nd_socket s, const std::string& bytes) {
    std::size_t off = 0;
    while (off < bytes.size()) {
#if defined(_WIN32)
        const int n = ::send(s, bytes.data() + off, static_cast<int>(bytes.size() - off), 0);
#else
        const int n = static_cast<int>(::send(s, bytes.data() + off, bytes.size() - off, 0));
#endif
        if (n <= 0) throw IOError("RemoteFile: send failed");
        off += static_cast<std::size_t>(n);
    }
}

std::string recv_some(nd_socket s) {
    char buf[8192];
#if defined(_WIN32)
    const int n = ::recv(s, buf, sizeof(buf), 0);
#else
    const int n = static_cast<int>(::recv(s, buf, sizeof(buf), 0));
#endif
    if (n < 0) throw IOError("RemoteFile: recv failed");
    return std::string(buf, buf + (n > 0 ? n : 0));
}

HttpResult http_socket(const Endpoint& ep, const std::string& extra_headers, std::uint64_t begin,
                       std::uint64_t end_inclusive, bool have_range) {
    nd_socket s = dial(ep.host, ep.port);
    std::string req = "GET " + ep.path;
    if (!ep.query.empty()) req += "?" + ep.query;
    req += " HTTP/1.1\r\nHost: " + ep.host + "\r\nConnection: close\r\nAccept-Encoding: identity\r\n";
    if (have_range) {
        req += "Range: bytes=" + std::to_string(begin) + "-" + std::to_string(end_inclusive) + "\r\n";
    }
    req += extra_headers;
    req += "\r\n";
    send_all(s, req);
    std::string raw;
    for (;;) {
        const std::string part = recv_some(s);
        if (part.empty()) break;
        raw += part;
    }
    close_socket(s);
    const auto sep = raw.find("\r\n\r\n");
    if (sep == std::string::npos || raw.size() < 12) {
        throw IOError("RemoteFile: truncated HTTP response");
    }
    HttpResult r;
    const std::string headers = raw.substr(0, sep);
    const auto sp = headers.find(' ');
    if (sp == std::string::npos) throw IOError("RemoteFile: bad status line");
    r.status = std::atoi(headers.c_str() + sp + 1);
    note_length(r, headers);
    std::string body = raw.substr(sep + 4);
    const std::string cl = header_value(headers, "content-length");
    if (!cl.empty()) {
        const auto need = std::strtoull(cl.c_str(), nullptr, 10);
        if (body.size() < need) {
            throw IOError("RemoteFile: short HTTP body");
        }
        body.resize(static_cast<std::size_t>(need));
    }
    r.body.assign(body.begin(), body.end());
    if (r.status == 200 && !r.have_total) {
        r.total = r.body.size();
        r.have_total = true;
    }
    return r;
}

#if defined(_WIN32)
std::wstring widen(const std::string& s) {
    return std::wstring(s.begin(), s.end());
}

HttpResult http_winhttp(const Endpoint& ep, const std::wstring& headers, std::uint64_t begin,
                        std::uint64_t end_inclusive, bool have_range) {
    HINTERNET ses = WinHttpOpen(L"nexusdata", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY, WINHTTP_NO_PROXY_NAME,
                                WINHTTP_NO_PROXY_BYPASS, 0);
    if (!ses) throw IOError("RemoteFile: WinHttpOpen failed");
    HINTERNET con = WinHttpConnect(ses, widen(ep.host).c_str(), static_cast<INTERNET_PORT>(std::atoi(ep.port.c_str())), 0);
    if (!con) {
        WinHttpCloseHandle(ses);
        throw IOError("RemoteFile: WinHttpConnect failed");
    }
    const DWORD flags = ep.tls ? WINHTTP_FLAG_SECURE : 0;
    std::string path = ep.path;
    if (!ep.query.empty()) path += "?" + ep.query;
    HINTERNET req = WinHttpOpenRequest(con, L"GET", widen(path).c_str(), nullptr, WINHTTP_NO_REFERER,
                                       WINHTTP_DEFAULT_ACCEPT_TYPES, flags);
    if (!req) {
        WinHttpCloseHandle(con);
        WinHttpCloseHandle(ses);
        throw IOError("RemoteFile: WinHttpOpenRequest failed");
    }
    std::wstring hdr = headers;
    if (have_range) {
        hdr += L"Range: bytes=" + std::to_wstring(begin) + L"-" + std::to_wstring(end_inclusive) + L"\r\n";
    }
    if (!WinHttpSendRequest(req, hdr.empty() ? WINHTTP_NO_ADDITIONAL_HEADERS : hdr.c_str(),
                            hdr.empty() ? 0 : static_cast<DWORD>(-1), WINHTTP_NO_REQUEST_DATA, 0, 0, 0) ||
        !WinHttpReceiveResponse(req, nullptr)) {
        WinHttpCloseHandle(req);
        WinHttpCloseHandle(con);
        WinHttpCloseHandle(ses);
        throw IOError("RemoteFile: HTTPS request failed");
    }
    DWORD status = 0, slen = sizeof(status);
    WinHttpQueryHeaders(req, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_HEADER_NAME_BY_INDEX,
                        &status, &slen, WINHTTP_NO_HEADER_INDEX);
    wchar_t cr[128] = {};
    DWORD crlen = sizeof(cr);
    HttpResult r;
    r.status = static_cast<int>(status);
    if (WinHttpQueryHeaders(req, WINHTTP_QUERY_CONTENT_RANGE, WINHTTP_HEADER_NAME_BY_INDEX, cr, &crlen,
                            WINHTTP_NO_HEADER_INDEX)) {
        std::wstring wcr(cr);
        const auto slash = wcr.rfind(L'/');
        if (slash != std::wstring::npos) {
            r.total = std::wcstoull(wcr.c_str() + slash + 1, nullptr, 10);
            r.have_total = true;
        }
    }
    for (;;) {
        DWORD avail = 0;
        if (!WinHttpQueryDataAvailable(req, &avail)) break;
        if (avail == 0) break;
        const std::size_t at = r.body.size();
        r.body.resize(at + avail);
        DWORD read = 0;
        if (!WinHttpReadData(req, r.body.data() + at, avail, &read)) {
            WinHttpCloseHandle(req);
            WinHttpCloseHandle(con);
            WinHttpCloseHandle(ses);
            throw IOError("RemoteFile: WinHttpReadData failed");
        }
        r.body.resize(at + read);
    }
    WinHttpCloseHandle(req);
    WinHttpCloseHandle(con);
    WinHttpCloseHandle(ses);
    if (r.status == 200 && !r.have_total) {
        r.total = r.body.size();
        r.have_total = true;
    }
    return r;
}
#endif

HttpResult exchange(const Endpoint& ep, const RemoteOptions& opt, std::uint64_t begin, std::uint64_t end_inclusive,
                    bool have_range) {
    std::string extra;
    if (ep.sign) {
        SigV4Request sig;
        sig.method = "GET";
        sig.canonical_uri = ep.path.empty() ? "/" : ep.path;
        sig.canonical_query = ep.query;
        sig.host = ep.host;
        // Date is fixed only when the caller is a unit test of sigv4_sign. Live reads use a UTC stamp.
        // A stable stamp is not required for correctness of the server check; build it from the system clock.
        const std::time_t now = std::time(nullptr);
        std::tm tm{};
#if defined(_WIN32)
        gmtime_s(&tm, &now);
#else
        gmtime_r(&now, &tm);
#endif
        char stamp[32];
        std::strftime(stamp, sizeof(stamp), "%Y%m%dT%H%M%SZ", &tm);
        sig.amz_date = stamp;
        std::uint8_t empty_hash[32];
        sha256(nullptr, 0, empty_hash);
        sig.payload_hash = hex32(empty_hash);
        sig.region = opt.region;
        sig.service = ep.service.empty() ? "s3" : ep.service;
        sig.access_key = opt.access_key;
        sig.secret_key = opt.secret_key;
        if (have_range) {
            sig.headers.emplace_back("range", "bytes=" + std::to_string(begin) + "-" + std::to_string(end_inclusive));
        }
        const SigV4Result signed_req = sigv4_sign(sig);
        extra += "x-amz-date: " + sig.amz_date + "\r\n";
        extra += "x-amz-content-sha256: " + sig.payload_hash + "\r\n";
        extra += "Authorization: " + signed_req.authorization + "\r\n";
    }
#if defined(_WIN32)
    if (ep.tls) {
        return http_winhttp(ep, std::wstring(extra.begin(), extra.end()), begin, end_inclusive, have_range);
    }
#else
    if (ep.tls) {
        throw IOError("RemoteFile: https requires WinHTTP (Windows) or a future libcurl build");
    }
#endif
    return http_socket(ep, extra, begin, end_inclusive, have_range);
}

} // namespace

SigV4Result sigv4_sign(const SigV4Request& request) {
    if (request.access_key.empty() || request.secret_key.empty() || request.amz_date.size() < 8) {
        throw InvalidArgumentError("sigv4_sign: access key, secret and amz_date are required");
    }
    std::vector<std::pair<std::string, std::string>> hs = request.headers;
    hs.emplace_back("host", request.host);
    hs.emplace_back("x-amz-content-sha256", request.payload_hash);
    hs.emplace_back("x-amz-date", request.amz_date);
    for (auto& h : hs) {
        h.first = lower_copy(h.first);
        h.second = trim(h.second);
    }
    std::sort(hs.begin(), hs.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    std::string signed_headers;
    std::string canonical_headers;
    for (std::size_t i = 0; i < hs.size(); ++i) {
        if (i) signed_headers += ";";
        signed_headers += hs[i].first;
        canonical_headers += hs[i].first + ":" + hs[i].second + "\n";
    }
    SigV4Result out;
    out.canonical_request = request.method + "\n" + (request.canonical_uri.empty() ? "/" : request.canonical_uri) +
                            "\n" + request.canonical_query + "\n" + canonical_headers + "\n" + signed_headers + "\n" +
                            request.payload_hash;
    std::uint8_t creq[32];
    sha256(reinterpret_cast<const std::uint8_t*>(out.canonical_request.data()), out.canonical_request.size(), creq);
    const std::string datestamp = request.amz_date.substr(0, 8);
    const std::string scope = datestamp + "/" + request.region + "/" + request.service + "/aws4_request";
    out.string_to_sign = "AWS4-HMAC-SHA256\n" + request.amz_date + "\n" + scope + "\n" + hex32(creq);
    const std::string seed = "AWS4" + request.secret_key;
    std::uint8_t kdate[32], kregion[32], kservice[32], ksigning[32], sig[32];
    hmac_sha256(reinterpret_cast<const std::uint8_t*>(seed.data()), seed.size(),
                reinterpret_cast<const std::uint8_t*>(datestamp.data()), datestamp.size(), kdate);
    hmac_sha256(kdate, 32, reinterpret_cast<const std::uint8_t*>(request.region.data()), request.region.size(),
                kregion);
    hmac_sha256(kregion, 32, reinterpret_cast<const std::uint8_t*>(request.service.data()), request.service.size(),
                kservice);
    const char* aws4 = "aws4_request";
    hmac_sha256(kservice, 32, reinterpret_cast<const std::uint8_t*>(aws4), 12, ksigning);
    hmac_sha256(ksigning, 32, reinterpret_cast<const std::uint8_t*>(out.string_to_sign.data()),
                out.string_to_sign.size(), sig);
    out.signature = hex32(sig);
    out.authorization = "AWS4-HMAC-SHA256 Credential=" + request.access_key + "/" + scope +
                        ", SignedHeaders=" + signed_headers + ", Signature=" + out.signature;
    return out;
}

struct RemoteFile::State {
    Endpoint ep;
    RemoteOptions opt;
    mutable std::uint64_t file_size = 0;
    mutable bool know_size = false;
    mutable std::uint64_t requests = 0;
    mutable std::vector<std::uint8_t> whole;
    mutable bool have_whole = false;
    struct Chunk {
        std::uint64_t index = 0;
        std::vector<std::uint8_t> data;
    };
    mutable std::list<Chunk> lru;
    mutable std::unordered_map<std::uint64_t, std::list<Chunk>::iterator> index;

    HttpResult fetch(std::uint64_t begin, std::uint64_t end_inclusive, bool have_range) const {
        ++requests;
        return exchange(ep, opt, begin, end_inclusive, have_range);
    }

    void ensure_size() const {
        if (know_size) return;
        HttpResult r = fetch(0, 0, true);
        if (r.status != 200 && r.status != 206) {
            throw IOError("RemoteFile: GET failed, status " + std::to_string(r.status));
        }
        if (r.status == 200) {
            whole = std::move(r.body);
            have_whole = true;
            file_size = whole.size();
            know_size = true;
            return;
        }
        if (!r.have_total) throw IOError("RemoteFile: response has no length");
        file_size = r.total;
        know_size = true;
    }

    void remember(std::uint64_t chunk_index, std::vector<std::uint8_t> data) const {
        if (opt.cache_chunks == 0) return;
        auto it = index.find(chunk_index);
        if (it != index.end()) {
            lru.splice(lru.begin(), lru, it->second);
            it->second->data = std::move(data);
            return;
        }
        lru.push_front(Chunk{chunk_index, std::move(data)});
        index[chunk_index] = lru.begin();
        while (lru.size() > opt.cache_chunks) {
            index.erase(lru.back().index);
            lru.pop_back();
        }
    }

    const std::vector<std::uint8_t>& chunk(std::uint64_t chunk_index) const {
        auto it = index.find(chunk_index);
        if (it != index.end()) {
            lru.splice(lru.begin(), lru, it->second);
            return it->second->data;
        }
        const std::uint64_t cs = opt.chunk_size == 0 ? 1 : opt.chunk_size;
        const std::uint64_t begin = chunk_index * cs;
        std::uint64_t end = begin + cs - 1;
        if (end >= file_size) end = file_size - 1;
        HttpResult r = fetch(begin, end, true);
        if (r.status != 206 && r.status != 200) {
            throw IOError("RemoteFile: range GET failed, status " + std::to_string(r.status));
        }
        if (r.status == 200) {
            whole = std::move(r.body);
            have_whole = true;
            file_size = whole.size();
            know_size = true;
            uncached.clear();
            return uncached;
        }
        if (opt.cache_chunks == 0) {
            uncached = std::move(r.body);
            return uncached;
        }
        remember(chunk_index, std::move(r.body));
        return index.find(chunk_index)->second->data;
    }

    mutable std::vector<std::uint8_t> uncached;
};

RemoteFile::RemoteFile(std::string uri, RemoteOptions opt) : uri_(std::move(uri)) {
    if (opt.chunk_size == 0) opt.chunk_size = 1;
    auto st = std::make_shared<State>();
    st->ep = resolve_uri(uri_, opt);
    st->opt = std::move(opt);
    state_ = std::move(st);
}

std::uint64_t RemoteFile::size() const {
    state_->ensure_size();
    return state_->file_size;
}

std::uint64_t RemoteFile::http_requests() const { return state_->requests; }

void RemoteFile::read_at(std::uint64_t offset, void* dst, std::size_t n) const {
    if (!dst && n > 0) throw InvalidArgumentError("RemoteFile::read_at: null destination");
    state_->ensure_size();
    if (n == 0) return;
    if (offset > state_->file_size || n > state_->file_size - offset) {
        throw IndexError("RemoteFile::read_at: range outside the file");
    }
    auto* out = static_cast<std::uint8_t*>(dst);
    if (state_->have_whole) {
        std::memcpy(out, state_->whole.data() + offset, n);
        return;
    }
    const std::uint64_t cs = state_->opt.chunk_size;
    std::size_t filled = 0;
    while (filled < n) {
        const std::uint64_t pos = offset + filled;
        const std::uint64_t ci = pos / cs;
        const auto& data = state_->chunk(ci);
        if (state_->have_whole) {
            std::memcpy(out + filled, state_->whole.data() + pos, n - filled);
            return;
        }
        const std::size_t local = static_cast<std::size_t>(pos - ci * cs);
        if (local >= data.size()) throw IOError("RemoteFile: short chunk");
        const std::size_t take = std::min(n - filled, data.size() - local);
        std::memcpy(out + filled, data.data() + local, take);
        filled += take;
    }
}

} // namespace nexusdata
