#include <doctest/doctest.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>

#include "nexusdata/dataset/odbc.hpp"
#include "nexusdata/dataset/postgres.hpp"
#include "nexusdata/image/media.hpp"
#include "nexusdata/io/remote_file.hpp"

using namespace nexusdata;

namespace {

void append_u16(std::vector<std::uint8_t>& b, std::uint16_t v) {
    b.push_back(static_cast<std::uint8_t>(v >> 8));
    b.push_back(static_cast<std::uint8_t>(v));
}
void append_u32(std::vector<std::uint8_t>& b, std::uint32_t v) {
    for (int s = 24; s >= 0; s -= 8) b.push_back(static_cast<std::uint8_t>((v >> s) & 0xff));
}
void append_u64(std::vector<std::uint8_t>& b, std::uint64_t v) {
    for (int s = 56; s >= 0; s -= 8) b.push_back(static_cast<std::uint8_t>((v >> s) & 0xff));
}
void append_f64(std::vector<std::uint8_t>& b, double v) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &v, 8);
    append_u64(b, bits);
}

std::vector<std::uint8_t> sample_copy() {
    std::vector<std::uint8_t> b = {'P', 'G', 'C', 'O', 'P', 'Y', '\n', 0xFF, '\r', '\n', 0};
    append_u32(b, 0);
    append_u32(b, 0);
    auto row = [&](double x, const char* name, std::int32_t y) {
        append_u16(b, 3);
        append_u32(b, 8);
        append_f64(b, x);
        append_u32(b, static_cast<std::uint32_t>(std::strlen(name)));
        b.insert(b.end(), name, name + std::strlen(name));
        append_u32(b, 4);
        append_u32(b, static_cast<std::uint32_t>(y));
    };
    row(1.5, "cat", 7);
    row(2.5, "dog", 8);
    append_u16(b, static_cast<std::uint16_t>(-1));
    return b;
}

std::vector<SqlColumn> schema() {
    SqlColumn x;
    x.name = "x";
    x.type = SqlType::Float64;
    x.role = SqlRole::Input;
    SqlColumn name;
    name.name = "name";
    name.type = SqlType::Text;
    SqlColumn y;
    y.name = "y";
    y.type = SqlType::Int32;
    y.role = SqlRole::Label;
    return {x, name, y};
}

struct LocalHttp {
    SOCKET listen_sock = INVALID_SOCKET;
    int port = 0;
    std::atomic<bool> run{true};
    std::thread thread;
    std::string body = "abcdefghijklmnopqrstuvwxyz";

    LocalHttp() {
        WSADATA w{};
        REQUIRE(WSAStartup(MAKEWORD(2, 2), &w) == 0);
        listen_sock = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        REQUIRE(listen_sock != INVALID_SOCKET);
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port = 0;
        REQUIRE(::bind(listen_sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0);
        REQUIRE(::listen(listen_sock, 8) == 0);
        int len = sizeof(addr);
        REQUIRE(::getsockname(listen_sock, reinterpret_cast<sockaddr*>(&addr), &len) == 0);
        port = ntohs(addr.sin_port);
        thread = std::thread([this] { loop(); });
    }
    ~LocalHttp() {
        run = false;
        closesocket(listen_sock);
        if (thread.joinable()) thread.join();
    }

    void loop() {
        while (run) {
            fd_set fds;
            FD_ZERO(&fds);
            FD_SET(listen_sock, &fds);
            timeval tv{0, 200000};
            if (select(0, &fds, nullptr, nullptr, &tv) <= 0) continue;
            SOCKET client = accept(listen_sock, nullptr, nullptr);
            if (client == INVALID_SOCKET) continue;
            std::string req;
            char buf[1024];
            while (req.find("\r\n\r\n") == std::string::npos && req.size() < 8192) {
                const int n = recv(client, buf, sizeof(buf), 0);
                if (n <= 0) break;
                req.append(buf, buf + n);
            }
            std::uint64_t a = 0;
            std::uint64_t bend = body.empty() ? 0 : body.size() - 1;
            const auto range = req.find("Range:");
            if (range != std::string::npos) {
                const auto bytes = req.find("bytes=", range);
                if (bytes != std::string::npos) {
                    char* end = nullptr;
                    a = std::strtoull(req.c_str() + bytes + 6, &end, 10);
                    if (end && *end == '-') bend = std::strtoull(end + 1, nullptr, 10);
                }
            }
            if (a >= body.size()) a = body.size();
            if (bend >= body.size() && !body.empty()) bend = body.size() - 1;
            const std::string slice = a < body.size() && a <= bend ? body.substr(static_cast<std::size_t>(a),
                                                                                  static_cast<std::size_t>(bend - a + 1))
                                                                   : std::string();
            std::string msg = "HTTP/1.1 206 Partial Content\r\nContent-Length: " + std::to_string(slice.size()) +
                              "\r\nContent-Range: bytes " + std::to_string(a) + "-" + std::to_string(bend) + "/" +
                              std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n" + slice;
            std::size_t off = 0;
            while (off < msg.size()) {
                const int n = send(client, msg.data() + off, static_cast<int>(msg.size() - off), 0);
                if (n <= 0) break;
                off += static_cast<std::size_t>(n);
            }
            closesocket(client);
        }
    }
};

} // namespace

TEST_CASE("COPY BINARY decodes rows and streams them") {
    const auto bytes = sample_copy();
    PostgresCopyDataset ds(bytes.data(), bytes.size(), schema());
    CHECK(ds.size() == 2);
    CHECK(ds.column_names().size() == 3);
    CHECK(ds.get(0).input.data<double>()[0] == doctest::Approx(1.5));
    CHECK(ds.get(0).metadata.at("name") == "cat");
    CHECK(ds.get(0).label.data<std::int64_t>()[0] == 7);
    CHECK(ds.get(1).metadata.at("name") == "dog");
    CHECK(ds.get(1).label.data<std::int64_t>()[0] == 8);

    PostgresCopyIterable it(bytes.data(), bytes.size(), schema());
    auto cur = it.make_iterator();
    CHECK(cur->has_next());
    CHECK(cur->next().input.data<double>()[0] == doctest::Approx(1.5));
    Sample second = cur->next();
    CHECK(second.input.data<double>()[0] == doctest::Approx(2.5));
    CHECK_FALSE(cur->has_next());
}

TEST_CASE("Postgres and ODBC constructors name the CMake flag when disabled") {
    if (!postgres_support_enabled()) {
        PostgresOptions opt;
        opt.conninfo = "host=127.0.0.1";
        opt.query = "SELECT 1";
        SqlColumn c;
        c.name = "x";
        opt.columns = {c};
        CHECK_THROWS_AS((PostgresDataset(opt)), InvalidArgumentError);
    }
    if (!odbc_support_enabled()) {
        OdbcOptions opt;
        opt.connection = "DSN=none";
        opt.query = "SELECT 1";
        SqlColumn c;
        c.name = "x";
        opt.columns = {c};
        CHECK_THROWS_AS((OdbcDataset(opt)), InvalidArgumentError);
    }
    CHECK_THROWS_AS(PostgresCopyDataset(nullptr, 4, schema()), InvalidArgumentError);
}

TEST_CASE("SigV4 matches the AWS GET Object example") {
    SigV4Request req;
    req.method = "GET";
    req.canonical_uri = "/test.txt";
    req.host = "examplebucket.s3.amazonaws.com";
    req.amz_date = "20130524T000000Z";
    req.payload_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    req.region = "us-east-1";
    req.service = "s3";
    req.access_key = "AKIAIOSFODNN7EXAMPLE";
    req.secret_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY";
    req.headers.emplace_back("range", "bytes=0-9");
    const SigV4Result sig = sigv4_sign(req);
    CHECK(sig.signature == "f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41");
}

TEST_CASE("RemoteFile range reads hit the chunk cache") {
    LocalHttp server;
    RemoteOptions opt;
    opt.chunk_size = 8;
    opt.cache_chunks = 4;
    RemoteFile file("http://127.0.0.1:" + std::to_string(server.port) + "/blob", opt);
    CHECK(file.size() == 26);
    char buf[8] = {};
    file.read_at(0, buf, 4);
    CHECK(std::string(buf, 4) == "abcd");
    const auto after_first = file.http_requests();
    file.read_at(0, buf, 4);
    CHECK(file.http_requests() == after_first);
    file.read_at(10, buf, 3);
    CHECK(std::string(buf, 3) == "klm");
    CHECK(file.http_requests() == after_first + 1);
}

TEST_CASE("media sniff routes stb formats; truncated webp fails, avif names its flag") {
    const std::uint8_t jpeg[] = {0xFF, 0xD8, 0xFF, 0xD9};
    CHECK(sniff_media(jpeg, sizeof(jpeg)) == MediaFormat::Jpeg);
    const std::uint8_t png[] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    CHECK(sniff_media(png, sizeof(png)) == MediaFormat::Png);
    const std::uint8_t webp[] = {'R', 'I', 'F', 'F', 0, 0, 0, 0, 'W', 'E', 'B', 'P'};
    CHECK(sniff_media(webp, sizeof(webp)) == MediaFormat::Webp);
    CHECK_THROWS_AS(static_cast<void>(decode_media_image(webp, sizeof(webp))), IOError);
    const std::uint8_t avif[] = {0, 0, 0, 0x18, 'f', 't', 'y', 'p', 'a', 'v', 'i', 'f'};
    CHECK(sniff_media(avif, sizeof(avif)) == MediaFormat::Avif);
    bool named = false;
    try {
        (void)decode_media_image(avif, sizeof(avif));
    } catch (const InvalidArgumentError& e) {
        named = std::string(e.what()).find("NEXUSDATA_WITH_AVIF") != std::string::npos;
    }
    CHECK(named);
    const std::uint8_t flac[] = {'f', 'L', 'a', 'C'};
    CHECK(sniff_media(flac, 4) == MediaFormat::Flac);
}
