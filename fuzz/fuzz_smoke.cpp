/// Portable deterministic fuzz smoke (no libFuzzer required).
/// Mutates seed corpora and ensures parsers only throw nexusdata::Error (or succeed).

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "nexusdata/fuzz/harness.hpp"

using namespace nexusdata;

namespace {

std::uint32_t xorshift(std::uint32_t& s) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}

void mutate(std::vector<std::uint8_t>& buf, std::uint32_t& rng) {
    if (buf.empty()) {
        buf.push_back(static_cast<std::uint8_t>(xorshift(rng)));
        return;
    }
    const std::uint32_t op = xorshift(rng) % 4;
    if (op == 0) {
        buf[xorshift(rng) % buf.size()] ^= static_cast<std::uint8_t>(xorshift(rng));
    } else if (op == 1 && buf.size() < 4096) {
        buf.insert(buf.begin() + static_cast<std::ptrdiff_t>(xorshift(rng) % buf.size()),
                   static_cast<std::uint8_t>(xorshift(rng)));
    } else if (op == 2 && buf.size() > 1) {
        buf.erase(buf.begin() + static_cast<std::ptrdiff_t>(xorshift(rng) % buf.size()));
    } else {
        buf[xorshift(rng) % buf.size()] = static_cast<std::uint8_t>(xorshift(rng));
    }
}

std::vector<std::uint8_t> seed_wav() {
    // Minimal valid-looking RIFF header (will still fail parse deeper — OK)
    std::vector<std::uint8_t> w = {'R', 'I', 'F', 'F', 36, 0, 0, 0, 'W', 'A', 'V', 'E',
                                   'f', 'm', 't', ' ', 16, 0, 0, 0, 1, 0, 1, 0,
                                   0x40, 0x1F, 0, 0, 0x80, 0x3E, 0, 0, 2, 0, 16, 0,
                                   'd', 'a', 't', 'a', 4, 0, 0, 0, 0, 0, 0, 0};
    return w;
}

std::vector<std::uint8_t> seed_npy() {
    // Bad magic with NUMPY-ish bytes
    std::vector<std::uint8_t> b = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0, 0, 0};
    return b;
}

bool run_buf(bool (*fn)(const std::uint8_t*, std::size_t), std::vector<std::uint8_t> buf,
             std::uint32_t seed, int iters) {
    std::uint32_t rng = seed;
    for (int i = 0; i < iters; ++i) {
        try {
            (void)fn(buf.data(), buf.size());
        } catch (const std::exception& e) {
            std::cerr << "UNEXPECTED non-Error exception: " << e.what() << "\n";
            return false;
        } catch (...) {
            std::cerr << "UNEXPECTED unknown exception\n";
            return false;
        }
        mutate(buf, rng);
    }
    return true;
}

bool run_file(bool (*fn)(const std::string&), std::vector<std::uint8_t> buf, std::uint32_t seed,
              int iters, const char* path) {
    std::uint32_t rng = seed;
    for (int i = 0; i < iters; ++i) {
        {
            std::ofstream out(path, std::ios::binary);
            out.write(reinterpret_cast<const char*>(buf.data()),
                      static_cast<std::streamsize>(buf.size()));
        }
        try {
            (void)fn(path);
        } catch (const std::exception& e) {
            std::cerr << "UNEXPECTED non-Error exception: " << e.what() << "\n";
            return false;
        } catch (...) {
            std::cerr << "UNEXPECTED unknown exception\n";
            return false;
        }
        mutate(buf, rng);
    }
    std::remove(path);
    return true;
}

} // namespace

int main() {
    const int iters = 200;
    if (!run_buf(fuzz::try_npy, seed_npy(), 1, iters)) {
        return 1;
    }
    if (!run_buf(fuzz::try_wav, seed_wav(), 2, iters)) {
        return 1;
    }
    if (!run_file(fuzz::try_csv_file, {'a', ',', 'b', '\n', '1', ',', '2', '\n'}, 3, iters,
                  "fuzz_smoke.csv")) {
        return 1;
    }
    if (!run_file(fuzz::try_webdataset_file, std::vector<std::uint8_t>(512, 0), 4, iters,
                  "fuzz_smoke.tar")) {
        return 1;
    }
    if (!run_buf(fuzz::try_json, {'{', '}'}, 5, iters)) {
        return 1;
    }
    if (!run_buf(fuzz::try_postgres_copy, {'P', 'G', 'C', 'O', 'P', 'Y'}, 6, iters)) {
        return 1;
    }
    if (!run_buf(fuzz::try_audio, {'I', 'D', '3'}, 7, iters)) {
        return 1;
    }
    if (!run_buf(fuzz::try_media, {'R', 'I', 'F', 'F', 0, 0, 0, 0, 'W', 'E', 'B', 'P'}, 8, iters)) {
        return 1;
    }
    std::cout << "fuzz_smoke: ok (" << iters << " iters x 8 targets)\n";
    return 0;
}
