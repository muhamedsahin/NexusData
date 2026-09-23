#include "nexusdata/backend/cpu_dispatch.hpp"

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <cpuid.h>
#endif
#endif

#if defined(__linux__) && (defined(__aarch64__) || defined(_M_ARM64))
#include <sys/auxv.h>
#ifndef HWCAP_SVE
#define HWCAP_SVE (1 << 22)
#endif
#endif

namespace nexusdata {

namespace {

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

void cpuid(unsigned leaf, unsigned sub, unsigned regs[4]) {
#if defined(_MSC_VER)
    int info[4] = {};
    __cpuidex(info, static_cast<int>(leaf), static_cast<int>(sub));
    for (int i = 0; i < 4; ++i) {
        regs[i] = static_cast<unsigned>(info[i]);
    }
#else
    unsigned a = 0, b = 0, c = 0, d = 0;
    __cpuid_count(leaf, sub, a, b, c, d);
    regs[0] = a;
    regs[1] = b;
    regs[2] = c;
    regs[3] = d;
#endif
}

unsigned max_leaf() {
#if defined(_MSC_VER)
    int info[4] = {};
    __cpuid(info, 0);
    return static_cast<unsigned>(info[0]);
#else
    return __get_cpuid_max(0, nullptr);
#endif
}

std::uint64_t read_xcr0() {
#if defined(_MSC_VER)
    return _xgetbv(0);
#else
    unsigned lo = 0, hi = 0;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return (static_cast<std::uint64_t>(hi) << 32) | lo;
#endif
}

#endif

CpuFeatures detect() {
    CpuFeatures f;
#if defined(__aarch64__) || defined(_M_ARM64)
    f.neon = true;
#if defined(__linux__)
    f.sve = (getauxval(AT_HWCAP) & HWCAP_SVE) != 0;
#endif
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    const unsigned top = max_leaf();
    if (top < 1) {
        return f;
    }
    unsigned r[4] = {};
    cpuid(1, 0, r);
    f.sse2 = (r[3] & (1u << 26)) != 0;
    const bool osxsave = (r[2] & (1u << 27)) != 0;
    const bool cpu_avx = (r[2] & (1u << 28)) != 0;
    const bool cpu_fma = (r[2] & (1u << 12)) != 0;
    bool os_ymm = false;
    bool os_zmm = false;
    if (osxsave) {
        const std::uint64_t xcr0 = read_xcr0();
        os_ymm = (xcr0 & 0x6) == 0x6;    // XMM + YMM state
        os_zmm = (xcr0 & 0xE6) == 0xE6;  // + opmask + ZMM_Hi256 + Hi16_ZMM
    }
    f.avx = cpu_avx && os_ymm;
    f.fma = f.avx && cpu_fma;
    if (top >= 7) {
        cpuid(7, 0, r);
        f.avx2 = f.avx && (r[1] & (1u << 5)) != 0;
        f.avx512f = os_zmm && (r[1] & (1u << 16)) != 0;
        f.avx512bw = f.avx512f && (r[1] & (1u << 30)) != 0;
    }
#endif
    return f;
}

} // namespace

const CpuFeatures& cpu_features() {
    static const CpuFeatures cached = detect();
    return cached;
}

} // namespace nexusdata
