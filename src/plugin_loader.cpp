#include "nexusdata/core/plugin_loader.hpp"

#include <cstring>
#include <string>

#include "nexusdata/core/error.hpp"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace nexusdata {
namespace {

using VersionFn = const char* (*)();
using RegisterFn = void (*)(DataSourceRegistry&);

} // namespace

void load_plugin(const std::string& path) {
    if (path.empty()) {
        throw InvalidArgumentError("load_plugin: empty path");
    }
#if defined(_WIN32)
    HMODULE lib = LoadLibraryA(path.c_str());
    if (!lib) {
        throw IOError("load_plugin: LoadLibrary failed for " + quote_path(path));
    }
    FARPROC version_raw = GetProcAddress(lib, "nexusdata_plugin_abi_version");
    FARPROC register_raw = GetProcAddress(lib, "nexusdata_plugin_register");
    VersionFn version = nullptr;
    RegisterFn reg = nullptr;
    static_assert(sizeof(version) == sizeof(version_raw), "function pointer width");
    std::memcpy(&version, &version_raw, sizeof(version));
    std::memcpy(&reg, &register_raw, sizeof(reg));
#else
    void* lib = dlopen(path.c_str(), RTLD_NOW);
    if (!lib) {
        const char* msg = dlerror();
        throw IOError(std::string("load_plugin: dlopen failed: ") + (msg ? msg : path.c_str()));
    }
    auto version = reinterpret_cast<VersionFn>(dlsym(lib, "nexusdata_plugin_abi_version"));
    auto reg = reinterpret_cast<RegisterFn>(dlsym(lib, "nexusdata_plugin_register"));
#endif
    if (!version || !reg) {
        throw IOError("load_plugin: missing nexusdata_plugin_abi_version or nexusdata_plugin_register");
    }
    const char* abi = version();
    if (!abi || std::string(abi) != kPluginAbiVersion) {
        throw InvalidArgumentError(std::string("load_plugin: ABI mismatch, plugin has '") +
                                   (abi ? abi : "") + "', host wants " + kPluginAbiVersion);
    }
    reg(DataSourceRegistry::instance());
}

} // namespace nexusdata
