#pragma once

#include <string>

#include "nexusdata/core/registry.hpp"

namespace nexusdata {

/// ABI token plugins must return from nexusdata_plugin_abi_version().
inline constexpr const char* kPluginAbiVersion = "2.0";

/// Load a shared library and call
///   extern "C" const char* nexusdata_plugin_abi_version();
///   extern "C" void nexusdata_plugin_register(nexusdata::DataSourceRegistry&);
/// The library stays loaded so factories and the datasets they return remain valid.
/// The plugin must be built with the same compiler and C++ runtime as this process.
void load_plugin(const std::string& path);

} // namespace nexusdata
