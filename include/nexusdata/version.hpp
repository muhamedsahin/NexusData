#pragma once

/// NexusData version (stable 1.x SemVer).
#define NEXUSDATA_VERSION_MAJOR 1
#define NEXUSDATA_VERSION_MINOR 0
#define NEXUSDATA_VERSION_PATCH 0
#define NEXUSDATA_VERSION_STRING "1.0.0"

namespace nexusdata {

[[nodiscard]] inline const char* version_string() noexcept {
    return NEXUSDATA_VERSION_STRING;
}

[[nodiscard]] inline int version_major() noexcept { return NEXUSDATA_VERSION_MAJOR; }
[[nodiscard]] inline int version_minor() noexcept { return NEXUSDATA_VERSION_MINOR; }
[[nodiscard]] inline int version_patch() noexcept { return NEXUSDATA_VERSION_PATCH; }

} // namespace nexusdata
