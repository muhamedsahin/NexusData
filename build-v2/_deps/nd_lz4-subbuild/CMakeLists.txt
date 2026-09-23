# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 4.3.1)

# Reject any attempt to use a toolchain file. We must not use one because
# we could be downloading it here. If the CMAKE_TOOLCHAIN_FILE environment
# variable is set, the cache variable will have been initialized from it.
unset(CMAKE_TOOLCHAIN_FILE CACHE)
unset(ENV{CMAKE_TOOLCHAIN_FILE})

# We name the project and the target for the ExternalProject_Add() call
# to something that will highlight to the user what we are working on if
# something goes wrong and an error message is produced.

project(nd_lz4-populate NONE)



include(ExternalProject)
ExternalProject_Add(nd_lz4-populate
                     "UPDATE_DISCONNECTED" "False" "DOWNLOAD_EXTRACT_TIMESTAMP" "ON" "EXTERNALPROJECT_INTERNAL_ARGUMENT_SEPARATOR" "URL" "https://github.com/lz4/lz4/releases/download/v1.10.0/lz4-1.10.0.tar.gz" "URL_HASH" "SHA256=537512904744b35e232912055ccf8ec66d768639ff3abe5788d90d792ec5f48b" "SOURCE_SUBDIR" "_nexusdata_no_cmake"
                    SOURCE_DIR          "C:/Users/muham/Desktop/Nexus/lib/NexusData/build-v2/_deps/nd_lz4-src"
                    BINARY_DIR          "C:/Users/muham/Desktop/Nexus/lib/NexusData/build-v2/_deps/nd_lz4-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    USES_TERMINAL_DOWNLOAD  YES
                    USES_TERMINAL_UPDATE    YES
                    USES_TERMINAL_PATCH     YES
)


