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

project(nd_libwebp-populate NONE)



include(ExternalProject)
ExternalProject_Add(nd_libwebp-populate
                     "UPDATE_DISCONNECTED" "False" "DOWNLOAD_EXTRACT_TIMESTAMP" "ON" "EXTERNALPROJECT_INTERNAL_ARGUMENT_SEPARATOR" "URL" "C:/Users/muham/Desktop/Nexus/lib/NexusData/third_party/downloads/libwebp-1.4.0.tar.gz" "URL_HASH" "SHA256=12af50c45530f0a292d39a88d952637e43fb2d4ab1883c44ae729840f7273381" "SOURCE_SUBDIR" "_nexusdata_no_cmake"
                    SOURCE_DIR          "C:/Users/muham/Desktop/Nexus/lib/NexusData/build-v2/_deps/nd_libwebp-src"
                    BINARY_DIR          "C:/Users/muham/Desktop/Nexus/lib/NexusData/build-v2/_deps/nd_libwebp-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    USES_TERMINAL_DOWNLOAD  YES
                    USES_TERMINAL_UPDATE    YES
                    USES_TERMINAL_PATCH     YES
)


