# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "C:/Users/muham/Desktop/Nexus/lib/NexusData/build-v2/_deps/nd_zstd-src")
  file(MAKE_DIRECTORY "C:/Users/muham/Desktop/Nexus/lib/NexusData/build-v2/_deps/nd_zstd-src")
endif()
file(MAKE_DIRECTORY
  "C:/Users/muham/Desktop/Nexus/lib/NexusData/build-v2/_deps/nd_zstd-build"
  "C:/Users/muham/Desktop/Nexus/lib/NexusData/build-v2/_deps/nd_zstd-subbuild/nd_zstd-populate-prefix"
  "C:/Users/muham/Desktop/Nexus/lib/NexusData/build-v2/_deps/nd_zstd-subbuild/nd_zstd-populate-prefix/tmp"
  "C:/Users/muham/Desktop/Nexus/lib/NexusData/build-v2/_deps/nd_zstd-subbuild/nd_zstd-populate-prefix/src/nd_zstd-populate-stamp"
  "C:/Users/muham/Desktop/Nexus/lib/NexusData/build-v2/_deps/nd_zstd-subbuild/nd_zstd-populate-prefix/src"
  "C:/Users/muham/Desktop/Nexus/lib/NexusData/build-v2/_deps/nd_zstd-subbuild/nd_zstd-populate-prefix/src/nd_zstd-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "C:/Users/muham/Desktop/Nexus/lib/NexusData/build-v2/_deps/nd_zstd-subbuild/nd_zstd-populate-prefix/src/nd_zstd-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "C:/Users/muham/Desktop/Nexus/lib/NexusData/build-v2/_deps/nd_zstd-subbuild/nd_zstd-populate-prefix/src/nd_zstd-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
