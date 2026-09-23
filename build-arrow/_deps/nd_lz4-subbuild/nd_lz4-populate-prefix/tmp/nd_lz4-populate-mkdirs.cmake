# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "C:/Users/muham/Desktop/Nexus/lib/NexusData/build-arrow/_deps/nd_lz4-src")
  file(MAKE_DIRECTORY "C:/Users/muham/Desktop/Nexus/lib/NexusData/build-arrow/_deps/nd_lz4-src")
endif()
file(MAKE_DIRECTORY
  "C:/Users/muham/Desktop/Nexus/lib/NexusData/build-arrow/_deps/nd_lz4-build"
  "C:/Users/muham/Desktop/Nexus/lib/NexusData/build-arrow/_deps/nd_lz4-subbuild/nd_lz4-populate-prefix"
  "C:/Users/muham/Desktop/Nexus/lib/NexusData/build-arrow/_deps/nd_lz4-subbuild/nd_lz4-populate-prefix/tmp"
  "C:/Users/muham/Desktop/Nexus/lib/NexusData/build-arrow/_deps/nd_lz4-subbuild/nd_lz4-populate-prefix/src/nd_lz4-populate-stamp"
  "C:/Users/muham/Desktop/Nexus/lib/NexusData/build-arrow/_deps/nd_lz4-subbuild/nd_lz4-populate-prefix/src"
  "C:/Users/muham/Desktop/Nexus/lib/NexusData/build-arrow/_deps/nd_lz4-subbuild/nd_lz4-populate-prefix/src/nd_lz4-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "C:/Users/muham/Desktop/Nexus/lib/NexusData/build-arrow/_deps/nd_lz4-subbuild/nd_lz4-populate-prefix/src/nd_lz4-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "C:/Users/muham/Desktop/Nexus/lib/NexusData/build-arrow/_deps/nd_lz4-subbuild/nd_lz4-populate-prefix/src/nd_lz4-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
