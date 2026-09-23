# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "C:/Users/muham/Desktop/Nexus/lib/NexusData/build-v2/_deps/nd_zlib-src")
  file(MAKE_DIRECTORY "C:/Users/muham/Desktop/Nexus/lib/NexusData/build-v2/_deps/nd_zlib-src")
endif()
file(MAKE_DIRECTORY
  "C:/Users/muham/Desktop/Nexus/lib/NexusData/build-v2/_deps/nd_zlib-build"
  "C:/Users/muham/Desktop/Nexus/lib/NexusData/build-v2/_deps/nd_zlib-subbuild/nd_zlib-populate-prefix"
  "C:/Users/muham/Desktop/Nexus/lib/NexusData/build-v2/_deps/nd_zlib-subbuild/nd_zlib-populate-prefix/tmp"
  "C:/Users/muham/Desktop/Nexus/lib/NexusData/build-v2/_deps/nd_zlib-subbuild/nd_zlib-populate-prefix/src/nd_zlib-populate-stamp"
  "C:/Users/muham/Desktop/Nexus/lib/NexusData/build-v2/_deps/nd_zlib-subbuild/nd_zlib-populate-prefix/src"
  "C:/Users/muham/Desktop/Nexus/lib/NexusData/build-v2/_deps/nd_zlib-subbuild/nd_zlib-populate-prefix/src/nd_zlib-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "C:/Users/muham/Desktop/Nexus/lib/NexusData/build-v2/_deps/nd_zlib-subbuild/nd_zlib-populate-prefix/src/nd_zlib-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "C:/Users/muham/Desktop/Nexus/lib/NexusData/build-v2/_deps/nd_zlib-subbuild/nd_zlib-populate-prefix/src/nd_zlib-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
