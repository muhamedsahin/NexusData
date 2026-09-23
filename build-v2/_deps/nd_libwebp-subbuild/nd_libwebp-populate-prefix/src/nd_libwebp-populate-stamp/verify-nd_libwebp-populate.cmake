# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

if("C:/Users/muham/Desktop/Nexus/lib/NexusData/third_party/downloads/libwebp-1.4.0.tar.gz" STREQUAL "")
  message(FATAL_ERROR "LOCAL can't be empty")
endif()

if(NOT EXISTS "C:/Users/muham/Desktop/Nexus/lib/NexusData/third_party/downloads/libwebp-1.4.0.tar.gz")
  message(FATAL_ERROR "File not found: C:/Users/muham/Desktop/Nexus/lib/NexusData/third_party/downloads/libwebp-1.4.0.tar.gz")
endif()

if("SHA256" STREQUAL "")
  message(WARNING "File cannot be verified since no URL_HASH specified")
  return()
endif()

if("12af50c45530f0a292d39a88d952637e43fb2d4ab1883c44ae729840f7273381" STREQUAL "")
  message(FATAL_ERROR "EXPECT_VALUE can't be empty")
endif()

message(VERBOSE "verifying file...
     file='C:/Users/muham/Desktop/Nexus/lib/NexusData/third_party/downloads/libwebp-1.4.0.tar.gz'")

file("SHA256" "C:/Users/muham/Desktop/Nexus/lib/NexusData/third_party/downloads/libwebp-1.4.0.tar.gz" actual_value)

if(NOT "${actual_value}" STREQUAL "12af50c45530f0a292d39a88d952637e43fb2d4ab1883c44ae729840f7273381")
  message(FATAL_ERROR "error: SHA256 hash of
  C:/Users/muham/Desktop/Nexus/lib/NexusData/third_party/downloads/libwebp-1.4.0.tar.gz
does not match expected value
  expected: '12af50c45530f0a292d39a88d952637e43fb2d4ab1883c44ae729840f7273381'
    actual: '${actual_value}'
")
endif()

message(VERBOSE "verifying file... done")
