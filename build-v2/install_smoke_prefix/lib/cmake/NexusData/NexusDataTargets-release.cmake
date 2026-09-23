#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "nexusdata::nexusdata" for configuration "Release"
set_property(TARGET nexusdata::nexusdata APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(nexusdata::nexusdata PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C;CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libnexusdata.a"
  )

list(APPEND _cmake_import_check_targets nexusdata::nexusdata )
list(APPEND _cmake_import_check_files_for_nexusdata::nexusdata "${_IMPORT_PREFIX}/lib/libnexusdata.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
