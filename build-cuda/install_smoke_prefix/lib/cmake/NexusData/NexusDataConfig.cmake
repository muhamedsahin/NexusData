
####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was NexusDataConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

include(CMakeFindDependencyMacro)
find_dependency(Threads)

set(NexusData_WITH_CUDA "ON")
set(NexusData_WITH_NVJPEG "ON")
if(NexusData_WITH_CUDA)
  find_dependency(CUDAToolkit)
endif()
set(NexusData_WITH_SQLITE "OFF")
set(NexusData_WITH_PYTHON "OFF")
set(NexusData_WITH_MATRIXFLASH "OFF")

include("${CMAKE_CURRENT_LIST_DIR}/NexusDataTargets.cmake")

if(NOT TARGET nexusdata::nexusdata)
  message(FATAL_ERROR "NexusData: exported target nexusdata::nexusdata missing")
endif()

check_required_components(NexusData)
