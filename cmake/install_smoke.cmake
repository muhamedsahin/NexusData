# Install NexusData to a local prefix and build example/find_package_smoke against it.
if(NOT NexusData_SOURCE_DIR OR NOT NexusData_BINARY_DIR)
  message(FATAL_ERROR "install_smoke: missing source/binary dirs")
endif()

set(PREFIX "${NexusData_BINARY_DIR}/install_smoke_prefix")
set(SMOKE_BUILD "${NexusData_BINARY_DIR}/install_smoke_build")

execute_process(
  COMMAND "${CMAKE_COMMAND}" --install "${NexusData_BINARY_DIR}" --prefix "${PREFIX}"
  RESULT_VARIABLE rc
)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "install_smoke: cmake --install failed (${rc})")
endif()

file(REMOVE_RECURSE "${SMOKE_BUILD}")
file(MAKE_DIRECTORY "${SMOKE_BUILD}")

set(gen_args "")
if(NexusData_GENERATOR)
  set(gen_args -G "${NexusData_GENERATOR}")
endif()
# The consumer must use the library's compiler: an MSVC-built nexusdata.lib cannot link into a MinGW binary.
if(NexusData_CXX_COMPILER)
  list(APPEND gen_args "-DCMAKE_CXX_COMPILER=${NexusData_CXX_COMPILER}")
endif()
# MSVC also refuses to mix /MD and /MDd runtimes across the library and the consumer.
if(NexusData_BUILD_TYPE)
  list(APPEND gen_args "-DCMAKE_BUILD_TYPE=${NexusData_BUILD_TYPE}")
endif()

set(prefix_path "${PREFIX}")
if(NexusData_ARROW_PREFIX)
  set(prefix_path "${PREFIX};${NexusData_ARROW_PREFIX}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" ${gen_args}
          -S "${NexusData_SOURCE_DIR}/example/find_package_smoke"
          -B "${SMOKE_BUILD}"
          "-DCMAKE_PREFIX_PATH=${prefix_path}"
  RESULT_VARIABLE rc
)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "install_smoke: configure consumer failed (${rc})")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" --build "${SMOKE_BUILD}"
  RESULT_VARIABLE rc
)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "install_smoke: build consumer failed (${rc})")
endif()

message(STATUS "install_smoke: OK")
