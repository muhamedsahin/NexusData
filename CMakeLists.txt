cmake_minimum_required(VERSION 3.20)
project(NexusData VERSION 1.0.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(NEXUSDATA_BUILD_TESTS "Build unit tests" ON)
option(NEXUSDATA_BUILD_EXAMPLES "Build examples" ON)
option(NEXUSDATA_BUILD_BENCH "Build microbenchmarks" ON)
option(NEXUSDATA_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)
option(NEXUSDATA_ENABLE_AVX2 "Compile AVX2 kernels for byte_scan (runtime dispatched)" ON)
option(NEXUSDATA_WITH_CUDA "Build CUDA backend (kernels, GPU executor). Default OFF." OFF)
option(NEXUSDATA_WITH_NVJPEG "GPU JPEG decoding via nvJPEG (requires NEXUSDATA_WITH_CUDA)" OFF)
option(NEXUSDATA_WITH_SQLITE "Link SQLite amalgamation for SqliteDataset" OFF)
option(NEXUSDATA_WITH_ARROW "Enable Parquet/Arrow dataset (requires Arrow)" OFF)
option(NEXUSDATA_WITH_POSTGRES "Enable PostgresDataset via libpq COPY BINARY" OFF)
option(NEXUSDATA_WITH_ODBC "Enable OdbcDataset (Windows driver manager)" OFF)
option(NEXUSDATA_WITH_HDF5 "Enable HDF5 dataset (requires HDF5)" OFF)
option(NEXUSDATA_WITH_PYTHON "Build pybind11 Python module nexusdata_py" OFF)
option(NEXUSDATA_WITH_MATRIXFLASH "Declare MatrixFlash adapter linked (Flash SDK)" OFF)
option(NEXUSDATA_BUILD_FUZZ "Build fuzz_smoke (+ libFuzzer targets when Clang)" ON)
# v2.0 Phase 3: compression codecs (cmake/NexusDataCompression.cmake).
option(NEXUSDATA_FETCH_DEPS "Download pinned third-party sources instead of using system packages" ON)
option(NEXUSDATA_WITH_ZLIB "gzip / zlib decompression (zlib)" ON)
option(NEXUSDATA_WITH_ZSTD "Zstandard decompression (zstd)" ON)
option(NEXUSDATA_WITH_LZ4 "LZ4 frame decompression (lz4)" ON)
option(NEXUSDATA_WITH_BZIP2 "bzip2 decompression (libbz2)" ON)
option(NEXUSDATA_WITH_LZMA "xz / lzma decompression (liblzma)" OFF)
# v2.0 Phase 4: media. WebP is fetched and compiled in. The others stay off
# until the matching dev package is installed.
option(NEXUSDATA_WITH_WEBP "Decode/encode WebP via libwebp (fetched, built-in)" ON)
option(NEXUSDATA_WITH_JPEGTURBO "JPEG via libjpeg-turbo (turbojpeg) instead of stb" OFF)
option(NEXUSDATA_WITH_AVIF "AVIF via libavif" OFF)
option(NEXUSDATA_WITH_FFMPEG "VideoIterable via libavformat/libavcodec/libswscale" OFF)
option(NEXUSDATA_WITH_NVDEC "NVDEC hardware decode (requires CUDA and FFmpeg)" OFF)
set(NEXUSDATA_SANITIZER "" CACHE STRING "Sanitizer: Address, Undefined, Thread, or empty")
set_property(CACHE NEXUSDATA_SANITIZER PROPERTY STRINGS "" Address Undefined Thread)

set(NEXUSDATA_SOURCES
    src/allocator.cpp
    src/ndarray.cpp
    src/random.cpp
    src/in_memory.cpp
    src/subset.cpp
    src/split.cpp
    src/stats.cpp
    src/csv_dataset.cpp
    src/collate.cpp
    src/dataloader.cpp
    src/mapped_file.cpp
    src/image_decode.cpp
    src/image_folder.cpp
    src/mnist.cpp
    src/cifar.cpp
    src/npy.cpp
    src/cpu_dispatch.cpp
    src/thread_pool.cpp
    src/byte_scan.cpp
    src/jsonl.cpp
    src/device.cpp
    src/pinned.cpp
    src/backend.cpp
    src/text.cpp
    src/audio.cpp
    src/sqlite.cpp
    src/webdataset.cpp
    src/pipeline.cpp
    src/parquet.cpp
    src/arrow_ipc.cpp
    src/arrow_columns.cpp
    src/hdf5.cpp
    src/dlpack.cpp
    src/c_api.cpp
    # v2.0 Phase 1: performance core
    src/numeric_ops.cpp
    src/numeric_ops_avx2.cpp
    src/image_ops.cpp
    src/arena_allocator.cpp
    src/object_pool.cpp
    src/cost_model.cpp
    src/fusion.cpp
    # v2.0 Phase 2: GPU execution (compiled in every build; CUDA calls go through cuda_api)
    src/gpu_executor.cpp
    src/gpu_telemetry.cpp
    src/gpu_transforms.cpp
    src/augment.cpp
    src/nvjpeg_decoder.cpp
    # v2.0 Phase 3: formats
    src/byte_source.cpp
    src/compression.cpp
    src/json_value.cpp
    src/json_dataset.cpp
    src/postgres.cpp
    src/odbc.cpp
    src/remote_file.cpp
    src/registry.cpp
    src/plugin_loader.cpp
    src/pipeline_builder.cpp
    src/metrics.cpp
    src/media.cpp
    src/audio_codec.cpp
    src/tokenizer.cpp
    src/video.cpp
)

if(NEXUSDATA_WITH_NVJPEG AND NOT NEXUSDATA_WITH_CUDA)
    message(FATAL_ERROR "NEXUSDATA_WITH_NVJPEG=ON requires NEXUSDATA_WITH_CUDA=ON")
endif()

if(NEXUSDATA_WITH_CUDA)
    # Must be set before enable_language(CUDA). CUDA 13 dropped sm < 75.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
            set(CMAKE_CUDA_ARCHITECTURES native)
        else()
            set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
        endif()
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    list(APPEND NEXUSDATA_SOURCES cuda/cuda_api.cu cuda/cuda_kernels.cu)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    message(STATUS "NexusData: CUDA ${CUDAToolkit_VERSION}, architectures ${CMAKE_CUDA_ARCHITECTURES}")
else()
    list(APPEND NEXUSDATA_SOURCES src/cuda_api_stub.cpp)
endif()

add_library(nexusdata ${NEXUSDATA_SOURCES})
add_library(nexusdata::nexusdata ALIAS nexusdata)
include(cmake/NexusDataCompression.cmake)
include(cmake/NexusDataMedia.cmake)

if(NEXUSDATA_WITH_CUDA)
    add_library(nexusdata_cuda INTERFACE)
    add_library(nexusdata::cuda ALIAS nexusdata_cuda)
    target_link_libraries(nexusdata_cuda INTERFACE nexusdata)
    target_compile_definitions(nexusdata PUBLIC NEXUSDATA_WITH_CUDA=1)
    # Static runtime: no cudart DLL next to consumers' binaries.
    target_link_libraries(nexusdata PUBLIC CUDA::cudart_static)
    # Bit-exact with the CPU kernels: no multiply-add contraction, IEEE div/sqrt.
    target_compile_options(nexusdata PRIVATE
        "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-prec-div=true;-prec-sqrt=true>")
    if(NEXUSDATA_WITH_NVJPEG)
        target_link_libraries(nexusdata PUBLIC CUDA::nvjpeg)
        target_compile_definitions(nexusdata PUBLIC NEXUSDATA_WITH_NVJPEG=1)
    endif()
endif()

if(NOT WIN32)
    target_link_libraries(nexusdata PRIVATE ${CMAKE_DL_LIBS}) # NVML is loaded at runtime
endif()

target_include_directories(nexusdata
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb
)

target_link_libraries(nexusdata PUBLIC Threads::Threads)
if(WIN32)
    target_link_libraries(nexusdata PUBLIC ws2_32 winhttp)
endif()

if(NEXUSDATA_WITH_SQLITE)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/sqlite/sqlite3.c")
        message(FATAL_ERROR "NEXUSDATA_WITH_SQLITE=ON but third_party/sqlite/sqlite3.c is missing")
    endif()
    target_sources(nexusdata PRIVATE third_party/sqlite/sqlite3.c)
    target_include_directories(nexusdata PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/sqlite)
    target_compile_definitions(nexusdata PUBLIC NEXUSDATA_WITH_SQLITE=1)
endif()

if(NEXUSDATA_WITH_ARROW)
    find_package(Arrow CONFIG REQUIRED)
    find_package(Parquet CONFIG REQUIRED)
    target_link_libraries(nexusdata PUBLIC Arrow::arrow_shared Parquet::parquet_shared)
    target_compile_definitions(nexusdata PUBLIC NEXUSDATA_WITH_ARROW=1)
    # Static nexusdata still needs the Arrow DLLs at test runtime.
    if(WIN32)
        foreach(_nd_arrow_tgt Arrow::arrow_shared Parquet::parquet_shared)
            if(NOT TARGET ${_nd_arrow_tgt})
                continue()
            endif()
            get_target_property(_nd_arrow_loc ${_nd_arrow_tgt} IMPORTED_LOCATION)
            if(NOT _nd_arrow_loc)
                get_target_property(_nd_arrow_loc ${_nd_arrow_tgt} IMPORTED_LOCATION_RELEASE)
            endif()
            if(_nd_arrow_loc)
                get_filename_component(_nd_arrow_bin "${_nd_arrow_loc}" DIRECTORY)
                list(APPEND NEXUSDATA_ARROW_BINS "${_nd_arrow_bin}")
            endif()
        endforeach()
        if(NEXUSDATA_ARROW_BINS)
            list(REMOVE_DUPLICATES NEXUSDATA_ARROW_BINS)
            string(REPLACE ";" "\\;" NEXUSDATA_ARROW_BIN "${NEXUSDATA_ARROW_BINS}")
        endif()
    endif()
    # install_find_package_smoke must find Arrow's transitive configs (thrift, ...).
    if(Arrow_DIR)
        get_filename_component(NEXUSDATA_ARROW_PREFIX "${Arrow_DIR}/../../.." ABSOLUTE)
    endif()
endif()

if(NEXUSDATA_WITH_POSTGRES)
    find_path(NEXUSDATA_LIBPQ_INCLUDE libpq-fe.h)
    find_library(NEXUSDATA_LIBPQ_LIBRARY pq)
    if(NOT NEXUSDATA_LIBPQ_INCLUDE OR NOT NEXUSDATA_LIBPQ_LIBRARY)
        message(FATAL_ERROR "NEXUSDATA_WITH_POSTGRES=ON but libpq was not found")
    endif()
    target_include_directories(nexusdata PRIVATE ${NEXUSDATA_LIBPQ_INCLUDE})
    target_link_libraries(nexusdata PUBLIC ${NEXUSDATA_LIBPQ_LIBRARY})
    target_compile_definitions(nexusdata PUBLIC NEXUSDATA_WITH_POSTGRES=1)
endif()

if(NEXUSDATA_WITH_ODBC)
    if(NOT WIN32)
        message(FATAL_ERROR "NEXUSDATA_WITH_ODBC=ON is implemented for the Windows driver manager")
    endif()
    target_link_libraries(nexusdata PUBLIC odbc32)
    target_compile_definitions(nexusdata PUBLIC NEXUSDATA_WITH_ODBC=1)
endif()

if(NEXUSDATA_WITH_HDF5)
    find_package(HDF5 REQUIRED COMPONENTS C)
    target_link_libraries(nexusdata PUBLIC HDF5::HDF5)
    target_compile_definitions(nexusdata PUBLIC NEXUSDATA_WITH_HDF5=1)
endif()

if(NEXUSDATA_WITH_MATRIXFLASH)
    target_compile_definitions(nexusdata PUBLIC NEXUSDATA_WITH_MATRIXFLASH=1)
    message(STATUS "NexusData: MatrixFlash adapter flag ON (view API; link Flash in consumer)")
endif()

target_compile_definitions(nexusdata PUBLIC "NEXUSDATA_VERSION_STRING=\"${PROJECT_VERSION}\"")

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.1)
    target_link_libraries(nexusdata PUBLIC stdc++fs)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    target_link_libraries(nexusdata PUBLIC c++fs)
endif()

if(MSVC)
    target_compile_options(nexusdata PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/W4;/permissive->")
    if(NEXUSDATA_WARNINGS_AS_ERRORS)
        target_compile_options(nexusdata PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/WX>)
    endif()
else()
    target_compile_options(nexusdata PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>")
    if(NEXUSDATA_WARNINGS_AS_ERRORS)
        target_compile_options(nexusdata PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Werror>)
    endif()
endif()

if(NOT MSVC)
    set_source_files_properties(src/image_decode.cpp PROPERTIES
        COMPILE_FLAGS "-Wno-unused-function -Wno-sign-compare")
endif()

# Numeric kernels promise bit-exact results vs. the scalar reference: forbid
# multiply-add contraction everywhere they are compiled.
set(_nd_exact_fp_sources src/numeric_ops.cpp src/numeric_ops_avx2.cpp src/image_ops.cpp src/fusion.cpp)
if(NOT MSVC)
    set_property(SOURCE ${_nd_exact_fp_sources} APPEND PROPERTY COMPILE_OPTIONS "-ffp-contract=off")
endif()

if(NEXUSDATA_ENABLE_AVX2)
    # Runtime-dispatched AVX2 translation units (no -mfma: see numeric_ops_avx2.cpp).
    set(_nd_avx2_sources src/byte_scan.cpp src/numeric_ops_avx2.cpp)
    if(MSVC)
        set_property(SOURCE ${_nd_avx2_sources} APPEND PROPERTY COMPILE_OPTIONS "/arch:AVX2")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64|i.86")
        set_property(SOURCE ${_nd_avx2_sources} APPEND PROPERTY COMPILE_OPTIONS "-mavx2")
        if(MINGW)
            # MinGW-w64 GCC assumes 32-byte stack alignment for AVX spills that the
            # Windows ABI does not guarantee; emit unaligned moves instead.
            include(CheckCXXCompilerFlag)
            check_cxx_compiler_flag("-Wa,-muse-unaligned-vector-move" NEXUSDATA_HAS_UNALIGNED_VMOV)
            if(NEXUSDATA_HAS_UNALIGNED_VMOV)
                set_property(SOURCE ${_nd_avx2_sources} APPEND PROPERTY COMPILE_OPTIONS
                             "-Wa,-muse-unaligned-vector-move")
            endif()
        endif()
    endif()
endif()

# --- Sanitizers (GCC/Clang; limited MSVC support) ---------------------------
if(NEXUSDATA_SANITIZER)
    if(MSVC)
        if(NEXUSDATA_SANITIZER STREQUAL "Address")
            target_compile_options(nexusdata PUBLIC /fsanitize=address)
            message(STATUS "NexusData: MSVC AddressSanitizer enabled")
        else()
            message(WARNING "NexusData: sanitizer '${NEXUSDATA_SANITIZER}' not supported on MSVC; ignoring")
        endif()
    else()
        set(_nd_san_flags "")
        if(NEXUSDATA_SANITIZER STREQUAL "Address")
            set(_nd_san_flags -fsanitize=address -fno-omit-frame-pointer)
        elseif(NEXUSDATA_SANITIZER STREQUAL "Undefined")
            set(_nd_san_flags -fsanitize=undefined -fno-omit-frame-pointer)
        elseif(NEXUSDATA_SANITIZER STREQUAL "Thread")
            set(_nd_san_flags -fsanitize=thread)
        else()
            message(FATAL_ERROR "NEXUSDATA_SANITIZER must be Address, Undefined, Thread, or empty")
        endif()
        target_compile_options(nexusdata PUBLIC ${_nd_san_flags})
        target_link_options(nexusdata PUBLIC ${_nd_san_flags})
        message(STATUS "NexusData: sanitizer ${NEXUSDATA_SANITIZER} enabled")
    endif()
endif()

set_target_properties(nexusdata PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    EXPORT_NAME nexusdata
    OUTPUT_NAME nexusdata
)

if(NEXUSDATA_BUILD_TESTS)
    include(FetchContent)
    # doctest v2.4.11 declares cmake_minimum_required(< 3.5), rejected by CMake >= 4.0.
    if(NOT DEFINED CMAKE_POLICY_VERSION_MINIMUM)
        set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
    endif()
    FetchContent_Declare(
        doctest
        GIT_REPOSITORY https://github.com/doctest/doctest.git
        GIT_TAG        v2.4.11
    )
    FetchContent_MakeAvailable(doctest)

    enable_testing()

    function(nexusdata_add_test name src)
        add_executable(${name} ${src})
        target_link_libraries(${name} PRIVATE nexusdata doctest::doctest)
        target_compile_definitions(${name} PRIVATE DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN)
        add_test(NAME ${name} COMMAND ${name})
        if(NEXUSDATA_ARROW_BIN)
            string(REPLACE ";" "\\;" _nd_old_path "$ENV{PATH}")
            set_tests_properties(${name} PROPERTIES
                ENVIRONMENT "PATH=${NEXUSDATA_ARROW_BIN}\\;${_nd_old_path}")
        endif()
    endfunction()

    nexusdata_add_test(test_core          test/test_core.cpp)
    nexusdata_add_test(test_ndarray       test/test_ndarray.cpp)
    nexusdata_add_test(test_random        test/test_random.cpp)
    nexusdata_add_test(test_dataset       test/test_dataset.cpp)
    nexusdata_add_test(test_dataloader    test/test_dataloader.cpp)
    nexusdata_add_test(test_csv           test/test_csv.cpp)
    nexusdata_add_test(test_split         test/test_split.cpp)
    nexusdata_add_test(test_transform     test/test_transform.cpp)
    nexusdata_add_test(test_preprocessing test/test_preprocessing.cpp)
    nexusdata_add_test(test_stats         test/test_stats.cpp)
    nexusdata_add_test(test_image         test/test_image.cpp)
    nexusdata_add_test(test_mnist_cifar   test/test_mnist_cifar.cpp)
    nexusdata_add_test(test_npy           test/test_npy.cpp)
    nexusdata_add_test(test_perf          test/test_perf.cpp)
    nexusdata_add_test(test_cuda          test/test_cuda.cpp)
    nexusdata_add_test(test_v06           test/test_v06.cpp)
    nexusdata_add_test(test_hardening     test/test_hardening.cpp)
    nexusdata_add_test(test_perf_v08      test/test_perf_v08.cpp)
    nexusdata_add_test(test_adapters      test/test_adapters.cpp)
    # v2.0 Phase 1
    nexusdata_add_test(test_simd          test/test_simd.cpp)
    nexusdata_add_test(test_memory        test/test_memory.cpp)
    nexusdata_add_test(test_stage_graph   test/test_stage_graph.cpp)
    nexusdata_add_test(test_fusion        test/test_fusion.cpp)
    nexusdata_add_test(test_cost_model    test/test_cost_model.cpp)
    # v2.0 Phase 2
    nexusdata_add_test(test_gpu           test/test_gpu.cpp)
    target_include_directories(test_gpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb)
    # v2.0 Phase 3
    nexusdata_add_test(test_sample_fields test/test_sample_fields.cpp)
    nexusdata_add_test(test_compression   test/test_compression.cpp)
    nexusdata_add_test(test_json          test/test_json.cpp)
    nexusdata_add_test(test_arrow         test/test_arrow.cpp)
    nexusdata_add_test(test_phase3_io     test/test_phase3_io.cpp)
    nexusdata_add_test(test_phase4        test/test_phase4.cpp)
    nexusdata_add_test(test_phase6        test/test_phase6.cpp)
    nexusdata_add_test(test_extensibility test/test_extensibility.cpp)

    add_library(nd_demo_plugin SHARED test/demo_plugin.cpp)
    target_link_libraries(nd_demo_plugin PRIVATE nexusdata)
    if(MINGW)
        target_link_options(nd_demo_plugin PRIVATE "-Wl,--export-all-symbols")
    endif()
    add_dependencies(test_extensibility nd_demo_plugin)
    add_custom_command(TARGET test_extensibility POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:nd_demo_plugin>
            $<TARGET_FILE_DIR:test_extensibility>)
    target_compile_definitions(test_extensibility PRIVATE
        "ND_DEMO_PLUGIN_NAME=\"$<TARGET_FILE_NAME:nd_demo_plugin>\"")

    if(NEXUSDATA_WITH_CUDA AND WIN32)
        # CUDA 13 keeps its DLLs (nvJPEG) in bin/x64; older toolkits in bin.
        get_property(_nd_tests DIRECTORY PROPERTY TESTS)
        set_tests_properties(${_nd_tests} PROPERTIES ENVIRONMENT_MODIFICATION
            "PATH=path_list_prepend:${CUDAToolkit_BIN_DIR};PATH=path_list_prepend:${CUDAToolkit_BIN_DIR}/x64")
    endif()
endif()

if(NEXUSDATA_WITH_PYTHON)
    find_package(pybind11 CONFIG QUIET)
    if(NOT pybind11_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG        v2.13.6
        )
        FetchContent_MakeAvailable(pybind11)
    endif()
    pybind11_add_module(nexusdata_py python/bindings.cpp)
    target_link_libraries(nexusdata_py PRIVATE nexusdata)
    message(STATUS "NexusData: Python module nexusdata_py enabled")
endif()

if(NEXUSDATA_BUILD_FUZZ)
    add_executable(fuzz_smoke fuzz/fuzz_smoke.cpp)
    target_link_libraries(fuzz_smoke PRIVATE nexusdata)
    if(NEXUSDATA_BUILD_TESTS)
        add_test(NAME fuzz_smoke COMMAND fuzz_smoke)
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        add_executable(fuzz_npy fuzz/fuzz_npy.cpp)
        target_link_libraries(fuzz_npy PRIVATE nexusdata)
        target_compile_options(fuzz_npy PRIVATE -fsanitize=fuzzer,address -fno-omit-frame-pointer)
        target_link_options(fuzz_npy PRIVATE -fsanitize=fuzzer,address)
        add_executable(fuzz_wav fuzz/fuzz_wav.cpp)
        target_link_libraries(fuzz_wav PRIVATE nexusdata)
        target_compile_options(fuzz_wav PRIVATE -fsanitize=fuzzer,address -fno-omit-frame-pointer)
        target_link_options(fuzz_wav PRIVATE -fsanitize=fuzzer,address)
        message(STATUS "NexusData: libFuzzer targets fuzz_npy / fuzz_wav enabled")
    endif()
endif()

if(NEXUSDATA_BUILD_EXAMPLES)
    add_executable(nexusdata_example example/example.cpp)
    target_link_libraries(nexusdata_example PRIVATE nexusdata)
    add_executable(nexusdata_hello example/hello_formats.cpp)
    target_link_libraries(nexusdata_hello PRIVATE nexusdata)
endif()

if(NEXUSDATA_BUILD_BENCH)
    add_executable(bench_csv_scan bench/bench_csv_scan.cpp)
    target_link_libraries(bench_csv_scan PRIVATE nexusdata)
    add_executable(bench_suite bench/bench_suite.cpp)
    target_link_libraries(bench_suite PRIVATE nexusdata)
    add_executable(bench_phase1 bench/bench_phase1.cpp)
    target_link_libraries(bench_phase1 PRIVATE nexusdata)
    add_executable(bench_phase3 bench/bench_phase3.cpp)
    target_link_libraries(bench_phase3 PRIVATE nexusdata)
    add_executable(bench_phase5 bench/bench_phase5.cpp)
    target_link_libraries(bench_phase5 PRIVATE nexusdata)
    add_executable(bench_gpu bench/bench_gpu.cpp)
    target_link_libraries(bench_gpu PRIVATE nexusdata)
    target_include_directories(bench_gpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb)
endif()

install(TARGETS nexusdata
    EXPORT NexusDataTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/nexusdata
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT NexusDataTargets
    FILE NexusDataTargets.cmake
    NAMESPACE nexusdata::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NexusData
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/NexusDataConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/NexusDataConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NexusData
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/NexusDataConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/NexusDataConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/NexusDataConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NexusData
)

install(FILES LICENSE CHANGELOG.md README.md
    DESTINATION ${CMAKE_INSTALL_DOCDIR}
)

# pkg-config
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/nexusdata.pc.in
    ${CMAKE_CURRENT_BINARY_DIR}/nexusdata.pc
    @ONLY
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/nexusdata.pc
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
)

# find_package consumer smoke (install to local prefix, then configure+build)
if(NEXUSDATA_BUILD_TESTS)
    add_test(
        NAME install_find_package_smoke
        COMMAND ${CMAKE_COMMAND}
            -DNexusData_SOURCE_DIR=${CMAKE_SOURCE_DIR}
            -DNexusData_BINARY_DIR=${CMAKE_BINARY_DIR}
            -DNexusData_GENERATOR=${CMAKE_GENERATOR}
            -DNexusData_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DNexusData_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DNexusData_ARROW_PREFIX=${NEXUSDATA_ARROW_PREFIX}
            -P ${CMAKE_SOURCE_DIR}/cmake/install_smoke.cmake
    )
    set_tests_properties(install_find_package_smoke PROPERTIES TIMEOUT 300)
endif()

