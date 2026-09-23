# v2.0 Phase 3: transparent compression codecs (plan 7.5).
#
# NEXUSDATA_FETCH_DEPS=ON (default): codec sources are downloaded at configure time
# (pinned release tarballs + SHA256) and compiled into nexusdata itself, so the
# installed package has no extra runtime or link dependencies.
# NEXUSDATA_FETCH_DEPS=OFF: use system packages via find_package().

set(_nd_codecs ZLIB ZSTD LZ4 BZIP2 LZMA)
set(_nd_any_codec OFF)
foreach(_c IN LISTS _nd_codecs)
    if(NEXUSDATA_WITH_${_c})
        set(_nd_any_codec ON)
    endif()
endforeach()
if(NOT _nd_any_codec)
    return()
endif()

enable_language(C)
include(FetchContent)

# Codec C sources: built without the nexusdata warning flags (third-party code).
function(_nd_codec_objects name)
    cmake_parse_arguments(A "" "" "SOURCES;INCLUDES;DEFINES" ${ARGN})
    add_library(${name} OBJECT ${A_SOURCES})
    target_include_directories(${name} PUBLIC ${A_INCLUDES})
    target_compile_definitions(${name} PUBLIC ${A_DEFINES})
    set_target_properties(${name} PROPERTIES POSITION_INDEPENDENT_CODE ON C_VISIBILITY_PRESET hidden)
    if(MSVC)
        target_compile_options(${name} PRIVATE /w)
    else()
        target_compile_options(${name} PRIVATE -w)
    endif()
    target_sources(nexusdata PRIVATE $<TARGET_OBJECTS:${name}>)
    target_include_directories(nexusdata PRIVATE ${A_INCLUDES})
    target_compile_definitions(nexusdata PRIVATE ${A_DEFINES})
endfunction()

# Download only; the codec's own CMakeLists (tests, install rules) is not used.
function(_nd_fetch name url sha256)
    FetchContent_Declare(${name} URL ${url} URL_HASH SHA256=${sha256}
                         SOURCE_SUBDIR _nexusdata_no_cmake DOWNLOAD_EXTRACT_TIMESTAMP ON)
    FetchContent_MakeAvailable(${name})
    set(${name}_SRC "${${name}_SOURCE_DIR}" PARENT_SCOPE)
endfunction()

set(NEXUSDATA_CODEC_DEPS "")

if(NEXUSDATA_WITH_ZLIB)
    if(NEXUSDATA_FETCH_DEPS)
        _nd_fetch(nd_zlib "https://github.com/madler/zlib/releases/download/v1.3.1/zlib-1.3.1.tar.gz"
                  9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23)
        set(_z ${nd_zlib_SRC})
        # gz* (stdio file API) is not needed: gzip framing is handled by inflate itself.
        _nd_codec_objects(nd_zlib_obj
            SOURCES ${_z}/adler32.c ${_z}/compress.c ${_z}/crc32.c ${_z}/deflate.c ${_z}/infback.c
                    ${_z}/inffast.c ${_z}/inflate.c ${_z}/inftrees.c ${_z}/trees.c ${_z}/uncompr.c
                    ${_z}/zutil.c
            INCLUDES ${_z})
    else()
        find_package(ZLIB REQUIRED)
        target_link_libraries(nexusdata PRIVATE ZLIB::ZLIB)
        list(APPEND NEXUSDATA_CODEC_DEPS ZLIB)
    endif()
    target_compile_definitions(nexusdata PUBLIC NEXUSDATA_WITH_ZLIB=1)
endif()

if(NEXUSDATA_WITH_ZSTD)
    if(NEXUSDATA_FETCH_DEPS)
        _nd_fetch(nd_zstd "https://github.com/facebook/zstd/releases/download/v1.5.7/zstd-1.5.7.tar.gz"
                  eb33e51f49a15e023950cd7825ca74a4a2b43db8354825ac24fc1b7ee09e6fa3)
        set(_z ${nd_zstd_SRC}/lib)
        file(GLOB _zstd_src ${_z}/common/*.c ${_z}/compress/*.c ${_z}/decompress/*.c)
        set(_zstd_defs ZSTD_DISABLE_ASM=1)
        # The BMI2 Huffman decoder is GNU assembler for ELF/Mach-O only.
        if(UNIX AND NOT MSVC AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
            enable_language(ASM)
            list(APPEND _zstd_src ${_z}/decompress/huf_decompress_amd64.S)
            set(_zstd_defs "")
        endif()
        _nd_codec_objects(nd_zstd_obj SOURCES ${_zstd_src} INCLUDES ${_z} DEFINES ${_zstd_defs})
    else()
        find_package(zstd CONFIG REQUIRED)
        if(TARGET zstd::libzstd_static)
            target_link_libraries(nexusdata PRIVATE zstd::libzstd_static)
        else()
            target_link_libraries(nexusdata PRIVATE zstd::libzstd_shared)
        endif()
        list(APPEND NEXUSDATA_CODEC_DEPS zstd)
    endif()
    target_compile_definitions(nexusdata PUBLIC NEXUSDATA_WITH_ZSTD=1)
endif()

if(NEXUSDATA_WITH_LZ4)
    if(NEXUSDATA_FETCH_DEPS)
        _nd_fetch(nd_lz4 "https://github.com/lz4/lz4/releases/download/v1.10.0/lz4-1.10.0.tar.gz"
                  537512904744b35e232912055ccf8ec66d768639ff3abe5788d90d792ec5f48b)
        set(_z ${nd_lz4_SRC}/lib)
        # XXH_NAMESPACE keeps lz4's xxhash symbols apart from zstd's copy.
        _nd_codec_objects(nd_lz4_obj
            SOURCES ${_z}/lz4.c ${_z}/lz4hc.c ${_z}/lz4frame.c ${_z}/xxhash.c
            INCLUDES ${_z}
            DEFINES XXH_NAMESPACE=LZ4_)
    else()
        find_package(lz4 CONFIG REQUIRED)
        target_link_libraries(nexusdata PRIVATE LZ4::lz4)
        list(APPEND NEXUSDATA_CODEC_DEPS lz4)
    endif()
    target_compile_definitions(nexusdata PUBLIC NEXUSDATA_WITH_LZ4=1)
endif()

if(NEXUSDATA_WITH_BZIP2)
    if(NEXUSDATA_FETCH_DEPS)
        _nd_fetch(nd_bzip2 "https://sourceware.org/pub/bzip2/bzip2-1.0.8.tar.gz"
                  ab5a03176ee106d3f0fa90e381da478ddae405918153cca248e682cd0c4a2269)
        set(_z ${nd_bzip2_SRC})
        _nd_codec_objects(nd_bzip2_obj
            SOURCES ${_z}/blocksort.c ${_z}/huffman.c ${_z}/crctable.c ${_z}/randtable.c
                    ${_z}/compress.c ${_z}/decompress.c ${_z}/bzlib.c
            INCLUDES ${_z})
    else()
        find_package(BZip2 REQUIRED)
        target_link_libraries(nexusdata PRIVATE BZip2::BZip2)
        list(APPEND NEXUSDATA_CODEC_DEPS BZip2)
    endif()
    target_compile_definitions(nexusdata PUBLIC NEXUSDATA_WITH_BZIP2=1)
endif()

if(NEXUSDATA_WITH_LZMA)
    if(NEXUSDATA_FETCH_DEPS)
        # liblzma's build needs xz's own configure checks, so its CMake project is used;
        # the resulting objects are still embedded into nexusdata.
        set(XZ_TOOL_XZ OFF CACHE BOOL "" FORCE)
        set(XZ_TOOL_XZDEC OFF CACHE BOOL "" FORCE)
        set(XZ_TOOL_LZMADEC OFF CACHE BOOL "" FORCE)
        set(XZ_TOOL_LZMAINFO OFF CACHE BOOL "" FORCE)
        set(XZ_TOOL_SCRIPTS OFF CACHE BOOL "" FORCE)
        set(XZ_TOOL_SYMLINKS OFF CACHE BOOL "" FORCE)
        set(XZ_TOOL_SYMLINKS_LZMA OFF CACHE BOOL "" FORCE)
        set(XZ_DOC OFF CACHE BOOL "" FORCE)
        set(XZ_NLS OFF CACHE BOOL "" FORCE)
        set(XZ_DOXYGEN OFF CACHE BOOL "" FORCE)
        set(_nd_saved_shared ${BUILD_SHARED_LIBS})
        set(_nd_saved_testing ${BUILD_TESTING})
        set(BUILD_SHARED_LIBS OFF)
        set(BUILD_TESTING OFF)
        FetchContent_Declare(nd_xz
            URL "https://github.com/tukaani-project/xz/releases/download/v5.8.1/xz-5.8.1.tar.gz"
            URL_HASH SHA256=507825b599356c10dca1cd720c9d0d0c9d5400b9de300af00e4d1ea150795543
            DOWNLOAD_EXTRACT_TIMESTAMP ON EXCLUDE_FROM_ALL)
        FetchContent_MakeAvailable(nd_xz)
        set(BUILD_SHARED_LIBS ${_nd_saved_shared})
        set(BUILD_TESTING ${_nd_saved_testing})
        set_target_properties(liblzma PROPERTIES POSITION_INDEPENDENT_CODE ON)
        target_sources(nexusdata PRIVATE $<TARGET_OBJECTS:liblzma>)
        target_include_directories(nexusdata PRIVATE ${nd_xz_SOURCE_DIR}/src/liblzma/api)
        target_compile_definitions(nexusdata PRIVATE LZMA_API_STATIC)
    else()
        find_package(LibLZMA REQUIRED)
        target_link_libraries(nexusdata PRIVATE LibLZMA::LibLZMA)
        list(APPEND NEXUSDATA_CODEC_DEPS LibLZMA)
    endif()
    target_compile_definitions(nexusdata PUBLIC NEXUSDATA_WITH_LZMA=1)
endif()
