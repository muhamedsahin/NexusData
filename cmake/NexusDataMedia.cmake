# v2.0 Phase 4: image / audio / video codecs.
#
# WebP is compiled into nexusdata (OBJECT merge, same pattern as the compression
# codecs) so the installed package does not grow a find_dependency. JPEG-turbo,
# AVIF, FFmpeg and NVDEC stay off unless the user turns the flag on; a missing
# package is then a configure error.

if(NEXUSDATA_WITH_NVDEC AND NOT (NEXUSDATA_WITH_CUDA AND NEXUSDATA_WITH_FFMPEG))
    message(FATAL_ERROR "NEXUSDATA_WITH_NVDEC=ON requires NEXUSDATA_WITH_CUDA=ON and NEXUSDATA_WITH_FFMPEG=ON")
endif()

enable_language(C)

# --- vendored audio decoders (minimp3, dr_flac, stb_vorbis) ---
add_library(nd_audio_dec OBJECT
    src/codec/nd_mp3.c
    src/codec/nd_flac.c
    src/codec/nd_vorbis.c)
target_include_directories(nd_audio_dec PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/codec
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/minimp3
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/dr_flac
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb)
set_target_properties(nd_audio_dec PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
if(MSVC)
    target_compile_options(nd_audio_dec PRIVATE /w)
else()
    target_compile_options(nd_audio_dec PRIVATE -w)
endif()
target_sources(nexusdata PRIVATE $<TARGET_OBJECTS:nd_audio_dec>)
target_include_directories(nexusdata PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/codec)

# --- libwebp, portable C only (no NEON / SSE / MIPS sources) ---
if(NEXUSDATA_WITH_WEBP)
    include(FetchContent)
    set(_nd_webp_url "https://github.com/webmproject/libwebp/archive/refs/tags/v1.4.0.tar.gz")
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/downloads/libwebp-1.4.0.tar.gz")
        set(_nd_webp_url "${CMAKE_CURRENT_SOURCE_DIR}/third_party/downloads/libwebp-1.4.0.tar.gz")
    endif()
    FetchContent_Declare(nd_libwebp
        URL "${_nd_webp_url}"
        URL_HASH SHA256=12af50c45530f0a292d39a88d952637e43fb2d4ab1883c44ae729840f7273381
        SOURCE_SUBDIR _nexusdata_no_cmake
        DOWNLOAD_EXTRACT_TIMESTAMP ON)
    FetchContent_MakeAvailable(nd_libwebp)

    set(_nd_webp_cfg "${CMAKE_BINARY_DIR}/nd_webp_gen/src/webp/config.h")
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/nd_webp_gen/src/webp")
    if(MSVC)
        file(WRITE "${_nd_webp_cfg}" [[
#ifndef WEBP_CONFIG_H_
#define WEBP_CONFIG_H_
#define HAVE_CONFIG_H 1
#define WEBP_NEAR_LOSSLESS 1
#endif
]])
    else()
        file(WRITE "${_nd_webp_cfg}" [[
#ifndef WEBP_CONFIG_H_
#define WEBP_CONFIG_H_
#define HAVE_CONFIG_H 1
#define HAVE_BUILTIN_BSWAP16 1
#define HAVE_BUILTIN_BSWAP32 1
#define HAVE_BUILTIN_BSWAP64 1
#define HAVE_UNISTD_H 1
#define WEBP_NEAR_LOSSLESS 1
#endif
]])
    endif()

    set(_wp ${nd_libwebp_SOURCE_DIR})
    set(_nd_webp_sources
        ${_wp}/src/dec/alpha_dec.c ${_wp}/src/dec/buffer_dec.c ${_wp}/src/dec/frame_dec.c
        ${_wp}/src/dec/idec_dec.c ${_wp}/src/dec/io_dec.c ${_wp}/src/dec/quant_dec.c
        ${_wp}/src/dec/tree_dec.c ${_wp}/src/dec/vp8_dec.c ${_wp}/src/dec/vp8l_dec.c
        ${_wp}/src/dec/webp_dec.c
        ${_wp}/src/enc/alpha_enc.c ${_wp}/src/enc/analysis_enc.c
        ${_wp}/src/enc/backward_references_cost_enc.c ${_wp}/src/enc/backward_references_enc.c
        ${_wp}/src/enc/config_enc.c ${_wp}/src/enc/cost_enc.c ${_wp}/src/enc/filter_enc.c
        ${_wp}/src/enc/frame_enc.c ${_wp}/src/enc/histogram_enc.c ${_wp}/src/enc/iterator_enc.c
        ${_wp}/src/enc/near_lossless_enc.c ${_wp}/src/enc/picture_csp_enc.c
        ${_wp}/src/enc/picture_enc.c ${_wp}/src/enc/picture_psnr_enc.c
        ${_wp}/src/enc/picture_rescale_enc.c ${_wp}/src/enc/picture_tools_enc.c
        ${_wp}/src/enc/predictor_enc.c ${_wp}/src/enc/quant_enc.c ${_wp}/src/enc/syntax_enc.c
        ${_wp}/src/enc/token_enc.c ${_wp}/src/enc/tree_enc.c ${_wp}/src/enc/vp8l_enc.c
        ${_wp}/src/enc/webp_enc.c
        ${_wp}/src/utils/bit_reader_utils.c ${_wp}/src/utils/bit_writer_utils.c
        ${_wp}/src/utils/color_cache_utils.c ${_wp}/src/utils/filters_utils.c
        ${_wp}/src/utils/huffman_encode_utils.c ${_wp}/src/utils/huffman_utils.c
        ${_wp}/src/utils/palette.c ${_wp}/src/utils/quant_levels_dec_utils.c
        ${_wp}/src/utils/quant_levels_utils.c ${_wp}/src/utils/random_utils.c
        ${_wp}/src/utils/rescaler_utils.c ${_wp}/src/utils/thread_utils.c ${_wp}/src/utils/utils.c
        ${_wp}/src/dsp/alpha_processing.c ${_wp}/src/dsp/cost.c ${_wp}/src/dsp/cpu.c
        ${_wp}/src/dsp/dec.c ${_wp}/src/dsp/dec_clip_tables.c ${_wp}/src/dsp/enc.c
        ${_wp}/src/dsp/filters.c ${_wp}/src/dsp/lossless.c ${_wp}/src/dsp/lossless_enc.c
        ${_wp}/src/dsp/rescaler.c ${_wp}/src/dsp/ssim.c ${_wp}/src/dsp/upsampling.c ${_wp}/src/dsp/yuv.c
        ${_wp}/sharpyuv/sharpyuv.c ${_wp}/sharpyuv/sharpyuv_cpu.c ${_wp}/sharpyuv/sharpyuv_csp.c
        ${_wp}/sharpyuv/sharpyuv_dsp.c ${_wp}/sharpyuv/sharpyuv_gamma.c)

    add_library(nd_webp_obj OBJECT ${_nd_webp_sources})
    target_include_directories(nd_webp_obj PUBLIC
        "${CMAKE_BINARY_DIR}/nd_webp_gen"
        "${_wp}")
    target_compile_definitions(nd_webp_obj PRIVATE HAVE_CONFIG_H=1)
    set_target_properties(nd_webp_obj PROPERTIES
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON)
    if(MSVC)
        target_compile_options(nd_webp_obj PRIVATE /w)
    else()
        target_compile_options(nd_webp_obj PRIVATE -w)
    endif()
    target_sources(nexusdata PRIVATE $<TARGET_OBJECTS:nd_webp_obj>)
    target_include_directories(nexusdata PRIVATE
        "${CMAKE_BINARY_DIR}/nd_webp_gen"
        "${_wp}")
    target_compile_definitions(nexusdata PUBLIC NEXUSDATA_WITH_WEBP=1)
endif()

if(NEXUSDATA_WITH_JPEGTURBO)
    find_path(NEXUSDATA_TURBOJPEG_INCLUDE turbojpeg.h)
    find_library(NEXUSDATA_TURBOJPEG_LIB turbojpeg)
    if(NOT NEXUSDATA_TURBOJPEG_INCLUDE OR NOT NEXUSDATA_TURBOJPEG_LIB)
        message(FATAL_ERROR "NEXUSDATA_WITH_JPEGTURBO=ON but turbojpeg was not found")
    endif()
    target_include_directories(nexusdata PRIVATE ${NEXUSDATA_TURBOJPEG_INCLUDE})
    target_link_libraries(nexusdata PUBLIC ${NEXUSDATA_TURBOJPEG_LIB})
    target_compile_definitions(nexusdata PUBLIC NEXUSDATA_WITH_JPEGTURBO=1)
endif()

if(NEXUSDATA_WITH_AVIF)
    find_path(NEXUSDATA_AVIF_INCLUDE avif/avif.h)
    find_library(NEXUSDATA_AVIF_LIB avif)
    if(NOT NEXUSDATA_AVIF_INCLUDE OR NOT NEXUSDATA_AVIF_LIB)
        message(FATAL_ERROR "NEXUSDATA_WITH_AVIF=ON but libavif was not found")
    endif()
    target_include_directories(nexusdata PRIVATE ${NEXUSDATA_AVIF_INCLUDE})
    target_link_libraries(nexusdata PUBLIC ${NEXUSDATA_AVIF_LIB})
    target_compile_definitions(nexusdata PUBLIC NEXUSDATA_WITH_AVIF=1)
endif()

if(NEXUSDATA_WITH_FFMPEG)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(NEXUSDATA_FFMPEG REQUIRED IMPORTED_TARGET
        libavformat libavcodec libavutil libswscale)
    target_link_libraries(nexusdata PUBLIC PkgConfig::NEXUSDATA_FFMPEG)
    target_compile_definitions(nexusdata PUBLIC NEXUSDATA_WITH_FFMPEG=1)
endif()

if(NEXUSDATA_WITH_NVDEC)
    target_compile_definitions(nexusdata PUBLIC NEXUSDATA_WITH_NVDEC=1)
endif()
