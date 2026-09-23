#pragma once

/// Fuzz entry helpers: catch NexusData::Error and return false (never abort).

#include <cstddef>
#include <cstdint>
#include <string>

#include "nexusdata/core/error.hpp"
#include "nexusdata/dataset/audio.hpp"
#include "nexusdata/dataset/csv/csv_dataset.hpp"
#include "nexusdata/dataset/npy.hpp"
#include "nexusdata/dataset/postgres.hpp"
#include "nexusdata/dataset/sql_column.hpp"
#include "nexusdata/dataset/webdataset.hpp"
#include "nexusdata/image/media.hpp"
#include "nexusdata/io/json.hpp"

namespace nexusdata {
namespace fuzz {

inline bool try_npy(const std::uint8_t* data, std::size_t size) {
    try {
        (void)load_npy_from_buffer(data, size);
        return true;
    } catch (const Error&) {
        return false;
    }
}

inline bool try_wav(const std::uint8_t* data, std::size_t size) {
    try {
        (void)load_wav_from_buffer(data, size, nullptr);
        return true;
    } catch (const Error&) {
        return false;
    }
}

inline bool try_csv_file(const std::string& path) {
    try {
        CSVOptions o;
        o.has_header = false;
        (void)CSVDataset(path, o);
        return true;
    } catch (const Error&) {
        return false;
    }
}

inline bool try_json(const std::uint8_t* data, std::size_t size) {
    try {
        const char* p = (data && size) ? reinterpret_cast<const char*>(data) : "";
        (void)json::parse(std::string_view(p, (data && size) ? size : 0));
        return true;
    } catch (const Error&) {
        return false;
    }
}

inline bool try_postgres_copy(const std::uint8_t* data, std::size_t size) {
    try {
        SqlColumn col;
        col.name = "x";
        col.type = SqlType::Float64;
        (void)PostgresCopyDataset(data, size, {col});
        return true;
    } catch (const Error&) {
        return false;
    }
}

inline bool try_audio(const std::uint8_t* data, std::size_t size) {
    try {
        const std::uint8_t* p = data ? data : reinterpret_cast<const std::uint8_t*>("");
        (void)load_audio_from_buffer(p, data ? size : 0, nullptr);
        return true;
    } catch (const Error&) {
        return false;
    }
}

inline bool try_media(const std::uint8_t* data, std::size_t size) {
    try {
        const std::uint8_t* p = data ? data : reinterpret_cast<const std::uint8_t*>("");
        (void)decode_media_image(p, data ? size : 0);
        return true;
    } catch (const Error&) {
        return false;
    }
}

inline bool try_webdataset_file(const std::string& path) {
    try {
        (void)WebDataset(path);
        return true;
    } catch (const Error&) {
        return false;
    }
}

} // namespace fuzz
} // namespace nexusdata
