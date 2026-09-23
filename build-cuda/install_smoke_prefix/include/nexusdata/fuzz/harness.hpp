#pragma once

/// Fuzz entry helpers: catch NexusData::Error and return false (never abort).

#include <cstddef>
#include <cstdint>
#include <string>

#include "nexusdata/core/error.hpp"
#include "nexusdata/dataset/audio.hpp"
#include "nexusdata/dataset/csv/csv_dataset.hpp"
#include "nexusdata/dataset/npy.hpp"
#include "nexusdata/dataset/webdataset.hpp"

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
