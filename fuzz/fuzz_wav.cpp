/// libFuzzer target: wav buffer parser.
#include <cstddef>
#include <cstdint>

#include "nexusdata/fuzz/harness.hpp"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    (void)nexusdata::fuzz::try_wav(data, size);
    return 0;
}
