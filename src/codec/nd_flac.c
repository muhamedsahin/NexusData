/* dr_flac: public-domain single header. DR_FLAC_NO_STDIO keeps file I/O out;
 * callers pass a memory buffer. f32 samples are already in [-1, 1]. */
#define DR_FLAC_IMPLEMENTATION
#define DR_FLAC_NO_STDIO
#include "dr_flac.h"

#include "nd_audio_c.h"

#include <string.h>

int nd_flac_decode(const unsigned char* data, size_t size, NdPcm* out) {
    unsigned int channels = 0;
    unsigned int rate = 0;
    drflac_uint64 frames = 0;
    float* pcm;

    if (!out) return -1;
    memset(out, 0, sizeof(*out));
    if (!data || size == 0) return -1;

    pcm = drflac_open_memory_and_read_pcm_frames_f32(data, size, &channels, &rate, &frames, NULL);
    if (!pcm || channels == 0 || frames == 0 || frames > 0x7fffffff) {
        if (pcm) drflac_free(pcm, NULL);
        return -1;
    }
    out->samples = pcm;
    out->frames = (int)frames;
    out->channels = (int)channels;
    out->sample_rate = (int)rate;
    return 0;
}
