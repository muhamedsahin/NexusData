/* stb_vorbis: public-domain single file. stb_vorbis_decode_memory returns the
 * frame count (per channel); the buffer is interleaved short samples. */
#define STB_VORBIS_NO_STDIO
#include "stb_vorbis.c"

#include "nd_audio_c.h"

#include <stdlib.h>
#include <string.h>

int nd_vorbis_decode(const unsigned char* data, size_t size, NdPcm* out) {
    int channels = 0;
    int rate = 0;
    short* pcm = NULL;
    int frames;
    size_t n;
    float* samples;

    if (!out) return -1;
    memset(out, 0, sizeof(*out));
    if (!data || size == 0 || size > 0x7fffffff) return -1;

    frames = stb_vorbis_decode_memory(data, (int)size, &channels, &rate, &pcm);
    if (frames <= 0 || channels <= 0 || !pcm) {
        free(pcm);
        return -1;
    }
    n = (size_t)frames * (size_t)channels;
    samples = (float*)malloc(n * sizeof(float));
    if (!samples) {
        free(pcm);
        return -1;
    }
    for (size_t i = 0; i < n; ++i) {
        samples[i] = (float)pcm[i] / 32768.0f;
    }
    free(pcm);
    out->samples = samples;
    out->frames = frames;
    out->channels = channels;
    out->sample_rate = rate;
    return 0;
}
