/* minimp3: public-domain, one header. Chosen over libsndfile so MP3 needs no
 * system package. MINIMP3_IMPLEMENTATION must live in exactly one TU. */
#define MINIMP3_IMPLEMENTATION
#include "minimp3.h"

#include "nd_audio_c.h"

#include <stdlib.h>
#include <string.h>

int nd_mp3_decode(const unsigned char* data, size_t size, NdPcm* out) {
    mp3dec_t dec;
    size_t cap = 0;
    size_t count = 0;
    size_t off = 0;
    float* samples = NULL;
    int channels = 0;
    int rate = 0;
    int frames = 0;
    int saw = 0;

    if (!out) return -1;
    memset(out, 0, sizeof(*out));
    if (!data || size == 0 || size > 0x7fffffff) return -1;

    mp3dec_init(&dec);
    while (off < size) {
        mp3d_sample_t pcm[MINIMP3_MAX_SAMPLES_PER_FRAME];
        mp3dec_frame_info_t info;
        int n;
        int inter;
        memset(&info, 0, sizeof(info));
        n = mp3dec_decode_frame(&dec, data + off, (int)(size - off), pcm, &info);
        if (info.frame_bytes <= 0) break;
        off += (size_t)info.frame_bytes;
        if (n <= 0 || info.channels <= 0) continue;
        if (!saw) {
            channels = info.channels;
            rate = info.hz;
            saw = 1;
        }
        if (info.channels != channels) {
            free(samples);
            return -1;
        }
        inter = n * channels;
        if (count + (size_t)inter > cap) {
            size_t nc = cap ? cap * 2 : 16384;
            float* grown;
            while (nc < count + (size_t)inter) nc *= 2;
            grown = (float*)realloc(samples, nc * sizeof(float));
            if (!grown) {
                free(samples);
                return -1;
            }
            samples = grown;
            cap = nc;
        }
        for (int i = 0; i < inter; ++i) {
            samples[count + (size_t)i] = (float)pcm[i] / 32768.0f;
        }
        count += (size_t)inter;
        frames += n;
    }
    if (!saw || frames <= 0 || !samples) {
        free(samples);
        return -1;
    }
    out->samples = samples;
    out->frames = frames;
    out->channels = channels;
    out->sample_rate = rate;
    return 0;
}

void nd_pcm_free(NdPcm* pcm) {
    if (!pcm) return;
    free(pcm->samples);
    memset(pcm, 0, sizeof(*pcm));
}
