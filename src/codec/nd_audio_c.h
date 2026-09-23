#ifndef ND_AUDIO_C_H
#define ND_AUDIO_C_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Interleaved float samples in [-1, 1], malloc'd. */
typedef struct NdPcm {
    float* samples;
    int frames;
    int channels;
    int sample_rate;
} NdPcm;

int nd_mp3_decode(const unsigned char* data, size_t size, NdPcm* out);
int nd_flac_decode(const unsigned char* data, size_t size, NdPcm* out);
int nd_vorbis_decode(const unsigned char* data, size_t size, NdPcm* out);
void nd_pcm_free(NdPcm* pcm);

#ifdef __cplusplus
}
#endif

#endif
