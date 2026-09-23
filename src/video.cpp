#include "nexusdata/dataset/video.hpp"

#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "nexusdata/core/error.hpp"

#if defined(NEXUSDATA_WITH_FFMPEG)
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/error.h>
#include <libavutil/imgutils.h>
#include <libavutil/rational.h>
#include <libswscale/swscale.h>
#if defined(NEXUSDATA_WITH_NVDEC)
#include <libavutil/hwcontext.h>
#endif
}
#endif

namespace nexusdata {

VideoIterable::VideoIterable(std::string path, VideoOptions opt)
    : path_(std::move(path)), opt_(opt) {
    if (opt_.hw_decode) {
#if !defined(NEXUSDATA_WITH_NVDEC)
        throw InvalidArgumentError(
            "VideoIterable: NVDEC is not linked. Reconfigure with -DNEXUSDATA_WITH_NVDEC=ON");
#else
        (void)path_;
#endif
    }
#if !defined(NEXUSDATA_WITH_FFMPEG)
    throw InvalidArgumentError(
        "VideoIterable: FFmpeg is not linked. Reconfigure with -DNEXUSDATA_WITH_FFMPEG=ON (" +
        path_ + ")");
#else
    if (path_.empty()) throw InvalidArgumentError("VideoIterable: empty path");
#endif
}

#if defined(NEXUSDATA_WITH_FFMPEG)

namespace {

AVPixelFormat dest_fmt(ImageChannels ch) {
    switch (ch) {
        case ImageChannels::Gray: return AV_PIX_FMT_GRAY8;
        case ImageChannels::RGBA: return AV_PIX_FMT_RGBA;
        case ImageChannels::Keep:
        case ImageChannels::RGB: return AV_PIX_FMT_RGB24;
    }
    return AV_PIX_FMT_RGB24;
}

int dest_channels(AVPixelFormat fmt) {
    if (fmt == AV_PIX_FMT_GRAY8) return 1;
    if (fmt == AV_PIX_FMT_RGBA) return 4;
    return 3;
}

#if defined(NEXUSDATA_WITH_NVDEC)
AVPixelFormat nvdec_get_format(AVCodecContext*, const AVPixelFormat* pix) {
    for (const AVPixelFormat* p = pix; *p != AV_PIX_FMT_NONE; ++p) {
        if (*p == AV_PIX_FMT_CUDA) return *p;
    }
    return AV_PIX_FMT_NONE;
}
#endif

struct AvCloser {
    AVFormatContext* fmt = nullptr;
    AVCodecContext* ctx = nullptr;
    AVPacket* pkt = nullptr;
    AVFrame* frame = nullptr;
    AVFrame* sw = nullptr;
    AVFrame* xfer = nullptr;
    SwsContext* sws = nullptr;
#if defined(NEXUSDATA_WITH_NVDEC)
    AVBufferRef* hw = nullptr;
#endif
    ~AvCloser() {
        sws_freeContext(sws);
        av_frame_free(&xfer);
        av_frame_free(&sw);
        av_frame_free(&frame);
        av_packet_free(&pkt);
        avcodec_free_context(&ctx);
#if defined(NEXUSDATA_WITH_NVDEC)
        av_buffer_unref(&hw);
#endif
        avformat_close_input(&fmt);
    }
};

} // namespace

struct VideoIterable::It::Impl {
    VideoOptions opt;
    AvCloser av;
    int stream = -1;
    AVRational time_base{};
    double next_time = 0;
    std::size_t emitted = 0;
    bool eof = false;
    std::optional<Sample> pending;

    explicit Impl(std::string path, VideoOptions o) : opt(o) {
        if (avformat_open_input(&av.fmt, path.c_str(), nullptr, nullptr) < 0) {
            throw IOError("VideoIterable: cannot open " + quote_path(path));
        }
        if (avformat_find_stream_info(av.fmt, nullptr) < 0) {
            throw IOError("VideoIterable: no stream info");
        }
        stream = av_find_best_stream(av.fmt, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        if (stream < 0) throw IOError("VideoIterable: no video stream");
        AVStream* st = av.fmt->streams[stream];
        time_base = st->time_base;
        const AVCodec* codec = avcodec_find_decoder(st->codecpar->codec_id);
        if (!codec) throw IOError("VideoIterable: no decoder for the video stream");
        av.ctx = avcodec_alloc_context3(codec);
        if (!av.ctx || avcodec_parameters_to_context(av.ctx, st->codecpar) < 0) {
            throw IOError("VideoIterable: cannot allocate the decoder");
        }
#if defined(NEXUSDATA_WITH_NVDEC)
        if (opt.hw_decode) {
            if (av_hwdevice_ctx_create(&av.hw, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0) < 0) {
                throw IOError("VideoIterable: cannot create a CUDA device for NVDEC");
            }
            av.ctx->hw_device_ctx = av_buffer_ref(av.hw);
            av.ctx->get_format = nvdec_get_format;
        }
#endif
        if (avcodec_open2(av.ctx, codec, nullptr) < 0) {
            throw IOError("VideoIterable: avcodec_open2 failed");
        }
        av.pkt = av_packet_alloc();
        av.frame = av_frame_alloc();
        av.sw = av_frame_alloc();
        av.xfer = av_frame_alloc();
        if (!av.pkt || !av.frame || !av.sw || !av.xfer) {
            throw IOError("VideoIterable: allocation failed");
        }
        pull();
    }

    Sample take_frame(AVFrame* src) {
        const AVPixelFormat fmt = dest_fmt(opt.channels);
        const int ch = dest_channels(fmt);
        av.sws = sws_getCachedContext(av.sws, src->width, src->height,
                                      static_cast<AVPixelFormat>(src->format), src->width,
                                      src->height, fmt, SWS_BILINEAR, nullptr, nullptr, nullptr);
        if (!av.sws) throw IOError("VideoIterable: swscale failed");
        av_frame_unref(av.sw);
        av.sw->format = fmt;
        av.sw->width = src->width;
        av.sw->height = src->height;
        if (av_frame_get_buffer(av.sw, 1) < 0) throw IOError("VideoIterable: frame buffer failed");
        if (sws_scale(av.sws, src->data, src->linesize, 0, src->height, av.sw->data,
                      av.sw->linesize) <= 0) {
            throw IOError("VideoIterable: sws_scale failed");
        }
        Image img;
        img.width = src->width;
        img.height = src->height;
        img.channels = ch;
        img.data = NDArray(Shape{static_cast<std::size_t>(src->height),
                                 static_cast<std::size_t>(src->width),
                                 static_cast<std::size_t>(ch)},
                           DType::UInt8);
        auto* dst = img.data.data<std::uint8_t>();
        for (int y = 0; y < src->height; ++y) {
            std::memcpy(dst + static_cast<std::size_t>(y) * static_cast<std::size_t>(src->width) *
                                  static_cast<std::size_t>(ch),
                        av.sw->data[0] + static_cast<std::size_t>(y) * static_cast<std::size_t>(av.sw->linesize[0]),
                        static_cast<std::size_t>(src->width) * static_cast<std::size_t>(ch));
        }
        const double pts = (src->pts == AV_NOPTS_VALUE) ? 0.0 : src->pts * av_q2d(time_base);
        Sample s;
        s.input = std::move(img.data);
        s.label = NDArray(Shape{1}, DType::Int64);
        s.label.data<std::int64_t>()[0] = static_cast<std::int64_t>(emitted);
        s.metadata.set("pts", std::to_string(pts));
        ++emitted;
        if (opt.fps > 0) next_time += 1.0 / opt.fps;
        return s;
    }

    bool accept(AVFrame* frame) {
        if (opt.max_frames > 0 && emitted >= opt.max_frames) return false;
        const double t = (frame->pts == AV_NOPTS_VALUE) ? next_time : frame->pts * av_q2d(time_base);
        if (opt.fps > 0 && t + 1e-6 < next_time) return false;
        return true;
    }

    bool receive_one() {
        for (;;) {
            const int rc = avcodec_receive_frame(av.ctx, av.frame);
            if (rc == AVERROR(EAGAIN) || rc == AVERROR_EOF) return false;
            if (rc < 0) throw IOError("VideoIterable: decode error");
            AVFrame* src = av.frame;
#if defined(NEXUSDATA_WITH_NVDEC)
            if (av.frame->format == AV_PIX_FMT_CUDA) {
                av_frame_unref(av.xfer);
                if (av_hwframe_transfer_data(av.xfer, av.frame, 0) < 0) {
                    throw IOError("VideoIterable: NVDEC transfer to host memory failed");
                }
                src = av.xfer;
            }
#endif
            const bool keep = accept(src);
            if (keep) pending = take_frame(src);
            av_frame_unref(av.frame);
            if (keep) return true;
            if (opt.max_frames > 0 && emitted >= opt.max_frames) return false;
        }
    }

    void pull() {
        if (pending || eof) return;
        while (!eof) {
            if (receive_one()) return;
            const int rd = av_read_frame(av.fmt, av.pkt);
            if (rd < 0) {
                avcodec_send_packet(av.ctx, nullptr);
                if (!receive_one()) eof = true;
                return;
            }
            if (av.pkt->stream_index == stream) {
                if (avcodec_send_packet(av.ctx, av.pkt) < 0) {
                    av_packet_unref(av.pkt);
                    throw IOError("VideoIterable: send_packet failed");
                }
            }
            av_packet_unref(av.pkt);
        }
    }
};

VideoIterable::It::It(std::string path, VideoOptions opt)
    : impl_(std::make_unique<Impl>(std::move(path), opt)) {}

VideoIterable::It::~It() = default;

bool VideoIterable::It::has_next() const { return impl_ && impl_->pending.has_value(); }

Sample VideoIterable::It::next() {
    if (!has_next()) throw IndexError("VideoIterable: exhausted");
    Sample s = std::move(*impl_->pending);
    impl_->pending.reset();
    impl_->pull();
    return s;
}

#else

struct VideoIterable::It::Impl {};

VideoIterable::It::It(std::string, VideoOptions) {
    throw InvalidArgumentError(
        "VideoIterable: FFmpeg is not linked. Reconfigure with -DNEXUSDATA_WITH_FFMPEG=ON");
}
VideoIterable::It::~It() = default;
bool VideoIterable::It::has_next() const { return false; }
Sample VideoIterable::It::next() {
    throw IndexError("VideoIterable: exhausted");
}

#endif

std::unique_ptr<IterableDataset::Iterator> VideoIterable::make_iterator() const {
    return std::make_unique<It>(path_, opt_);
}

} // namespace nexusdata
