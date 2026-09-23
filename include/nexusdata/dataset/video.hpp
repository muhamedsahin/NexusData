#pragma once

#include <string>

#include "nexusdata/dataset/iterable.hpp"
#include "nexusdata/image/image.hpp"

namespace nexusdata {

struct VideoOptions {
    /// 0 emits every decoded frame. A positive value keeps frames nearest to
    /// that rate, using the stream time base.
    double fps = 0;
    /// 0 means no cap.
    std::size_t max_frames = 0;
    /// Requests NVDEC. Without NEXUSDATA_WITH_NVDEC the constructor throws.
    bool hw_decode = false;
    ImageChannels channels = ImageChannels::RGB;
};

/// Frame extractor. Sample.input is HWC UInt8, label is the emitted frame
/// index, metadata["pts"] is the stream timestamp in seconds.
///
/// FFmpeg (libavformat/libavcodec/libswscale) is compiled only when
/// NEXUSDATA_WITH_FFMPEG=ON. Otherwise the constructor throws
/// InvalidArgumentError naming that flag. NVDEC (NEXUSDATA_WITH_NVDEC) decodes
/// on the GPU; the frame is copied back to host memory so the rest of the
/// pipeline can keep using a CPU NDArray.
class VideoIterable : public IterableDataset {
public:
    explicit VideoIterable(std::string path, VideoOptions opt = {});

    class It : public Iterator {
    public:
        It(std::string path, VideoOptions opt);
        ~It() override;
        [[nodiscard]] bool has_next() const override;
        [[nodiscard]] Sample next() override;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

    [[nodiscard]] std::unique_ptr<Iterator> make_iterator() const override;

private:
    std::string path_;
    VideoOptions opt_;
};

} // namespace nexusdata
