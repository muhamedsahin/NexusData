#include <doctest/doctest.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "nexusdata/dataset/arrow_ipc.hpp"
#include "nexusdata/dataset/audio.hpp"
#include "nexusdata/dataset/compose.hpp"
#include "nexusdata/dataset/hdf5.hpp"
#include "nexusdata/dataset/in_memory.hpp"
#include "nexusdata/dataset/iterable.hpp"
#include "nexusdata/dataset/parquet.hpp"
#include "nexusdata/dataset/sqlite.hpp"
#include "nexusdata/dataset/text.hpp"
#include "nexusdata/dataset/webdataset.hpp"
#include "nexusdata/loading/batch.hpp"
#include "nexusdata/pipeline/persist.hpp"
#include "nexusdata/preprocessing/scaler.hpp"
#include "nexusdata/sampling/advanced.hpp"
#include "nexusdata/sampling/sequential.hpp"

using namespace nexusdata;

namespace {

std::string write_temp(const std::string& name, const std::string& body) {
    std::ofstream out(name, std::ios::binary);
    out.write(body.data(), static_cast<std::streamsize>(body.size()));
    return name;
}

std::shared_ptr<InMemoryDataset> make_range_ds(int n) {
    NDArray feats(Shape{static_cast<std::size_t>(n), 1}, DType::Float32);
    NDArray labs(Shape{static_cast<std::size_t>(n)}, DType::Int64);
    for (int i = 0; i < n; ++i) {
        feats.data<float>()[i] = static_cast<float>(i);
        labs.data<std::int64_t>()[i] = i % 2;
    }
    return std::make_shared<InMemoryDataset>(std::move(feats), std::move(labs));
}

void write_ustar_member(std::string& tar, const std::string& name, const std::string& data) {
    char hdr[512] = {};
    std::memcpy(hdr, name.c_str(), std::min<std::size_t>(name.size(), 100));
    std::snprintf(hdr + 100, 8, "%07o", 0644);
    std::snprintf(hdr + 108, 8, "%07o", 0);
    std::snprintf(hdr + 116, 8, "%07o", 0);
    std::snprintf(hdr + 124, 12, "%011o", static_cast<unsigned>(data.size()));
    std::snprintf(hdr + 136, 12, "%011o", 0);
    hdr[156] = '0';
    std::memcpy(hdr + 257, "ustar", 5);
    hdr[262] = '0';
    hdr[263] = '0';
    unsigned sum = 0;
    std::memset(hdr + 148, ' ', 8);
    for (int i = 0; i < 512; ++i) {
        sum += static_cast<unsigned char>(hdr[i]);
    }
    std::snprintf(hdr + 148, 8, "%06o", sum);
    hdr[154] = '\0';
    hdr[155] = ' ';
    tar.append(hdr, 512);
    tar.append(data);
    while (tar.size() % 512 != 0) {
        tar.push_back('\0');
    }
}

std::string tiny_bmp() {
    const unsigned char bmp[] = {
        0x42, 0x4D, 0x3A, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x36, 0x00, 0x00, 0x00, 0x28, 0x00,
        0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x18, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x13, 0x0B, 0x00, 0x00, 0x13, 0x0B, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0x00};
    return std::string(reinterpret_cast<const char*>(bmp), sizeof(bmp));
}

std::string tiny_wav_pcm16() {
    std::string w;
    auto put_u32 = [&](std::uint32_t v) {
        w.push_back(static_cast<char>(v & 0xff));
        w.push_back(static_cast<char>((v >> 8) & 0xff));
        w.push_back(static_cast<char>((v >> 16) & 0xff));
        w.push_back(static_cast<char>((v >> 24) & 0xff));
    };
    auto put_u16 = [&](std::uint16_t v) {
        w.push_back(static_cast<char>(v & 0xff));
        w.push_back(static_cast<char>((v >> 8) & 0xff));
    };
    const std::uint32_t data_sz = 16;
    w += "RIFF";
    put_u32(36 + data_sz);
    w += "WAVE";
    w += "fmt ";
    put_u32(16);
    put_u16(1);
    put_u16(1);
    put_u32(8000);
    put_u32(16000);
    put_u16(2);
    put_u16(16);
    w += "data";
    put_u32(data_sz);
    for (int i = 0; i < 8; ++i) {
        put_u16(static_cast<std::uint16_t>(i * 1000));
    }
    return w;
}

} // namespace

TEST_CASE("IterableFromMap and ShuffleBuffer") {
    auto ds = make_range_ds(5);
    auto itds = std::make_shared<IterableFromMap>(ds);
    auto it = itds->make_iterator();
    int count = 0;
    while (it->has_next()) {
        auto s = it->next();
        CHECK(s.input.data<float>()[0] == static_cast<float>(count));
        ++count;
    }
    CHECK(count == 5);

    auto shuf = std::make_shared<ShuffleBufferIterable>(itds, 3, 42);
    auto it2 = shuf->make_iterator();
    std::vector<float> vals;
    while (it2->has_next()) {
        vals.push_back(it2->next().input.data<float>()[0]);
    }
    CHECK(vals.size() == 5);
}

TEST_CASE("Filter Zip Concat take skip Repeat") {
    auto ds = make_range_ds(6);
    FilterDataset filt(ds, [](const Sample& s) {
        return s.label.data<std::int64_t>()[0] == 0;
    });
    CHECK(filt.size() == 3);

    auto a = make_range_ds(6);
    auto b = make_range_ds(6);
    ZipDataset zip(a, b);
    CHECK(zip.size() == 6);

    ConcatDataset cat({a, b});
    CHECK(cat.size() == 12);

    auto t = take(ds, 2);
    CHECK(t.size() == 2);
    auto sk = skip(ds, 4);
    CHECK(sk.size() == 2);
    RepeatDataset rep(ds, 2);
    CHECK(rep.size() == 12);
}

TEST_CASE("Weighted Bucket Distributed Batch samplers") {
    WeightedRandomSampler w({1.0, 2.0, 3.0}, 20, 7, true);
    auto wi = w.indices();
    CHECK(wi.size() == 20);
    for (auto i : wi) {
        CHECK(i < 3);
    }

    BucketSampler buck({5, 1, 3, 2, 4}, 2, 1, false);
    auto bi = buck.indices();
    CHECK(bi.size() == 5);

    DistributedSampler dist(10, 0, 2, false, 0);
    auto d0 = dist.indices();
    DistributedSampler dist1(10, 1, 2, false, 0);
    auto d1 = dist1.indices();
    CHECK(d0.size() == d1.size());
    CHECK(d0[0] == 0);
    CHECK(d1[0] == 1);

    auto seq = std::make_shared<SequentialSampler>(8);
    BatchSampler bs(seq, 3, true);
    CHECK(bs.size() == 2);
    CHECK(bs.indices().size() == 6);
    CHECK(bs.batches().size() == 2);
}

TEST_CASE("pad_sequence and ragged collate") {
    std::vector<Sample> batch;
    for (int len : {2, 4, 3}) {
        Sample s;
        s.input = NDArray(Shape{static_cast<std::size_t>(len)}, DType::Float32);
        for (int i = 0; i < len; ++i) {
            s.input.data<float>()[i] = static_cast<float>(i + 1);
        }
        s.label = NDArray(Shape{1}, DType::Int64);
        s.label.data<std::int64_t>()[0] = len;
        batch.push_back(std::move(s));
    }
    auto b = collate_pad_sequence(batch);
    CHECK(b.inputs.shape().size() == 2);
    CHECK(b.inputs.shape()[0] == 3);
    CHECK(b.inputs.shape()[1] == 4);
    CHECK(b.mask.shape()[1] == 4);
    CHECK(b.mask.data<std::uint8_t>()[0 * 4 + 1] == 1);
    CHECK(b.mask.data<std::uint8_t>()[0 * 4 + 3] == 0);

    auto r = collate_ragged(batch);
    CHECK(r.inputs.size() == 3);
    CHECK(r.inputs[1].numel() == 4);
}

TEST_CASE("TextLineDataset and tokenizer") {
    const auto path = write_temp("test_v06_lines.txt", "hello world\n\nfoo\n");
    TextOptions opt;
    TextLineDataset ds(path, opt);
    CHECK(ds.size() == 2);
    auto s = ds.get(0);
    CHECK(s.input.dtype() == DType::UInt8);
    CHECK(s.input.numel() == 11);

    TextLineIterable itds(path, opt);
    auto it = itds.make_iterator();
    CHECK(it->has_next());
    (void)it->next();
    CHECK(it->has_next());
    (void)it->next();
    CHECK_FALSE(it->has_next());

    WhitespaceHashTokenizer tok(100);
    auto ids = tok.encode("hello world");
    CHECK(ids.size() == 2);
    std::remove(path.c_str());
}

TEST_CASE("WavDataset") {
    const auto path = write_temp("test_v06.wav", tiny_wav_pcm16());
    WavInfo info;
    auto arr = load_wav(path, &info);
    CHECK(info.sample_rate == 8000);
    CHECK(info.channels == 1);
    CHECK(arr.shape()[0] == 8);
    WavDataset ds(path);
    CHECK(ds.size() == 1);
    auto s = ds.get(0);
    CHECK(s.input.dtype() == DType::Float32);
    std::remove(path.c_str());
}

TEST_CASE("WebDataset ustar") {
    std::string tar;
    write_ustar_member(tar, "a.bmp", tiny_bmp());
    write_ustar_member(tar, "a.cls", "7\n");
    write_ustar_member(tar, "b.bmp", tiny_bmp());
    write_ustar_member(tar, "b.cls", "3\n");
    tar.append(1024, '\0');
    const auto path = write_temp("test_v06.tar", tar);
    WebDatasetOptions opt;
    opt.image_decode.channels = ImageChannels::RGB;
    WebDataset ds(path, opt);
    CHECK(ds.size() == 2);
    auto s0 = ds.get(0);
    CHECK(s0.input.shape().size() == 3);
    CHECK(s0.label.data<std::int64_t>()[0] == 7);
    std::remove(path.c_str());
}

TEST_CASE("Sqlite Parquet HDF5 optional stubs") {
    CHECK_FALSE(sqlite_support_enabled());
    CHECK_FALSE(hdf5_support_enabled());
    CHECK_THROWS_AS(SqliteDataset("x.db", {}), InvalidArgumentError);
    CHECK_THROWS_AS(Hdf5Dataset("x.h5", {}), InvalidArgumentError);
#if defined(NEXUSDATA_WITH_ARROW)
    CHECK(parquet_support_enabled());
    CHECK(arrow_ipc_support_enabled());
#else
    CHECK_FALSE(parquet_support_enabled());
    CHECK_FALSE(arrow_ipc_support_enabled());
    CHECK_THROWS_AS(ParquetDataset("x.parquet", {}), InvalidArgumentError);
    CHECK_THROWS_AS(ArrowIpcDataset("x.feather", {}), InvalidArgumentError);
#endif
}

TEST_CASE("Pipeline save load") {
    NDArray X(Shape{4, 2}, DType::Float64);
    for (std::size_t i = 0; i < 8; ++i) {
        X.data<double>()[i] = static_cast<double>(i);
    }
    auto scaler = std::make_shared<StandardScaler>();
    scaler->fit(X);

    Pipeline pipe;
    pipe.add_transformer(scaler);
    pipe.add_compose_step("Clip -10 10");
    pipe.rebuild_compose_from_steps();

    const auto path = "test_v06_pipe.txt";
    pipe.save_file(path);

    Pipeline loaded;
    loaded.load_file(path);
    CHECK(loaded.transformers().size() == 1);
    auto Y = loaded.transform_matrix(X);
    CHECK(Y.shape()[0] == 4);
    Sample s;
    s.input = NDArray(Shape{2}, DType::Float64);
    s.input.data<double>()[0] = 100.0;
    s.input.data<double>()[1] = -100.0;
    s.label = NDArray(Shape{1}, DType::Int64);
    auto out = loaded.apply(s);
    CHECK(out.input.data<double>()[0] == 10.0);
    CHECK(out.input.data<double>()[1] == -10.0);
    std::remove(path);
}
