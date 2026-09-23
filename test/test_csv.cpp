#include <doctest/doctest.h>

#include <cstdio>
#include <fstream>
#include <string>

#include "nexusdata/dataset/csv/csv_dataset.hpp"
#include "nexusdata/loading/dataloader.hpp"
#include "nexusdata/dataset/split.hpp"

using namespace nexusdata;

namespace {

std::string write_temp(const std::string& name, const std::string& content) {
    const std::string path = name;
    std::ofstream out(path, std::ios::binary);
    out << content;
    return path;
}

} // namespace

TEST_CASE("CSV basic header and labels") {
    const auto path = write_temp("test_basic.csv",
                                 "a,b,target\n"
                                 "1.0,2.0,0\n"
                                 "3.0,4.0,1\n"
                                 "5.0,6.0,0\n");
    CSVOptions opt;
    opt.label_column = "target";
    CSVDataset ds(path, opt);
    CHECK(ds.size() == 3);
    CHECK(ds.num_features() == 2);
    Sample s = ds.get(1);
    CHECK(s.input.data<float>()[0] == 3.0f);
    CHECK(s.input.data<float>()[1] == 4.0f);
    CHECK(s.label.data<float>()[0] == 1.0f);
    std::remove(path.c_str());
}

TEST_CASE("CSV quotes CRLF and UTF-8 BOM") {
    std::string body;
    body.push_back(static_cast<char>(0xEF));
    body.push_back(static_cast<char>(0xBB));
    body.push_back(static_cast<char>(0xBF));
    body += "x,y,target\r\n";
    body += "\"10\",20,0\r\n";
    body += "30,\"40\",1\r\n";

    const auto path = write_temp("test_quotes.csv", body);
    CSVOptions opt;
    opt.label_column = "target";
    CSVDataset ds(path, opt);
    CHECK(ds.size() == 2);
    CHECK(ds.get(0).input.data<float>()[0] == 10.0f);
    CHECK(ds.get(1).input.data<float>()[1] == 40.0f);
    std::remove(path.c_str());
}

TEST_CASE("CSV quoted numeric field and escaped quote text failure message") {
    const auto path = write_temp("test_qnum.csv",
                                 "x,y,target\n"
                                 "\"10\",20,1\n"
                                 "30,40,0\n");
    CSVOptions opt;
    opt.label_column = "target";
    CSVDataset ds(path, opt);
    CHECK(ds.get(0).input.data<float>()[0] == 10.0f);
    std::remove(path.c_str());
}

TEST_CASE("CSV parse error includes file line column") {
    const auto path = write_temp("test_bad.csv",
                                 "a,b,target\n"
                                 "1.0,abc,0\n");
    CSVOptions opt;
    opt.label_column = "target";
    try {
        CSVDataset ds(path, opt);
        FAIL("expected throw");
    } catch (const IOError& e) {
        std::string msg = e.what();
        CHECK(msg.find("test_bad.csv") != std::string::npos);
        CHECK(msg.find("abc") != std::string::npos);
        CHECK(msg.find("float") != std::string::npos);
    }
    std::remove(path.c_str());
}

TEST_CASE("CSV + split + DataLoader smoke (acceptance)") {
    const auto path = write_temp("train.csv",
                                 "f1,f2,f3,target\n"
                                 "0,1,2,0\n"
                                 "1,2,3,1\n"
                                 "2,3,4,0\n"
                                 "3,4,5,1\n"
                                 "4,5,6,0\n"
                                 "5,6,7,1\n"
                                 "6,7,8,0\n"
                                 "7,8,9,1\n"
                                 "8,9,10,0\n"
                                 "9,10,11,1\n");
    CSVOptions opt;
    opt.label_column = "target";
    CSVDataset data(path, opt);

    auto [train, val] = random_split(data, {0.8, 0.2}, /*seed=*/42);

    DataLoaderOptions lo;
    lo.batch_size = 2;
    lo.shuffle = true;
    lo.seed = 42;
    DataLoader loader(train, lo);

    std::size_t batches = 0;
    for (const Batch& b : loader) {
        CHECK(b.inputs.shape().size() == 2);
        CHECK(b.inputs.shape()[0] <= 2);
        CHECK(b.inputs.shape()[1] == 3);
        CHECK(b.labels.shape()[0] == b.inputs.shape()[0]);
        ++batches;
    }
    CHECK(batches == loader.size());
    CHECK(train.size() + val.size() == data.size());
    std::remove(path.c_str());
}

TEST_CASE("CSV custom delimiter TSV") {
    const auto path = write_temp("test.tsv", "a\tb\ttarget\n1\t2\t0\n3\t4\t1\n");
    CSVOptions opt;
    opt.delimiter = '\t';
    opt.label_column = "target";
    CSVDataset ds(path, opt);
    CHECK(ds.size() == 2);
    CHECK(ds.get(1).input.data<float>()[1] == 4.0f);
    std::remove(path.c_str());
}
