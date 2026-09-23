#include <doctest/doctest.h>

#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>

#include "nexusdata/preprocessing/encoder.hpp"
#include "nexusdata/preprocessing/imputer.hpp"
#include "nexusdata/preprocessing/scaler.hpp"

using namespace nexusdata;

static NDArray make_X() {
    // rows: [0, 2], [2, 0], [4, 4]
    NDArray X({3, 2}, DType::Float32);
    float* p = X.data<float>();
    p[0] = 0; p[1] = 2;
    p[2] = 2; p[3] = 0;
    p[4] = 4; p[5] = 4;
    return X;
}

TEST_CASE("StandardScaler mean std and inverse-ish") {
    auto X = make_X();
    StandardScaler sc;
    auto Xt = sc.fit_transform(X);
    CHECK(sc.is_fitted());
    // Column 0 mean = 2, std = sqrt((4+0+4)/3)=sqrt(8/3)
    CHECK(sc.mean()[0] == doctest::Approx(2.0));
    CHECK(sc.mean()[1] == doctest::Approx(2.0));
    double sum0 = 0;
    for (std::size_t i = 0; i < 3; ++i) {
        sum0 += Xt.data<double>()[i * 2];
    }
    CHECK(sum0 == doctest::Approx(0.0).epsilon(1e-9));
}

TEST_CASE("StandardScaler save/load roundtrip") {
    auto X = make_X();
    StandardScaler sc;
    sc.fit(X);
    std::stringstream ss;
    sc.save(ss);
    StandardScaler sc2;
    sc2.load(ss);
    auto a = sc.transform(X);
    auto b = sc2.transform(X);
    for (std::size_t i = 0; i < a.numel(); ++i) {
        CHECK(a.data<double>()[i] == doctest::Approx(b.data<double>()[i]));
    }
}

TEST_CASE("MinMaxScaler to [0,1]") {
    auto X = make_X();
    MinMaxScaler mm;
    auto Xt = mm.fit_transform(X);
    CHECK(Xt.data<double>()[0] == doctest::Approx(0.0)); // col0 min
    CHECK(Xt.data<double>()[4] == doctest::Approx(1.0)); // col0 max
}

TEST_CASE("RobustScaler and MaxAbsScaler") {
    auto X = make_X();
    RobustScaler rs;
    rs.fit(X);
    CHECK(rs.is_fitted());
    auto Xt = rs.transform(X);
    CHECK(Xt.numel() == 6);

    MaxAbsScaler ma;
    auto X2 = ma.fit_transform(X);
    CHECK(std::abs(X2.data<double>()[4]) <= 1.0 + 1e-9);
}

TEST_CASE("Imputer mean fills NaN") {
    NDArray X({3, 1}, DType::Float32);
    X.data<float>()[0] = 1.0f;
    X.data<float>()[1] = std::numeric_limits<float>::quiet_NaN();
    X.data<float>()[2] = 3.0f;
    Imputer imp(ImputeStrategy::Mean);
    auto Xt = imp.fit_transform(X);
    CHECK(Xt.data<double>()[1] == doctest::Approx(2.0));
}

TEST_CASE("LabelEncoder") {
    NDArray y({4}, DType::Float32);
    y.data<float>()[0] = 10;
    y.data<float>()[1] = 20;
    y.data<float>()[2] = 10;
    y.data<float>()[3] = 30;
    LabelEncoder le;
    auto yt = le.fit_transform(y);
    CHECK(le.classes().size() == 3);
    CHECK(yt.data<std::int64_t>()[0] == yt.data<std::int64_t>()[2]);
    CHECK(yt.data<std::int64_t>()[1] != yt.data<std::int64_t>()[0]);
}

TEST_CASE("OneHotEncoder") {
    NDArray X({3, 1}, DType::Float32);
    X.data<float>()[0] = 0;
    X.data<float>()[1] = 1;
    X.data<float>()[2] = 0;
    OneHotEncoder oh;
    auto Xt = oh.fit_transform(X);
    CHECK(oh.n_features_out() == 2);
    CHECK(Xt.shape() == Shape{3, 2});
    CHECK(Xt.data<double>()[0] == doctest::Approx(1.0));
    CHECK(Xt.data<double>()[1] == doctest::Approx(0.0));
}

TEST_CASE("OrdinalEncoder Binarizer Normalizer") {
    NDArray X({2, 2}, DType::Float32);
    X.data<float>()[0] = 1; X.data<float>()[1] = 2;
    X.data<float>()[2] = 1; X.data<float>()[3] = 3;
    OrdinalEncoder oe;
    auto Xt = oe.fit_transform(X);
    CHECK(Xt.data<double>()[0] == doctest::Approx(0.0));

    Binarizer bin(1.5);
    bin.fit(X);
    auto Xb = bin.transform(X);
    CHECK(Xb.data<double>()[1] == doctest::Approx(1.0));

    Normalizer norm(NormType::L2);
    auto Xn = norm.fit_transform(X);
    double n0 = std::sqrt(Xn.data<double>()[0] * Xn.data<double>()[0] +
                          Xn.data<double>()[1] * Xn.data<double>()[1]);
    CHECK(n0 == doctest::Approx(1.0));
}

TEST_CASE("no leakage: scaler fit on train only") {
    NDArray train({2, 1}, DType::Float32);
    train.data<float>()[0] = 0;
    train.data<float>()[1] = 2;
    NDArray test({1, 1}, DType::Float32);
    test.data<float>()[0] = 4;
    StandardScaler sc;
    sc.fit(train);
    auto te = sc.transform(test);
    // mean=1, std=1 -> (4-1)/1 = 3
    CHECK(te.data<double>()[0] == doctest::Approx(3.0));
}
