#include <cstdint>
#include <iostream>

#include "nexusdata/bench/harness.hpp"
#include "nexusdata/dataset/function_dataset.hpp"

using namespace nexusdata;

int main() {
    constexpr std::size_t n = 200000;
    DatasetPtr ds = make_dataset(n, [](std::size_t i) {
        Sample s;
        s.input = NDArray(Shape{1}, DType::Float32);
        s.input.data<float>()[0] = static_cast<float>(i);
        s.label = NDArray(Shape{1}, DType::Int64);
        s.label.data<std::int64_t>()[0] = static_cast<std::int64_t>(i);
        return s;
    });
    const double seconds = bench::time_best([&] {
        double sum = 0;
        for (std::size_t i = 0; i < n; ++i) sum += ds->get(i).input.data<float>()[0];
        if (sum < 0) std::cerr << sum;
    });
    std::cout << "function_dataset_200k_gets " << seconds << " s\n";
    return 0;
}
