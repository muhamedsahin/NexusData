#include <iostream>

#include "nexusdata/nexusdata.hpp"
#include "nexusdata/version.hpp"

int main() {
    using namespace nexusdata;
    std::cout << "NexusData " << version_string() << " major=" << version_major() << "\n";
    if (version_major() < 1) {
        return 1;
    }
    NDArray a(Shape{2, 2}, DType::Float32);
    a.data<float>()[0] = 1.0f;
    Batch b;
    b.inputs = a;
    b.labels = NDArray(Shape{2}, DType::Int64);
    auto view = MatrixFlashAdapter::view_batch(b);
    if (view.inputs.shape.size() != 2) {
        return 2;
    }
    std::cout << "find_package smoke OK\n";
    return 0;
}
