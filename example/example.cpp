#include <iostream>
#include <fstream>

#include "nexusdata/nexusdata.hpp"

using namespace nexusdata;

int main() {
    // Write a tiny train.csv next to the executable cwd for the demo.
    {
        std::ofstream out("train.csv");
        out << "f1,f2,f3,target\n";
        for (int i = 0; i < 32; ++i) {
            out << i << "," << (i * 2) << "," << (i * 3) << "," << (i % 2) << "\n";
        }
    }

    CSVOptions opt;
    opt.label_column = "target";
    CSVDataset data("train.csv", opt);

    auto [train, val] = random_split(data, {0.8, 0.2}, /*seed=*/42);

    DataLoaderOptions lo;
    lo.batch_size = 8;
    lo.shuffle = true;
    lo.seed = 42;
    DataLoader loader(train, lo);

    std::cout << "NexusData " << version_string()
              << " train=" << train.size() << " val=" << val.size()
              << " batches=" << loader.size() << "\n";

    for (const Batch& b : loader) {
        std::cout << "batch inputs=" << shape_to_string(b.inputs.shape())
                  << " labels=" << shape_to_string(b.labels.shape()) << "\n";
    }
    return 0;
}
