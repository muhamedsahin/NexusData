#include "nexusdata/dataset/hdf5.hpp"

namespace nexusdata {

bool hdf5_support_enabled() {
#if defined(NEXUSDATA_WITH_HDF5)
    return true;
#else
    return false;
#endif
}

Hdf5Dataset::Hdf5Dataset(const std::string&, Hdf5Options) {
#if defined(NEXUSDATA_WITH_HDF5)
    throw InvalidArgumentError("Hdf5Dataset: HDF5 backend not linked in this build");
#else
    throw InvalidArgumentError(
        "Hdf5Dataset: built without HDF5. Reconfigure with -DNEXUSDATA_WITH_HDF5=ON");
#endif
}

std::size_t Hdf5Dataset::size() const {
    return n_;
}

Sample Hdf5Dataset::get(std::size_t) const {
    throw IndexError("Hdf5Dataset::get: not available");
}

} // namespace nexusdata
