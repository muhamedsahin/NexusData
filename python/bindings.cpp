#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>

#include "nexusdata/adapter/dlpack.hpp"
#include "nexusdata/core/ndarray.hpp"
#include "nexusdata/dataset/csv/csv_dataset.hpp"
#include "nexusdata/dataset/in_memory.hpp"
#include "nexusdata/loading/dataloader.hpp"

namespace py = pybind11;
using namespace nexusdata;

namespace {

py::array ndarray_to_numpy(const NDArray& a) {
    if (!a.device().is_host()) {
        throw std::runtime_error("numpy export requires host NDArray");
    }
    std::vector<py::ssize_t> shape(a.shape().begin(), a.shape().end());
    if (a.dtype() == DType::Float32) {
        py::array_t<float> out(shape);
        std::memcpy(out.mutable_data(), a.data(), a.nbytes());
        return std::move(out);
    }
    if (a.dtype() == DType::Float64) {
        py::array_t<double> out(shape);
        std::memcpy(out.mutable_data(), a.data(), a.nbytes());
        return std::move(out);
    }
    if (a.dtype() == DType::Int64) {
        py::array_t<std::int64_t> out(shape);
        std::memcpy(out.mutable_data(), a.data(), a.nbytes());
        return std::move(out);
    }
    throw std::runtime_error("ndarray_to_numpy: dtype not wrapped in v0.9");
}

NDArray numpy_to_ndarray(py::array arr) {
    arr = py::array::ensure(arr);
    Shape shape;
    for (py::ssize_t i = 0; i < arr.ndim(); ++i) {
        shape.push_back(static_cast<std::size_t>(arr.shape(i)));
    }
    if (py::isinstance<py::array_t<float>>(arr)) {
        auto a = py::array_t<float>::ensure(arr);
        NDArray out(shape, DType::Float32);
        std::memcpy(out.data(), a.data(), out.nbytes());
        return out;
    }
    if (py::isinstance<py::array_t<double>>(arr)) {
        auto a = py::array_t<double>::ensure(arr);
        NDArray out(shape, DType::Float64);
        std::memcpy(out.data(), a.data(), out.nbytes());
        return out;
    }
    if (py::isinstance<py::array_t<std::int64_t>>(arr)) {
        auto a = py::array_t<std::int64_t>::ensure(arr);
        NDArray out(shape, DType::Int64);
        std::memcpy(out.data(), a.data(), out.nbytes());
        return out;
    }
    throw std::runtime_error("numpy_to_ndarray: unsupported dtype");
}

} // namespace

PYBIND11_MODULE(nexusdata_py, m) {
    m.doc() = "NexusData Python bindings (v0.9 optional)";
    m.attr("__version__") = "1.0.0";

    py::enum_<DType>(m, "DType")
        .value("Float32", DType::Float32)
        .value("Float64", DType::Float64)
        .value("Int64", DType::Int64)
        .value("UInt8", DType::UInt8)
        .export_values();

    py::class_<NDArray>(m, "NDArray")
        .def(py::init([](const std::vector<std::size_t>& shape, DType dt) {
            return NDArray(Shape(shape.begin(), shape.end()), dt);
        }))
        .def_property_readonly("shape",
                               [](const NDArray& a) {
                                   return std::vector<std::size_t>(a.shape().begin(), a.shape().end());
                               })
        .def_property_readonly("dtype", &NDArray::dtype)
        .def_property_readonly("numel", &NDArray::numel)
        .def("numpy", &ndarray_to_numpy)
        .def_static("from_numpy", &numpy_to_ndarray);

    m.def("to_dlpack", [](const NDArray& a) {
        DLManagedTensor* t = ndarray_to_dlpack(a);
        return py::capsule(t, "dltensor", [](void* p) {
            auto* m = static_cast<DLManagedTensor*>(p);
            if (m && m->deleter) {
                m->deleter(m);
            }
        });
    });
}
