#include "include/aru/core/bindings/mapping_bindings.h"
#include <pybind11/pybind11.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

int add(int i, int j) { return i + j; }

namespace py = pybind11;
using namespace aru::core::bindings;

PYBIND11_MODULE(aru_py_mapping, m) {
  py::class_<PyMapping>(m, "PyMapping")
      .def(py::init<const std::string &, const std::string &>())
      .def("fuse_depth", &PyMapping::FuseColourDepth, py::arg
           ("image_depth"),
           py::arg("image_rgb"),py::arg("position"))
      .def("add_image_at_time", &PyMapping::AddImageAtTime, py::arg
           ("image_depth"),
           py::arg("image_rgb"),py::arg("time"))
      .def("add_transform", &PyMapping::AddTransform, py::arg
           ("position"),py::arg("dest_time"),py::arg("src_time"))
      .def("save_ply", &PyMapping::SavePly,
           py::arg("output_ply_filename"));
}
