#include "include/aru/core/bindings/mesh_bindings.h"
#include <pybind11/pybind11.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

int add(int i, int j) { return i + j; }

namespace py = pybind11;
using namespace aru::core::bindings;

PYBIND11_MODULE(aru_py_mesh, m) {
  py::class_<PyMesh>(m, "PyDepth")
      .def(py::init<const std::string &>())
      .def("estimate_depth", &PyMesh::EstimateDepth, py::arg("image_left"),
           py::arg("image_right"))
      .def("create_dense_depth", &PyMesh::CreateDenseDepth,
           py::arg("sparse_depth"))
      .def("create_sparse_depth", &PyMesh::CreateSparseDepth,
           py::arg("image_left"), py::arg("image_right"))
      .def("draw_wireframe", &PyMesh::DrawWireFrame, py::arg("image_left"),
           py::arg("image_features"));
}
