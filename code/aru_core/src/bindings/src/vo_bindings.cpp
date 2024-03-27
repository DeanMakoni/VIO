#include "include/aru/core/bindings/vo_bindings.h"
#include <pybind11/pybind11.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

int add(int i, int j) { return i + j; }

namespace py = pybind11;
using namespace aru::core::bindings;

PYBIND11_MODULE(aru_py_vo, m) {
  py::class_<PyVO>(m, "PyVO")
      .def(py::init<const std::string &, const std::string &>())
      .def("stereo_odometry", &PyVO::StereoOdometry, py::arg("image_1_left"),
           py::arg("image_1_right"), py::arg("timestamp_1"),
           py::arg("image_2_left"), py::arg("image_2_right"),
           py::arg("timestamp_2"))
      .def("stereo_matches", &PyVO::StereoMatches, py::arg("image_1_left"),
           py::arg("image_1_right"))
      .def("motion_estimation", &PyVO::MotionEstimation, py::arg("landmarks"),
           py::arg("keypoints"));
}
