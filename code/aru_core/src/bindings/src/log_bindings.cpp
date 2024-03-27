#include "include/aru/core/bindings/log_bindings.h"
#include <pybind11/pybind11.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

int add(int i, int j) { return i + j; }

namespace py = pybind11;
using namespace aru::core::bindings;


PYBIND11_MODULE(aru_py_logger, m) {

  py::class_<StereoImageLogger>(m, "StereoImageLogger")
      .def(py::init<const std::string &,bool >())
      .def("read_from_file", &StereoImageLogger::ReadFromFile)
      .def("read_index_from_file", &StereoImageLogger::ReadIndexFromFile,py::arg
           ("index"))
      .def("write_to_file", &StereoImageLogger::WriteToFile,
           py::arg("image_left"), py::arg("image_right"), py::arg("timestamp"))
      .def("end_of_file", &StereoImageLogger::EndOfFile);

  py::class_<MonoImageLogger>(m, "MonoImageLogger")
      .def(py::init<const std::string &,bool >())
      .def("read_from_file", &MonoImageLogger::ReadFromFile)
      .def("read_index_from_file", &MonoImageLogger::ReadIndexFromFile,py::arg
           ("index"))
      .def("read_channel_from_file", &MonoImageLogger::ReadChannelFromFile)
      .def("write_to_file", &MonoImageLogger::WriteToFile,
           py::arg("image"), py::arg("timestamp"))
      .def("end_of_file", &MonoImageLogger::EndOfFile);

  py::class_<LaserLogger>(m, "LaserLogger")
      .def(py::init<const std::string &,bool >())
      .def("read_from_file", &LaserLogger::ReadFromFile)
      .def("read_index_from_file", &LaserLogger::ReadIndexFromFile,py::arg
           ("index"))
      .def("write_to_file", &LaserLogger::WriteToFile,
           py::arg("points"), py::arg("timestamp"));

  py::class_<TransformLogger>(m, "TransformLogger")
      .def(py::init<const std::string &,bool >())
      .def("read_from_file", &TransformLogger::ReadFromFile)
      .def("read_index_from_file", &TransformLogger::ReadIndexFromFile,py::arg
           ("index"))
      .def("write_to_file", &TransformLogger::WriteToFile,
           py::arg("transform"), py::arg("source_timestamp"),py::arg
           ("dest_timestamp"))
      .def("end_of_file", &TransformLogger::EndOfFile);

    py::class_<ExperienceLogger>(m, "ExperienceLogger")
            .def(py::init<const std::string &,bool >())
            .def("read_from_file", &ExperienceLogger::ReadFromFile)
            .def("write_to_file", &ExperienceLogger::WriteToFile, py::arg("timestamp"),
                 py::arg("left_image"), py::arg("cv_keypoints"),py::arg
                         ("cv_desc"), py::arg("landmarks"), py::arg("query_desc"))
            .def("end_of_file", &ExperienceLogger::EndOfFile);
}
