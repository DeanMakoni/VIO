#include "include/aru/core/bindings/localisation_bindings.h"
#include <pybind11/pybind11.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;
using namespace aru::core::bindings;

PYBIND11_MODULE(aru_py_localisation, m) {
  py::class_<PyLocalisation>(m, "PyLocalisation")
      .def(py::init<const std::string &, const std::string &,
                    const std::string &>())
      .def("add_vocab_training_image",
           &PyLocalisation::AddVocabularyTrainingImage, py::arg("image"))
      .def("add_query_image", &PyLocalisation::AddQueryImage, py::arg("image"))
      .def("add_sample_image", &PyLocalisation::AddSampleImage,
           py::arg("image"))
      .def("train_save_vocabulary", &PyLocalisation::TrainAndSaveVocabulary)
      .def("save_tree", &PyLocalisation::TrainAndSaveTree)
      .def("save_query_descriptors", &PyLocalisation::SaveQueryDescriptors,
           py::arg("query_filename"))
      .def("save_sample_descriptors", &PyLocalisation::SaveSampleDescriptors,
           py::arg("sample_filename"))
      .def("add_sample_data", &PyLocalisation::AddSampleData,
           py::arg("sample_filename"))
      .def("add_query_data", &PyLocalisation::AddQueryData,
           py::arg("query_filename"))
      .def("add_bow_data", &PyLocalisation::AddBowData,
           py::arg("query_bow_desc"))
      .def("initialise_localisation", &PyLocalisation::InitLocalisation)
      .def("localise_against", &PyLocalisation::LocaliseAgainst,
           py::arg("image"))
      .def("loop_close", &PyLocalisation::LoopClose, py::arg("image"),
           py::arg("max_index"));
}