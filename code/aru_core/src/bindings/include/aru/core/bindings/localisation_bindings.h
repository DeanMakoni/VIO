#ifndef ARU_CORE_LOCASATION_BINDINGS_H_
#define ARU_CORE_LOCASATION_BINDINGS_H_

#include <aru/core/bindings/conversions.h>
#include <aru/core/localisation/localisation.h>

#include <Eigen/Dense>
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>
#include <glog/logging.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <utility>

#include <opencv2/opencv.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace aru {
namespace core {
namespace bindings {

class PyLocalisation {
public:
  PyLocalisation(std::string vocabulary_filename, std::string chow_liu_filename,
                 std::string settings_filename);

  virtual ~PyLocalisation() = default;

  void AddVocabularyTrainingImage(pybind11::array_t<unsigned char> &image);

  void AddSampleImage(pybind11::array_t<unsigned char> &image);

  pybind11::array_t<float> AddQueryImage(pybind11::array_t<unsigned char> &image);

  void SaveQueryDescriptors(const std::string &query_descriptors_file);

  void SaveSampleDescriptors(const std::string &sample_descriptors_file);

  void TrainAndSaveTree();

  void InitLocalisation();

  void AddSampleData(const std::string &sample_descriptors_file);

  void AddQueryData(const std::string &query_descriptors_file);

  void AddBowData(pybind11::array_t<float> &query_bow_desc);

  void InitialiseLocalisation(const std::string &query_descriptors_file,
                              const std::string &sample_descriptors_file);

  void TrainAndSaveVocabulary();
  
  std::tuple<int, float>
  LocaliseAgainst(pybind11::array_t<unsigned char> &image);

  std::tuple<int, float> LoopClose(pybind11::array_t<unsigned char> &image,
                                   int max_index);


private:
  std::string vocab_filename_;
  std::string chow_liu_filename_;
  std::string settings_filename_;

  boost::shared_ptr<aru::core::localisation::Localisation> localiser_;
};

//------------------------------------------------------------------------------
PyLocalisation::PyLocalisation(std::string vocabulary_filename,
                               std::string chow_liu_filename,
                               std::string settings_filename)
    : vocab_filename_(std::move(vocabulary_filename)),
      chow_liu_filename_(std::move(chow_liu_filename)),
      settings_filename_(std::move(settings_filename)) {

  localiser_ = boost::make_shared<aru::core::localisation::Localisation>(
      vocab_filename_, chow_liu_filename_, settings_filename_);
}

//------------------------------------------------------------------------------
void PyLocalisation::AddVocabularyTrainingImage(
    pybind11::array_t<unsigned char> &image) {
  cv::Mat image_mat = numpy_uint8_3c_to_cv_mat(image);
  localiser_->AddVocabularyTrainingImage(image_mat);
}

//------------------------------------------------------------------------------
void PyLocalisation::TrainAndSaveVocabulary() {
  localiser_->TrainAndSaveVocabulary();
}

//------------------------------------------------------------------------------
pybind11::array_t<float> PyLocalisation::AddQueryImage(pybind11::array_t<unsigned char> &image) {
  cv::Mat query_image = numpy_uint8_3c_to_cv_mat(image);
  auto bow_out = localiser_->AddQueryImage(query_image);
//  std::cout << "BOW DESC ROWS\t " << bow_out.rows << "\t BOW DESC COLS \t" << bow_out.cols << "\n";
  return cv_mat_float_1c_to_numpy(bow_out);
}

//------------------------------------------------------------------------------
void PyLocalisation::AddSampleImage(pybind11::array_t<unsigned char> &image) {
  cv::Mat sample_image = numpy_uint8_3c_to_cv_mat(image);
  localiser_->AddSampleImage(sample_image);
}

//------------------------------------------------------------------------------
void PyLocalisation::TrainAndSaveTree() { localiser_->TrainAndSaveTree(); }

//------------------------------------------------------------------------------
void PyLocalisation::InitialiseLocalisation(
    const std::string &query_descriptors_file,
    const std::string &sample_descriptors_file) {}

//------------------------------------------------------------------------------
void PyLocalisation::InitLocalisation() { localiser_->InitLocalisation(); }
//------------------------------------------------------------------------------
void PyLocalisation::AddSampleData(const std::string &sample_descriptors_file) {
  localiser_->AddSampleData(sample_descriptors_file);
}
//------------------------------------------------------------------------------
void PyLocalisation::AddQueryData(const std::string &query_descriptors_file) {
  localiser_->AddQueryData(query_descriptors_file);
}

//------------------------------------------------------------------------------
void PyLocalisation::AddBowData(pybind11::array_t<float> &query_bow_desc) {
    cv::Mat bow_desc = numpy_float_1c_to_cv_mat(query_bow_desc);
//    std::cout << "BOW DESC ROWS\t " << bow_desc.rows << "\t BOW DESC COLS \t" << bow_desc.cols << "\n";
    localiser_->AddBowData(bow_desc);
}

//------------------------------------------------------------------------------
void PyLocalisation::SaveQueryDescriptors(
    const std::string &query_descriptors_file) {
  localiser_->SaveQueryDescriptors(query_descriptors_file);
}

//------------------------------------------------------------------------------
void PyLocalisation::SaveSampleDescriptors(
    const std::string &sample_descriptors_file) {
  localiser_->SaveSampleDescriptors(sample_descriptors_file);
}

//------------------------------------------------------------------------------
std::tuple<int, float>
PyLocalisation::LocaliseAgainst(pybind11::array_t<unsigned char> &image) {
  cv::Mat test_image = numpy_uint8_3c_to_cv_mat(image);
  auto bestMatch = localiser_->FindClosestImage(test_image);
  return std::make_tuple(bestMatch.first, bestMatch.second);
}

//------------------------------------------------------------------------------

std::tuple<int, float>
PyLocalisation::LoopClose(pybind11::array_t<unsigned char> &image,
                          int max_index) {
  cv::Mat test_image = numpy_uint8_3c_to_cv_mat(image);
  auto bestMatch = localiser_->FindLoopClosure(test_image, max_index);
  return std::make_tuple(bestMatch.first, bestMatch.second);

}
} // namespace bindings
} // namespace core
} // namespace aru

#endif // ARU_CORE_LOCASATION_BINDINGS_H_
