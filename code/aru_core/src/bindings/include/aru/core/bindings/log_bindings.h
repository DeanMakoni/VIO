#ifndef ARU_CORE_BINDINGS_H_
#define ARU_CORE_BINDINGS_H_

#include <aru/core/bindings/conversions.h>
#include <aru/core/utilities/image/imageprotocolbufferadaptor.h>
#include <aru/core/utilities/laser/laserprotocolbufferadaptor.h>
#include <aru/core/utilities/navigation/experienceprotocolbufferadaptor.h>
#include <aru/core/utilities/logging/index_log.h>
#include <aru/core/utilities/transforms/transformprotocolbufferadaptor.h>

#include <Eigen/Dense>
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>
#include <glog/logging.h>
#include <google/protobuf/util/delimited_message_util.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <pbExperience.pb.h>
#include <pbLaser.pb.h>
#include <pbStereoImage.pb.h>
#include <utility>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace aru {
namespace core {
namespace bindings {

class StereoImageLogger {
public:
  StereoImageLogger(const std::string &filename, bool overwrite);

  virtual ~StereoImageLogger() = default;

  void WriteToFile(pybind11::array_t<unsigned char> &image_left,
                   pybind11::array_t<unsigned char> &image_right,
                   int64 timestamp);

  std::tuple<pybind11::array_t<unsigned char>, pybind11::array_t<unsigned char>,
             int64>
  ReadFromFile();


  std::tuple<pybind11::array_t<unsigned char>, pybind11::array_t<unsigned char>,
             int64>
  ReadIndexFromFile(int index);

  bool EndOfFile() { return stereo_logger_->EndOfFile(); };

private:
  std::shared_ptr<aru::core::utilities::logging::IndexLogger<
      datatype ::image::pbStereoImage>>
      stereo_logger_;

  datatype::image::pbStereoImage pb_stereo_image_;
};
//------------------------------------------------------------------------------
StereoImageLogger::StereoImageLogger(const std::string &filename,
                                     bool overwrite) {
  stereo_logger_ = std::make_shared<aru::core::utilities::logging::IndexLogger<
      datatype::image::pbStereoImage>>(filename, overwrite);

  if (!overwrite)
    pb_stereo_image_ = stereo_logger_->ReadFromFile();
}
//------------------------------------------------------------------------------
void StereoImageLogger::WriteToFile(
    pybind11::array_t<unsigned char> &image_left,
    pybind11::array_t<unsigned char> &image_right, int64 timestamp) {
  aru::core::utilities::image::StereoImage stereo_image;

  stereo_image.first = aru::core::utilities::image::Image(
      timestamp, numpy_uint8_3c_to_cv_mat(image_left));

  stereo_image.second = aru::core::utilities::image::Image(
      timestamp, numpy_uint8_3c_to_cv_mat(image_right));

  datatype::image::pbStereoImage pb_stereo_image = aru::core::utilities::image::
      ImageProtocolBufferAdaptor ::ReadStereoToProtocolBuffer(stereo_image);

  stereo_logger_->WriteToFile(pb_stereo_image);
}
//------------------------------------------------------------------------------
std::tuple<pybind11::array_t<unsigned char>, pybind11::array_t<unsigned char>,
           int64>
StereoImageLogger::ReadFromFile() {

  aru::core::utilities::image::StereoImage stereo_image =
      aru::core::utilities::image::ImageProtocolBufferAdaptor ::
          ReadStereoFromProtocolBuffer(pb_stereo_image_);
  cv::Mat image_left_cv = stereo_image.first.GetImage();
  cv::Mat image_right_cv = stereo_image.second.GetImage();

  pb_stereo_image_ = stereo_logger_->ReadFromFile();
  return std::make_tuple(cv_mat_uint8_3c_to_numpy(image_left_cv),
                         cv_mat_uint8_3c_to_numpy(image_right_cv),
                         stereo_image.first.GetTimeStamp());
}
//-----------------------------------------------------------------------------
// Commented out as duplicate definition
//bool StereoImageLogger::EndOfFile() { return stereo_logger_->EndOfFile(); }
//------------------------------------------------------------------------------
std::tuple<pybind11::array_t<unsigned char>, pybind11::array_t<unsigned char>,
           int64>
StereoImageLogger::ReadIndexFromFile(int index) {

  pb_stereo_image_ = stereo_logger_->ReadIndex(index);
  aru::core::utilities::image::StereoImage stereo_image =
      aru::core::utilities::image::ImageProtocolBufferAdaptor ::
          ReadStereoFromProtocolBuffer(pb_stereo_image_);
  cv::Mat image_left_cv = stereo_image.first.GetImage();
  cv::Mat image_right_cv = stereo_image.second.GetImage();

  return std::make_tuple(cv_mat_uint8_3c_to_numpy(image_left_cv),
                         cv_mat_uint8_3c_to_numpy(image_right_cv),
                         stereo_image.first.GetTimeStamp());
}

//------------------------------------------------------------------------------
class MonoImageLogger {
public:
  MonoImageLogger(const std::string &filename, bool overwrite);

  virtual ~MonoImageLogger() = default;

  void WriteToFile(pybind11::array_t<unsigned char> &image, int64 timestamp);

  std::tuple<pybind11::array_t<unsigned char>, int64> ReadFromFile();

  std::tuple<pybind11::array_t<unsigned char>, int64> ReadIndexFromFile(int
                                                                            index);

  std::tuple<pybind11::array_t<unsigned char>, int64> ReadChannelFromFile();

  bool EndOfFile() { return mono_logger_->EndOfFile(); };

private:
  std::shared_ptr<
      aru::core::utilities::logging::IndexLogger<datatype ::image::pbImage>>
      mono_logger_;
  datatype::image::pbImage pb_image_;
};
//------------------------------------------------------------------------------
MonoImageLogger::MonoImageLogger(const std::string &filename, bool overwrite) {
  mono_logger_ = std::make_shared<
      aru::core::utilities::logging::IndexLogger<datatype::image::pbImage>>(
      filename, overwrite);
  if (!overwrite)
    pb_image_ = mono_logger_->ReadFromFile();
}
//------------------------------------------------------------------------------
void MonoImageLogger::WriteToFile(pybind11::array_t<unsigned char> &image,
                                  int64 timestamp) {
  aru::core::utilities::image::Image mono_image(
      timestamp, numpy_uint8_3c_to_cv_mat(image));

  datatype::image::pbImage pb_image = aru::core::utilities::image::
      ImageProtocolBufferAdaptor ::ReadToProtocolBuffer(mono_image);

  mono_logger_->WriteToFile(pb_image);
}
//------------------------------------------------------------------------------
std::tuple<pybind11::array_t<unsigned char>, int64>
MonoImageLogger::ReadFromFile() {

  aru::core::utilities::image::Image mono_image = aru::core::utilities::image::
      ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(pb_image_);
  cv::Mat image_left_cv = mono_image.GetImage();
  pb_image_ = mono_logger_->ReadFromFile();
  return std::make_tuple(cv_mat_uint8_3c_to_numpy(image_left_cv),
                         mono_image.GetTimeStamp());
}

//------------------------------------------------------------------------------
std::tuple<pybind11::array_t<unsigned char>, int64>
MonoImageLogger::ReadChannelFromFile() {
  datatype::image::pbImage pb_image = mono_logger_->ReadFromFile();
  aru::core::utilities::image::Image mono_image = aru::core::utilities::image::
      ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(pb_image);
  cv::Mat image_left_cv = mono_image.GetImage();
  return std::make_tuple(cv_mat_uint8_1c_to_numpy(image_left_cv),
                         mono_image.GetTimeStamp());
}
//------------------------------------------------------------------------------
std::tuple<pybind11::array_t<unsigned char>, int64> MonoImageLogger::ReadIndexFromFile(int index) {
  datatype::image::pbImage pb_image = mono_logger_->ReadIndex(index);
  aru::core::utilities::image::Image mono_image = aru::core::utilities::image::
      ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(pb_image);
  cv::Mat image_left_cv = mono_image.GetImage();
  return std::make_tuple(cv_mat_uint8_3c_to_numpy(image_left_cv),
                         mono_image.GetTimeStamp());
}
//------------------------------------------------------------------------------
class LaserLogger {
public:
  explicit LaserLogger(const std::string &filename, bool overwrite);

  virtual ~LaserLogger() = default;

  void WriteToFile(const pybind11::EigenDRef<Eigen::MatrixXd> &points,
                   int64 timestamp);

  std::tuple<Eigen::MatrixXd, int64> ReadFromFile();

  std::tuple<Eigen::MatrixXd, int64> ReadIndexFromFile(int index);

  bool EndOfFile() { return laser_logger_->EndOfFile(); };

private:
  std::shared_ptr<
      aru::core::utilities::logging::IndexLogger<datatype::laser::pbLaser>>
      laser_logger_;
  datatype::laser::pbLaser pb_laser_;
};
//------------------------------------------------------------------------------
LaserLogger::LaserLogger(const std::string &filename, bool overwrite) {
  laser_logger_ = std::make_shared<
      aru::core::utilities::logging::IndexLogger<datatype::laser::pbLaser>>(
      filename, overwrite);

  if (!overwrite)
    pb_laser_ = laser_logger_->ReadFromFile();
}
//------------------------------------------------------------------------------
void LaserLogger::WriteToFile(
    const pybind11::EigenDRef<Eigen::MatrixXd> &points, int64 timestamp) {

  aru::core::utilities::laser::Laser laser(timestamp, points.cast<float>());

  datatype::laser::pbLaser pb_laser = aru::core::utilities::laser::
      LaserProtocolBufferAdaptor::WriteToProtocolBuffer(laser);

  laser_logger_->WriteToFile(pb_laser);
}
//------------------------------------------------------------------------------
std::tuple<Eigen::MatrixXd, int64> LaserLogger::ReadFromFile() {

  aru::core::utilities::laser::Laser laser = aru::core::utilities::laser::
      LaserProtocolBufferAdaptor ::ReadFromProtocolBuffer(pb_laser_);
  pb_laser_ = laser_logger_->ReadFromFile();
  return std::make_tuple(laser.GetPoints().cast<double>(),
                         laser.GetTimeStamp());
}
//------------------------------------------------------------------------------
std::tuple<Eigen::MatrixXd, int64> LaserLogger::ReadIndexFromFile(int index) {
  pb_laser_ = laser_logger_->ReadIndex(index);
  aru::core::utilities::laser::Laser laser = aru::core::utilities::laser::
      LaserProtocolBufferAdaptor ::ReadFromProtocolBuffer(pb_laser_);

  return std::make_tuple(laser.GetPoints().cast<double>(),
                         laser.GetTimeStamp());
}
//------------------------------------------------------------------------------
class TransformLogger {
public:
  explicit TransformLogger(const std::string &filename, bool overwrite);

  virtual ~TransformLogger() = default;

  void WriteToFile(const pybind11::EigenDRef<Eigen::MatrixXd> &transform_matrix,
                   int64 source_timestamp, int64 dest_timestamp);

  bool EndOfFile();

  std::tuple<Eigen::MatrixXd, int64, int64> ReadFromFile();

  std::tuple<Eigen::MatrixXd, int64, int64> ReadIndexFromFile(int index);

private:
  std::shared_ptr<aru::core::utilities::logging::IndexLogger<
      datatype::transform::pbTransform>>
      transform_logger_;

  datatype::transform::pbTransform pb_transform_;
};
//------------------------------------------------------------------------------
TransformLogger::TransformLogger(const std::string &filename, bool overwrite) {
  transform_logger_ =
      std::make_shared<aru::core::utilities::logging::IndexLogger<
          datatype::transform::pbTransform>>(filename, overwrite);
  if (!overwrite)
    pb_transform_ = transform_logger_->ReadFromFile();
}
//------------------------------------------------------------------------------
void TransformLogger::WriteToFile(
    const pybind11::EigenDRef<Eigen::MatrixXd> &transform_matrix,
    int64 source_timestamp, int64 dest_timestamp) {

  Eigen::Affine3f transform_affine;
  transform_affine.matrix() = transform_matrix.cast<float>();
  aru::core::utilities::transform::Transform transforms(source_timestamp, dest_timestamp, transform_affine);

  datatype::transform::pbTransform pb_transform =
      aru::core::utilities::transform::TransformProtocolBufferAdaptor::
          ReadToProtocolBuffer(transforms);

  transform_logger_->WriteToFile(pb_transform);
}
//------------------------------------------------------------------------------
std::tuple<Eigen::MatrixXd, int64, int64> TransformLogger::ReadFromFile() {

  aru::core::utilities::transform::Transform transforms =
      aru::core::utilities::transform::TransformProtocolBufferAdaptor ::
          ReadFromProtocolBuffer(pb_transform_);

  pb_transform_ = transform_logger_->ReadFromFile();
  return std::make_tuple(transforms.GetTransform().matrix().cast<double>(),
                         transforms.GetSourceTimestamp(),
                         transforms.GetDestinationTimestamp());
}
//------------------------------------------------------------------------------
std::tuple<Eigen::MatrixXd, int64, int64> TransformLogger::ReadIndexFromFile(int index) {

  pb_transform_ = transform_logger_->ReadIndex(index);
  aru::core::utilities::transform::Transform transform =
      aru::core::utilities::transform::TransformProtocolBufferAdaptor ::
          ReadFromProtocolBuffer(pb_transform_);


  return std::make_tuple(transform.GetTransform().matrix().cast<double>(),
                         transform.GetSourceTimestamp(),
                         transform.GetDestinationTimestamp());
}
//------------------------------------------------------------------------------
bool TransformLogger::EndOfFile() { return transform_logger_->EndOfFile(); }
//-------------------------------------------------------------------------------
class ExperienceLogger {
public:
  ExperienceLogger(const std::string &filename, bool overwrite);

  virtual ~ExperienceLogger() = default;

  void WriteToFile(int64 timestamp,
                   pybind11::array_t<unsigned char> &left_image,
                   pybind11::array_t<float> &cv_keypoints,
                   pybind11::array_t<float> &landmarks,
                   pybind11::array_t<unsigned char> &cv_desc,
                   pybind11::array_t<float> &query_desc);

  std::tuple<int64, pybind11::array_t<unsigned char>, Eigen::MatrixXd,
             Eigen::MatrixXd, pybind11::array_t<unsigned char>,
             pybind11::array_t<float>>  ReadFromFile();

  bool EndOfFile();

private:
  boost::shared_ptr<aru::core::utilities::logging::ProtocolLogger<
      datatype::navigation::pbExperience>>
      experience_logger_;
  datatype::navigation::pbExperience pb_experience_;
};
//------------------------------------------------------------------------------
ExperienceLogger::ExperienceLogger(const std::string &filename,
                                   bool overwrite) {
  experience_logger_ =
      boost::make_shared<aru::core::utilities::logging::ProtocolLogger<
          datatype::navigation::pbExperience>>(filename, overwrite);
//  LOG(INFO)<<"Init experience logger";
  if (!overwrite)
    pb_experience_ = experience_logger_->ReadFromFile();
}
//------------------------------------------------------------------------------
void ExperienceLogger::WriteToFile(
    int64 timestamp, pybind11::array_t<unsigned char> &left_image,
    pybind11::array_t<float> &cv_keypoints,
    pybind11::array_t<float> &landmarks,
    pybind11::array_t<unsigned char> &cv_desc,
    pybind11::array_t<float> &bow_desc) {

        cv::Mat keypoints_cv = numpy_float_1c_to_cv_mat(cv_keypoints);
        cv::Mat landmarks_cv = numpy_float_1c_to_cv_mat(cv_keypoints);

        // TODO: Check that landmarks is 3 by n and keypoints is 2 by n
        Eigen::MatrixXf  landmarks_eigen,keypoints_eigen;
        cv::cv2eigen(keypoints_cv, keypoints_eigen);
        cv::cv2eigen(landmarks_cv, landmarks_eigen);

  cv::Mat image_cv = numpy_uint8_3c_to_cv_mat(left_image);
//  Eigen::MatrixXf keypoints = cv_keypoints.cast<float>();
//  Eigen::MatrixXf landmarks_ = landmarks.cast<float>();
  cv::Mat descriptors = numpy_uint8_1c_to_cv_mat(cv_desc);
//  std::cout << "CV DESC ROWS \t" << descriptors.rows << "\t CV DESC COLS \t" << descriptors.cols <<"\n";
  cv::Mat bow_descriptors = numpy_float_1c_to_cv_mat(bow_desc);
//  std::cout << "BOW DESC ROWS\t " << bow_descriptors.rows << "\t BOW DESC COLS \t" << bow_descriptors.cols << "\n";
  aru::core::utilities::navigation::Experience experience(
      timestamp, image_cv, keypoints_eigen, landmarks_eigen, descriptors, bow_descriptors);
//  LOG(INFO)<<"Written experience class";

  datatype::navigation::pbExperience pb_experience =
      aru::core::utilities::navigation::ExperienceProtocolBufferAdaptor::
          ReadToProtocolBuffer(experience);
//  LOG(INFO)<<"Experience pb written";
  experience_logger_->WriteToFile(pb_experience);
//  LOG(INFO)<<"File written";
}
//------------------------------------------------------------------------------
std::tuple<int64, pybind11::array_t<unsigned char>, Eigen::MatrixXd,
           Eigen::MatrixXd, pybind11::array_t<unsigned char>,
           pybind11::array_t<float>>
ExperienceLogger::ReadFromFile() {

  aru::core::utilities::navigation::Experience experience =
      aru::core::utilities::navigation::ExperienceProtocolBufferAdaptor ::
          ReadFromProtocolBuffer(pb_experience_);
  int64_t timestamp = experience.GetTimeStamp();
  cv::Mat image_left = experience.GetImage();
  cv::Mat cv_desc = experience.GetDescriptors();
  Eigen::MatrixXf keypoints = experience.GetKeypoints();
  Eigen::MatrixXf landmarks = experience.GetLandmarks();
  cv::Mat bow_desc = experience.GetBowDescriptors();

//  std::cout << "BOW DESC ROWS\t " << bow_desc.rows << "\t BOW DESC COLS \t" << bow_desc.cols << "\n";
//  std::cout << "Read Type " << bow_desc.type();

  pb_experience_ = experience_logger_->ReadFromFile();

//  std::cout << "CV DESC ROWS \t" << cv_desc.rows << "\t CV DESC COLS \t" << cv_desc.cols <<"\n";
//  std::cout << "BOW DESC ROWS\t " << bow_desc.rows << "\t BOW DESC COLS \t" << bow_desc.cols << "\n";

  return std::make_tuple(timestamp, cv_mat_uint8_3c_to_numpy(image_left),
                         keypoints.cast<double>(), landmarks.cast<double>(),
                         cv_mat_uint8_1c_to_numpy(cv_desc),
                         cv_mat_float_1c_to_numpy(bow_desc));
}

//------------------------------------------------------------------------------
bool ExperienceLogger::EndOfFile() { return experience_logger_->EndOfFile(); }
//------------------------------------------------------------------------------
} // namespace bindings
} // namespace core
} // namespace aru

#endif // ARU_CORE_BINDINGS_H_
