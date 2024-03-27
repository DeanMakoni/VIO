#ifndef ARU_CORE_VO_BINDINGS_H_
#define ARU_CORE_VO_BINDINGS_H_

#include <aru/core/bindings/conversions.h>
#include <aru/core/utilities/logging/log.h>
#include <aru/core/vo/vo.h>

#include <Eigen/Dense>
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>
#include <glog/logging.h>
#include <google/protobuf/util/delimited_message_util.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <pbStereoImage.pb.h>
#include <utility>

#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace aru {
namespace core {
namespace bindings {

class PyVO {
public:
  PyVO(std::string filename, std::string vocabulary_filename);

  virtual ~PyVO() = default;

  std::tuple<pybind11::array_t<float>, pybind11::array_t<float>,
             pybind11::array_t<float>, pybind11::array_t<float>>
  StereoOdometry(pybind11::array_t<unsigned char> &image_1_left,
                 pybind11::array_t<unsigned char> &image_1_right,
                 int64 timestamp_1,
                 pybind11::array_t<unsigned char> &image_2_left,
                 pybind11::array_t<unsigned char> &image_2_right,
                 int64 timestamp_2);

  std::tuple<pybind11::array_t<float>, pybind11::array_t<float>>
  StereoMatches(pybind11::array_t<unsigned char> &image_1_left,
                pybind11::array_t<unsigned char> &image_1_right);

  std::tuple<pybind11::array_t<float>, pybind11::array_t<float>,
             pybind11::array_t<float>>
  StereoSequentialMatches(pybind11::array_t<unsigned char> &image_1_left,
                          pybind11::array_t<unsigned char> &image_1_right,
                          pybind11::array_t<unsigned char> &image_2_left,
                          pybind11::array_t<unsigned char> &image_2_right);

  pybind11::array_t<float> MotionEstimation(pybind11::array_t<float> landmarks,
                                            pybind11::array_t<float> keypoints);

private:
  boost::shared_ptr<aru::core::vo::VO> vo_;
  std::string config_filename_;
  std::string vocab_filename_;
  utilities::image::StereoImage image_1_;
  utilities::image::StereoImage image_2_;
};
//------------------------------------------------------------------------------
PyVO::PyVO(std::string filename, std::string vocabulary_filename) : config_filename_(std::move(filename)),
vocab_filename_(std::move(vocabulary_filename)){
  vo_ = boost::make_shared<aru::core::vo::VO>(config_filename_, vocab_filename_);
}

//------------------------------------------------------------------------------
std::tuple<pybind11::array_t<float>, pybind11::array_t<float>,
           pybind11::array_t<float>, pybind11::array_t<float>>
PyVO::StereoOdometry(pybind11::array_t<unsigned char> &image_1_left,
                     pybind11::array_t<unsigned char> &image_1_right,
                     int64 timestamp_1,
                     pybind11::array_t<unsigned char> &image_2_left,
                     pybind11::array_t<unsigned char> &image_2_right,
                     int64 timestamp_2) {

  image_1_.first = utilities::image::Image(
      timestamp_1, numpy_uint8_1c_to_cv_mat(image_1_left));
  image_1_.second = utilities::image::Image(
      timestamp_1, numpy_uint8_1c_to_cv_mat(image_1_right));
  image_2_.first = utilities::image::Image(
      timestamp_2, numpy_uint8_1c_to_cv_mat(image_2_left));
  image_2_.second = utilities::image::Image(
      timestamp_2, numpy_uint8_1c_to_cv_mat(image_2_right));

  utilities::transform::Transform transform =
      vo_->EstimateMotion(image_1_, image_2_);

  cv::Mat left_cv, right_cv, left_2_cv;

  std::tuple<Eigen::MatrixXf, Eigen::MatrixXf, Eigen::MatrixXf> keypoints =
      vo_->ObtainStereoSequentialPoints(
          numpy_uint8_1c_to_cv_mat(image_1_left),
          numpy_uint8_1c_to_cv_mat(image_1_right),
          numpy_uint8_1c_to_cv_mat(image_2_left),
          numpy_uint8_1c_to_cv_mat(image_2_right));

  cv::eigen2cv(std::get<0>(keypoints), left_cv);
  cv::eigen2cv(std::get<1>(keypoints), right_cv);
  cv::eigen2cv(std::get<2>(keypoints), left_2_cv);

  cv::Mat cv_transform;
  cv::eigen2cv(transform.GetTransform().matrix(), cv_transform);

  return std::make_tuple(
      cv_mat_float_1c_to_numpy(cv_transform), cv_mat_float_1c_to_numpy(left_cv),
      cv_mat_float_1c_to_numpy(right_cv), cv_mat_float_1c_to_numpy(left_2_cv));
  ;
}
//------------------------------------------------------------------------------
std::tuple<pybind11::array_t<float>, pybind11::array_t<float>,
           pybind11::array_t<float>>
PyVO::StereoSequentialMatches(pybind11::array_t<unsigned char> &image_1_left,
                              pybind11::array_t<unsigned char> &image_1_right,
                              pybind11::array_t<unsigned char> &image_2_left,
                              pybind11::array_t<unsigned char> &image_2_right) {

  cv::Mat left_cv, right_cv, left_2_cv;

  std::tuple<Eigen::MatrixXf, Eigen::MatrixXf, Eigen::MatrixXf> keypoints =
      vo_->ObtainStereoSequentialPoints(
          numpy_uint8_1c_to_cv_mat(image_1_left),
          numpy_uint8_1c_to_cv_mat(image_1_right),
          numpy_uint8_1c_to_cv_mat(image_2_left),
          numpy_uint8_1c_to_cv_mat(image_2_right));

  cv::eigen2cv(std::get<0>(keypoints), left_cv);
  cv::eigen2cv(std::get<1>(keypoints), right_cv);
  cv::eigen2cv(std::get<2>(keypoints), left_2_cv);

  return std::make_tuple(cv_mat_float_1c_to_numpy(left_cv),
                         cv_mat_float_1c_to_numpy(right_cv),
                         cv_mat_float_1c_to_numpy(left_2_cv));
}

//------------------------------------------------------------------------------
std::tuple<pybind11::array_t<float>, pybind11::array_t<float>>
PyVO::StereoMatches(pybind11::array_t<unsigned char> &image_1_left,
                    pybind11::array_t<unsigned char> &image_1_right) {

  cv::Mat left_cv, right_cv;

  std::pair<Eigen::MatrixXf, Eigen::MatrixXf> keypoints =
      vo_->ObtainStereoPoints(numpy_uint8_1c_to_cv_mat(image_1_left),
                              numpy_uint8_1c_to_cv_mat(image_1_right));

  cv::eigen2cv(keypoints.first, left_cv);
  cv::eigen2cv(keypoints.second, right_cv);

  return std::make_tuple(cv_mat_float_1c_to_numpy(left_cv),
                         cv_mat_float_1c_to_numpy(right_cv));
}

//------------------------------------------------------------------------------
pybind11::array_t<float>
PyVO::MotionEstimation(pybind11::array_t<float> landmarks,
                       pybind11::array_t<float> keypoints) {
  cv::Mat landmarks_cv = numpy_float_1c_to_cv_mat(landmarks);
  cv::Mat keypoints_cv = numpy_float_1c_to_cv_mat(keypoints);

  LOG(INFO) << "Size of landmarks is " << landmarks_cv.rows << " x "
            << landmarks_cv.cols;
  // TODO: Check that landmarks is 3 by n and keypoints is 2 by n
  Eigen::MatrixXf landmarks_eigen, keypoints_eigen;
  cv::cv2eigen(landmarks_cv, landmarks_eigen);
  cv::cv2eigen(keypoints_cv, keypoints_eigen);

  std::vector<cv::Point3d> points3d;
  std::vector<cv::Point2d> points2d;

  int num_points = landmarks_eigen.rows();

  for (int i = 0; i < num_points; ++i) {
    points3d.emplace_back(landmarks_eigen.row(i)(0), landmarks_eigen.row(i)(1),
                          landmarks_eigen.row(i)(2));
    points2d.emplace_back(keypoints_eigen.row(i)(0), keypoints_eigen.row(i)(1));
  }

  utilities::transform::Transform transform =
      vo_->ObtainTransform(points3d, points2d);
  cv::Mat cv_transform;
  cv::eigen2cv(transform.GetTransform().matrix(), cv_transform);

  return cv_mat_float_1c_to_numpy(cv_transform);
}

} // namespace bindings
} // namespace core
} // namespace aru

#endif // ARU_CORE_VO_BINDINGS_H_
