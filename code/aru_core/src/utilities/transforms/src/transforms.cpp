#include "aru/core/utilities/transforms/transforms.h"
#include <Eigen/Dense>
#include <boost/make_shared.hpp>
#include <chrono>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/mat.hpp>
#include <utility>

namespace aru {
namespace core {
namespace utilities {
namespace transform {

//------------------------------------------------------------------------------
Transform::Transform(std::string source, std::string destination,
                     const Eigen::Affine3f &transform)
    : source_frame_(std::move(std::move(source))),
      destination_frame_(std::move(destination)), transform_(transform) {
  source_timestamp_ = 0;
  destination_timestamp_ = 0;
}
//------------------------------------------------------------------------------
Transform::Transform(int64_t source_timestamp, int64_t dest_timestamp,
                     const Eigen::Affine3f &transform)
    : source_timestamp_(source_timestamp),
      destination_timestamp_(dest_timestamp), transform_(transform) {
  source_frame_ = "A";
  destination_frame_ = "A";
}
//------------------------------------------------------------------------------
Transform::Transform(std::string source, std::string destination,
                     int64_t source_timestamp, int64_t dest_timestamp,
                     const Eigen::Affine3f &transform)
    : source_frame_(std::move(source)),
      destination_frame_(std::move(destination)),
      source_timestamp_(source_timestamp),
      destination_timestamp_(dest_timestamp), transform_(transform) {}
//------------------------------------------------------------------------------
void Transform::RightCompose(Transform transform_right) {
  transform_ = transform_ * transform_right.transform_;
}
//------------------------------------------------------------------------------
Eigen::Matrix3f Transform::RotationMatrixFromRPY(Eigen::Vector3f rpy) {
  cv::Mat R_matrix = cv::Mat::zeros(3, 3, CV_64FC1); // rotation matrix
  cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
  // Read Rotation matrix and convert to vector

  cv::eigen2cv(rpy, rvec);
  cv::Rodrigues(rvec, R_matrix);
  // Convert rotation rpy to Matrix
  Eigen::Matrix3f Rot;
  cv::cv2eigen(R_matrix, Rot);
  return Rot;
}
//------------------------------------------------------------------------------
Eigen::Vector3f Transform::RPYFromRotationMatrix(Eigen::Matrix3f rot) {

  cv::Mat R_matrix = cv::Mat::zeros(3, 3, CV_64FC1); // rotation matrix
  cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
  // Read Rotation matrix and convert to vector
  cv::eigen2cv(rot, R_matrix);
  cv::Rodrigues(R_matrix, rvec);
  // Convert from CV to Eigen
  Eigen::Vector3f rpy;
  cv::cv2eigen(rvec, rpy);
  return rpy;
}
//------------------------------------------------------------------------------
// Transform::Transform(const Transform &transform_in) {
//  transform_=transform_in.transform_;
//  source_timestamp_=transform_in.source_timestamp_;
//  destination_timestamp_=transform_in.destination_timestamp_;
//}
} // namespace transform
} // namespace utilities
} // namespace core
} // namespace aru
