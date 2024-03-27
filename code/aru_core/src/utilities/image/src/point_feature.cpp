#include <aru/core/utilities/image/point_feature.h>

#include <Eigen/Dense>
#include <boost/make_shared.hpp>
#include <chrono>
#include <utility>

namespace aru {
namespace core {
namespace utilities {
namespace image {
//------------------------------------------------------------------------------
using namespace cv;
using namespace std;
//------------------------------------------------------------------------------
Feature::Feature() {
  uv_1_left_ = Eigen::Vector2d(0, 0);
  uv_1_right_ = Eigen::Vector2d(0, 0);
  uv_2_left_ = Eigen::Vector2d(0, 0);
  keypoint_1_left_ =
      cv::KeyPoint(cv::Point2f(uv_1_left_.x(), uv_1_left_.y()), 1);
  keypoint_1_right_ =
      cv::KeyPoint(cv::Point2f(uv_1_right_.x(), uv_1_right_.y()), 1);
  keypoint_2_left_ =
      cv::KeyPoint(cv::Point2f(uv_2_left_.x(), uv_2_left_.y()), 1);

  triangulated_point_ = Eigen::Vector3d(0, 0, 0);
  depth_ = -1;
  disparity_ = -1;
  octave = 0;
  patch_size = 0;
}
//------------------------------------------------------------------------------
Feature::Feature(const Eigen::Vector2d &point_uv) {
  uv_1_left_ = point_uv;
  uv_1_right_ = Eigen::Vector2d(0, 0);
  uv_2_left_ = Eigen::Vector2d(0, 0);
  keypoint_1_left_ =
      cv::KeyPoint(cv::Point2f(uv_1_left_.x(), uv_1_left_.y()), 1);
  keypoint_1_right_ =
      cv::KeyPoint(cv::Point2f(uv_1_right_.x(), uv_1_right_.y()), 1);
  keypoint_2_left_ =
      cv::KeyPoint(cv::Point2f(uv_2_left_.x(), uv_2_left_.y()), 1);

  triangulated_point_ = Eigen::Vector3d(0, 0, 0);
  depth_ = -1;
  disparity_ = -1;
  octave = 0;
  patch_size = 0;
}
//------------------------------------------------------------------------------
Feature::Feature(const Eigen::Vector2d &point_uv_1,
                 const Eigen::Vector2d &point_uv_2) {
  uv_1_left_ = point_uv_1;
  uv_1_right_ = point_uv_2;

  uv_2_left_ = Eigen::Vector2d(0, 0);
  keypoint_1_left_ =
      cv::KeyPoint(cv::Point2f(uv_1_left_.x(), uv_1_left_.y()), 1);
  keypoint_1_right_ =
      cv::KeyPoint(cv::Point2f(uv_1_right_.x(), uv_1_right_.y()), 1);
  keypoint_2_left_ =
      cv::KeyPoint(cv::Point2f(uv_2_left_.x(), uv_2_left_.y()), 1);

  triangulated_point_ = Eigen::Vector3d(0, 0, 0);
  depth_ = -1;
  disparity_ = -1;
}
//------------------------------------------------------------------------------
void Feature::TriangulatePoint(const Eigen::Matrix3d &K, const float baseline) {

  // baseline_ = baseline;
  // camera_intrinsic_ = K;
  // uv_1_right_ = Eigen::Vector2d(keypoint_1_right_.pt.y,
  // keypoint_1_right_.pt.x);
  // uv_1_left_ =
  // Eigen::Vector2d(keypoint_1_left_.pt.y, keypoint_1_left_.pt.x);
  Eigen::Vector3d point_world = Eigen::MatrixXd::Zero(3, 1);
  double disparity = uv_1_left_(0) - uv_1_right_(0);
  double baseline_over_disparity_1 = baseline / disparity;


  point_world(2) = baseline_over_disparity_1 * K(0, 0);
  point_world(0) = (uv_1_left_(0) - K(0, 2)) * baseline_over_disparity_1;
  point_world(1) = (uv_1_left_(1) - K(1, 2)) * baseline_over_disparity_1 *
                   (K(0, 0) / K(1, 1));
  triangulated_point_ = point_world;
  depth_ = point_world(2);
  disparity_ = disparity;
}
//------------------------------------------------------------------------------
// void Feature::UpdateTriangulatedPointDisparity() {
//  double baseline_over_disparity_1 = baseline_ / disparity_;
//  triangulated_point_(2) = baseline_over_disparity_1 * camera_intrinsic_(0,
//  0); triangulated_point_(0) =
//      (uv_1_left_(1) - camera_intrinsic_(0, 2)) * baseline_over_disparity_1;
//  triangulated_point_(1) = (uv_1_left_(0) - camera_intrinsic_(1, 2)) *
//                           baseline_over_disparity_1 *
//                           (camera_intrinsic_(0, 0) / camera_intrinsic_(1,
//                           1));
//}
} // namespace image
} // namespace utilities
} // namespace core
} // namespace aru
