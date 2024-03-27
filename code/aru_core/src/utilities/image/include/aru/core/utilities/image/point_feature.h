#ifndef ARU_CORE_UTILITIES_IMAGE_FEATURES_H_
#define ARU_CORE_UTILITIES_IMAGE_FEATURES_H_

#include <Eigen/Dense>
#include <boost/shared_ptr.hpp>
#include <glog/logging.h>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <utility>

namespace aru {
namespace core {
namespace utilities {
namespace image {

class Feature {
public:
  Feature();
  
  Feature(const Eigen::Vector2d &point_uv_1, const Eigen::Vector2d &point_uv_2);

  explicit Feature(const Eigen::Vector2d &point_uv);

  virtual ~Feature() = default;

  void SetMatchedKeypoint(cv::KeyPoint keypoint_1_right) {
    keypoint_1_right_ = std::move(keypoint_1_right);
    uv_1_right_ =
        Eigen::Vector2d(keypoint_1_right_.pt.x, keypoint_1_right_.pt.y);
  }

  void TriangulatePoint(const Eigen::Matrix3d &K, float baseline);

  void UpdateKeyPoint(const Eigen::Vector2d &point_uv_1);

  void SetSequentialKeyPoint(cv::KeyPoint key_point_2) {
    keypoint_2_left_ = std::move(key_point_2);
  }

  void SetKeyPoint(cv::KeyPoint key_point) {
    keypoint_1_left_ = std::move(key_point);
    uv_1_left_ =
        Eigen::Vector2d(keypoint_1_left_.pt.x, keypoint_1_left_.pt.y);
  }

  Eigen::Vector2d GetValue() { return uv_1_left_; }

  cv::KeyPoint GetKeyPoint() { return keypoint_1_left_; }

  cv::KeyPoint GetMatchedKeyPoint() { return keypoint_1_right_; }

  cv::KeyPoint GetSequentialKeyPoint() { return keypoint_2_left_; }

  Eigen::Vector3d GetTriangulatedPoint() { return triangulated_point_; }

  void UpdateTriangulatedPointDepth() { triangulated_point_[2] = depth_; }

  void UpdateTriangulatedPointDisparity();

  float GetDepth() const { return depth_; }

  float GetDisparity() const { return disparity_; }

  void UpdateDepth(float depth) { depth_ = depth; }

  void UpdateDisparity(float disparity) { disparity_ = disparity; }

  void SetDescriptor(cv::Mat descriptor) {
    point_1_descriptor = std::move(descriptor);
  }

private:
  // Vector and keypoint for stereo images 1
  Eigen::Vector2d uv_1_left_;
  Eigen::Vector2d uv_1_right_;

  cv::KeyPoint keypoint_1_left_;
  cv::KeyPoint keypoint_1_right_;

  // float baseline_{};
  // Eigen::Matrix3d camera_intrinsic_;

  // Triangulated 3d point from stereo
  Eigen::Vector3d triangulated_point_;

  float depth_{};
  float disparity_{};

  // Vector and keypoint for stereo image 2
  Eigen::Vector2d uv_2_left_;
  cv::KeyPoint keypoint_2_left_;

  cv::Mat point_1_descriptor;

  int octave{};
  int patch_size{};
};
typedef boost::shared_ptr<Feature> FeatureSPtr;
typedef std::vector<FeatureSPtr> FeatureSPtrVector;
typedef boost::shared_ptr<FeatureSPtrVector> FeatureSPtrVectorSptr;
} // namespace image
} // namespace utilities
} // namespace core
} // namespace aru
#endif // ARU_CORE_UTILITIES_IMAGE_FEATURES_H_
