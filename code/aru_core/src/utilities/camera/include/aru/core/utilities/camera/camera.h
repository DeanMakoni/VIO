#ifndef ARU_UTILITIES_CAMERA_H_
#define ARU_UTILITIES_CAMERA_H_

#include <Eigen/Geometry>
#include <Eigen/Sparse>
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Core/util/Constants.h>
#include <boost/shared_ptr.hpp>
#include <glog/logging.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <utility>

namespace aru {
namespace core {
namespace utilities {
namespace camera {

struct CameraParams {
  Eigen::Matrix3d K;
  float baseline = 0.24;
  int image_height;
  int image_width;
};

class CameraModel {
public:
  CameraModel(std::string filename);

  virtual ~CameraModel() = default;

  int GetHeight() const { return height_; }

  int GetWidth() const { return width_; }

  cv::Mat GetIntrinsicMatrix() { return intrinsic_matrix_; };
  cv::Mat GetProjectionMatrix() { return projection_matrix_; };

  cv::Mat UndistortImage(cv::Mat distorted_image);

  cv::Mat ProjectLidarToCamera(Eigen::MatrixXf point_cloud,
                               Eigen::Affine3f transform);

  cv::Mat UndistortRectifyImage(cv::Mat image_in);

private:
  std::string camera_name_;
  std::string filename_;
  int height_;
  int width_;
  cv::Mat intrinsic_matrix_;
  cv::Mat distortion_coefficients_;
  cv::Mat dist_camera_matrix_;
  cv::Mat projection_matrix_;
  cv::Mat map_x_;
  cv::Mat map_y_;
};

class StereoCameraModel {
public:
  StereoCameraModel(std::string filename_left, std::string filename_right);
  cv::Mat UndistortRectifyLeft(cv::Mat image_in) {return camera_left_.UndistortRectifyImage(image_in);};
  cv::Mat UndistortRectifyRight(cv::Mat image_in) {return camera_right_.UndistortRectifyImage(image_in);};

private:
  CameraModel camera_left_;
  CameraModel camera_right_;
};

} // namespace camera
} // namespace utilities
} // namespace core
} // namespace aru

#endif // ARU_UTILITIES_CAMERA_H_
