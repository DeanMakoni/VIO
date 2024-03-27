#include "aru/core/utilities/camera/camera.h"
#include <Eigen/Dense>
#include <boost/make_shared.hpp>
#include <chrono>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc.hpp>
#include <utility>

namespace aru {
namespace core {
namespace utilities {
namespace camera {

//------------------------------------------------------------------------------
CameraModel::CameraModel(std::string filename) : filename_(filename) {
  cv::FileStorage fs;
  std::cout<<filename_<<std::endl;
  fs.open(filename_, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    LOG(ERROR) << "Could not open camera model file: ";
  }
  std::cout<<filename_<<std::endl;
  LOG(INFO)<<"Opened file "<<filename_;
  height_ = fs["image_height"];
  LOG(INFO) << "Height is " << height_;
  width_ = fs["image_width"];

  fs["camera_matrix"] >> intrinsic_matrix_;
  LOG(INFO) << "Camera matrix is " << intrinsic_matrix_;
  fs["distortion_coefficients"] >> distortion_coefficients_;
  LOG(INFO) << "Camera distortion is " << distortion_coefficients_;
  fs["projection_matrix"] >> projection_matrix_;
  LOG(INFO) << "Camera projection matrix is " << projection_matrix_;

  fs["camera_name"] >> camera_name_;

  dist_camera_matrix_ =
      cv::getOptimalNewCameraMatrix(intrinsic_matrix_, distortion_coefficients_,
                                    cv::Size(width_, height_), 0);
  
  LOG(INFO)<<"Dist camera matrix is "<<dist_camera_matrix_;

  // Init maps for undistort rectify
  cv::Mat R;
  cv::initUndistortRectifyMap(intrinsic_matrix_, distortion_coefficients_, R, 
    projection_matrix_, cv::Size(width_, height_), cv::INTER_LINEAR_EXACT, 
    map_x_, map_y_);
    LOG(INFO)<<"Init maps";
}
//------------------------------------------------------------------------------
cv::Mat CameraModel::UndistortImage(cv::Mat distorted_image) {
  cv::Mat undistorted;
  cv::undistort(distorted_image, undistorted, intrinsic_matrix_,
                distortion_coefficients_);
  return undistorted;
}
//------------------------------------------------------------------------------
cv::Mat CameraModel::ProjectLidarToCamera(Eigen::MatrixXf point_cloud,
                                          Eigen::Affine3f transform) {
  for (int i = 0; i < point_cloud.rows(); ++i) {
    Eigen::Vector3f curr_row = point_cloud.row(i);
    point_cloud.row(i) = transform * curr_row;
  }

  Eigen::Matrix3f K_;
  cv::cv2eigen(intrinsic_matrix_,K_);
  for (int i = 0; i < point_cloud.rows(); ++i) {
    Eigen::Vector3f curr_row = point_cloud.row(i);
    point_cloud.row(i) = transform * curr_row;
  }
}
//------------------------------------------------------------------------------
cv::Mat CameraModel::UndistortRectifyImage(cv::Mat image_in){
  cv::Mat image_out;
  cv::remap(image_in, image_out, map_x_, map_y_, cv::INTER_LINEAR);
  return image_out;
}
//-------------------------------------------------------------------------------

StereoCameraModel::StereoCameraModel(std::string filename_left, std::string filename_right): 
  camera_left_(filename_left), camera_right_(filename_right){}


} // namespace camera
} // namespace utilities
} // namespace core
} // namespace aru
