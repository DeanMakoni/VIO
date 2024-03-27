#include "aru/core/repeat/controller.h"
#include <boost/make_shared.hpp>
#include <opencv2/core/eigen.hpp>

namespace aru {
namespace core {
namespace repeat {

using namespace cv;
using namespace std;
//------------------------------------------------------------------------------
// Constructor
Controller::Controller(std::string controller_config_file) {

  transform_map_ = std::make_shared<utilities::transform::TransformMap>();
}

//------------------------------------------------------------------------------
void Controller::AddTeachTransform(
    utilities::transform::TransformSPtr teach_transform) {
  transform_map_->AddTransform(teach_transform);
}
//------------------------------------------------------------------------------
Command Controller::TransformToCommand(
    utilities::transform::TransformSPtr teach_transform) {

  float dist = teach_transform->GetTransform().translation().norm();
  cv::Mat R_matrix = cv::Mat::zeros(3, 3, CV_64FC1); // rotation matrix
  cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
  // Read Rotation matrix and convert to vector
  cv::eigen2cv(teach_transform->GetRotation(), R_matrix);
  cv::Rodrigues(R_matrix, rvec);
  Eigen::Vector3f rpy;
  cv::cv2eigen(rvec, rpy);
  float rotation = rpy.norm();

  Command command = std::make_pair(dist, rotation);
  return command;
}
} // namespace repeat
} // namespace core
} // namespace aru
