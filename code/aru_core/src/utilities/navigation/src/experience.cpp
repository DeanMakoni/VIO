#include "aru/core/utilities/navigation/experience.h"
#include <Eigen/Dense>
#include <boost/make_shared.hpp>
#include <chrono>
#include <utility>

namespace aru {
namespace core {
namespace utilities {
namespace navigation {

//------------------------------------------------------------------------------
Experience::Experience(int64 timestamp,
                       const cv::Mat &image_cv,
                       Eigen::MatrixXf keypoints,
                       Eigen::MatrixXf landmarks,
                       cv::Mat descriptors,
                       cv::Mat bow_descriptors)
    : timestamp_(timestamp), image_(image_cv), landmarks_(landmarks),
      keypoints_(keypoints), descriptors_(descriptors),
      bow_descriptors_(bow_descriptors) {}
//------------------------------------------------------------------------------
Experience::Experience() {}

} // namespace navigation
} // namespace utilities
} // namespace core
} // namespace aru
