#ifndef ARU_UTILITIES_EXPERIENCE_H_
#define ARU_UTILITIES_EXPERIENCE_H_

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
namespace navigation {

class Experience {
private:
    cv::Mat image_;
    int64 timestamp_;
    Eigen::MatrixXf keypoints_;
    Eigen::MatrixXf landmarks_;
    cv::Mat descriptors_;
    cv::Mat bow_descriptors_;

public:

    Experience();

    Experience(int64_t timestamp,
               const cv::Mat &image_cv,
               Eigen::MatrixXf keypoints,
               Eigen::MatrixXf landmarks,
               cv::Mat descriptors,
               cv::Mat bow_descriptors);

    virtual ~Experience() = default;

    int64 GetTimeStamp() const { return timestamp_; }

    cv::Mat GetImage() { return image_; }
    cv::Mat GetDescriptors() { return descriptors_; }
    cv::Mat GetBowDescriptors() { return bow_descriptors_; }

    Eigen::MatrixXf GetLandmarks() { return landmarks_; }
    Eigen::MatrixXf GetKeypoints() { return keypoints_; }
};

} // namespace navigation
} // namespace utilities
} // namespace core
} // namespace aru

#endif // ARU_UTILITIES_EXPERIENCE_H_
