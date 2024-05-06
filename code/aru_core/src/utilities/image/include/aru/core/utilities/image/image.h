#ifndef ARU_UTILITIES_IMAGE_H_
#define ARU_UTILITIES_IMAGE_H_

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
namespace image {

class Image {
public:
  Image();

  Image(int64 timestamp, const cv::Mat &image_cv);

  virtual ~Image() = default;

  int GetHeight() const { return num_rows_; }

  int GetWidth() const { return num_cols_; }

  int64 GetTimeStamp() const { return timestamp_; }

  cv::Mat GetImage() { return image_; }

private:
  int num_rows_;
  int num_cols_;
  cv::Mat image_;
  int64 timestamp_;
};

typedef std::pair<Image,Image> StereoImage;
std::map<Image, StereoImage> Frames; 

} // namespace image
} // namespace utilities
} // namespace core
} // namespace aru

#endif // ARU_UTILITIES_IMAGE_H_
