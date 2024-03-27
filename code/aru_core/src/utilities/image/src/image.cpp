#include "aru/core/utilities/image/image.h"
#include <Eigen/Dense>
#include <boost/make_shared.hpp>
#include <chrono>
#include <utility>

namespace aru {
namespace core {
namespace utilities {
namespace image {

//------------------------------------------------------------------------------
Image::Image() {}
//------------------------------------------------------------------------------
Image::Image(int64 timestamp, const cv::Mat &image_cv)
    : timestamp_(timestamp), image_(image_cv) {
  num_rows_ = image_cv.rows;
  num_cols_ = image_cv.cols;
}

} // namespace image
} // namespace utilities
} // namespace core
} // namespace aru
