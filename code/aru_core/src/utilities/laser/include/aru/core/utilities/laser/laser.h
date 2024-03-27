#ifndef ARU_UTILITIES_LASER_H_
#define ARU_UTILITIES_LASER_H_

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
namespace laser {

class Laser {
public:
  Laser() = default;

  Laser(int64 timestamp, const Eigen::MatrixXf points);

  virtual ~Laser() = default;

  int64 GetTimeStamp() const { return timestamp_; }

  void SetPoints(Eigen::MatrixXf points) { points_ = std::move(points); }

  void SetTimestamp(int64 timestamp) { timestamp_ = timestamp; }

  Eigen::MatrixXf GetPoints() { return points_; }

  Eigen::MatrixXf GetIntensity() { return intensity_; }
  Eigen::MatrixXf GetReflectance() { return reflectance_; }

private:
  Eigen::MatrixXf points_;
  Eigen::MatrixXf intensity_;
  Eigen::MatrixXf reflectance_;

  int64 timestamp_;
};

} // namespace laser
} // namespace utilities
} // namespace core
} // namespace aru

#endif // ARU_UTILITIES_LASER_H_
