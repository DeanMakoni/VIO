#ifndef ARU_UTILITIES_LASER_PROTOCOL_BUFFER_H_
#define ARU_UTILITIES_LASER_PROTOCOL_BUFFER_H_

#include "laser.h"
#include "pbLaser.pb.h"
#include <Eigen/Sparse>
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Core/util/Constants.h>
#include <glog/logging.h>
#include <iostream>
#include <opencv2/core/mat.hpp>

namespace aru {
namespace core {
namespace utilities {
namespace laser {

class LaserProtocolBufferAdaptor {
public:
  LaserProtocolBufferAdaptor()=default;

  virtual ~LaserProtocolBufferAdaptor() = default;

  static datatype::laser::pbLaser
  WriteToProtocolBuffer(Laser laser);

  static Laser
  ReadFromProtocolBuffer(const datatype::laser::pbLaser& pb_laser);

private:
};
} // namespace image
} // namespace utilities
} // namespace core
} // namespace aru

#endif // ARU_UTILITIES_LASER_PROTOCOL_BUFFER_H_
