
#include "aru/core/utilities/laser/laserprotocolbufferadaptor.h"
#include <Eigen/Dense>
#include <boost/make_shared.hpp>
#include <chrono>
#include <opencv2/features2d.hpp>
#include <utility>

namespace aru {
namespace core {
namespace utilities {
namespace laser {
//------------------------------------------------------------------------------
Laser LaserProtocolBufferAdaptor::ReadFromProtocolBuffer(
    const datatype::laser::pbLaser &pb_laser) {

  Laser laser_out;
  laser_out.SetTimestamp(pb_laser.timestamp());

  Eigen::MatrixXf points = Eigen::MatrixXf::Zero(pb_laser.x_size(), 3);
  for (int i = 0; i < pb_laser.x_size(); ++i) {
    points(i, 0) = pb_laser.x(i);
    points(i, 1) = pb_laser.y(i);
    points(i, 2) = pb_laser.z(i);
  }
  laser_out.SetPoints(points);

  return laser_out;
}
//------------------------------------------------------------------------------
datatype::laser::pbLaser
LaserProtocolBufferAdaptor::WriteToProtocolBuffer(Laser laser) {
  datatype::laser::pbLaser pb_laser;
  pb_laser.set_timestamp(laser.GetTimeStamp());
  Eigen::MatrixXf points = laser.GetPoints();
  for (int i = 0; i < points.rows(); ++i) {
    pb_laser.add_x(points(i, 0));
    pb_laser.add_y(points(i, 1));
    pb_laser.add_z(points(i, 2));
    pb_laser.add_intensity(0);
    pb_laser.add_reflectance(0);
  }
  return pb_laser;
}
//------------------------------------------------------------------------------
} // namespace laser
} // namespace utilities
} // namespace core
} // namespace aru
