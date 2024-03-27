#include "aru/core/utilities/laser/laser.h"
#include <Eigen/Dense>
#include <boost/make_shared.hpp>
#include <chrono>
#include <utility>

namespace aru {
namespace core {
namespace utilities {
namespace laser {

//------------------------------------------------------------------------------
Laser::Laser(int64 timestamp, const Eigen::MatrixXf points)
    : timestamp_(timestamp), points_(points) {}
//------------------------------------------------------------------------------
} // namespace laser
} // namespace utilities
} // namespace core
} // namespace aru
