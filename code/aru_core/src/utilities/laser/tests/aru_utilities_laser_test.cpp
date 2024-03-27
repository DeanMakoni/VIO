#define BOOST_TEST_MODULE My Test
#include "aru/core/utilities/laser/laserprotocolbufferadaptor.h"
#include "aru/core/utilities/logging/log.h"
#include "glog/logging.h"
#include <boost/make_shared.hpp>
#include <boost/test/included/unit_test.hpp>

struct LaserFixture {


  int timestamp;

  Eigen::MatrixXf points;

  LaserFixture() {

    timestamp=1;

    points=Eigen::MatrixXf::Identity(3,3);
  }
};

BOOST_FIXTURE_TEST_CASE(BuildCheck, LaserFixture) {
  aru::core::utilities::laser::Laser laser(timestamp,points);
}
