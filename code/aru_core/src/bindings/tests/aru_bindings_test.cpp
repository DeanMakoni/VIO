#define BOOST_TEST_MODULE My Test
#include "aru/core/utilities/logging/log.h"
#include "glog/logging.h"
#include <boost/make_shared.hpp>
#include <boost/test/included/unit_test.hpp>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/util/delimited_message_util.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>

struct BindingsFixture {
  std::string filename;

  BindingsFixture() {
    filename = "/home/paulamayo/data/kitti/training/image.monolithic";
  }
};

BOOST_FIXTURE_TEST_CASE(ReadWriteFromProtocolBuffer, BindingsFixture) {


}
