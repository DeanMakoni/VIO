#define BOOST_TEST_MODULE My Test
#include "glog/logging.h"
#include "include/aru/core/utilities/camera/camera.h"
#include <boost/make_shared.hpp>
#include <boost/test/included/unit_test.hpp>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/util/delimited_message_util.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>

using namespace aru::core::utilities::camera;

// struct CameraFixture {
//
//   boost::shared_ptr<CameraModel> camera_model_;
//
//   CameraFixture() {
//
//     camera_model_=boost::make_shared<CameraModel>
//         ("/home/paulamayo/data/husky_data/aru-calibration-main/left.yaml");
//   }
// };
//
//  BOOST_FIXTURE_TEST_CASE(CameraConstructor, CameraFixture) {
//
//  }

BOOST_AUTO_TEST_CASE(CameraConstructor) {
  CameraModel camera_model_("/home/paulamayo/data/husky_data/aru-calibration-main/left.yaml");
}
