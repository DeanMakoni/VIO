
#include <Eigen/Dense>
#include <aru/core/vo/system.h>

#include <aru/core/utilities/image/image.h>
#include <aru/core/utilities/image/imageprotocolbufferadaptor.h>
#include <aru/core/utilities/logging/log.h>
#include <aru/core/utilities/viewer/vo_viewer.h>
#include <boost/make_shared.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <pbStereoImage.pb.h>
#include <thread>

using namespace aru::core;
using namespace aru::core::utilities;
using namespace aru::core::utilities::logging;
using namespace aru::core::utilities::image;
using namespace datatype::image;


int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  LOG(INFO) << "This is an info  message";

  auto *vo_ = new vo::System(
      "/home/paulamayo/data/husky_data/vo/vo_config_zed.yaml",
      "/home/paulamayo/data/husky_data/log/white_lab_left.monolithic",
      "/home/paulamayo/data/husky_data/log/white_lab_right.monolithic",
      "/home/paulamayo/data/husky_data/vo/white_lab_vo.monolithic");

  auto *viewer =
      new aru::core::utilities::viewer::VOViewer(640, 480, vo_->PoseChain());

  auto *viewer_thread = new thread(&utilities::viewer::VOViewer::Run, viewer);

  auto *vo_thread = new thread(&vo::System::Run, vo_);

  while (1) {
  }

  return 0;
}
