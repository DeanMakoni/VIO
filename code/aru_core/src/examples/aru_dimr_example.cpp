

#include "aru/core/dimr/dimr.h"

#include "opencv2/optflow.hpp"
#include "opencv2/ximgproc.hpp"
#include <Eigen/Dense>
#include <boost/make_shared.hpp>
#include <iostream>

#include <aru/core/mesh/mesh.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

using namespace aru::core;
using namespace aru::core::mapping;

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  LOG(INFO) << "This is an info  message";

  aru::core::dimr::DiMR distraction_map(
      "/home/paulamayo/code/aru-core/src/dimr/config/dimr.yaml");

  LOG(INFO) << "DiMR initialised";

  int start = 0;
  for (int i = start; i < 1200; ++i) {
    std::string directory_left(
        "/home/paulamayo/data/multi_vo/swinging_static/stereo/000");

    std::string directory_right(
        "/home/paulamayo/data/multi_vo/swinging_static/stereo/000");
    std::string zero_pad;
    if (i < 1000) {
      zero_pad = "";
    }
    if (i < 100) {
      zero_pad = "0";
    }

    if (i < 10) {
      zero_pad = "00";
    }
    LOG(INFO) << directory_left + zero_pad + std::to_string(i) + "_left.png";
    cv::Mat image_left;
    float resize = 1;
    cv::resize(cv::imread(directory_left + zero_pad + std::to_string(i) +
                          "_left"
                          ".png"),
               image_left, cv::Size(), resize, resize);

    cv::Mat image_right;
    cv::resize(cv::imread(directory_right + zero_pad + std::to_string(i) +
                          "_right.png"),
               image_right, cv::Size(), resize, resize);

    if (i == start) {
      distraction_map.InitialFrame(image_left, image_right);
    } else {
      distraction_map.UpdateFrame(image_left, image_right);
    }
  }
  return 0;
}
