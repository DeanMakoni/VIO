
#include <Eigen/Dense>
#include <boost/make_shared.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <include/aru/core/utilities/image/image.h>
#include <include/aru/core/utilities/image/imageprotocolbufferadaptor.h>
#include <include/aru/core/utilities/logging/log.h>
#include <iostream>
#include <pbStereoImage.pb.h>

#include "pbImage.pb.h"

using namespace aru;
using namespace aru::core::utilities;
using namespace aru::core::utilities::logging;
using namespace aru::core::utilities::image;
using namespace datatype::image;
int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  LOG(INFO) << "This is an info  message";
  std::string output_monolithic(
      "/home/paulamayo/data/kitti/dataset/sequences/00/image.monolithic");
  // Create the logger
  ProtocolLogger<datatype::image::pbStereoImage> logger(output_monolithic, true);
  for (int i = 0; i < 4000; ++i) {
    std::string directory_left(
        "/home/paulamayo/data/kitti/dataset/sequences/00/image_0/00");

    std::string directory_right(
        "/home/paulamayo/data/kitti/dataset/sequences/00/image_1/00");
    std::string zero_pad;

    if (i < 1000) {
      zero_pad = "0";
    }

    if (i < 100) {
      zero_pad = "00";
    }

    if (i < 10) {
      zero_pad = "000";
    }

    LOG(INFO) << directory_left + zero_pad + std::to_string(i) + "_left.png";
    cv::Mat image_left =
        cv::imread(directory_left + zero_pad + std::to_string(i) + ".png");
    cv::Mat image_right =
        cv::imread(directory_right + zero_pad + std::to_string(i) + ".png");

    // Create a wrapped image
    Image cv_image_left(i, image_left);
    Image cv_image_right(i, image_right);

    pbStereoImage pb_image_1 =
        ImageProtocolBufferAdaptor ::ReadStereoToProtocolBuffer(
            StereoImage {cv_image_left, cv_image_right});
    logger.WriteToFile(pb_image_1);

    cv::imshow("Image", image_left);
    cv::waitKey(20);
  }

  return 0;
}
