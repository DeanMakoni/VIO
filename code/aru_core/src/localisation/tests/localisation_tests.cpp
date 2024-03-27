#define BOOST_TEST_MODULE My Test
#include <aru/core/localisation/localisation.h>
#include <aru/core/localisation/loop_closer.h>
#include <aru/core/utilities/logging/index_log.h>
#include <include/aru/core/utilities/image/image.h>
#include <include/aru/core/utilities/image/imageprotocolbufferadaptor.h>
#include <include/aru/core/utilities/logging/log.h>

#include "glog/logging.h"
#include <boost/make_shared.hpp>
#include <boost/test/included/unit_test.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <pbStereoImage.pb.h>

using namespace aru::core::utilities;
using namespace aru::core::utilities::logging;
using namespace aru::core::utilities::image;
using namespace datatype::image;

struct LocalisationFixture {

  const double baseline = 0.5372;
  Eigen::MatrixXd K;

  const int n = 1000;
  const double noise = 3;

  LocalisationFixture() { K = Eigen::MatrixXd::Zero(3, 3); }
};

BOOST_AUTO_TEST_CASE(Localiser) {
  aru::core::localisation::Localisation localiser(
      "/home/paulamayo/data/husky_data/localisation/zoo_vocab.yml",
      "/home/paulamayo/data/husky_data/localisation/zoo_loop_chow_liu_tree.yml",
      "/home/paulamayo/data/husky_data/localisation/settings.yml");


  ProtocolLogger<datatype::image::pbImage> logger_left(
       "/home/paulamayo/data/husky_data/log/zoo_loop_left.monolithic", false);

  ProtocolLogger<datatype::image::pbImage> logger_right(
      "/home/paulamayo/data/husky_data/log/zoo_loop_left.monolithic", false);

   pbImage pb_image_left = logger_left.ReadFromFile();

//   for (int i = 0; i < 5; ++i) {
//     aru::core::utilities::image::Image image_left =
//         ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(pb_image_left);
//     localiser.AddVocabularyTrainingImage(image_left.GetImage());
//     pb_image_left = logger_left.ReadFromFile();
//   }
//   localiser.TrainAndSaveVocabulary();

   for (int i = 0; i < 5; ++i) {
     aru::core::utilities::image::Image image_left =
         ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(pb_image_left);
     localiser.AddQueryImage(image_left.GetImage());
     pb_image_left = logger_left.ReadFromFile();
   }
   localiser.TrainAndSaveTree();

   localiser.InitLocalisation();

   for (int i = 0; i < 5; ++i) {
     aru::core::utilities::image::Image image_left =
         ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(pb_image_left);
     cv::Mat img=image_left.GetImage();
     localiser.FindClosestImage(img);
     pb_image_left = logger_left.ReadFromFile();
   }

  LOG(INFO) << "Do something";
}
// BOOST_FIXTURE_TEST_CASE(TestLocalisation, LocalisationFixture) {
//   aru::core::localisation::LoopCloser loop_closer(
//       "/home/paulamayo/data/husky_data/vocabulary/ORBvoc/ORBvoc.yml",
//       "/home/paulamayo/code/aru-core/src/localisation/config/"
//       "loop_closure_config.yaml");
//
//   ProtocolLogger<datatype::image::pbImage> logger_left(
//       "/home/paulamayo/data/husky_data/log/zoo_loop_left.monolithic", false);
//   ProtocolLogger<datatype::image::pbImage> logger_right(
//       "/home/paulamayo/data/husky_data/log/zoo_loop_right.monolithic",
//       false);
//
//   pbImage pb_image_left = logger_left.ReadFromFile();
//   pbImage pb_image_right = logger_right.ReadFromFile();
//
//   for (int i = 0; i < 500; ++i) {
//     aru::core::utilities::image::Image image_left =
//         ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(pb_image_left);
//     aru::core::utilities::image::Image image_right =
//         ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(pb_image_right);
//
//     loop_closer.LoopDetect(image_left.GetImage(), image_right.GetImage());
//     pb_image_left = logger_left.ReadFromFile();
//     pb_image_right = logger_right.ReadFromFile();
//   }
//
//   for (int i = 0; i < 6500; ++i) {
//     pb_image_left = logger_left.ReadFromFile();
//     pb_image_right = logger_right.ReadFromFile();
//   }
//
//   while (!logger_right.EndOfFile() || !logger_left.EndOfFile()) {
//
//     aru::core::utilities::image::Image image_left =
//         ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(pb_image_left);
//     aru::core::utilities::image::Image image_right =
//         ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(pb_image_right);
//
//     loop_closer.LoopDetect(image_left.GetImage(), image_right.GetImage());
//
//     cv::imshow("Image left", image_left.GetImage());
//     cv::waitKey(5);
//
//     pb_image_left = logger_left.ReadFromFile();
//     pb_image_right = logger_right.ReadFromFile();
//   }
// }
