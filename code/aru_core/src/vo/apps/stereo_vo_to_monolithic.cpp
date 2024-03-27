#include <aru/core/vo/system.h>
#include <aru/core/utilities/viewer/vo_viewer.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <thread>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>

DEFINE_string(VOCAB, "/home/paulamayo/data/husky_data/vocabulary/ORBvoc.txt",
              "path to ORB vocab file");
DEFINE_string(VO_CONFIG,
              "/home/paulamayo/data/husky_data/vo/vo_config_zed.yaml",
              "path to VO Config file");
DEFINE_string(
    IMAGE_LEFT,
    "/home/paulamayo/data/husky_data/log/keyframe_farm_1_left.monolithic",
    "path to Image left monolithic file");
DEFINE_string(
    IMAGE_RIGHT,
    "/home/paulamayo/data/husky_data/log/keyframe_farm_1_right.monolithic",
    "path to Image right monolithic file");
DEFINE_string(VO_MONO,
              "/home/paulamayo/data/husky_data/vo/farm_1_vo.monolithic",
              "path to VO output monolithic file");
using namespace aru::core;

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  boost::filesystem::path vocab_path(FLAGS_VOCAB);
  if (!boost::filesystem::exists(vocab_path) ||
      !boost::filesystem::is_regular_file(vocab_path))
      LOG(FATAL) << "ORB vocab file " << vocab_path.native()
      << " does not exist or is not a regular file.";

  boost::filesystem::path config_path(FLAGS_VO_CONFIG);
  if (!boost::filesystem::exists(config_path) ||
      !boost::filesystem::is_regular_file(config_path))
    LOG(FATAL) << " Config file " << config_path.native()
               << " does not exist or is not a regular file.";

  boost::filesystem::path image_left_path(FLAGS_IMAGE_LEFT);
  if (!boost::filesystem::exists(image_left_path) ||
      !boost::filesystem::is_regular_file(image_left_path))
    LOG(FATAL) << " Image left file " << image_left_path.native()
               << " does not exist or is not a regular file.";

  boost::filesystem::path image_right_path(FLAGS_IMAGE_RIGHT);
  if (!boost::filesystem::exists(image_right_path) ||
      !boost::filesystem::is_regular_file(image_right_path))
    LOG(FATAL) << " Image right file " << image_right_path.native()
               << " does not exist or is not a regular file.";

  LOG(INFO) << "This is an info  message";

  auto *VisualOdometry = new vo::System(FLAGS_VO_CONFIG, FLAGS_VOCAB, FLAGS_IMAGE_LEFT,
                            FLAGS_IMAGE_RIGHT, FLAGS_VO_MONO);

  auto *viewer = new aru::core::utilities::viewer::VOViewer(
      640, 480, VisualOdometry->PoseChain());

//  auto *viewer_thread = new std::thread(&utilities::viewer::VOViewer::Run,
//                                      viewer);
//
//  auto *vo_thread = new std::thread(&vo::System::Run, VisualOdometry);
//
//  while (1) {
//  }

  VisualOdometry->Run();

  google::protobuf::ShutdownProtobufLibrary();

  return 0;
}
