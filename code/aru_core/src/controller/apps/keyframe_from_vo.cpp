#include <aru/core/utilities/viewer/vo_viewer.h>
#include <aru/core/vo/system.h>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <thread>

DEFINE_string(IMAGE_LEFT_MONO, "/home/david/data/Husky_Bags/lab/2022-07-04-10-28-32/log/white_lab_2_left.monolithic ", "path to image left monolithic file");
DEFINE_string(IMAGE_RIGHT_MONO, "/home/david/data/Husky_Bags/lab/2022-07-04-10-28-32/log/white_lab_2_right.monolithic", "path to image right monolithic file");
DEFINE_string(KEYFRAME_LEFT_MONO, "/home/david/data/Husky_Bags/lab/2022-07-04-10-28-32/log/left_kf.monolithic", "path to keyframe left monolithic file");
DEFINE_string(KEYFRAME_RIGHT_MONO, "/home/david/data/Husky_Bags/lab/2022-07-04-10-28-32/log/right_kf.monolithic", "path to keyframe right monolithic file");
DEFINE_string(VO_MONO, "/home/david/data/Husky_Bags/lab/2022-07-04-10-28-32/log/vo.monolithic", "path to VO monolithic file");

using namespace aru::core;

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  boost::filesystem::path image_left_path(FLAGS_IMAGE_LEFT_MONO);
  if (!boost::filesystem::exists(image_left_path) ||
      !boost::filesystem::is_regular_file(image_left_path))
    LOG(FATAL) << " Image left file " << image_left_path.native()
               << " does not exist or is not a regular file.";


  boost::filesystem::path image_right_path(FLAGS_IMAGE_RIGHT_MONO);
  if (!boost::filesystem::exists(image_right_path) ||
      !boost::filesystem::is_regular_file(image_right_path))
    LOG(FATAL) << " Image right file " << image_right_path.native()
               << " does not exist or is not a regular file.";


  boost::filesystem::path vo_mono_path(FLAGS_VO_MONO);
  if (!boost::filesystem::exists(vo_mono_path) ||
      !boost::filesystem::is_regular_file(vo_mono_path))
    LOG(FATAL) << " VO monolithic file " << vo_mono_path.native()
               << " does not exist or is not a regular file.";

  LOG(INFO) << "This is an info  message";
  auto *Keyframe = new vo::KeyframeSystem(FLAGS_VO_MONO,FLAGS_IMAGE_LEFT_MONO,
                                          FLAGS_IMAGE_RIGHT_MONO,
                                          FLAGS_KEYFRAME_LEFT_MONO,
                                          FLAGS_KEYFRAME_RIGHT_MONO);

  auto *viewer = new aru::core::utilities::viewer::VOViewer(
      640, 480, Keyframe->PoseChain());

  auto *viewer_thread = new std::thread(&utilities::viewer::VOViewer::Run,
                                      viewer);

  auto *vo_thread = new std::thread(&vo::KeyframeSystem::Run, Keyframe);
  // Pangolin viewer needs to be exited and a button pressed in cv Keyframe window for program to terminate
  // Cleaner option and future to-do might be to check if threads are finished runnning
  while (cv::waitKey(10) == -1) {
  }
  // Close window on completion
  cv::destroyWindow("Keyframe");
  return 0;
}
