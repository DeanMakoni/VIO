#include <aru/core/utilities/viewer/vo_viewer.h>
#include <aru/core/vo/system.h>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <thread>

DEFINE_string(EXP_MONO, "/home/david/data/Husky_Bags/New_lab/2022-08-31-05-01-56_exp.monolithic", "path to image left monolithic file");
DEFINE_string(KEYFRAME_EXP_MONO, "/home/david/data/Husky_Bags/New_lab/2022-08-31-05-01-56_keyframe_exp.monolithic", "path to keyframe left monolithic file");
DEFINE_string(VO_MONO, "/home/david/data/Husky_Bags/New_lab/2022-08-31-05-01-56_vo.monolithic", "path to VO monolithic file");

using namespace aru::core;

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  boost::filesystem::path experience_path(FLAGS_EXP_MONO);
  if (!boost::filesystem::exists(experience_path) ||
      !boost::filesystem::is_regular_file(experience_path))
      LOG(FATAL) << " Experience file " << experience_path.native()
                 << " does not exist or is not a regular file.";

  boost::filesystem::path vo_mono_path(FLAGS_VO_MONO);
  if (!boost::filesystem::exists(vo_mono_path) ||
      !boost::filesystem::is_regular_file(vo_mono_path))
    LOG(FATAL) << " Image right file " << vo_mono_path.native()
               << " does not exist or is not a regular file.";

  LOG(INFO) << "This is an info  message";

  auto *Keyframe = new vo::KeyframeExpSystem(FLAGS_EXP_MONO,
                                          FLAGS_VO_MONO,
                                          FLAGS_KEYFRAME_EXP_MONO);

  auto *viewer = new aru::core::utilities::viewer::VOViewer(
      640, 480, Keyframe->PoseChain());

  auto *viewer_thread = new std::thread(&utilities::viewer::VOViewer::Run,
                                      viewer);

  auto *vo_thread = new std::thread(&vo::KeyframeExpSystem::Run, Keyframe);
  // Pangolin viewer needs to be exited and a button pressed in cv Keyframe window for program to terminate
  // Cleaner option and future to-do might be to check if threads are finished runnning
  while (cv::waitKey(10) == -1) {
  }
  // Close window on completion
  cv::destroyWindow("Keyframe");

  return 0;
}
