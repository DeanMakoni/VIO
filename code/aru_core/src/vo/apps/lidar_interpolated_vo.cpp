#include <aru/core/vo/system.h>
#include <aru/core/utilities/viewer/vo_viewer.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <thread>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>


DEFINE_string(LASER_MONO, "", "path to Laser monolithic file");
DEFINE_string(INTERPOLATED_MONO, "", "path to Laser iterpolated monolithic "
                                    "file");
DEFINE_string(VO_MONO, "", "path to VO monolithic file");

using namespace aru::core;

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  boost::filesystem::path laser_path(FLAGS_LASER_MONO);
  if (!boost::filesystem::exists(laser_path) ||
      !boost::filesystem::is_regular_file(laser_path))
    LOG(FATAL) << " Image left file " << laser_path.native()
               << " does not exist or is not a regular file.";

  boost::filesystem::path vo_mono_path(FLAGS_VO_MONO);
  if (!boost::filesystem::exists(vo_mono_path) ||
      !boost::filesystem::is_regular_file(vo_mono_path))
    LOG(FATAL) << " Image right file " << vo_mono_path.native()
               << " does not exist or is not a regular file.";

  LOG(INFO) << "This is an info  message";

  auto *VisualOdometry = new vo::LaserSystem( FLAGS_LASER_MONO,
                            FLAGS_VO_MONO, FLAGS_INTERPOLATED_MONO);

  auto *viewer = new aru::core::utilities::viewer::VOViewer(
      640, 480, VisualOdometry->PoseChain());

  auto *viewer_thread = new std::thread(&utilities::viewer::VOViewer::Run,
                                      viewer);

  auto *vo_thread = new std::thread(&vo::LaserSystem::Run, VisualOdometry);

  while (1) {
  }

  return 0;
}
