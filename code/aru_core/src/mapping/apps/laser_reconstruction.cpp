#include <aru/core/mapping/mesh_mapping/mesh_map_system.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>

DEFINE_string(MAP_CONFIG, "", "path to Map Config file");
DEFINE_string(IMAGE_MONO, "", "path to Image RGB monolithic file");
DEFINE_string(LASER_MONO, "", "path to Image Depth monolithic file");
DEFINE_string(VO_MONO, "", "path to VO output monolithic file");
DEFINE_string(OUTPUT_PLY, "", "path to output ply file");

using namespace aru::core;

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  boost::filesystem::path config_path(FLAGS_MAP_CONFIG);
  if (!boost::filesystem::exists(config_path) ||
      !boost::filesystem::is_regular_file(config_path))
    LOG(FATAL) << " Config file " << config_path.native()
               << " does not exist or is not a regular file.";

  boost::filesystem::path image_path(FLAGS_IMAGE_MONO);
  if (!boost::filesystem::exists(image_path) ||
      !boost::filesystem::is_regular_file(image_path))
    LOG(FATAL) << " Image rgb file " << image_path.native()
               << " does not exist or is not a regular file.";

  boost::filesystem::path laser_path(FLAGS_LASER_MONO);
  if (!boost::filesystem::exists(laser_path) ||
      !boost::filesystem::is_regular_file(laser_path))
    LOG(FATAL) << " Laser file " << laser_path.native()
               << " does not exist or is not a regular file.";

  boost::filesystem::path vo_path(FLAGS_VO_MONO);
  if (!boost::filesystem::exists(vo_path) ||
      !boost::filesystem::is_regular_file(vo_path))
    LOG(FATAL) << " VO file " << vo_path.native()
               << " does not exist or is not a regular file.";

  LOG(INFO) << "This is an info  message";

  mapping::mesh_map::LaserSystem LaserReconstruction(
      FLAGS_MAP_CONFIG, FLAGS_IMAGE_MONO, FLAGS_LASER_MONO, FLAGS_VO_MONO,
      FLAGS_OUTPUT_PLY);

  LaserReconstruction.Run();

  return 0;
}
