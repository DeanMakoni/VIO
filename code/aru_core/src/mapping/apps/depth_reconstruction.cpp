
#include <aru/core/mapping/mesh_mapping/mesh_map_system.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>

DEFINE_string(MAP_CONFIG, "", "path to Map Config file");
DEFINE_string(VOCAB_FILE, "", "path to ORB vocab file");
DEFINE_string(IMAGE_RGB, "", "path to Image RGB monolithic file");
DEFINE_string(IMAGE_DEPTH, "", "path to Image Depth monolithic file");
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

  boost::filesystem::path vocab_path(FLAGS_VOCAB_FILE);
  if (!boost::filesystem::exists(vocab_path) ||
      !boost::filesystem::is_regular_file(vocab_path))
    LOG(FATAL) << " ORB vocab file " << vocab_path.native()
               << " does not exist or is not a regular file.";

  boost::filesystem::path image_rgb_path(FLAGS_IMAGE_RGB);
  if (!boost::filesystem::exists(image_rgb_path) ||
      !boost::filesystem::is_regular_file(image_rgb_path))
    LOG(FATAL) << " Image rgb file " << image_rgb_path.native()
               << " does not exist or is not a regular file.";

  boost::filesystem::path image_depth_path(FLAGS_IMAGE_DEPTH);
  if (!boost::filesystem::exists(image_depth_path) ||
      !boost::filesystem::is_regular_file(image_depth_path))
    LOG(FATAL) << " Image depth file " << image_depth_path.native()
               << " does not exist or is not a regular file.";

  boost::filesystem::path vo_path(FLAGS_VO_MONO);
  if (!boost::filesystem::exists(vo_path) ||
      !boost::filesystem::is_regular_file(vo_path))
    LOG(FATAL) << " VO file " << vo_path.native()
               << " does not exist or is not a regular file.";

  LOG(INFO) << "This is an info  message";

  mapping::mesh_map::DepthSystem DepthReconstruction(
      FLAGS_MAP_CONFIG, FLAGS_VOCAB_FILE, FLAGS_IMAGE_RGB, FLAGS_IMAGE_DEPTH, FLAGS_VO_MONO,
      FLAGS_OUTPUT_PLY);

  DepthReconstruction.Run();

  return 0;
}
