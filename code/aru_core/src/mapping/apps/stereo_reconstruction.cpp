#include <aru/core/mapping/mesh_mapping/mesh_map_system.h>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

DEFINE_string(MAP_CONFIG, "", "path to VO Config file");
DEFINE_string(VOCAB_FILE, "", "path to ORB vocab file");
DEFINE_string(IMAGE_LEFT, "", "path to Image left monolithic file");
DEFINE_string(IMAGE_RIGHT, "", "path to Image right monolithic file");
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

  mapping::mesh_map::StereoSystem StereoMapping(
      FLAGS_MAP_CONFIG, FLAGS_VOCAB_FILE, FLAGS_IMAGE_LEFT, FLAGS_IMAGE_RIGHT, FLAGS_OUTPUT_PLY);

  StereoMapping.Run();

  return 0;
}
