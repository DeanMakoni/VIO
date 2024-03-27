#include <aru/core/bundle_adjustment/system.h>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

DEFINE_string(BA_CONFIG, "", "path to VO Config file");
DEFINE_string(IMAGE_LEFT, "", "path to Image left monolithic file");
DEFINE_string(IMAGE_RIGHT, "", "path to Image right monolithic file");
DEFINE_string(VO_MONO, "", "path to Image right monolithic file");
DEFINE_string(BA_MONO, "", "path to Bundle Adjustment output monolithic file");

using namespace aru::core;

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  boost::filesystem::path config_path(FLAGS_BA_CONFIG);
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

  boost::filesystem::path vo_path(FLAGS_VO_MONO);
  if (!boost::filesystem::exists(vo_path) ||
      !boost::filesystem::is_regular_file(vo_path))
    LOG(FATAL) << " VO file " << vo_path.native()
               << " does not exist or is not a regular file.";

  LOG(INFO) << "This is an info  message";

  bundle_adjustment::System BundleAdjuster(FLAGS_BA_CONFIG, FLAGS_IMAGE_LEFT,
                                           FLAGS_IMAGE_RIGHT, FLAGS_VO_MONO,
                                           FLAGS_BA_MONO);

  BundleAdjuster.Run();

  return 0;
}