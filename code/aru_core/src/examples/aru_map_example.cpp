
#include <Eigen/Dense>
#include <aru/core/mapping/mesh_mapping/mesh_map_system.h>
#include <aru/core/vo/vo.h>

#include <aru/core/utilities/laser/laser.h>
#include <aru/core/utilities/laser/laserprotocolbufferadaptor.h>
#include <boost/make_shared.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <include/aru/core/utilities/image/image.h>
#include <include/aru/core/utilities/image/imageprotocolbufferadaptor.h>
#include <include/aru/core/utilities/logging/log.h>
#include <iostream>
#include <pbLaser.pb.h>
#include <pbStereoImage.pb.h>

using namespace aru::core;
using namespace aru::core::utilities;
using namespace aru::core::utilities::logging;
using namespace aru::core::utilities::image;
using namespace aru::core::utilities::laser;
using namespace datatype::image;
using namespace datatype::laser;

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  LOG(INFO) << "This is an info  message";

  mapping::mesh_map::System mesh_map(
      "/home/paulamayo/data/husky_data/mapping/viso_mapping_zed.yaml",
      "/home/paulamayo/data/husky_data/log/outdoor_zoo_3_left.monolithic",
      "/home/paulamayo/data/husky_data/log/outdoor_zoo_3_right.monolithic",
      "/home/paulamayo/data/husky_data/mapping/zoo_image_3_mesh.ply");

  mesh_map.Run();
  return 0;
}
