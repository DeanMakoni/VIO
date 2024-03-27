
#include <Eigen/Dense>
#include <aru/core/mapping/mesh_mapping/laser_mesh_map.h>
#include <aru/core/vo/vo.h>

#include <aru/core/utilities/image/image.h>
#include <aru/core/utilities/image/imageprotocolbufferadaptor.h>
#include <aru/core/utilities/laser/laser.h>
#include <aru/core/utilities/laser/laserprotocolbufferadaptor.h>
#include <aru/core/utilities/logging/log.h>
#include <aru/core/utilities/transforms/transformprotocolbufferadaptor.h>
#include <aru/core/utilities/transforms/transforms.h>
#include <boost/make_shared.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <iostream>
#include <pbLaser.pb.h>
#include <pbStereoImage.pb.h>
#include <pbTransform.pb.h>

using namespace aru::core;
using namespace aru::core::utilities;
using namespace aru::core::utilities::logging;
using namespace aru::core::utilities::image;
using namespace aru::core::utilities::laser;
using namespace datatype::image;
using namespace datatype::laser;
using namespace datatype::transform;

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  LOG(INFO) << "This is an info  message";

  // Mono Image Monolithics
  std::string laser_monolithic(
      "/home/paulamayo/data/husky_data/log/zoo_2_laser.monolithic");
  std::string image_monolithic(
      "/home/paulamayo/data/husky_data/log/outdoor_zoo_2_left.monolithic");
  std::string vo_monolithic(
      "/home/paulamayo/data/husky_data/vo/outdoor_zoo_2_vo.monolithic");

  std::string output_ply = "/home/paulamayo/data/husky_data/mapping/laser_mesh"
                           ".ply";
  // Create the loggers
  ProtocolLogger<pbLaser> laser_logger(laser_monolithic, false);
  ProtocolLogger<pbTransform> vo_logger(vo_monolithic, false);
  ProtocolLogger<pbImage> image_logger(image_monolithic, false);

  // Create the mesh mapper
  mapping::mesh_map::LaserMeshMap laser_mapper(
      "/home/paulamayo/data/husky_data/mapping/viso_mapping_zed.yaml");

  // Read the Vo monolithic into the mesh_map
  pbTransform pb_transform = vo_logger.ReadFromFile();
  while (!vo_logger.EndOfFile()) {
    transform::Transform curr_transform = aru::core::utilities::transform ::
        TransformProtocolBufferAdaptor::ReadFromProtocolBuffer(pb_transform);
    aru::core::utilities::transform::TransformSPtr curr_transform_sptr =
        boost::make_shared<aru::core::utilities::transform::Transform>(
            curr_transform);
    laser_mapper.ReadTransform(curr_transform_sptr);
    pb_transform = vo_logger.ReadFromFile();
  }

  image_logger.ReadFromFile();
  // Read the following images
  LOG(INFO) << "Updating the map";
  pbLaser pb_laser = laser_logger.ReadFromFile();
  pbImage pb_image_left = image_logger.ReadFromFile();
  while (!image_logger.EndOfFile()) {
    Image curr_image =
        ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(pb_image_left);
    while (curr_image.GetTimeStamp() > pb_laser.timestamp())
      pb_laser = laser_logger.ReadFromFile();
    Laser curr_laser =
        LaserProtocolBufferAdaptor::ReadFromProtocolBuffer(pb_laser);
    laser_mapper.UpdateMap(curr_image, curr_laser);
    pb_image_left = image_logger.ReadFromFile();
  }

  laser_mapper.SaveCurrentTsdf(output_ply);

  return 0;
}
