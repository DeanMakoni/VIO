//
// Created by paulamayo on 2022/04/14.
//

#include "aru/core/mapping/mesh_mapping/mesh_map_system.h"
#include <aru/core/utilities/image/imageprotocolbufferadaptor.h>
#include <aru/core/utilities/laser/laserprotocolbufferadaptor.h>
#include <aru/core/utilities/transforms/transformprotocolbufferadaptor.h>
#include <aru/core/utilities/viewer/viewer.h>
#include <boost/make_shared.hpp>
#include <utility>
using namespace datatype::image;
using namespace datatype::transform;
using namespace datatype::laser;
using namespace aru::core::utilities;

namespace aru {
namespace core {
namespace mapping {
namespace mesh_map {
//------------------------------------------------------------------------------
StereoSystem::StereoSystem(std::string mapping_config_file,
                           std::string vocab_file,
                           std::string image_left_monolithic,
                           std::string image_right_monolithic,
                           std::string output_ply)
    : output_ply_(std::move(output_ply)) {

  mesh_mapper_ = boost::make_shared<MeshMap>(mapping_config_file, vocab_file);
  image_left_logger_ = boost::make_shared<logging::ProtocolLogger<pbImage>>(
      image_left_monolithic, false);
  image_right_logger_ = boost::make_shared<logging::ProtocolLogger<pbImage>>(
      image_right_monolithic, false);
}
//------------------------------------------------------------------------------
void StereoSystem::Run() {

  // Read previous image
  pbImage image_left_prev = image_left_logger_->ReadFromFile();
  image::Image prev_image_left =
      image::ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(
          image_left_prev);
  pbImage image_right_prev = image_right_logger_->ReadFromFile();
  image::Image prev_image_right =
      image::ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(
          image_right_prev);

  cv::Mat image_1_left_grey, image_1_right_grey;

  mesh_mapper_->InitialiseMap(
      image::StereoImage(prev_image_left, prev_image_right));

  pbImage image_left_curr = image_left_logger_->ReadFromFile();
  pbImage image_right_curr = image_right_logger_->ReadFromFile();
  int num = 0;
  while (!image_left_logger_->EndOfFile() &&
         !image_right_logger_->EndOfFile()) {

    // Perform the estimation
    auto estimation_start = std::chrono::high_resolution_clock::now();
    image::Image curr_image_left =
        image::ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(
            image_left_curr);

    image::Image curr_image_right =
        image::ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(
            image_right_curr);

    mesh_mapper_->UpdateMap(
        image::StereoImage(curr_image_left, curr_image_right));

    cv::cvtColor(curr_image_left.GetImage(), image_1_left_grey,
                 cv::COLOR_BGR2GRAY);
    cv::cvtColor(curr_image_right.GetImage(), image_1_right_grey,
                 cv::COLOR_BGR2GRAY);

    image_left_curr = image_left_logger_->ReadFromFile();
    image_right_curr = image_right_logger_->ReadFromFile();
  }
  LOG(INFO) << "Saving mesh to ply";
  mesh_mapper_->SaveCurrentTsdf(output_ply_);
}

//------------------------------------------------------------------------------
DepthSystem::DepthSystem(std::string mapping_config_file,
                         std::string vocab_file,
                         std::string image_left_monolithic,
                         std::string image_depth_monolithic,
                         std::string vo_monolithic, std::string output_ply)
    : output_ply_(std::move(output_ply)) {

  mesh_mapper_ = boost::make_shared<MeshMap>(mapping_config_file, vocab_file);
  image_rgb_logger_ = boost::make_shared<logging::ProtocolLogger<pbImage>>(
      image_left_monolithic, false);
  image_depth_logger_ = boost::make_shared<logging::ProtocolLogger<pbImage>>(
      image_depth_monolithic, false);
  vo_logger_ = boost::make_shared<logging::ProtocolLogger<pbTransform>>(
      vo_monolithic, false);
}
//------------------------------------------------------------------------------
void DepthSystem::Run() {
  // Read the Vo monolithic into the mesh_map
  pbTransform pb_transform = vo_logger_->ReadFromFile();
  while (!vo_logger_->EndOfFile()) {
    transform::Transform curr_transform = aru::core::utilities::transform ::
        TransformProtocolBufferAdaptor::ReadFromProtocolBuffer(pb_transform);
    aru::core::utilities::transform::TransformSPtr curr_transform_sptr =
        boost::make_shared<aru::core::utilities::transform::Transform>(
            curr_transform);
    mesh_mapper_->ReadTransform(curr_transform_sptr);
    pb_transform = vo_logger_->ReadFromFile();
  }
  pbImage image_rgb_curr = image_rgb_logger_->ReadFromFile();
  pbImage image_depth_curr = image_depth_logger_->ReadFromFile();

  int num = 0;
  while (!image_rgb_logger_->EndOfFile() && !image_depth_logger_->EndOfFile()) {

    // Perform the estimation
    auto estimation_start = std::chrono::high_resolution_clock::now();
    image::Image curr_image_rgb =
        image::ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(
            image_rgb_curr);

    image::Image curr_image_depth =
        image::ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(
            image_depth_curr);

    mesh_mapper_->InsertDepthImage(curr_image_depth, curr_image_rgb);

    mesh_mapper_->DrawCurrentTsdf();

    image_rgb_curr = image_rgb_logger_->ReadFromFile();
    image_depth_curr = image_depth_logger_->ReadFromFile();
    LOG(INFO) << "Number is " << num;
    num++;
  }
  LOG(INFO) << "Saving mesh to ply";
  mesh_mapper_->SaveCurrentTsdf(output_ply_);
}
//------------------------------------------------------------------------------
LaserSystem::LaserSystem(std::string mapping_config_file,
                         std::string image_rgb_monolithic,
                         std::string laser_monolithic,
                         std::string vo_monolithic, std::string output_ply)
    : output_ply_(std::move(output_ply)) {

  laser_mesh_mapper_ = boost::make_shared<LaserMeshMap>(mapping_config_file);
  image_rgb_logger_ = boost::make_shared<logging::ProtocolLogger<pbImage>>(
      image_rgb_monolithic, false);
  laser_logger_ = boost::make_shared<logging::ProtocolLogger<pbLaser>>(
      laser_monolithic, false);
  vo_logger_ = boost::make_shared<logging::ProtocolLogger<pbTransform>>(
      vo_monolithic, false);
}
//------------------------------------------------------------------------------
void LaserSystem::Run() {
  // Read the Vo monolithic into the mesh_map
  pbTransform pb_transform = vo_logger_->ReadFromFile();
  while (!vo_logger_->EndOfFile()) {
    transform::Transform curr_transform = aru::core::utilities::transform ::
        TransformProtocolBufferAdaptor::ReadFromProtocolBuffer(pb_transform);
    aru::core::utilities::transform::TransformSPtr curr_transform_sptr =
        boost::make_shared<aru::core::utilities::transform::Transform>(
            curr_transform);
    laser_mesh_mapper_->ReadTransform(curr_transform_sptr);
    pb_transform = vo_logger_->ReadFromFile();
  }

  image_rgb_logger_->ReadFromFile();
  // Read the following images
  LOG(INFO) << "Updating the map";
  pbLaser pb_laser = laser_logger_->ReadFromFile();
  pbImage pb_image_left = image_rgb_logger_->ReadFromFile();
  //for (int i = 0; i < 90; ++i) { image_rgb_logger_->ReadFromFile();}
  while (!image_rgb_logger_->EndOfFile()) {
  //for (int i = 0; i < 100; ++i) {
    image::Image curr_image =
        image::ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(
            pb_image_left);
    laser::Laser curr_laser =
        laser::LaserProtocolBufferAdaptor::ReadFromProtocolBuffer(pb_laser);
    laser_mesh_mapper_->UpdateMap(curr_image, curr_laser);
    while (curr_image.GetTimeStamp() > pb_laser.timestamp() &&
           !laser_logger_->EndOfFile()) {
      laser::Laser curr_laser =
          laser::LaserProtocolBufferAdaptor::ReadFromProtocolBuffer(pb_laser);
      //      laser_mesh_mapper_->UpdateMap(curr_image, curr_laser);
      pb_laser = laser_logger_->ReadFromFile();
    }
    // pb_laser = laser_logger_->ReadFromFile();
    pb_image_left = image_rgb_logger_->ReadFromFile();

   // if(i%10==0) laser_mesh_mapper_->DrawCurrentTsdf();
  }

  laser_mesh_mapper_->SaveCurrentTsdf(output_ply_);
}

} // namespace mesh_map
} // namespace mapping
} // namespace core
} // namespace aru