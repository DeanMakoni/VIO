//
// Created by paulamayo on 2022/06/17.
//

#include "aru/core/bundle_adjustment/system.h"
#include <aru/core/utilities/image/imageprotocolbufferadaptor.h>
#include <aru/core/utilities/transforms/transformprotocolbufferadaptor.h>
#include <aru/core/utilities/viewer/viewer.h>
#include <boost/make_shared.hpp>
using namespace datatype::image;
using namespace datatype::transform;
using namespace aru::core::utilities;

namespace aru {
namespace core {
namespace bundle_adjustment {
//------------------------------------------------------------------------------
System::System(std::string ba_config_file, std::string image_left_monolithic,
               std::string image_right_monolithic,
               std::string vo_transform_monolithic,
               std::string transform_monolithic) {

  ba_ = boost::make_shared<BundleAdjustment>(ba_config_file);
  image_left_logger_ = boost::make_shared<logging::ProtocolLogger<pbImage>>(
      image_left_monolithic, false);
  image_right_logger_ = boost::make_shared<logging::ProtocolLogger<pbImage>>(
      image_right_monolithic, false);
  vo_transform_logger_ =
      boost::make_shared<logging::ProtocolLogger<pbTransform>>(
          vo_transform_monolithic, false);
  ba_transform_logger_ =
      boost::make_shared<logging::ProtocolLogger<pbTransform>>(
          transform_monolithic, true);

  pose_chain_ = boost::make_shared<transform::TransformSPtrVector>();
}
//------------------------------------------------------------------------------
void System::Run() {
  // Read VO monolithic
  pbTransform pb_transform = vo_transform_logger_->ReadFromFile();
  while (!vo_transform_logger_->EndOfFile()) {
    transform::Transform curr_transform = aru::core::utilities::transform ::
        TransformProtocolBufferAdaptor::ReadFromProtocolBuffer(pb_transform);
    aru::core::utilities::transform::TransformSPtr curr_transform_sptr =
        boost::make_shared<aru::core::utilities::transform::Transform>(
            curr_transform);
    ba_->AddPose(curr_transform_sptr);
    pb_transform = vo_transform_logger_->ReadFromFile();
  }

  // Read first image
  pbImage image_left_prev = image_left_logger_->ReadFromFile();
  image::Image prev_image_left =
      image::ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(
          image_left_prev);
  pbImage image_right_prev = image_right_logger_->ReadFromFile();
  image::Image prev_image_right =
      image::ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(
          image_right_prev);
  pbImage image_left_curr = image_left_logger_->ReadFromFile();
  pbImage image_right_curr = image_right_logger_->ReadFromFile();

  image::Image curr_image_left =
      image::ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(
          image_left_curr);
  image::Image curr_image_right =
      image::ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(
          image_right_curr);

  ba_->InitialFrame(image::StereoImage(prev_image_left, prev_image_right),
                    image::StereoImage(curr_image_left, curr_image_right));

  image_left_curr = image_left_logger_->ReadFromFile();
  image_right_curr = image_right_logger_->ReadFromFile();
  int num = 0;

  while (!image_left_logger_->EndOfFile() &&
         !image_right_logger_->EndOfFile()) {
   // for (int i = 0; i < 20; ++i) {
    // Perform the estimation
    auto estimation_start = std::chrono::high_resolution_clock::now();
    curr_image_left =
        image::ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(
            image_left_curr);
    curr_image_right =
        image::ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(
            image_right_curr);
    ba_->AddFrame(image::StereoImage(curr_image_left, curr_image_right));

    auto estimation_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = estimation_end - estimation_start;
    VLOG(2) << "Estimation takes " << elapsed.count() << " seconds";
    LOG(INFO) << "BA estimation runs at " << 1 / elapsed.count() << " Hz";

    image_left_curr = image_left_logger_->ReadFromFile();
    image_right_curr = image_right_logger_->ReadFromFile();
    LOG(INFO) << "Num is " << num;
    num++;
  }
  LOG(INFO) << "Solving BA Problem";
  BASolver solver;
  std::vector<Eigen::Affine3f> ba_poses = solver.SolveProblem(*ba_);
  LOG(INFO) << "Output to file";
  // TODO: correct timestamps and make relative
  for (auto pose : ba_poses) {
    transform::TransformSPtr init_transform =
        boost::make_shared<utilities ::transform::Transform>(0, 1, pose);
    pbTransform pb_init_transform = utilities::transform ::
        TransformProtocolBufferAdaptor::ReadToProtocolBuffer(*init_transform);
    ba_transform_logger_->WriteToFile(pb_init_transform);
  }
}

} // namespace bundle_adjustment
} // namespace core
} // namespace aru