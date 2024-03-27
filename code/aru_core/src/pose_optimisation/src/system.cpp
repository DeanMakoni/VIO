//
// Created by paulamayo on 2022/06/17.
//

#include "aru/core/pose_optimisation/system.h"
#include <aru/core/utilities/image/imageprotocolbufferadaptor.h>
#include <aru/core/utilities/transforms/transformprotocolbufferadaptor.h>
#include <aru/core/utilities/viewer/viewer.h>
#include <boost/make_shared.hpp>
using namespace datatype::image;
using namespace datatype::transform;
using namespace aru::core::utilities;

namespace aru {
namespace core {
namespace pose_optimisation {
//------------------------------------------------------------------------------
System::System(std::string ba_config_file, std::string image_left_monolithic,
               std::string image_right_monolithic,
               std::string vo_transform_monolithic,
               std::string transform_monolithic) {

  pose_optimisation_ = boost::make_shared<PoseOptimisation>(ba_config_file);
  image_left_logger_ = boost::make_shared<logging::ProtocolLogger<pbImage>>(
      image_left_monolithic, false);
  image_right_logger_ = boost::make_shared<logging::ProtocolLogger<pbImage>>(
      image_right_monolithic, false);
  vo_transform_logger_ =
      boost::make_shared<logging::ProtocolLogger<pbTransform>>(
          vo_transform_monolithic, false);
  opt_transform_logger_ =
      boost::make_shared<logging::ProtocolLogger<pbTransform>>(
          transform_monolithic, true);

  pose_chain_ = boost::make_shared<transform::TransformSPtrVector>();
}
//------------------------------------------------------------------------------
void System::Run() {
  // Read VO monolithic
  pbTransform pb_transform = vo_transform_logger_->ReadFromFile();
  int t_num = 0;
  while (!vo_transform_logger_->EndOfFile()) {
    transform::Transform curr_transform = aru::core::utilities::transform ::
        TransformProtocolBufferAdaptor::ReadFromProtocolBuffer(pb_transform);
    aru::core::utilities::transform::TransformSPtr curr_transform_sptr =
        boost::make_shared<aru::core::utilities::transform::Transform>(
            curr_transform);
    pose_optimisation_->AddPose(curr_transform_sptr);
    LOG(INFO) << "Number is " << t_num;
    pb_transform = vo_transform_logger_->ReadFromFile();
    t_num = t_num + 1;
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

  pose_optimisation_->InitialFrame(
      image::StereoImage(prev_image_left, prev_image_right));

  image_left_curr = image_left_logger_->ReadFromFile();
  image_right_curr = image_right_logger_->ReadFromFile();
  int num = 0;

  while (!image_left_logger_->EndOfFile() &&
         !image_right_logger_->EndOfFile()) {
    // Perform the estimation
    auto estimation_start = std::chrono::high_resolution_clock::now();
    curr_image_left = image::ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(
        image_left_curr);
    curr_image_right =
        image::ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(
            image_right_curr);
    pose_optimisation_->AddFrame(
        image::StereoImage(curr_image_left, curr_image_right));

    auto estimation_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = estimation_end - estimation_start;
    VLOG(2) << "Estimation takes " << elapsed.count() << " seconds";
    LOG(INFO) << "BA estimation runs at " << 1 / elapsed.count() << " Hz";

    image_left_curr = image_left_logger_->ReadFromFile();
    image_right_curr = image_right_logger_->ReadFromFile();
    LOG(INFO) << "Num is " << num;
    num++;
  }
  pose_optimisation_->OptimisePose();

  auto opt_poses = pose_optimisation_->GetOptPoses();
  auto timestamps = pose_optimisation_->GetTimestamps();
  int key_no = 0;
  gtsam::Pose3 p = opt_poses.at<gtsam::Pose3>(key_no);
  Eigen::Affine3d opt_pose;
  opt_pose.matrix() = p.matrix();
  transform::TransformSPtr init_transform =
      boost::make_shared<utilities ::transform::Transform>(
          timestamps[key_no], timestamps[key_no], opt_pose.cast<float>());
  pbTransform pb_init_transform = utilities::transform ::
      TransformProtocolBufferAdaptor::ReadToProtocolBuffer(*init_transform);
  opt_transform_logger_->WriteToFile(pb_init_transform);

  for (int next_key = 0; next_key < opt_poses.keys().size() - 1; ++next_key) {
    gtsam::Pose3 p_k = opt_poses.at<gtsam::Pose3>(next_key);
    gtsam::Pose3 p_k_1 = opt_poses.at<gtsam::Pose3>(next_key + 1);
    Eigen::Affine3d opt_pose_k;
    opt_pose_k.matrix() = p_k.matrix();
    Eigen::Affine3d opt_pose_k_1;
    opt_pose_k_1.matrix() = p_k_1.matrix();
    opt_pose = opt_pose_k.inverse() * opt_pose_k_1;
    LOG(INFO) << " Opt Pose " << next_key << " is \n" << opt_pose.matrix();

    transform::TransformSPtr opt_transform =
        boost::make_shared<utilities ::transform::Transform>(
            timestamps[next_key], timestamps[next_key + 1],
            opt_pose.cast<float>());
    pbTransform pb_opt_transform = utilities::transform ::
        TransformProtocolBufferAdaptor::ReadToProtocolBuffer(*opt_transform);
    opt_transform_logger_->WriteToFile(pb_opt_transform);
  }
}

} // namespace pose_optimisation
} // namespace core
} // namespace aru