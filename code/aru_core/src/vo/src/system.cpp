//
// Created by paulamayo on 2022/04/14.
//

#include "aru/core/vo/system.h"
//#include "aru/core/utilities//laser/laserprotocolbufferadaptor.h"
//#include <aru/core/utilities/image/imageprotocolbufferadaptor.h>
//#include <aru/core/utilities/transforms/transformprotocolbufferadaptor.h>
//#include <aru/core/utilities/navigation/experienceprotocolbufferadaptor.h>
#include <aru/core/utilities/viewer/viewer.h>
#include <boost/make_shared.hpp>
#include <opencv2/core/eigen.hpp>
using namespace datatype::image;
using namespace datatype::transform;
using namespace datatype::laser;
using namespace datatype::navigation;
using namespace aru::core::utilities;

namespace aru {
namespace core {
namespace vo {
//------------------------------------------------------------------------------
System::System(std::string vo_config_file, std::string vocab_file, std::string image_left_monolithic,
               std::string image_right_monolithic,
               std::string transform_monolithic) {

  vo_ = boost::make_shared<VO>(vo_config_file, vocab_file);
  image_left_logger_ = boost::make_shared<logging::ProtocolLogger<pbImage>>(
      image_left_monolithic, false);
  image_right_logger_ = boost::make_shared<logging::ProtocolLogger<pbImage>>(
      image_right_monolithic, false);
  transform_logger_ = boost::make_shared<logging::ProtocolLogger<pbTransform>>(
      transform_monolithic, true);

  pose_chain_ = boost::make_shared<transform::TransformSPtrVector>();

  // Initialise VISO

  cv::FileStorage fs;
  fs.open(vo_config_file, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    LOG(ERROR) << "Could not open vo model file: ";
  }

  // Extractor Parameters
  aru::core::utilities::image::ExtractorParams extractor_params_{};
  aru::core::utilities::image::MatcherParams matcher_params_{};
  extractor_params_.num_features = fs["FeatureExtractor"]["num_features"];
  extractor_params_.num_levels = fs["FeatureExtractor"]["num_levels"];
  extractor_params_.scale_factor = fs["FeatureExtractor"]["scale_factor"];
  extractor_params_.initial_fast_threshold =
      fs["FeatureExtractor"]["intitial_fast_threshold"];
  extractor_params_.minimum_fast_threshold = fs["FeatureExtractor"
                                                ""]["minimum_fast_threshold"];
  extractor_params_.patch_size = fs["FeatureExtractor"]["patch_size"];
  extractor_params_.half_patch_size = fs["FeatureExtractor"]["half_patch_size"];
  extractor_params_.edge_threshold = fs["FeatureExtractor"]["edge_threshold"];

  // Matcher parameters
  matcher_params_.match_threshold_high =
      fs["FeatureMatcher"]["match_threshold_high"];
  matcher_params_.match_threshold_low =
      fs["FeatureMatcher"]["match_threshold_low"];
  matcher_params_.stereo_baseline = fs["FeatureMatcher"]["stereo_baseline"];
  matcher_params_.focal_length = fs["FeatureMatcher"]["focal_length"];

  viso_extractor_ = boost::make_shared<utilities::image::VisoFeatureTracker>(
      matcher_params_, extractor_params_);
}
//------------------------------------------------------------------------------
void System::Run() {
  // Initialise starting position
  Eigen::Affine3f curr_position;
  curr_position.linear() = Eigen::MatrixXf::Identity(3, 3);
  curr_position.translation() = Eigen::VectorXf::Zero(3);

  // Read previous image
  image::StereoImage prev_image;
  pbImage image_left_prev = image_left_logger_->ReadFromFile();
  prev_image.first = image::ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(
      image_left_prev);
  // image_right_logger_->ReadFromFile();
  pbImage image_right_prev = image_right_logger_->ReadFromFile();
  prev_image.second = image::ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(
      image_right_prev);

  // check whether the timestamp for this is after the left image
  if (prev_image.second.GetTimeStamp() < prev_image.first.GetTimeStamp()) {
    image_right_prev = image_right_logger_->ReadFromFile();
    prev_image.second =
        image::ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(
            image_right_prev);
  }

  transform::TransformSPtr init_transform =
      boost::make_shared<utilities ::transform::Transform>(
          prev_image.first.GetTimeStamp(), prev_image.first.GetTimeStamp(),
          curr_position);
  pbTransform pb_init_transform = utilities::transform ::
      TransformProtocolBufferAdaptor::ReadToProtocolBuffer(*init_transform);
  transform_logger_->WriteToFile(pb_init_transform);

  pose_chain_->push_back(init_transform);

  pbImage image_left_curr = image_left_logger_->ReadFromFile();
  pbImage image_right_curr = image_right_logger_->ReadFromFile();

  int num = 0;

  while (!image_left_logger_->EndOfFile() &&
         !image_right_logger_->EndOfFile()) {

    if (num == 178) {
      LOG(INFO) << "Pause";
    }

    // Perform the estimation
    auto estimation_start = std::chrono::high_resolution_clock::now();
    image::StereoImage curr_image;
    curr_image.first =
        image::ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(
            image_left_curr);

    curr_image.second =
        image::ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(
            image_right_curr);

    // pose is T_prev_curr. Source is curr_image dest is prev_image
    utilities::transform::Transform pose =
        vo_->EstimateMotion(prev_image, curr_image);

    auto estimation_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = estimation_end - estimation_start;
    VLOG(2) << "Estimation takes " << elapsed.count() << " seconds";
    LOG(INFO) << "VO estimation runs at " << 1 / elapsed.count() << " Hz";

    pbTransform pb_transform = utilities::transform ::
        TransformProtocolBufferAdaptor::ReadToProtocolBuffer(pose);
    transform_logger_->WriteToFile(pb_transform);

    curr_position = curr_position * pose.GetTransform();

    transform::TransformSPtr curr_transform =
        boost::make_shared<utilities ::transform::Transform>(0, 0,
                                                             curr_position);
    prev_image = curr_image;
    pose_chain_->push_back(curr_transform);

    image_left_curr = image_left_logger_->ReadFromFile();
    image_right_curr = image_right_logger_->ReadFromFile();
    LOG(INFO) << "Num is " << num;
    num++;
  }

  for (int i = 0; i < 50; ++i) {
    transform_logger_->WriteToFile(pb_init_transform);
  }
}

//------------------------------------------------------------------------------
LaserSystem::LaserSystem(std::string laser_monolithic,
                         std::string vo_monolithic,
                         std::string interpolated_monolithic) {

  laser_logger_ = boost::make_shared<logging::ProtocolLogger<pbLaser>>(
      laser_monolithic, false);
  vo_logger_ = boost::make_shared<logging::ProtocolLogger<pbTransform>>(
      vo_monolithic, false);
  transform_logger_ = boost::make_shared<logging::ProtocolLogger<pbTransform>>(
      interpolated_monolithic, true);

  pose_chain_ = boost::make_shared<transform::TransformSPtrVector>();

  transform_map_ = boost::make_shared<utilities::transform::TransformMap>();
}
//------------------------------------------------------------------------------
void LaserSystem::Run() {

  // Read the Vo monolithic into the transform map
  pbTransform pb_transform = vo_logger_->ReadFromFile();
  while (!vo_logger_->EndOfFile()) {
    transform::Transform curr_transform = aru::core::utilities::transform ::
        TransformProtocolBufferAdaptor::ReadFromProtocolBuffer(pb_transform);
    aru::core::utilities::transform::TransformSPtr curr_transform_sptr =
        boost::make_shared<aru::core::utilities::transform::Transform>(
            curr_transform);
    transform_map_->AddTransform(curr_transform_sptr);
    pb_transform = vo_logger_->ReadFromFile();
  }

  // Initialise starting position
  Eigen::Affine3f curr_position;
  curr_position.linear() = Eigen::MatrixXf::Identity(3, 3);
  curr_position.translation() = Eigen::VectorXf::Zero(3);

  // Read previous image
  pbLaser pb_laser = laser_logger_->ReadFromFile();
  laser::Laser curr_laser =
      laser::LaserProtocolBufferAdaptor::ReadFromProtocolBuffer(pb_laser);

  transform::TransformSPtr init_transform =
      boost::make_shared<utilities ::transform::Transform>(
          curr_laser.GetTimeStamp(), curr_laser.GetTimeStamp(), curr_position);
  pbTransform pb_init_transform = utilities::transform ::
      TransformProtocolBufferAdaptor::ReadToProtocolBuffer(*init_transform);
  transform_logger_->WriteToFile(pb_init_transform);
  pose_chain_->push_back(init_transform);

  pb_laser = laser_logger_->ReadFromFile();
  int num = 0;

  while (!laser_logger_->EndOfFile()) {

    // Perform the estimation
    auto estimation_start = std::chrono::high_resolution_clock::now();
    laser::Laser curr_laser =
        laser::LaserProtocolBufferAdaptor::ReadFromProtocolBuffer(pb_laser);

    // pose is T_prev_curr. Source is curr_image dest is prev_image
    utilities::transform::TransformSPtr pose =
        transform_map_->Interpolate(curr_laser.GetTimeStamp());
    if (pose) {

      auto estimation_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = estimation_end - estimation_start;
      VLOG(2) << "Estimation takes " << elapsed.count() << " seconds";
      LOG(INFO) << "VO estimation runs at " << 1 / elapsed.count() << " Hz";

      pbTransform pb_transform = utilities::transform ::
          TransformProtocolBufferAdaptor::ReadToProtocolBuffer(*pose);
      transform_logger_->WriteToFile(pb_transform);

      transform::TransformSPtr curr_transform =
          boost::make_shared<utilities ::transform::Transform>(
              0, 0, pose->GetTransform());
      pose_chain_->push_back(curr_transform);
    }

    pb_laser = laser_logger_->ReadFromFile();
    LOG(INFO) << "Num is " << num;
    num++;
  }
}
//------------------------------------------------------------------------------
    KeyframeSystem::KeyframeSystem(std::string vo_transform_monolithic,
                                   std::string image_left_monolithic,
                                   std::string image_right_monolithic,
                                   std::string keyframe_left_monolithic,
                                   std::string keyframe_right_monolithic) {

      image_left_logger_ = boost::make_shared<logging::ProtocolLogger<pbImage>>(
              image_left_monolithic, false);
      image_right_logger_ = boost::make_shared<logging::ProtocolLogger<pbImage>>(
              image_right_monolithic, false);
      keyframe_left_logger_ = boost::make_shared<logging::ProtocolLogger<pbImage>>(
              keyframe_left_monolithic, true);
      keyframe_right_logger_ = boost::make_shared<logging::ProtocolLogger<pbImage>>(
              keyframe_right_monolithic, true);
      vo_logger_ = boost::make_shared<logging::ProtocolLogger<pbTransform>>(
              vo_transform_monolithic, false);

      pose_chain_ = boost::make_shared<transform::TransformSPtrVector>();

      min_distance_ = 5.0;
      min_rotation_ = 0.2536;

      transform_map_ = boost::make_shared<utilities::transform::TransformMap>();
    }
//------------------------------------------------------------------------------
  void KeyframeSystem::Run() {

    // Read the Vo monolithic into the transform map
    pbTransform pb_transform = vo_logger_->ReadFromFile();
    while (!vo_logger_->EndOfFile()) {
      transform::Transform curr_transform = aru::core::utilities::transform ::
      TransformProtocolBufferAdaptor::ReadFromProtocolBuffer(pb_transform);
      aru::core::utilities::transform::TransformSPtr curr_transform_sptr =
              boost::make_shared<aru::core::utilities::transform::Transform>(
                      curr_transform);
      transform_map_->AddTransform(curr_transform_sptr);
      pb_transform = vo_logger_->ReadFromFile();
    }

    // Read previous image
    // Read previous image
    image::StereoImage prev_image;
    pbImage image_left_prev = image_left_logger_->ReadFromFile();
    prev_image.first = image::ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(
            image_left_prev);
    // image_right_logger_->ReadFromFile();
    pbImage image_right_prev = image_right_logger_->ReadFromFile();
    prev_image.second = image::ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(
            image_right_prev);

    if (prev_image.second.GetTimeStamp() < prev_image.first.GetTimeStamp()) {
      image_right_prev = image_right_logger_->ReadFromFile();
      prev_image.second =
              image::ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(
                      image_right_prev);
    }

    // Write the first images to monolithic
    keyframe_left_logger_->WriteToFile(image_left_prev);
    keyframe_right_logger_->WriteToFile(image_right_prev);
    // Initialise starting position
    Eigen::Affine3f curr_position;
    curr_position.linear() = Eigen::MatrixXf::Identity(3, 3);
    curr_position.translation() = Eigen::VectorXf::Zero(3);
    transform::TransformSPtr init_transform =
            boost::make_shared<utilities ::transform::Transform>(
                    prev_image.first.GetTimeStamp(), prev_image.first.GetTimeStamp(),
                    curr_position);
    pbTransform pb_init_transform = utilities::transform ::
    TransformProtocolBufferAdaptor::ReadToProtocolBuffer(*init_transform);
    pose_chain_->push_back(init_transform);

    prev_timestamp = prev_image.first.GetTimeStamp();

    pbImage image_left_curr = image_left_logger_->ReadFromFile();
    pbImage image_right_curr = image_right_logger_->ReadFromFile();

    int num = 0;

    while (!image_left_logger_->EndOfFile() &&
           !image_right_logger_->EndOfFile()) {

      // Perform the estimation
      image::StereoImage curr_image;
      curr_image.first =
              image::ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(
                      image_left_curr);

      curr_image.second =
              image::ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(
                      image_right_curr);

      // pose is T_prev_curr. Source is curr_image dest is prev_image
      utilities::transform::TransformSPtr pose = transform_map_->Interpolate(
              prev_timestamp, curr_image.first.GetTimeStamp());
      if (pose) {
        float dist = pose->GetTransform().translation().norm();
        cv::Mat R_matrix = cv::Mat::zeros(3, 3, CV_64FC1); // rotation matrix
        cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
        // Read Rotation matrix and convert to vector
        cv::eigen2cv(pose->GetRotation(), R_matrix);
        cv::Rodrigues(R_matrix, rvec);
        Eigen::Vector3f rpy;
        cv::cv2eigen(rvec, rpy);
        float rotation = rpy.norm();
        LOG(INFO) << "Rotation is " << rotation;
        LOG(INFO) << " Dist is " << dist;
        if (dist > min_distance_ || rotation > min_rotation_) {
          keyframe_left_logger_->WriteToFile(image_left_curr);
          keyframe_right_logger_->WriteToFile(image_right_curr);
          prev_timestamp = curr_image.first.GetTimeStamp();
          utilities::transform::TransformSPtr key_frame_position =
                  transform_map_->Interpolate(prev_timestamp);
          pose_chain_->push_back(key_frame_position);

          cv::namedWindow("Keyframe", cv::WINDOW_NORMAL);
          cv::resizeWindow("Keyframe", curr_image.first.GetImage().cols / 2, curr_image.first.GetImage().rows / 2);
          cv::imshow("Keyframe", curr_image.first.GetImage());
          cv::waitKey(5);
        }
      }

      image_left_curr = image_left_logger_->ReadFromFile();
      image_right_curr = image_right_logger_->ReadFromFile();

      LOG(INFO) << "Num is " << num;
      num++;
    }
  }
//------------------------------------------------------------------------------
    KeyframeExpSystem::KeyframeExpSystem(std::string experience_monolithic,
                                         std::string vo_transform_monolithic,
                                         std::string keyframe_experience_monolithic) {
      experience_logger_ = boost::make_shared<logging::ProtocolLogger<datatype::navigation::pbExperience>>(
              experience_monolithic, false);
      keyframe_experience_logger_ = boost::make_shared<logging::ProtocolLogger<datatype::navigation::pbExperience>>(
              keyframe_experience_monolithic, true);
      vo_logger_ = boost::make_shared<logging::ProtocolLogger<pbTransform>>(
              vo_transform_monolithic, false);

      pose_chain_ = boost::make_shared<transform::TransformSPtrVector>();

      min_distance_ = 2.0;
      min_rotation_ = 0.2536;

      transform_map_ = boost::make_shared<utilities::transform::TransformMap>();
    }
  void KeyframeExpSystem::Run() {

    // Read the Vo monolithic into the transform map
    pbTransform pb_transform = vo_logger_->ReadFromFile();
    while (!vo_logger_->EndOfFile()) {
      transform::Transform curr_transform = aru::core::utilities::transform ::
          TransformProtocolBufferAdaptor::ReadFromProtocolBuffer(pb_transform);
      aru::core::utilities::transform::TransformSPtr curr_transform_sptr =
          boost::make_shared<aru::core::utilities::transform::Transform>(
              curr_transform);
      transform_map_->AddTransform(curr_transform_sptr);
      pb_transform = vo_logger_->ReadFromFile();
    }

    // Read previous image
    navigation::Experience prev_experience;
    pbExperience experience_prev = experience_logger_->ReadFromFile();
    prev_experience = navigation::ExperienceProtocolBufferAdaptor::ReadFromProtocolBuffer
            (experience_prev);

    // Write the first images to monolithic
    keyframe_experience_logger_->WriteToFile(experience_prev);
    // Initialise starting position
    Eigen::Affine3f curr_position;
    curr_position.linear() = Eigen::MatrixXf::Identity(3, 3);
    curr_position.translation() = Eigen::VectorXf::Zero(3);
    transform::TransformSPtr init_transform =
            boost::make_shared<utilities ::transform::Transform>(
                    prev_experience.GetTimeStamp(), prev_experience.GetTimeStamp(),
                    curr_position);
    pbTransform pb_init_transform = utilities::transform ::
        TransformProtocolBufferAdaptor::ReadToProtocolBuffer(*init_transform);
    pose_chain_->push_back(init_transform);

    prev_timestamp = prev_experience.GetTimeStamp();
    pbExperience curr_experience = experience_logger_->ReadFromFile();

    int num = 0;
    while (!experience_logger_->EndOfFile()) {

      // Perform the estimation
      navigation::Experience curr_exp;
      curr_exp = navigation::ExperienceProtocolBufferAdaptor::ReadFromProtocolBuffer(curr_experience);

      // pose is T_prev_curr. Source is curr_image dest is prev_image
      utilities::transform::TransformSPtr pose = transform_map_->Interpolate(
              prev_timestamp, curr_exp.GetTimeStamp());
      if (pose) {
        float dist = pose->GetTransform().translation().norm();
        cv::Mat R_matrix = cv::Mat::zeros(3, 3, CV_64FC1); // rotation matrix
        cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
        // Read Rotation matrix and convert to vector
        cv::eigen2cv(pose->GetRotation(), R_matrix);
        cv::Rodrigues(R_matrix, rvec);
        Eigen::Vector3f rpy;
        cv::cv2eigen(rvec, rpy);
        float rotation = rpy.norm();
        LOG(INFO) << "Rotation is " << rotation;
        LOG(INFO) << " Dist is " << dist;
        if (dist > min_distance_ || rotation > min_rotation_) {
          keyframe_experience_logger_->WriteToFile(curr_experience);
          prev_timestamp = curr_exp.GetTimeStamp();
          utilities::transform::TransformSPtr key_frame_position =
              transform_map_->Interpolate(prev_timestamp);
          pose_chain_->push_back(key_frame_position);
          cv::namedWindow("Keyframe", cv::WINDOW_NORMAL);
          cv::resizeWindow("Keyframe", curr_exp.GetImage().cols / 2, curr_exp.GetImage().rows / 2);
          cv::imshow("Keyframe", curr_exp.GetImage());
          cv::waitKey(5);
        }
      }
      curr_experience = experience_logger_->ReadFromFile();

      LOG(INFO) << "Num is " << num;
      num++;
    }
  }

} // namespace vo
} // namespace core
} // namespace aru
