//
// Created by paulamayo on 2022/04/14.
//

#include "aru/core/repeat/system.h"
#include "aru/core/utilities//laser/laserprotocolbufferadaptor.h"
#include <aru/core/utilities/image/imageprotocolbufferadaptor.h>
#include <aru/core/utilities/navigation/experienceprotocolbufferadaptor.h>
#include <aru/core/utilities/transforms/transformprotocolbufferadaptor.h>
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
namespace repeat {
//------------------------------------------------------------------------------
System::System(std::string repeat_config_file, std::string vocab_file,
               std::string image_left_teach_monolithic,
               std::string image_right_teach_monolithic,
               std::string image_left_repeat_monolithic,
               std::string image_right_repeat_monolithic,
               std::string transform_monolithic) {

  repeat_ = std::make_shared<Repeat>(repeat_config_file, vocab_file);
  image_left_teach_logger_ =
      std::make_shared<logging::ProtocolLogger<pbImage>>(
          image_left_teach_monolithic, false);
  image_right_teach_logger_ =
      std::make_shared<logging::ProtocolLogger<pbImage>>(
          image_right_teach_monolithic, false);
  image_left_repeat_logger_ =
      std::make_shared<logging::ProtocolLogger<pbImage>>(
          image_left_repeat_monolithic, false);
  image_right_repeat_logger_ =
      std::make_shared<logging::ProtocolLogger<pbImage>>(
          image_right_repeat_monolithic, false);
  transform_logger_ = std::make_shared<logging::ProtocolLogger<pbTransform>>(
      transform_monolithic, false);

 // Read the config file
  cv::FileStorage fs;
  fs.open(repeat_config_file, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    LOG(ERROR) << "Could not open repeat model file: ";
  }

  // Initialise Viewer
  // viewer_ = boost::make_shared<utilities::viewer::Viewer>();
}

//------------------------------------------------------------------------------
void System::Run() {
  LOG(INFO)<<"Running Teach Transforms";
  // Read the Vo monolithic into the Repeat Map
  pbTransform pb_transform = transform_logger_->ReadFromFile();
  while (!transform_logger_->EndOfFile()) {
    transform::Transform curr_transform = aru::core::utilities::transform ::
        TransformProtocolBufferAdaptor::ReadFromProtocolBuffer(pb_transform);
    aru::core::utilities::transform::TransformSPtr curr_transform_sptr =
        boost::make_shared<aru::core::utilities::transform::Transform>(
            curr_transform);
    repeat_->AddTeachTransform(curr_transform_sptr);
    pb_transform = transform_logger_->ReadFromFile();
  }

  LOG(INFO)<<"Running Teach Keyframes";
  // Read previous image
  image::StereoImage init_image;
  pbImage image_left_prev = image_left_teach_logger_->ReadFromFile();
  init_image.first = image::ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(
      image_left_prev);
  // image_right_logger_->ReadFromFile();
  pbImage image_right_prev = image_right_teach_logger_->ReadFromFile();
  init_image.second = image::ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(
      image_right_prev);

  // check whether the timestamp for this is after the left image
  if (init_image.second.GetTimeStamp() < init_image.first.GetTimeStamp()) {
    image_right_prev = image_right_teach_logger_->ReadFromFile();
    init_image.second =
        image::ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(
            image_right_prev);
  }
  repeat_->InitialiseMap(init_image);
  LOG(INFO)<<"Initialised Map";


  pbImage image_left_curr = image_left_teach_logger_->ReadFromFile();
  pbImage image_right_curr = image_right_teach_logger_->ReadFromFile();

  int num = 0;

  while (!image_left_teach_logger_->EndOfFile() &&
         !image_right_teach_logger_->EndOfFile()) {

    // Add the teach keyframes to the map
    image::StereoImage curr_image;
    curr_image.first =
        image::ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(
            image_left_curr);
    curr_image.second =
        image::ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(
            image_right_curr);
    repeat_->AddTeachKeyframe(curr_image);
    image_left_curr = image_left_teach_logger_->ReadFromFile();
    image_right_curr = image_right_teach_logger_->ReadFromFile();
    LOG(INFO) << "Num is " << num;
    num++;
  }

  LOG(INFO)<<"Read all the teach keyframes";

  pbImage image_left_repeat = image_left_repeat_logger_->ReadFromFile();
  pbImage image_right_repeat = image_right_repeat_logger_->ReadFromFile();

  while (!image_left_repeat_logger_->EndOfFile() &&
         !image_right_repeat_logger_->EndOfFile()) {
    // Add the repeat keyframes to the map
    image::StereoImage curr_image;
    curr_image.first =
        image::ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(
            image_left_repeat);
    curr_image.second =
        image::ImageProtocolBufferAdaptor::ReadFromProtocolBuffer(
            image_right_repeat);
    repeat_->QueryRepeatframe(curr_image);
    image_left_repeat = image_left_repeat_logger_->ReadFromFile();
    image_right_repeat = image_right_repeat_logger_->ReadFromFile();
    LOG(INFO)<<"Queried the repeat keyframe";
  }

  // TODO: Add the repeat keyframes
}


} // namespace repeat
} // namespace core
} // namespace aru