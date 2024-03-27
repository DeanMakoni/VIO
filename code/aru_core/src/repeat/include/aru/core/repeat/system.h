//
// Created by paulamayo on 2022/04/14.
//

#ifndef ARU_CORE_REPEAT_SYSTEM_H
#define ARU_CORE_REPEAT_SYSTEM_H

#include <aru/core/utilities/image/feature_tracker.h>
#include <aru/core/utilities/logging/log.h>
#include <aru/core/utilities/navigation/experience.h>
#include <aru/core/utilities/transforms/transforms.h>
#include <aru/core/repeat/repeat.h>
#include <pbExperience.pb.h>
#include <pbLaser.pb.h>
#include <pbStereoImage.pb.h>
#include <pbTransform.pb.h>

#include "aru/core/utilities/transforms/transform_map.h"
#include <Eigen/Dense>
#include <glog/logging.h>
#include <iostream>

namespace aru {
namespace core {
namespace repeat {

class System {

public:
  System(std::string repeat_config_file, std::string vocab_file,
         std::string image_left_teach_monolithic,
         std::string image_right_teach_monolithic,
         std::string image_left_repeat_monolithic,
         std::string image_right_repeat_monolithic,
         std::string transform_monolithic);

  void Run();

  utilities::transform::TransformSPtrVectorSptr PoseChain() {
    return pose_chain_;
  }

  ~System() = default;

private:
  std::string vo_config_file_;
  std::string image_left_monolithic_;

  std::shared_ptr<Repeat> repeat_;

  utilities::transform::TransformSPtrVectorSptr pose_chain_;
  std::shared_ptr<
      utilities::logging::ProtocolLogger<datatype::image::pbImage>>
      image_left_teach_logger_;
  std::shared_ptr<
      utilities::logging::ProtocolLogger<datatype::image::pbImage>>
      image_right_teach_logger_;
  std::shared_ptr<
      utilities::logging::ProtocolLogger<datatype::image::pbImage>>
      image_left_repeat_logger_;
  std::shared_ptr<
      utilities::logging::ProtocolLogger<datatype::image::pbImage>>
      image_right_repeat_logger_;

  std::shared_ptr<
      utilities::logging::ProtocolLogger<datatype::transform::pbTransform>>
      transform_logger_;
};


} // namespace system
} // namespace core
} // namespace aru

#endif // ARU_CORE_REPEAT_SYSTEM_H
