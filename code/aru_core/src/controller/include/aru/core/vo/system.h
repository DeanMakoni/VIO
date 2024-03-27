//
// Created by paulamayo on 2022/04/14.
//

#ifndef ARU_CORE_VO_SYSTEM_H
#define ARU_CORE_VO_SYSTEM_H

#include <aru/core/utilities/image/feature_tracker.h>
#include <aru/core/utilities/logging/log.h>
#include <aru/core/utilities/transforms/transforms.h>
#include <aru/core/utilities/navigation/experience.h>
#include <aru/core/vo/vo.h>
#include <pbLaser.pb.h>
#include <pbStereoImage.pb.h>
#include <pbTransform.pb.h>
#include <pbExperience.pb.h>

#include "aru/core/utilities/transforms/transform_map.h"
#include <Eigen/Dense>
#include <glog/logging.h>
#include <iostream>

namespace aru {
namespace core {
namespace vo {

class System {

public:
  System(std::string vo_config_file, std::string vocab_file, std::string image_left_monolithic,
         std::string image_right_monolithic, std::string transform_monolithic);

  void Run();

  utilities::transform::TransformSPtrVectorSptr PoseChain() {
    return pose_chain_;
  }

  ~System() = default;

private:
  std::string vo_config_file_;
  std::string image_left_monolithic_;

  boost::shared_ptr<VO> vo_;

  boost::shared_ptr<utilities::image::VisoFeatureTracker> viso_extractor_;
  utilities::transform::TransformSPtrVectorSptr pose_chain_;
  boost::shared_ptr<
      utilities::logging::ProtocolLogger<datatype::image::pbImage>>
      image_left_logger_;
  boost::shared_ptr<
      utilities::logging::ProtocolLogger<datatype::image::pbImage>>
      image_right_logger_;
  boost::shared_ptr<
      utilities::logging::ProtocolLogger<datatype::transform::pbTransform>>
      transform_logger_;
};

class LaserSystem {

public:
  LaserSystem(std::string laser_monolithic, std::string vo_monolithic,
              std::string interpolated_monolithic);

  void Run();

  utilities::transform::TransformSPtrVectorSptr PoseChain() {
    return pose_chain_;
  }

  ~LaserSystem() = default;

private:
  boost::shared_ptr<utilities::transform::TransformMap> transform_map_;

  utilities::transform::TransformSPtrVectorSptr pose_chain_;
  boost::shared_ptr<
      utilities::logging::ProtocolLogger<datatype::laser::pbLaser>>
      laser_logger_;
  boost::shared_ptr<
      utilities::logging::ProtocolLogger<datatype::transform::pbTransform>>
      vo_logger_;
  boost::shared_ptr<
      utilities::logging::ProtocolLogger<datatype::transform::pbTransform>>
      transform_logger_;
};

class KeyframeSystem {

public:
  KeyframeSystem(std::string vo_transform_monolithic,
                 std::string image_left_monolithic,
                 std::string image_right_monolithic,
                 std::string keyframe_left_monolithic,
                 std::string keyframe_right_monolithic);

  void Run();

  utilities::transform::TransformSPtrVectorSptr PoseChain() {
    return pose_chain_;
  }

  ~KeyframeSystem() = default;

private:
  boost::shared_ptr<utilities::transform::TransformMap> transform_map_;
  utilities::transform::TransformSPtrVectorSptr pose_chain_;
  boost::shared_ptr<
      utilities::logging::ProtocolLogger<datatype::image::pbImage>>
      image_left_logger_;
  boost::shared_ptr<
      utilities::logging::ProtocolLogger<datatype::image::pbImage>>
      image_right_logger_;
  boost::shared_ptr<
      utilities::logging::ProtocolLogger<datatype::image::pbImage>>
      keyframe_left_logger_;
  boost::shared_ptr<
      utilities::logging::ProtocolLogger<datatype::image::pbImage>>
      keyframe_right_logger_;
  boost::shared_ptr<
      utilities::logging::ProtocolLogger<datatype::transform::pbTransform>>
      vo_logger_;

  float min_distance_;
  float min_rotation_;

  long prev_timestamp;
};

class KeyframeExpSystem {

public:
    KeyframeExpSystem(std::string vo_transform_monolithic,
                   std::string experience_monolithic,
                   std::string keyframe_experience_monolithic);

    void Run();

    utilities::transform::TransformSPtrVectorSptr PoseChain() {
      return pose_chain_;
    }

    ~KeyframeExpSystem() = default;

private:
    boost::shared_ptr<utilities::transform::TransformMap> transform_map_;
    utilities::transform::TransformSPtrVectorSptr pose_chain_;
    boost::shared_ptr<
            utilities::logging::ProtocolLogger<datatype::navigation::pbExperience>>
            experience_logger_;
    boost::shared_ptr<
            utilities::logging::ProtocolLogger<datatype::navigation::pbExperience>>
            keyframe_experience_logger_;
    boost::shared_ptr<
            utilities::logging::ProtocolLogger<datatype::transform::pbTransform>>
            vo_logger_;

    float min_distance_;
    float min_rotation_;

    long prev_timestamp;
};
} // namespace vo
} // namespace core
} // namespace aru

#endif // ARU_CORE_VO_SYSTEM_H
