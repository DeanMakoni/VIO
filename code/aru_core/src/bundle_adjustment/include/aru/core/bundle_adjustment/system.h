//
// Created by paulamayo on 2022/06/17.
//

#ifndef ARU_CORE_BA_SYSTEM_H
#define ARU_CORE_BA_SYSTEM_H

#include <aru/core/bundle_adjustment/bundle_adjustment.h>
#include <pbStereoImage.pb.h>
#include <pbTransform.pb.h>

#include <Eigen/Dense>
#include <aru/core/utilities/logging/log.h>
#include <glog/logging.h>
#include <iostream>

namespace aru {
namespace core {
namespace bundle_adjustment {

class System {

public:
  System(std::string ba_config_file, std::string image_left_monolithic,
         std::string image_right_monolithic,
         std::string vo_transform_monolithic, std::string transform_monolithic);

  void Run();

  utilities::transform::TransformSPtrVectorSptr PoseChain() {
    return pose_chain_;
  }

  ~System() = default;

private:
  std::string ba_config_file_;

  boost::shared_ptr<BundleAdjustment> ba_;

  utilities::transform::TransformSPtrVectorSptr pose_chain_;
  boost::shared_ptr<
      utilities::logging::ProtocolLogger<datatype::image::pbImage>>
      image_left_logger_;
  boost::shared_ptr<
      utilities::logging::ProtocolLogger<datatype::image::pbImage>>
      image_right_logger_;
  boost::shared_ptr<
      utilities::logging::ProtocolLogger<datatype::transform::pbTransform>>
      vo_transform_logger_;
  boost::shared_ptr<
      utilities::logging::ProtocolLogger<datatype::transform::pbTransform>>
      ba_transform_logger_;
};
} // namespace bundle_adjustment
} // namespace core
} // namespace aru

#endif // ARU_CORE_BA_SYSTEM_H
