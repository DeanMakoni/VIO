//
// Created by paulamayo on 2022/06/14.
//

#ifndef ARU_CORE_MESH_SYSTEM_H
#define ARU_CORE_MESH_SYSTEM_H


#include <aru/core/utilities/image/feature_tracker.h>
#include <aru/core/utilities/logging/log.h>
#include <aru/core/utilities/transforms/transforms.h>
#include <aru/core/mesh/mesh.h>
#include <pbStereoImage.pb.h>
#include <pbTransform.pb.h>

#include <Eigen/Dense>
#include <glog/logging.h>
#include <iostream>

namespace aru {
namespace core {
namespace mesh {

class System {

public:
  System(std::string mesh_config_file, std::string image_left_monolithic,
         std::string image_right_monolithic, std::string mesh_depth_monolithic);

  void Run();


  ~System() = default;

private:

  boost::shared_ptr<Mesh> mesh_;

  boost::shared_ptr<
      utilities::logging::ProtocolLogger<datatype::image::pbImage>>
      image_left_logger_;
  boost::shared_ptr<
      utilities::logging::ProtocolLogger<datatype::image::pbImage>>
      image_right_logger_;
  boost::shared_ptr<
      utilities::logging::ProtocolLogger<datatype::image::pbImage>>
      mesh_depth_logger_;
};
} // namespace vo
} // namespace core
} // namespace aru
#endif // ARU_CORE_MESH_SYSTEM_H
