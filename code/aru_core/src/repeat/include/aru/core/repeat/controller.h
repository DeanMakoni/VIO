#ifndef ARU_CORE_CONTROLLER_H_
#define ARU_CORE_CONTROLLER_H_

#include <aru/core/utilities/image/feature_matcher.h>
#include <aru/core/utilities/image/image.h>
#include <aru/core/utilities/transforms/transforms.h>
#include <aru/core/vo/vo.h>

#include "aru/core/utilities/transforms/transform_map.h"
#include <Eigen/Dense>
#include <glog/logging.h>
#include <iostream>

namespace aru {
namespace core {
namespace repeat {

typedef std::pair<float,float> Command;
class Controller {

public:
  explicit Controller(std::string controller_config_file);

  Command TransformToCommand(utilities::transform::TransformSPtr
                                teach_transform);

  void AddTeachTransform(utilities::transform::TransformSPtr teach_transform);

  void Navigate();

  ~Controller() = default;

private:
  std::shared_ptr<utilities::transform::TransformMap> transform_map_;

};

} // namespace repeat
} // namespace core
} // namespace aru

#endif // ARU_CORE_CONTROLLER_H_
