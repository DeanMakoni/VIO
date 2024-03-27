#ifndef ARU_UTILITIES_EXPERIENCE_PROTOCOL_BUFFER_H_
#define ARU_UTILITIES_EXPERIENCE_PROTOCOL_BUFFER_H_

#include "experience.h"
#include "pbExperience.pb.h"
#include "pbStereoImage.pb.h"
#include <Eigen/Dense>
#include <glog/logging.h>
#include <iostream>
#include <opencv2/core/mat.hpp>

namespace aru {
namespace core {
namespace utilities {
namespace navigation {

class ExperienceProtocolBufferAdaptor {
public:
    ExperienceProtocolBufferAdaptor();

  virtual ~ExperienceProtocolBufferAdaptor() = default;

  static Experience
  ReadFromProtocolBuffer(const datatype::navigation::pbExperience& pb_experience);

  static datatype::navigation::pbExperience
  ReadToProtocolBuffer(Experience experience);

};
} // namespace image
} // namespace utilities
} // namespace core
} // namespace aru

#endif // ARU_UTILITIES_EXPERIENCE_PROTOCOL_BUFFER_H_
