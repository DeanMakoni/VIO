//
// Created by paulamayo on 2022/07/13.
//

#ifndef ARU_CORE_LOCALISATION_SYSTEM_H
#define ARU_CORE_LOCALISATION_SYSTEM_H

#include <aru/core/localisation/localisation.h>
#include <aru/core/utilities/logging/log.h>
#include <pbStereoImage.pb.h>
#include <pbTransform.pb.h>

#include <Eigen/Dense>
#include <glog/logging.h>
#include <iostream>

namespace aru {
namespace core {
namespace localisation {

class System {

public:
  System(std::string image_query_monolithic,
         std::string image_curr_monolithic);

  void Run();


  ~System() = default;

private:
  cv::Mat GenerateVocabData();

  cv::Mat GenerateBoWData();

  boost::shared_ptr<Localisation> localiser_;

  boost::shared_ptr<
      utilities::logging::ProtocolLogger<datatype::image::pbImage>>
      image_query_logger_;
  boost::shared_ptr<
      utilities::logging::ProtocolLogger<datatype::image::pbImage>>
      image_current_logger_;
};
} // namespace localisation
} // namespace core
} // namespace aru

#endif // ARU_CORE_LOCALISATION_SYSTEM_H
