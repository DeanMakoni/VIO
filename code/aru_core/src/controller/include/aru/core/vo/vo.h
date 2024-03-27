#ifndef ARU_CORE_VO_H_
#define ARU_CORE_VO_H_

#include <aru/core/utilities/image/feature_matcher.h>
#include <aru/core/utilities/image/image.h>
#include <aru/core/utilities/transforms/transforms.h>
#include <aru/core/vo/solver.h>

#include <Eigen/Dense>
#include <glog/logging.h>
#include <iostream>

namespace aru {
namespace core {
namespace vo {

class VO {

public:
  explicit VO(std::string vo_config_file, std::string vocab_file);

  VO(std::string vocab_file, utilities::image::ExtractorParams extractor_params,
     utilities::image::MatcherParams matcher_params,
     SolverParams solver_params);

  utilities::transform::Transform
  EstimateMotion(utilities::image::StereoImage image_1,
                 utilities::image::StereoImage image_2);

  utilities::transform::Transform
  EstimateMotion(const utilities::image::FeatureSPtrVectorSptr &image_features);

  utilities::transform::Transform
  EstimateMotion(const utilities::image::FeatureSPtrVectorSptr &image_features,
                 int64_t source_timestamp, int64_t dest_timestamp);

  utilities::transform::Transform
  ObtainTransform(std::vector<cv::Point3d> landmarks,
                  std::vector<cv::Point2d> uv_points);

  ~VO() = default;

  std::pair<Eigen::MatrixXf, Eigen::MatrixXf>
  ObtainStereoPoints(cv::Mat image_left, cv::Mat image_right);

  std::tuple<Eigen::MatrixXf, Eigen::MatrixXf, Eigen::MatrixXf>
  ObtainStereoSequentialPoints(cv::Mat image_1_left, cv::Mat image_1_right,
                               cv::Mat image_2_left, cv::Mat image_2_right);

private:
  utilities::image::ExtractorParams extractor_params_{};
  utilities::image::MatcherParams matcher_params_{};
  SolverParams solver_params_;

  std::string vo_config_file_;
  std::string vocab_file_;
  boost::shared_ptr<VOSolver> vo_solver_;

};
} // namespace vo
} // namespace core
} // namespace aru

#endif // ARU_CORE_VO_H_
