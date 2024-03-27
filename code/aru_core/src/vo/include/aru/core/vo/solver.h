#ifndef ARU_CORE_VO_SOLVER_H_
#define ARU_CORE_VO_SOLVER_H_

#include <aru/core/utilities/image/feature_matcher.h>

#include <Eigen/Dense>
#include <glog/logging.h>
#include <iostream>

namespace aru {
namespace core {
namespace vo {

struct SolverParams {
  float ransac_prob;

  float threshold;

  float ransac_max_iterations;

  int min_inliers_;

  Eigen::Matrix3d camera_matrix;
};

typedef Eigen::Matrix3d Rot;
typedef Eigen::Vector3d Trans;

class VOSolver {
public:
  VOSolver(std::string vocabulary_filename,
           SolverParams solver_params,
           utilities::image::ExtractorParams extractor_params,
           utilities::image::MatcherParams matcher_params);

  ~VOSolver() = default;

public:
  Eigen::Affine3d MatchAndSolve(const cv::Mat &image_1_left,
                                const cv::Mat &image_1_right,
                                const cv::Mat &image_2_left,
                                const cv::Mat &image_2_right);

  Eigen::Affine3d MatchAndSolveBow(const cv::Mat &image_1_left,
                                const cv::Mat &image_1_right,
                                const cv::Mat &image_2_left,
                                const cv::Mat &image_2_right);

  Eigen::Affine3d Solve(const std::vector<cv::Point3d> &landmarks,
                        const std::vector<cv::Point2d> &uv_points);

  Eigen::Affine3d Refine(const std::vector<cv::Point3d> &landmarks,
                         const std::vector<cv::Point2d> &uv_points);

  Eigen::Affine3d SolveRansac(std::vector<cv::Point3d> landmarks,
                              std::vector<cv::Point2d> uv_points);

  Eigen::MatrixXd ReprojectionError(const Eigen::Affine3d &transform,
                                    std::vector<cv::Point3d> landmarks,
                                    std::vector<cv::Point2d> uv_points) const;

  int NumIterations(){return num_iterations_;}

  int NumInliers(){return num_inliers_;}

  Eigen::Matrix3d GetRotation() { return R_curr_; }

  Eigen::Vector3d GetTranslation() { return t_curr_; }

  utilities::image::FeatureSPtrVectorSptr
  ObtainStereoPoints(cv::Mat image_left, cv::Mat image_right);

  utilities::image::FeatureSPtrVectorSptr
  ObtainStereoPoints(cv::Mat image_1_left, cv::Mat image_1_right,
                     cv::Mat image_2_left, cv::Mat image_2_right);

private:
  SolverParams solver_params_;
  // camera intrinsics
  double u_c_, v_c_, f_u_, f_v_;

  Eigen::Matrix3f K_;
  float stereo_baseline_;

  // ransac parameters
  // double ransac_prob_;
  // int ransac_max_iterations_;
  // double threshold_;

  int min_inliers_;

  // Initial Guess
  Rot R_est_;
  Trans t_est_;

  Eigen::Affine3d transform_est_;

  // Current Estimation
  Rot R_curr_;
  Trans t_curr_;

  Eigen::Affine3d transform_;

  // Feature matcher
  boost::shared_ptr<utilities::image::VisoMatcher> sequential_matcher_;
  boost::shared_ptr<utilities::image::OrbFeatureMatcher> distance_matcher_;

  // Solver diagnostics
  int num_iterations_;
  int num_inliers_;
};
} // namespace vo
} // namespace core
} // namespace aru

#endif // ARU_CORE_VO_SOLVER_H_
