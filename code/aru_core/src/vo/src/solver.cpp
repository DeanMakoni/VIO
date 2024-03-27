#include "aru/core/vo/solver.h"
#include <Eigen/Dense>
#include <aru/core/utilities/viewer/viewer.h>
#include <boost/make_shared.hpp>
#include <chrono>
#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d.hpp>
#include <utility>

namespace aru {
namespace core {
namespace vo {

using namespace cv;
using namespace std;
//------------------------------------------------------------------------------
VOSolver::VOSolver(std::string vocabulary_filename, SolverParams solver_params,
                   utilities::image::ExtractorParams extractor_params,
                   utilities::image::MatcherParams matcher_params)
    : solver_params_(std::move(solver_params)) {
  f_u_ = solver_params_.camera_matrix(0, 0);
  u_c_ = solver_params_.camera_matrix(0, 2);
  f_v_ = solver_params_.camera_matrix(1, 1);
  v_c_ = solver_params_.camera_matrix(1, 2);

  stereo_baseline_ = matcher_params.stereo_baseline;
  // Create the matchers
  sequential_matcher_ = boost::make_shared<utilities::image::VisoMatcher>(
      matcher_params, extractor_params);

  distance_matcher_ = boost::make_shared<utilities::image::OrbFeatureMatcher>(
      matcher_params, extractor_params, vocabulary_filename);
}
//------------------------------------------------------------------------------
Eigen::Affine3d VOSolver::Refine(const std::vector<cv::Point3d> &landmarks,
                                 const std::vector<cv::Point2d> &uv_points) {
  cv::Mat K_matrix =
      cv::Mat::zeros(3, 3, CV_64FC1); // intrinsic camera parameters
  K_matrix.at<double>(0, 0) = f_u_;   //      [ fx   0  cx ]
  K_matrix.at<double>(1, 1) = f_v_;   //      [  0  fy  cy ]
  K_matrix.at<double>(0, 2) = u_c_;   //      [  0   0   1 ]
  K_matrix.at<double>(1, 2) = v_c_;
  K_matrix.at<double>(2, 2) = 1;
  cv::Mat R_matrix = cv::Mat::zeros(3, 3, CV_64FC1); // rotation matrix
  cv::Mat t_matrix = cv::Mat::zeros(3, 1, CV_64FC1);
  Eigen::Matrix3d K = Eigen::MatrixXd::Zero(3, 3);

  cv::Mat distCoeffs =
      cv::Mat::zeros(4, 1, CV_64FC1); // vector of distortion coefficients
  cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1); // output rotation vector
  cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1); // output translation vector
  bool useExtrinsicGuess = true;

  // if true the function uses the provided rvec and tvec values as
  // initial approximations of the rotation and translation vectors
  // TODO: Solve using Ransac
  if (landmarks.size() >= 4) {

    cv::eigen2cv(R_curr_, R_matrix);
    cv::Rodrigues(R_matrix, rvec);
    cv::eigen2cv(t_curr_, tvec);
    cv::solvePnP(landmarks, uv_points, K_matrix, distCoeffs, rvec, tvec,
                 useExtrinsicGuess, cv::SOLVEPNP_ITERATIVE);
    cv::Rodrigues(rvec, R_matrix); // converts Rotation Vector to Matrix
    t_matrix = tvec;               // set translation matrix

    R_curr_ = Eigen::MatrixXd::Zero(3, 3);
    t_curr_ = Eigen::MatrixXd::Zero(3, 1);
    cv::cv2eigen(R_matrix, R_curr_);
    cv::cv2eigen(t_matrix, t_curr_);
  }

  Eigen::Affine3d transform;
  transform.linear() = R_curr_;
  transform.translation() = t_curr_;

  return transform;
}
//------------------------------------------------------------------------------
Eigen::Affine3d VOSolver::Solve(const std::vector<cv::Point3d> &landmarks,
                                const std::vector<cv::Point2d> &uv_points) {
  cv::Mat K_matrix =
      cv::Mat::zeros(3, 3, CV_64FC1); // intrinsic camera parameters
  K_matrix.at<double>(0, 0) = f_u_;   //      [ fx   0  cx ]
  K_matrix.at<double>(1, 1) = f_v_;   //      [  0  fy  cy ]
  K_matrix.at<double>(0, 2) = u_c_;   //      [  0   0   1 ]
  K_matrix.at<double>(1, 2) = v_c_;
  K_matrix.at<double>(2, 2) = 1;
  cv::Mat R_matrix = cv::Mat::zeros(3, 3, CV_64FC1); // rotation matrix
  cv::Mat t_matrix = cv::Mat::zeros(3, 1, CV_64FC1);
  Eigen::Matrix3d K = Eigen::MatrixXd::Zero(3, 3);

  cv::Mat distCoeffs =
      cv::Mat::zeros(4, 1, CV_64FC1); // vector of distortion coefficients
  cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1); // output rotation vector
  cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1); // output translation vector
  bool useExtrinsicGuess = false;

  // if true the function uses the provided rvec and tvec values as
  // initial approximations of the rotation and translation vectors

  // TODO: Solve using Ransac

  if (landmarks.size() >= 4) {
    cv::solvePnP(landmarks, uv_points, K_matrix, distCoeffs, rvec, tvec,
                 useExtrinsicGuess, cv::SOLVEPNP_EPNP);
    cv::Rodrigues(rvec, R_matrix); // converts Rotation Vector to Matrix
    t_matrix = tvec;               // set translation matrix

    R_curr_ = Eigen::MatrixXd::Zero(3, 3);
    t_curr_ = Eigen::MatrixXd::Zero(3, 1);
    cv::cv2eigen(R_matrix, R_curr_);
    cv::cv2eigen(t_matrix, t_curr_);
  }

  Eigen::Affine3d transform;
  transform.linear() = R_curr_;
  transform.translation() = t_curr_;

  return transform;
}
//------------------------------------------------------------------------------
Eigen::MatrixXd
VOSolver::ReprojectionError(const Eigen::Affine3d &transform,
                            std::vector<cv::Point3d> landmarks,
                            std::vector<cv::Point2d> uv_points) const {

  int num_features = landmarks.size();
  Eigen::MatrixXd error_matrix = Eigen::MatrixXd::Zero(num_features, 1);
  Eigen::MatrixXd K = Eigen::MatrixXd::Zero(3, 3);

  K << f_u_, 0, u_c_, 0, f_v_, v_c_, 0, 0, 1;
  for (int i = 0; i < num_features; ++i) {
    Eigen::Vector3d point(landmarks[i].x, landmarks[i].y, landmarks[i].z);
    Eigen::Vector3d transformed_point_1 = transform * point;
    Eigen::Vector3d uv_est = K * transformed_point_1;
    uv_est = uv_est / uv_est(2);

    double u = uv_points[i].x;
    double v = uv_points[i].y;

    error_matrix(i) = sqrt((u - uv_est(0)) * (u - uv_est(0)) +
                           (v - uv_est(1)) * (v - uv_est(1)));
  }
  return error_matrix;
}
//------------------------------------------------------------------------------
Eigen::Affine3d VOSolver::SolveRansac(std::vector<cv::Point3d> landmarks,
                                      std::vector<cv::Point2d> uv_points) {
  VLOG(2) << "VO solve";
  int DOF = 4;
  int num_points = landmarks.size();
  float current_prob = (float)DOF / num_points;

  uint max_iterations_prob =
      log(1 - solver_params_.ransac_prob) / log(1 - pow(current_prob, DOF));
  //  LOG(INFO) << "Max iterations prob is " << max_iterations_prob;
  //  LOG(INFO) << "Max iterations is " << solver_params_.ransac_max_iterations;
   num_iterations_ = 0;

  int max_inliers = 0;
  Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> output_inlier_matrix =
      (Eigen::VectorXf::Zero(num_points).array() > 1).matrix();

  while (num_iterations_ < max_iterations_prob &&
         num_iterations_ < solver_params_.ransac_max_iterations) {

    Eigen::VectorXi ind_features = Eigen::VectorXi::Constant(DOF, num_points);
    std::vector<cv::Point3d> curr_landmarks;
    std::vector<cv::Point2d> curr_uv_points;
    for (int i = 0; i < DOF; ++i) {
      int ind_random_feature = rand() % num_points;
      while ((ind_features.array() == ind_random_feature).any())
        ind_random_feature = rand() % num_points;
      ind_features(i) = ind_random_feature;
      curr_landmarks.push_back(landmarks[ind_random_feature]);
      curr_uv_points.push_back(uv_points[ind_random_feature]);
    }

    Eigen::Affine3d curr_transform = Solve(curr_landmarks, curr_uv_points);

    Eigen::MatrixXd feature_costs =
        ReprojectionError(curr_transform, landmarks, uv_points);
    uint num_inliers =
        (feature_costs.array() < solver_params_.threshold).count();

    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> inlier_matrix =
        (feature_costs.array() < solver_params_.threshold).matrix();

    if (num_inliers > max_inliers) {
      max_inliers = num_inliers;
      output_inlier_matrix = inlier_matrix;
      current_prob = (float)max_inliers / num_points;
      transform_est_ = curr_transform;

      max_iterations_prob =
          log(1 - solver_params_.ransac_prob) / log(1 - pow(current_prob, DOF));
    }
    num_iterations_++;
  }

  LOG(INFO) << "Number of inliers is " << max_inliers << " out of "
            << num_points;
  num_inliers_=max_inliers;
  if (max_inliers < 4) {
    R_curr_ = Eigen::MatrixXd::Identity(3, 3);
    t_curr_ = Eigen::MatrixXd::Zero(3, 1);

    Eigen::Affine3d transform;
    transform.linear() = R_curr_;
    transform.translation() = t_curr_;

    return transform;
  }
  std::vector<cv::Point3d> inlier_landmarks;
  std::vector<cv::Point2d> inlier_uv_points;
  for (int i = 0; i < num_points; ++i) {
    if (output_inlier_matrix(i)) {
      inlier_landmarks.push_back(landmarks[i]);
      inlier_uv_points.push_back(uv_points[i]);
    }
  }
  
  //LOG(INFO)<<"Transform est is "<< transform_est_.matrix();
  Eigen::Affine3d refined_transform = Solve(inlier_landmarks, inlier_uv_points);
  Eigen::MatrixXd feature_costs =
      ReprojectionError(transform_est_, landmarks, uv_points);
      
  //LOG(INFO)<<"Feature costs are "<<feature_costs;
  uint ref_inliers = (feature_costs.array() < solver_params_.threshold).count();
  if (ref_inliers > max_inliers) {
    transform_est_ = refined_transform;
  }
  //LOG(INFO)<<"Refined inilers is "<<ref_inliers;
  //LOG(INFO)<<"Transform refined is "<< refined_transform.matrix();
  return transform_est_;
}
//------------------------------------------------------------------------------
Eigen::Affine3d VOSolver::MatchAndSolveBow(const cv::Mat &image_1_left,
                                           const cv::Mat &image_1_right,
                                           const cv::Mat &image_2_left,
                                           const cv::Mat &image_2_right) {
  // Check for greyscale
  cv::Mat image_1_left_grey = image_1_left.clone();
  if (image_1_left_grey.channels() > 1) {
    cv::cvtColor(image_1_left, image_1_left_grey, cv::COLOR_BGR2GRAY);
  }
  cv::Mat image_1_right_grey = image_1_right.clone();
  if (image_1_right_grey.channels() > 1) {
    cv::cvtColor(image_1_right, image_1_right_grey, cv::COLOR_BGR2GRAY);
  }
  cv::Mat image_2_left_grey = image_2_left.clone();
  if (image_2_left_grey.channels() > 1) {
    cv::cvtColor(image_2_left, image_2_left_grey, cv::COLOR_BGR2GRAY);
  }
  cv::Mat image_2_right_grey = image_2_right.clone();
  if (image_2_right_grey.channels() > 1) {
    cv::cvtColor(image_2_right, image_2_right_grey, cv::COLOR_BGR2GRAY);
  }

  utilities::image::FeatureSPtrVectorSptr matched_features =
      distance_matcher_->ComputeMatches(image_1_left_grey, image_1_right_grey,
                                        image_2_left_grey, image_2_right_grey);

  std::vector<cv::Point3d> points3d;
  std::vector<cv::Point2d> points2d;
  auto solve_start = std::chrono::high_resolution_clock::now();
  for (const auto &feature : *matched_features) {
    feature->TriangulatePoint(solver_params_.camera_matrix, stereo_baseline_);
    Eigen::Vector3d triangulated_point = feature->GetTriangulatedPoint();
    points3d.emplace_back(triangulated_point(0), triangulated_point(1),
                          triangulated_point(2));
    points2d.emplace_back(feature->GetSequentialKeyPoint().pt);
  }

  //  aru::core::utilities::viewer::Viewer::ViewImageFeatures(image_1_left,
  //                                                          matched_features);

  transform_ = SolveRansac(points3d, points2d);
  auto solve_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> solve_elapsed = solve_end - solve_start;
  VLOG(2) << "Solve takes " << solve_elapsed.count() << " seconds";

  return transform_;
}
//------------------------------------------------------------------------------
Eigen::Affine3d VOSolver::MatchAndSolve(const cv::Mat &image_1_left,
                                        const cv::Mat &image_1_right,
                                        const cv::Mat &image_2_left,
                                        const cv::Mat &image_2_right) {

  // Check for greyscale
  cv::Mat image_1_left_grey = image_1_left.clone();
  if (image_1_left_grey.channels() > 1) {
    cv::cvtColor(image_1_left, image_1_left_grey, cv::COLOR_BGR2GRAY);
  }
  cv::Mat image_1_right_grey = image_1_right.clone();
  if (image_1_right_grey.channels() > 1) {
    cv::cvtColor(image_1_right, image_1_right_grey, cv::COLOR_BGR2GRAY);
  }
  cv::Mat image_2_left_grey = image_2_left.clone();
  if (image_2_left_grey.channels() > 1) {
    cv::cvtColor(image_2_left, image_2_left_grey, cv::COLOR_BGR2GRAY);
  }
  cv::Mat image_2_right_grey = image_2_right.clone();
  if (image_2_right_grey.channels() > 1) {
    cv::cvtColor(image_2_right, image_2_right_grey, cv::COLOR_BGR2GRAY);
  }

  utilities::image::FeatureSPtrVectorSptr matched_features =
      sequential_matcher_->ComputeSequentialMatches(
          image_1_left_grey, image_2_left_grey, image_1_right_grey,
          image_2_right_grey);

  std::vector<cv::Point3d> points3d;
  std::vector<cv::Point2d> points2d;
  auto solve_start = std::chrono::high_resolution_clock::now();
  for (const auto &feature : *matched_features) {
    feature->TriangulatePoint(solver_params_.camera_matrix, stereo_baseline_);
    Eigen::Vector3d triangulated_point = feature->GetTriangulatedPoint();
    points3d.emplace_back(triangulated_point(0), triangulated_point(1),
                          triangulated_point(2));
    points2d.emplace_back(feature->GetSequentialKeyPoint().pt);
  }

  //  aru::core::utilities::viewer::Viewer::ViewImageFeatures(image_2_left,
  //                                                          matched_features);

  transform_ = SolveRansac(points3d, points2d);
  auto solve_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> solve_elapsed = solve_end - solve_start;
  VLOG(2) << "Solve takes " << solve_elapsed.count() << " seconds";

  return transform_;
}

//------------------------------------------------------------------------------
utilities::image::FeatureSPtrVectorSptr
VOSolver::ObtainStereoPoints(cv::Mat image_left, cv::Mat image_right) {
  return sequential_matcher_->ComputeStereoMatches(image_left, image_right);
}
//------------------------------------------------------------------------------
utilities::image::FeatureSPtrVectorSptr
VOSolver::ObtainStereoPoints(cv::Mat image_1_left, cv::Mat image_1_right,
                             cv::Mat image_2_left, cv::Mat image_2_right) {
  return sequential_matcher_->ComputeSequentialMatches(
      image_1_left, image_2_left, image_1_right, image_2_right);
}

} // namespace vo
} // namespace core
} // namespace aru
