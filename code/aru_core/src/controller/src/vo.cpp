#include <aru/core/vo/vo.h>

#include <Eigen/Dense>
#include <boost/make_shared.hpp>
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <utility>
namespace aru {
namespace core {
namespace vo {

//------------------------------------------------------------------------------
VO::VO(std::string vo_config_file, std::string vocab_file)
    : vo_config_file_(std::move(vo_config_file)), vocab_file_(std::move(vocab_file)) {

  cv::FileStorage fs;
  fs.open(vo_config_file_, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    LOG(ERROR) << "Could not open vo model file: ";
  }

  // Extractor Parameters
  extractor_params_.num_features = fs["FeatureExtractor"]["num_features"];
  extractor_params_.num_levels = fs["FeatureExtractor"]["num_levels"];
  extractor_params_.scale_factor = fs["FeatureExtractor"]["scale_factor"];
  extractor_params_.initial_fast_threshold =
      fs["FeatureExtractor"]["intitial_fast_threshold"];
  extractor_params_.minimum_fast_threshold = fs["FeatureExtractor"
                                                ""]["minimum_fast_threshold"];
  extractor_params_.patch_size = fs["FeatureExtractor"]["patch_size"];
  extractor_params_.half_patch_size = fs["FeatureExtractor"]["half_patch_size"];
  extractor_params_.edge_threshold = fs["FeatureExtractor"]["edge_threshold"];

  // Matcher parameters
  matcher_params_.match_threshold_high =
      fs["FeatureMatcher"]["match_threshold_high"];
  matcher_params_.match_threshold_low =
      fs["FeatureMatcher"]["match_threshold_low"];
  matcher_params_.stereo_baseline = fs["FeatureMatcher"]["stereo_baseline"];
  matcher_params_.focal_length = fs["FeatureMatcher"]["focal_length"];

  // Solver Parameters
  solver_params_.ransac_prob = fs["FeatureSolver"]["ransac_prob"];
  solver_params_.ransac_max_iterations =
      fs["FeatureSolver"]["ransac_max_iterations"];
  solver_params_.threshold = fs["FeatureSolver"]["inlier_threshold"];

  // Camera params
  cv::Mat camera_mat;
  fs["FeatureSolver"]["CameraMatrix"] >> camera_mat;
  cv::cv2eigen(camera_mat, solver_params_.camera_matrix);


  // Create the vo estimator
  vo_solver_ = boost::make_shared<VOSolver>(vocab_file_ ,solver_params_, extractor_params_,
                                            matcher_params_);
}
//------------------------------------------------------------------------------
VO::VO(std::string vocab_file, utilities::image::ExtractorParams extractor_params,
       utilities::image::MatcherParams matcher_params,
       SolverParams solver_params)
    : vocab_file_(vocab_file), extractor_params_(extractor_params), matcher_params_(matcher_params),
      solver_params_(solver_params) {
  // Create the mesh estimator
  vo_solver_ = boost::make_shared<VOSolver>(vocab_file_,solver_params_, extractor_params_,
                                            matcher_params_);
}
//------------------------------------------------------------------------------
utilities::transform::Transform
VO::EstimateMotion(utilities::image::StereoImage image_dest,
                   utilities::image::StereoImage image_source) {
  // Check if images are in grey scale

  // Perform the estimation
  auto estimation_start = std::chrono::high_resolution_clock::now();
  Eigen::Affine3d motion_est = vo_solver_->MatchAndSolve(
      image_source.first.GetImage(), image_source.second.GetImage(),
      image_dest.first.GetImage(), image_dest.second.GetImage());
  auto estimation_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = estimation_end - estimation_start;
  VLOG(2) << "Estimation takes " << elapsed.count() << " seconds";
  LOG(INFO) << "Estimation is running at " << 1 / elapsed.count() << " Hz";

  utilities::transform::Transform transform(image_source.first.GetTimeStamp(),
                                            image_dest.first.GetTimeStamp(),
                                            motion_est.cast<float>().inverse());

  return transform;
}
//------------------------------------------------------------------------------
utilities::transform::Transform VO::EstimateMotion(
    const utilities::image::FeatureSPtrVectorSptr &image_features) {
  std::vector<cv::Point3d> points3d;
  std::vector<cv::Point2d> points2d;
  // Perform the estimation
  auto estimation_start = std::chrono::high_resolution_clock::now();

  Eigen::Matrix3d K = solver_params_.camera_matrix;

  for (const auto &feature : *image_features) {
    feature->TriangulatePoint(K, matcher_params_.stereo_baseline);
    Eigen::Vector3d triangulated_point = feature->GetTriangulatedPoint();
    points3d.emplace_back(triangulated_point(0), triangulated_point(1),
                          triangulated_point(2));
    points2d.emplace_back(feature->GetSequentialKeyPoint().pt);
  }
  // Perform the estimation
  Eigen::Affine3d motion_est = vo_solver_->SolveRansac(points3d, points2d);
  auto estimation_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = estimation_end - estimation_start;
  VLOG(2) << "Solving takes " << elapsed.count() << " seconds";
  VLOG(2) << "Solving is running at " << 1 / elapsed.count() << " Hz";
  utilities::transform::Transform transform(0, 1, motion_est.cast<float>());

  return transform;
}

//------------------------------------------------------------------------------
utilities::transform::Transform VO::EstimateMotion(
    const utilities::image::FeatureSPtrVectorSptr &image_features,
    int64_t source_timestamp, int64_t dest_timestamp) {
  std::vector<cv::Point3d> points3d;
  std::vector<cv::Point2d> points2d;
  // Perform the estimation
  auto estimation_start = std::chrono::high_resolution_clock::now();

  Eigen::Matrix3d K = solver_params_.camera_matrix;

  for (const auto &feature : *image_features) {
    feature->TriangulatePoint(K, matcher_params_.stereo_baseline);
    Eigen::Vector3d triangulated_point = feature->GetTriangulatedPoint();
    points3d.emplace_back(triangulated_point(0), triangulated_point(1),
                          triangulated_point(2));
    points2d.emplace_back(feature->GetSequentialKeyPoint().pt);
  }
  // Perform the estimation
  Eigen::Affine3d motion_est = vo_solver_->SolveRansac(points3d, points2d);
  auto estimation_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = estimation_end - estimation_start;
  VLOG(2) << "Solving takes " << elapsed.count() << " seconds";
  VLOG(2) << "Solving is running at " << 1 / elapsed.count() << " Hz";
  utilities::transform::Transform transform(source_timestamp, dest_timestamp,
                                            motion_est.cast<float>());

  return transform;
}
//------------------------------------------------------------------------------
utilities::transform::Transform
VO::ObtainTransform(std::vector<cv::Point3d> landmarks,
                    std::vector<cv::Point2d> uv_points) {
  // Perform the estimation
  Eigen::Affine3d motion_est = vo_solver_->SolveRansac(landmarks, uv_points);

  utilities::transform::Transform transform(0, 1, motion_est.cast<float>());

  return transform;
}
//------------------------------------------------------------------------------

std::tuple<Eigen::MatrixXf, Eigen::MatrixXf, Eigen::MatrixXf>
VO::ObtainStereoSequentialPoints(cv::Mat image_1_left, cv::Mat image_1_right,
                                 cv::Mat image_2_left, cv::Mat image_2_right) {

  utilities::image::FeatureSPtrVectorSptr features =
      vo_solver_->ObtainStereoPoints(image_1_left, image_1_right, image_2_left,
                                     image_2_right);

  int num_points = features->size();

  Eigen::MatrixXf feat_left = Eigen::MatrixXf::Zero(num_points, 2);
  Eigen::MatrixXf feat_right = Eigen::MatrixXf::Zero(num_points, 2);
  Eigen::MatrixXf feat_2_left = Eigen::MatrixXf::Zero(num_points, 2);

  int num = 0;
  for (auto feat : *features) {
    cv::KeyPoint keypoint_left = feat->GetKeyPoint();
    cv::KeyPoint keypoint_right = feat->GetMatchedKeyPoint();
    cv::KeyPoint keypoint_2_left = feat->GetSequentialKeyPoint();

    feat_left(num, 0) = keypoint_left.pt.x;
    feat_left(num, 1) = keypoint_left.pt.y;

    feat_2_left(num, 0) = keypoint_2_left.pt.x;
    feat_2_left(num, 1) = keypoint_2_left.pt.y;

    feat_right(num, 0) = keypoint_right.pt.x;
    feat_right(num, 1) = keypoint_right.pt.y;

    num++;
  }
  return std::make_tuple(feat_left, feat_right,feat_2_left);
}
//------------------------------------------------------------------------------
std::pair<Eigen::MatrixXf, Eigen::MatrixXf>
VO::ObtainStereoPoints(cv::Mat image_left, cv::Mat image_right) {

  utilities::image::FeatureSPtrVectorSptr features =
      vo_solver_->ObtainStereoPoints(image_left, image_right);

  int num_points = features->size();

  Eigen::MatrixXf feat_left = Eigen::MatrixXf::Zero(num_points, 2);
  Eigen::MatrixXf feat_right = Eigen::MatrixXf::Zero(num_points, 2);

  int num = 0;
  for (auto feat : *features) {
    cv::KeyPoint keypoint_left = feat->GetKeyPoint();
    cv::KeyPoint keypoint_right = feat->GetMatchedKeyPoint();

    feat_left(num, 0) = keypoint_left.pt.x;
    feat_left(num, 1) = keypoint_left.pt.y;

    feat_right(num, 0) = keypoint_right.pt.x;
    feat_right(num, 1) = keypoint_right.pt.y;

    num++;
  }

  return std::make_pair(feat_left, feat_right);
}

} // namespace vo
} // namespace core
} // namespace aru
