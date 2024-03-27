#include "aru/core/bundle_adjustment/bundle_adjustment.h"

#include <Eigen/Dense>
#include <boost/make_shared.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/persistence.hpp>
#include <sophus/ceres_local_parameterization.hpp>
#include <utility>

using namespace ceres;
using namespace aru::core::utilities;
using namespace aru::core::utilities::image;
namespace aru {
namespace core {
namespace bundle_adjustment {
//------------------------------------------------------------------------------
BundleAdjustment::BundleAdjustment(std::string bundle_settings_file)
    : bundle_settings_file_(std::move(bundle_settings_file)) {

  cv::FileStorage fs;
  fs.open(bundle_settings_file_, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    LOG(ERROR) << "Could not open mesh map settings file: ";
  }

  // Camera params
  LOG(INFO) << "Camera params found";
  camera_params_.baseline = fs["Camera"]["baseline"];
  camera_params_.image_height = fs["Camera"]["height"];
  camera_params_.image_width = fs["Camera"]["width"];
  cv::Mat camera_mat;
  fs["Camera"]["CameraMatrix"] >> camera_mat;
  cv::cv2eigen(camera_mat, camera_params_.K);

  // Matcher params
  matcher_params_.focal_length = fs["FeatureMatcher"]["focal_length"];
  matcher_params_.stereo_baseline = fs["FeatureMatcher"]["stereo_baseline"];
  matcher_params_.match_threshold_low = fs["FeatureMatcher"
                                           ""]["match_threshold_low"];
  matcher_params_.match_threshold_high = fs["FeatureMatcher"
                                            ""]["match_threshold_high"];

  // Extractor params
  extractor_params_.patch_size = fs["FeatureExtractor"]["patch_size"];
  extractor_params_.half_patch_size = fs["FeatureExtractor"]["half_patch_size"];
  extractor_params_.num_levels = fs["FeatureExtractor"]["num_levels"];
  extractor_params_.scale_factor = fs["FeatureExtractor"]["scale_factor"];
  extractor_params_.edge_threshold = fs["FeatureExtractor"]["edge_threshold"];
  extractor_params_.num_features = fs["FeatureExtractor"]["num_features"];
  extractor_params_.initial_fast_threshold = fs["FeatureExtractor"
                                                ""]["initial_fast_threshold"];
  extractor_params_.minimum_fast_threshold = fs["FeatureExtractor"
                                                ""]["minimum_fast_threshold"];

  // Initialise VISO
  viso_extractor_ = boost::make_shared<utilities::image::VisoFeatureTracker>(
      matcher_params_, extractor_params_);

  transform_map_ = boost::make_shared<utilities::transform::TransformMap>();

  num_cameras_ = 0;
}
//------------------------------------------------------------------------------
void BundleAdjustment::InitialFrame(
    aru::core::utilities::image::StereoImage dest_image_frame,
    aru::core::utilities::image::StereoImage src_image_frame) {

  cv::Mat image_dest_left_grey, image_dest_right_grey;
  cv::cvtColor(dest_image_frame.first.GetImage(), image_dest_left_grey,
               cv::COLOR_BGR2GRAY);
  cv::cvtColor(dest_image_frame.second.GetImage(), image_dest_right_grey,
               cv::COLOR_BGR2GRAY);

  cv::Mat image_src_left_grey, image_src_right_grey;
  cv::cvtColor(src_image_frame.first.GetImage(), image_src_left_grey,
               cv::COLOR_BGR2GRAY);
  cv::cvtColor(src_image_frame.second.GetImage(), image_src_right_grey,
               cv::COLOR_BGR2GRAY);

  // Obtain Features
  viso_extractor_->InitialiseFeatures(
      image_dest_left_grey, image_dest_right_grey, image_src_left_grey,
      image_src_right_grey);

  // Add pose 1
  Eigen::Affine3f curr_position;
  curr_position.linear() = Eigen::MatrixXf::Identity(3, 3);
  curr_position.translation() = Eigen::VectorXf::Zero(3);
  init_poses.push_back(curr_position);
  num_cameras_++;
  // Add pose 2
  // Get position
  utilities::transform::TransformSPtr position =
      transform_map_->Interpolate(src_image_frame.first.GetTimeStamp());
  if (position) {
    // Obtain Features
    init_poses.push_back(position->GetTransform());
    num_cameras_++;
  }
}
//------------------------------------------------------------------------------
void BundleAdjustment::AddFrame(
    aru::core::utilities::image::StereoImage image_frame) {
  cv::Mat image_left_grey, image_right_grey;
  cv::cvtColor(image_frame.first.GetImage(), image_left_grey,
               cv::COLOR_BGR2GRAY);
  cv::cvtColor(image_frame.second.GetImage(), image_right_grey,
               cv::COLOR_BGR2GRAY);

  // Get position
  utilities::transform::TransformSPtr curr_position =
      transform_map_->Interpolate(image_frame.first.GetTimeStamp());
  if (curr_position) {
    // Obtain Features
    viso_extractor_->UpdateFeatures(image_left_grey, image_right_grey);
    init_poses.push_back(curr_position->GetTransform());
    num_cameras_++;
  }
}
//------------------------------------------------------------------------------
void BundleAdjustment::AddPose(
    aru::core::utilities::transform::TransformSPtr transform) {
  transform_map_->AddTransform(transform);
}
//------------------------------------------------------------------------------
void BundleAdjustment::Optimise() {

  LOG(INFO) << "Get all tracks";
  std::vector<FeatureTrack> feature_tracks = viso_extractor_->GetAllTracks();
  num_cameras_ = init_poses.size();
  num_points_ = feature_tracks.size();
  num_observations_ = 0;
  for (const auto &track : feature_tracks) {
    num_observations_ = num_observations_ + track.feature_track_->size();
  }

  LOG(INFO) << "Number of cameras is " << num_cameras_;
  LOG(INFO) << "Number of points is " << num_points_;
  LOG(INFO) << "Number of observations is " << num_observations_;

  point_index_ = new int[num_observations_];
  camera_index_ = new int[num_observations_];
  observations_ = new double[2 * num_observations_];

  num_parameters_ = 6 * num_cameras_ + 3 * num_points_;
  parameters_ = new double[num_parameters_];

  int observ_no = 0;
  int point_no = 0;
  for (const auto &track : feature_tracks) {
    int num_feature_tracks = track.feature_track_->size();
    for (int feat = 0; feat < num_feature_tracks; ++feat) {
      point_index_[observ_no] = point_no;
      camera_index_[observ_no] = track.frame_track_->at(feat);
      observations_[observ_no * 2] =
          track.feature_track_->at(feat)->GetKeyPoint().pt.x;
      observations_[observ_no * 2 + 1] =
          track.feature_track_->at(feat)->GetKeyPoint().pt.y;
      observ_no++;
    }
    point_no++;
  }

  // Add the cameras to the parameters
  for (int camera = 0; camera < num_cameras_; ++camera) {
    Eigen::Affine3f curr_position = init_poses[camera].inverse();
    Eigen::Vector3f rpy =
        utilities::transform::Transform::RPYFromRotationMatrix(
            curr_position.rotation());

    Eigen::Vector3f xyz = curr_position.translation();
    // Add the initial rpy
    parameters_[camera * 6 + 0] = rpy(0);
    parameters_[camera * 6 + 1] = rpy(1);
    parameters_[camera * 6 + 2] = rpy(2);
    // Add the initial xyz
    parameters_[camera * 6 + 3] = xyz.x();
    parameters_[camera * 6 + 4] = xyz.y();
    parameters_[camera * 6 + 5] = xyz.z();
    //    LOG(INFO) << "Camera number " << camera;
  }
  // LOG(INFO) << "Number of init_poses is " << init_poses.size();
  // Add the points to the parameters
  int offset = num_cameras_ * 6;
  for (int point = 0; point < num_points_; ++point) {
    // Find the first time the point was observed
    FeatureSPtr feat = feature_tracks[point].feature_track_->at(0);
    feat->TriangulatePoint(camera_params_.K, camera_params_.baseline);
    Eigen::Vector3f point_xyz = (feat->GetTriangulatedPoint()).cast<float>();
    int frame_index = feature_tracks[point].frame_track_->at(0);
    point_xyz = init_poses[frame_index] * point_xyz;
    parameters_[offset + point * 3 + 0] = point_xyz.x();
    parameters_[offset + point * 3 + 1] = point_xyz.y();
    parameters_[offset + point * 3 + 2] = point_xyz.z();
  }
  CalculateResiduals();
}
//------------------------------------------------------------------------------
void BundleAdjustment::CalculateResiduals() {
  float res_error = 0;
  int offset = num_cameras_ * CameraBlockSize();
  for (int observ = 0; observ < num_observations_; observ++) {

    int curr_camera_index = camera_index_[observ];
    int curr_point_index = point_index_[observ];

    //    LOG(INFO) << "Camera index of observed point is " <<
    //    curr_camera_index; LOG(INFO) << "Point index of observed point is " <<
    //    curr_point_index;

    Eigen::Vector2f observation(observations_[observ * 2],
                                observations_[observ * 2 + 1]);
    double obs_x = observation.x();
    double obs_y = observation.y();

    double *camera =
        MutableCameras() + CameraBlockSize() * camera_index_[observ];
    double *point = MutablePoints() + PointBlockSize() * point_index_[observ];

    double p[3];
    ceres::AngleAxisRotatePoint(camera, point, p);
    // camera[3,4,5] are the translation.
    p[0] += camera[3];
    p[1] += camera[4];
    p[2] += camera[5];
    // Compute final projected point position.
    const double predicted_x =
        p[0] / p[2] * camera_params_.K(0, 0) + camera_params_.K(0, 2);
    const double predicted_y =
        p[1] / p[2] * camera_params_.K(1, 1) + camera_params_.K(1, 2);
    // The error is the difference between the predicted and observed position.
    double resid_x = predicted_x - obs_x;
    double resid_y = predicted_y - obs_y;

    //    LOG(INFO) << "Ceres observed is " << obs_x << "," << obs_y;
    //    LOG(INFO) << "Ceres point is " << point[0] << "," << point[1] << ","
    //              << point[2];
    //    LOG(INFO) << " Ceres Angle axis is " << camera[0] << "," << camera[1]
    //    << ","
    //              << camera[2];
    // LOG(INFO) << "Ceres proj is " << p[0] << "," << p[1] << "," << p[2];
    // LOG(INFO) << "Ceres predicted is " << predicted_x << "," << predicted_y;
    //    LOG(INFO) << "Ceres residual " << observ << " is " << resid_x << ","
    //              << resid_y;

    // LOG(INFO) << "Observation is " << observation.transpose();

    Eigen::Vector3f point_proj(parameters_[offset + curr_point_index * 3 + 0],
                               parameters_[offset + curr_point_index * 3 + 1],
                               parameters_[offset + curr_point_index * 3 + 2]);
    // LOG(INFO) << "Point is " << point_proj.transpose();

    Eigen::Affine3f rand_position;
    Eigen::Matrix3f eigen_rot =
        utilities::transform::Transform::RotationMatrixFromRPY(
            Eigen::Vector3f(parameters_[curr_camera_index * 6 + 0],
                            parameters_[curr_camera_index * 6 + 1],
                            parameters_[curr_camera_index * 6 + 2]));
    rand_position.linear() = eigen_rot;
    rand_position.translation() =
        Eigen::Vector3f(parameters_[curr_camera_index * 6 + 3],
                        parameters_[curr_camera_index * 6 + 4],
                        parameters_[curr_camera_index * 6 + 5]);
    //    LOG(INFO) << "Camera position is \n" << rand_position.matrix();
    //    LOG(INFO)
    //        << "Position error is \n "
    //        << (rand_position * init_poses[curr_camera_index]).matrix();

    Eigen::Vector3f proj = rand_position * point_proj;
    // LOG(INFO) << "Point projection is " << proj.transpose();
    Eigen::Vector2f proj_2d;
    proj_2d.x() =
        proj(0) / proj(2) * camera_params_.K(0, 0) + camera_params_.K(0, 2);
    proj_2d.y() =
        proj(1) / proj(2) * camera_params_.K(1, 1) + camera_params_.K(1, 2);
    // LOG(INFO) << "Proj 2d is " << proj_2d.transpose();
    //    LOG(INFO) << "Point Residual error is "
    //              << (observation - proj_2d).transpose();
  }
}
//------------------------------------------------------------------------------
BASolver::BASolver() {
  trust_region_strategy = "levenberg_marquardt";
  inner_iterations = false;
  dogleg = "traditional_dogleg";
  blocks_for_inner_iterations = "automatic";
  linear_solver = "sparse_schur";
  explicit_schur_complement = false;
  preconditioner = "jacobi";
  visibility_clustering = "canonical_views";
  sparse_linear_algebra_library = "suite_sparse";
  dense_linear_algebra_library = "eigen";
  ordering_ = "automatic";
  // use_quaternions = true;
  use_local_parameterization = false;
  robustify = false;
  eta = 1e-2;
  num_threads = 8;
  num_iterations = 500;
  max_solver_time = 1e32;
  nonmonotonic_steps = false;
  rotation_sigma = 0.1;
  translation_sigma = 0.1;
  point_sigma = 0.1;
  random_seed = 38401;
  line_search = false;
  mixed_precision_solves = false;
  max_num_refinement_iterations = 0;
}
//------------------------------------------------------------------------------
void BASolver::SetLinearSolver(ceres::Solver::Options *options) {
  CHECK(
      StringToLinearSolverType((linear_solver), &options->linear_solver_type));
  CHECK(StringToPreconditionerType((preconditioner),
                                   &options->preconditioner_type));
  CHECK(StringToVisibilityClusteringType((visibility_clustering),
                                         &options->visibility_clustering_type));
  CHECK(StringToSparseLinearAlgebraLibraryType(
      (sparse_linear_algebra_library),
      &options->sparse_linear_algebra_library_type));
  CHECK(StringToDenseLinearAlgebraLibraryType(
      (dense_linear_algebra_library),
      &options->dense_linear_algebra_library_type));
  options->use_explicit_schur_complement = (explicit_schur_complement);
  options->use_mixed_precision_solves = (mixed_precision_solves);
  options->max_num_refinement_iterations = (max_num_refinement_iterations);
}
//------------------------------------------------------------------------------
void BASolver::SetOrdering(BundleAdjustment bal,
                           ceres::Solver::Options *options) {
  const int num_points = bal.NumPoints();
  const int point_block_size = bal.PointBlockSize();
  double *points = bal.MutablePoints();
  const int num_cameras = bal.NumCameras();
  const int camera_block_size = bal.CameraBlockSize();
  double *cameras = bal.MutableCameras();
  if (options->use_inner_iterations) {
    if ((blocks_for_inner_iterations) == "cameras") {
      LOG(INFO) << "Camera blocks for inner iterations";
      options->inner_iteration_ordering.reset(
          new ceres::ParameterBlockOrdering);
      for (int i = 0; i < num_cameras; ++i) {
        options->inner_iteration_ordering->AddElementToGroup(
            cameras + camera_block_size * i, 0);
      }
    } else if ((blocks_for_inner_iterations) == "points") {
      LOG(INFO) << "Point blocks for inner iterations";
      options->inner_iteration_ordering.reset(
          new ceres::ParameterBlockOrdering);
      for (int i = 0; i < num_points; ++i) {
        options->inner_iteration_ordering->AddElementToGroup(
            points + point_block_size * i, 0);
      }
    } else if ((blocks_for_inner_iterations) == "cameras,points") {
      LOG(INFO) << "Camera followed by point blocks for inner iterations";
      options->inner_iteration_ordering.reset(
          new ceres::ParameterBlockOrdering);
      for (int i = 0; i < num_cameras; ++i) {
        options->inner_iteration_ordering->AddElementToGroup(
            cameras + camera_block_size * i, 0);
      }
      for (int i = 0; i < num_points; ++i) {
        options->inner_iteration_ordering->AddElementToGroup(
            points + point_block_size * i, 1);
      }
    } else if ((blocks_for_inner_iterations) == "points,cameras") {
      LOG(INFO) << "Point followed by camera blocks for inner iterations";
      options->inner_iteration_ordering.reset(
          new ceres::ParameterBlockOrdering);
      for (int i = 0; i < num_cameras; ++i) {
        options->inner_iteration_ordering->AddElementToGroup(
            cameras + camera_block_size * i, 1);
      }
      for (int i = 0; i < num_points; ++i) {
        options->inner_iteration_ordering->AddElementToGroup(
            points + point_block_size * i, 0);
      }
    } else if ((blocks_for_inner_iterations) == "automatic") {
      LOG(INFO) << "Choosing automatic blocks for inner iterations";
    } else {
      LOG(FATAL) << "Unknown block type for inner iterations: "
                 << (blocks_for_inner_iterations);
    }
  }
  // Bundle adjustment problems have a sparsity structure that makes
  // them amenable to more specialized and much more efficient
  // solution strategies. The SPARSE_SCHUR, DENSE_SCHUR and
  // ITERATIVE_SCHUR solvers make use of this specialized
  // structure.
  //
  // This can either be done by specifying Options::ordering_type =
  // ceres::SCHUR, in which case Ceres will automatically determine
  // the right ParameterBlock ordering, or by manually specifying a
  // suitable ordering vector and defining
  // Options::num_eliminate_blocks.
  if ((ordering_) == "automatic") {
    return;
  }
  ceres::ParameterBlockOrdering *ordering = new ceres::ParameterBlockOrdering;
  // The points come before the cameras.
  for (int i = 0; i < num_points; ++i) {
    ordering->AddElementToGroup(points + point_block_size * i, 0);
  }
  for (int i = 0; i < num_cameras; ++i) {
    // When using axis-angle, there is a single parameter block for
    // the entire camera.
    ordering->AddElementToGroup(cameras + camera_block_size * i, 1);
  }
  options->linear_solver_ordering.reset(ordering);
}
//------------------------------------------------------------------------------
void BASolver::SetMinimizerOptions(ceres::Solver::Options *options) {
  options->max_num_iterations = (num_iterations);
  options->minimizer_progress_to_stdout = true;
  options->num_threads = (num_threads);
  options->eta = (eta);
  options->max_solver_time_in_seconds = (max_solver_time);
  options->use_nonmonotonic_steps = (nonmonotonic_steps);
  if ((line_search)) {
    options->minimizer_type = ceres::LINE_SEARCH;
  }
  CHECK(StringToTrustRegionStrategyType((trust_region_strategy),
                                        &options->trust_region_strategy_type));
  CHECK(StringToDoglegType((dogleg), &options->dogleg_type));
  options->use_inner_iterations = (inner_iterations);
}
//------------------------------------------------------------------------------
void BASolver::SetSolverOptions(BundleAdjustment bal,
                                ceres::Solver::Options *options) {
  SetMinimizerOptions(options);
  SetLinearSolver(options);
  SetOrdering(bal, options);
}
//------------------------------------------------------------------------------
void BASolver::BuildProblem(BundleAdjustment bal, ceres::Problem *problem) {
  const Eigen::Matrix3d intrinsics = bal.GetIntrinsics();

  const int point_block_size = bal.PointBlockSize();
  const int camera_block_size = bal.CameraBlockSize();
  double *points = bal.MutablePoints();
  LOG(INFO) << "Points zero is " << points[0];
  double *cameras = bal.MutableCameras();
  // Observations is 2*num_observations long array observations =
  // [u_1, u_2, ... , u_n], where each u_i is two dimensional, the x
  // and y positions of the observation.
  const double *observations = bal.Observations();
  for (int i = 0; i < bal.NumObservations(); ++i) {
    CostFunction *cost_function;
    // Each Residual block takes a point and a camera as input and
    // outputs a 2 dimensional residual.
    cost_function = SnavelyReprojectionError::Create(
        observations[2 * i + 0], observations[2 * i + 1], intrinsics);
    // If enabled use Huber's loss function.
    LossFunction *loss_function = (robustify) ? new HuberLoss(1.0) : NULL;
    // Each observation correponds to a pair of a camera and a point
    // which are identified by camera_index()[i] and point_index()[i]
    // respectively.
    double *camera = cameras + camera_block_size * bal.CameraIndex()[i];
    double *point = points + point_block_size * bal.PointIndex()[i];
    problem->AddResidualBlock(cost_function, loss_function, camera, point);
  }
  //  if ((use_quaternions) && (use_local_parameterization)) {
  //    LocalParameterization *camera_parameterization =
  //        new ProductParameterization(new QuaternionParameterization(),
  //                                    new IdentityParameterization(6));
  //    for (int i = 0; i < bal.NumCameras(); ++i) {
  //      problem->SetParameterization(cameras + camera_block_size * i,
  //                                   camera_parameterization);
  //    }
  //  }
}

//------------------------------------------------------------------------------
std::vector<Eigen::Affine3f> BASolver::SolveProblem(BundleAdjustment bal) {
  bal.Optimise();

  Problem problem;
  srand((random_seed));
  // bal.Normalize();
  //bal.Perturb((rotation_sigma), (translation_sigma), (point_sigma));
  BuildProblem(bal, &problem);
  Solver::Options options;
  SetSolverOptions(bal, &options);
  options.gradient_tolerance = 1e-16;
  options.function_tolerance = 1e-16;
  Solver::Summary summary;
  LOG(INFO) << "Solving";
  Solve(options, &problem, &summary);
  LOG(INFO) << summary.FullReport();

  std::vector<Eigen::Affine3f> output_poses;

  bal.CalculateResiduals();

  Eigen::Affine3f T_0w;
  T_0w.linear() = utilities::transform::Transform::RotationMatrixFromRPY(
      Eigen::Vector3f(bal.MutableCameras()[0], bal.MutableCameras()[1],
                      bal.MutableCameras()[2]));
  T_0w.translation() =
      Eigen::Vector3f(bal.MutableCameras()[3], bal.MutableCameras()[4],
                      bal.MutableCameras()[5]);
  for (int camera = 0; camera < bal.NumCameras(); ++camera) {
    Eigen::Affine3f curr_position;
    //= init_poses[camera].inverse();
    curr_position.linear() =
        utilities::transform::Transform::RotationMatrixFromRPY(
            Eigen::Vector3f(bal.MutableCameras()[camera * 6 + 0],
                            bal.MutableCameras()[camera * 6 + 1],
                            bal.MutableCameras()[camera * 6 + 2]));
    curr_position.translation() =
        Eigen::Vector3f(bal.MutableCameras()[camera * 6 + 3],
                        bal.MutableCameras()[camera * 6 + 4],
                        bal.MutableCameras()[camera * 6 + 5]);
    curr_position = T_0w * curr_position.inverse();
    output_poses.push_back(curr_position);
    LOG(INFO) << "VO xyz is "
              << bal.GetPoses()[camera].translation().transpose();
    LOG(INFO) << " BA xyz is " << curr_position.translation().transpose();
  }
  return output_poses;
}

//------------------------------------------------------------------------------
inline double RandDouble() {
  double r = static_cast<double>(rand());
  return r / RAND_MAX;
}
//------------------------------------------------------------------------------
// Marsaglia Polar method for generation standard normal (pseudo)
// random numbers http://en.wikipedia.org/wiki/Marsaglia_polar_method
inline double RandNormal() {
  double x1, x2, w;
  do {
    x1 = 2.0 * RandDouble() - 1.0;
    x2 = 2.0 * RandDouble() - 1.0;
    w = x1 * x1 + x2 * x2;
  } while (w >= 1.0 || w == 0.0);

  w = sqrt((-2.0 * log(w)) / w);
  return x1 * w;
}
//------------------------------------------------------------------------------
void PerturbPoint3(const double sigma, double *point) {
  for (int i = 0; i < 3; ++i) {
    point[i] += RandNormal() * sigma;
  }
}
//------------------------------------------------------------------------------
double Median(std::vector<double> *data) {
  int n = data->size();
  std::vector<double>::iterator mid_point = data->begin() + n / 2;
  std::nth_element(data->begin(), mid_point, data->end());
  return *mid_point;
}
//------------------------------------------------------------------------------
void BundleAdjustment::CameraToAngleAxisAndCenter(const double *camera,
                                                  double *angle_axis,
                                                  double *center) const {
  VectorRef angle_axis_ref(angle_axis, 3);
  ceres::QuaternionToAngleAxis(camera, angle_axis);

  // c = -R't
  Eigen::VectorXd inverse_rotation = -angle_axis_ref;
  AngleAxisRotatePoint(inverse_rotation.data(), camera + CameraBlockSize() - 6,
                       center);
  VectorRef(center, 3) *= -1.0;
}
//------------------------------------------------------------------------------
void BundleAdjustment::AngleAxisAndCenterToCamera(const double *angle_axis,
                                                  const double *center,
                                                  double *camera) const {
  ConstVectorRef angle_axis_ref(angle_axis, 3);

  AngleAxisToQuaternion(angle_axis, camera);

  // t = -R * c
  AngleAxisRotatePoint(angle_axis, center, camera + CameraBlockSize() - 6);
  VectorRef(camera + CameraBlockSize() - 6, 3) *= -1.0;
}
//------------------------------------------------------------------------------
void BundleAdjustment::Normalize() {
  // Compute the marginal median of the geometry.
  std::vector<double> tmp(num_points_);
  Eigen::Vector3d median;
  double *points = MutablePoints();
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < num_points_; ++j) {
      tmp[j] = points[3 * j + i];
    }
    median(i) = Median(&tmp);
  }

  for (int i = 0; i < num_points_; ++i) {
    VectorRef point(points + 3 * i, 3);
    tmp[i] = (point - median).lpNorm<1>();
  }

  const double median_absolute_deviation = Median(&tmp);

  // Scale so that the median absolute deviation of the resulting
  // reconstruction is 100.
  const double scale = 100.0 / median_absolute_deviation;

  VLOG(2) << "median: " << median.transpose();
  VLOG(2) << "median absolute deviation: " << median_absolute_deviation;
  VLOG(2) << "scale: " << scale;

  // X = scale * (X - median)
  for (int i = 0; i < num_points_; ++i) {
    VectorRef point(points + 3 * i, 3);
    point = scale * (point - median);
  }

  double *cameras = MutableCameras();
  double angle_axis[3];
  double center[3];
  for (int i = 0; i < num_cameras_; ++i) {
    double *camera = cameras + CameraBlockSize() * i;
    CameraToAngleAxisAndCenter(camera, angle_axis, center);
    // center = scale * (center - median)
    VectorRef(center, 3) = scale * (VectorRef(center, 3) - median);
    AngleAxisAndCenterToCamera(angle_axis, center, camera);
  }
}
//------------------------------------------------------------------------------
void BundleAdjustment::Perturb(const double rotation_sigma,
                               const double translation_sigma,
                               const double point_sigma) {
  CHECK_GE(point_sigma, 0.0);
  CHECK_GE(rotation_sigma, 0.0);
  CHECK_GE(translation_sigma, 0.0);

  double *points = MutablePoints();
  if (point_sigma > 0) {
    for (int i = 0; i < num_points_; ++i) {
      PerturbPoint3(point_sigma, points + 3 * i);
    }
  }

  for (int i = 0; i < num_cameras_; ++i) {
    double *camera = MutableCameras() + CameraBlockSize() * i;

    double angle_axis[3];
    double center[3];
    // Perturb in the rotation of the camera in the angle-axis
    // representation.
    CameraToAngleAxisAndCenter(camera, angle_axis, center);
    if (rotation_sigma > 0.0) {
      PerturbPoint3(rotation_sigma, angle_axis);
    }
    AngleAxisAndCenterToCamera(angle_axis, center, camera);

    if (translation_sigma > 0.0) {
      PerturbPoint3(translation_sigma, camera + CameraBlockSize() - 6);
    }
  }
}
} // namespace bundle_adjustment
} // namespace core
} // namespace aru
