#include "aru/core/pose_optimisation/pose_optimisation.h"

#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>

// We will also need a camera object to hold calibration information and perform
// projections.
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/PinholeCamera.h>
#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/slam/ProjectionFactor.h>

#include <Eigen/Dense>
#include <boost/make_shared.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/persistence.hpp>
#include <utility>

using namespace gtsam;
using namespace std;
using namespace aru::core::utilities;
using namespace aru::core::utilities::image;
namespace aru {
namespace core {
namespace pose_optimisation {
//------------------------------------------------------------------------------
PoseOptimisation::PoseOptimisation(std::string bundle_settings_file)
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

  keyframe_map_ = boost::make_shared<KeyFrameMap>();

  graph_ = boost::make_shared<gtsam::NonlinearFactorGraph>();

  num_cameras_ = 0;

  //  min_distance_ = 5;
  //  min_rotation_ = 0.5236;
  //
  localiser_ = boost::make_shared<localisation::Localisation>(
      "/home/paulamayo/data/husky_data/localisation/zoo_vocab.yml",
      "/home/paulamayo/data/husky_data/localisation/zoo_chow_liu_tree.yml",
      "/home/paulamayo/data/husky_data/localisation/settings.yml");

  localiser_->InitLocalisation();

  localiser_->AddSampleData("/home/paulamayo/data/husky_data/localisation"
                            "/sample_descriptors.yml");

  vo_ = boost::make_shared<vo::VO>(bundle_settings_file_);

  max_inter_frames_ = 50;
}
//------------------------------------------------------------------------------
void PoseOptimisation::InitialFrame(
    aru::core::utilities::image::StereoImage dest_image_frame) {

  cv::Mat image_dest_left_grey, image_dest_right_grey;
  cv::cvtColor(dest_image_frame.first.GetImage(), image_dest_left_grey,
               cv::COLOR_BGR2GRAY);
  cv::cvtColor(dest_image_frame.second.GetImage(), image_dest_right_grey,
               cv::COLOR_BGR2GRAY);

  // Add First frame as keyframe
  keyframe_map_->insert(KeyFrameMap::value_type(
      dest_image_frame.first.GetTimeStamp(), dest_image_frame));
  prev_timestamp = dest_image_frame.first.GetTimeStamp();

  // Add Frame 1 to graph
  auto poseNoise = gtsam::noiseModel::Diagonal::Sigmas(
      (Vector(6) << Vector3::Constant(0.01), Vector3::Constant(0.03))
          .finished());

  gtsam::Pose3 init_pose =
      gtsam::Pose3(gtsam::Rot3::Ypr(0, 0, 0), gtsam::Point3(0, 0, 0));
  graph_->addPrior(num_cameras_, init_pose, poseNoise);

  // Add  query to localiser
  localiser_->AddQueryImage(dest_image_frame.first.GetImage());

  stereo_vector.push_back(dest_image_frame);

  // Add initial estimate
  init_estimate_.insert(num_cameras_, init_pose);

  //Add timestamp
  timestamps_vec_.push_back((dest_image_frame.first.GetTimeStamp()));

  num_cameras_++;
}
//------------------------------------------------------------------------------
void PoseOptimisation::AddFrame(
    aru::core::utilities::image::StereoImage image_frame) {
  cv::Mat image_left_grey, image_right_grey;
  cv::cvtColor(image_frame.first.GetImage(), image_left_grey,
               cv::COLOR_BGR2GRAY);
  cv::cvtColor(image_frame.second.GetImage(), image_right_grey,
               cv::COLOR_BGR2GRAY);

  // Get position
  utilities::transform::TransformSPtr curr_position =
      transform_map_->Interpolate(prev_timestamp,
                                  image_frame.first.GetTimeStamp());

  auto poseNoise = gtsam::noiseModel::Diagonal::Sigmas(
      (Vector(6) << Vector3::Constant(0.01), Vector3::Constant(0.03))
          .finished());

  if (curr_position) {
    Eigen::Vector3f xyz = curr_position->GetTranslation();
    gtsam::Rot3 curr_rot(curr_position->GetRotation().cast<double>());
    gtsam::Point3 curr_trans(xyz.cast<double>());
    gtsam::Pose3 curr_pose(curr_rot, curr_trans);

    // Check for loop closure
    cv::Mat image_curr = image_frame.first.GetImage();
    auto loop_closure = localiser_->FindLoopClosure(
        image_curr, num_cameras_ - max_inter_frames_);
    LOG(INFO) << "Image " << num_cameras_ << " localised to image "
              << loop_closure.first << " with probability "
              << loop_closure.second;
    if (loop_closure.first > 0) {
      utilities::transform::Transform pose =
          vo_->EstimateMotion(stereo_vector[loop_closure.first], image_frame);

      if (pose.GetTranslation().norm() > 0 &&
          pose.GetTranslation().norm() < 2) {
        LOG(INFO) << "Transform is " << pose.GetTransform().matrix();
        gtsam::Rot3 loop_rot(pose.GetRotation().cast<double>());
        gtsam::Point3 loop_trans(pose.GetTranslation().cast<double>());
        gtsam::Pose3 loop_pose(loop_rot, loop_trans);
        graph_->emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
            loop_closure.first, num_cameras_, curr_pose, poseNoise);

        cv::Mat localised;
        cv::hconcat(image_curr,
                    stereo_vector[loop_closure.first].first.GetImage(),
                    localised);

        cv::resize(localised, localised, cv::Size(), 0.5, 0.5);
        cv::imshow("Localisation", localised);
        cv::waitKey(0);
      }
    }

    localiser_->AddQueryImage(image_frame.first.GetImage());
    timestamps_vec_.push_back((image_frame.first.GetTimeStamp()));
    stereo_vector.push_back(image_frame);

    graph_->emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
        num_cameras_ - 1, num_cameras_, curr_pose, poseNoise);

    utilities::transform::TransformSPtr global_position =
        transform_map_->Interpolate(image_frame.first.GetTimeStamp());

    gtsam::Rot3 global_rot(global_position->GetRotation().cast<double>());
    gtsam::Point3 global_trans(
        global_position->GetTranslation().cast<double>());
    gtsam::Pose3 global_pose(global_rot, global_trans);

    init_estimate_.insert(num_cameras_, global_pose);

    prev_timestamp = image_frame.first.GetTimeStamp();
    num_cameras_++;
  }
}
//------------------------------------------------------------------------------
void PoseOptimisation::AddPose(
    aru::core::utilities::transform::TransformSPtr transform) {
  transform_map_->AddTransform(transform);
}
//------------------------------------------------------------------------------
void PoseOptimisation::OptimisePose() {
  // optimization. We will set a few parameters as a demonstration.
  gtsam::GaussNewtonParams parameters;
  // Stop iterating once the change in error between steps is less than this
  // value
  parameters.relativeErrorTol = 1e-5;
  // Do not perform more than N iteration steps
  parameters.maxIterations = 100;
  // Create the optimizer ...
  gtsam::GaussNewtonOptimizer optimizer(*graph_, init_estimate_, parameters);
  // ... and optimize
  opt_result_ = optimizer.optimize();
  LOG(INFO) << "Optimization complete";
  LOG(INFO) << "initial error = " << graph_->error(init_estimate_) << std::endl;
  LOG(INFO) << "final error = " << graph_->error(opt_result_) << std::endl;

}
//------------------------------------------------------------------------------


/* ************************************************************************* */
std::vector<gtsam::Point3> createPoints() {

  // Create the set of ground-truth landmarks
  std::vector<gtsam::Point3> points;
  points.push_back(gtsam::Point3(10.0, 10.0, 10.0));
  points.push_back(gtsam::Point3(-10.0, 10.0, 10.0));
  points.push_back(gtsam::Point3(-10.0, -10.0, 10.0));
  points.push_back(gtsam::Point3(10.0, -10.0, 10.0));
  points.push_back(gtsam::Point3(10.0, 10.0, -10.0));
  points.push_back(gtsam::Point3(-10.0, 10.0, -10.0));
  points.push_back(gtsam::Point3(-10.0, -10.0, -10.0));
  points.push_back(gtsam::Point3(10.0, -10.0, -10.0));

  return points;
}

/* ************************************************************************* */
std::vector<gtsam::Pose3> createPoses(
    const gtsam::Pose3 &init = gtsam::Pose3(gtsam::Rot3::Ypr(M_PI / 2, 0,
                                                             -M_PI / 2),
                                            gtsam::Point3(30, 0, 0)),
    const gtsam::Pose3 &delta = gtsam::Pose3(
        gtsam::Rot3::Ypr(0, -M_PI / 4, 0),
        gtsam::Point3(sin(M_PI / 4) * 30, 0, 30 * (1 - sin(M_PI / 4)))),
    int steps = 8) {

  // Create the set of ground-truth poses
  // Default values give a circular trajectory, radius 30 at pi/4 intervals,
  // always facing the circle center
  std::vector<gtsam::Pose3> poses;
  int i = 1;
  poses.push_back(init);
  for (; i < steps; ++i) {
    poses.push_back(poses[i - 1].compose(delta));
  }

  return poses;
}
//------------------------------------------------------------------------------
void PoseOptimisation::OptimiseBundle() {
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

  // Define the camera calibration parameters
  gtsam::Cal3_S2::shared_ptr K(
      new gtsam::Cal3_S2(531.14774, 531.26312, 0, 637.87114, 331.27469));

  // Define the camera observation noise model
  auto measurementNoise =
      gtsam::noiseModel::Isotropic::Sigma(2, 1.0); // one pixel in u and v

  std::vector<gtsam::Pose3> poses; // = createPoses();

  // Add the camera positions to the poses
  for (int camera = 0; camera < num_cameras_; ++camera) {
    Eigen::Affine3f curr_position = init_poses[camera];
    Eigen::Vector3f rpy =
        utilities::transform::Transform::RPYFromRotationMatrix(
            curr_position.rotation());
    Eigen::Vector3f xyz = curr_position.translation();
    gtsam::Rot3 curr_rot(curr_position.rotation().cast<double>());
    gtsam::Point3 curr_trans(xyz.cast<double>());
    gtsam::Pose3 curr_pose(curr_rot, curr_trans);
    poses.push_back(curr_pose);
  }

  // Add a prior on pose x1. This indirectly specifies where the origin is.
  auto poseNoise = gtsam::noiseModel::Diagonal::Sigmas(
      (Vector(6) << Vector3::Constant(0.01), Vector3::Constant(0.03))
          .finished()); // 30cm std on x,y,z 0.1 rad on roll,pitch,yaw
  graph_->addPrior(Symbol('x', 0), poses[0], poseNoise); // add directly to
                                                         // graph

  int observ_no = 0;
  int point_no = 0;
  for (const auto &track : feature_tracks) {
    // for (int track_no = 0; track_no < 10; ++track_no) {
    //  auto track = feature_tracks[track_no];
    int num_feature_tracks = track.feature_track_->size();
    for (int feat = 0; feat < num_feature_tracks; ++feat) {
      int pose_no = track.frame_track_->at(feat);
      float u = track.feature_track_->at(feat)->GetKeyPoint().pt.x;
      float v = track.feature_track_->at(feat)->GetKeyPoint().pt.y;
      Point2 measurement(u, v);
      graph_->emplace_shared<GenericProjectionFactor<Pose3, Point3, Cal3_S2>>(
          measurement, measurementNoise, Symbol('x', pose_no),
          Symbol('l', point_no), K);
      observ_no++;
    }
    point_no++;
  }

  //

  // LOG(INFO) << "Number of init_poses is " << init_poses.size();
  // Add the points to the parameters
  std::vector<gtsam::Point3> points; // = createPoints();
  for (int point = 0; point < num_points_; ++point) {
    // Find the first time the point was observed
    FeatureSPtr feat = feature_tracks[point].feature_track_->at(0);
    feat->TriangulatePoint(camera_params_.K, camera_params_.baseline);
    Eigen::Vector3f point_xyz = (feat->GetTriangulatedPoint()).cast<float>();
    int frame_index = feature_tracks[point].frame_track_->at(0);
    point_xyz = init_poses[frame_index] * point_xyz;
    points.emplace_back(point_xyz.cast<double>());
  }

  //  for (size_t i = 0; i < poses.size(); ++i) {
  //    PinholeCamera<Cal3_S2> camera(poses[i], *K);
  //    for (size_t j = 0; j < points.size(); ++j) {
  //      Point2 measurement = camera.project(points[j]);
  //      LOG(INFO) << "Measurement "<<j<<" is " << measurement;
  //      graph_->emplace_shared<GenericProjectionFactor<Pose3, Point3,
  //      Cal3_S2>>(
  //          measurement, measurementNoise, Symbol('x', i), Symbol('l', j), K);
  //    }
  //  }

  auto pointNoise = noiseModel::Isotropic::Sigma(3, 0.1);
  graph_->addPrior(Symbol('l', 0), points[0],
                   pointNoise); // add directly to graph
  graph_->print("Factor Graph:\n");

  // Create the data structure to hold the initial estimate to the solution
  // Intentionally initialize the variables off from the ground truth
  Values initialEstimate;
  for (size_t i = 0; i < poses.size(); ++i) {
    auto corrupted_pose = poses[i];
    initialEstimate.insert(Symbol('x', i), corrupted_pose);
  }
  for (size_t j = 0; j < points.size(); ++j) {
    Point3 corrupted_point = points[j];
    initialEstimate.insert<Point3>(Symbol('l', j), corrupted_point);
  }
  initialEstimate.print("Initial Estimates:\n");

  /* Optimize the graph and print results */
  Values result;
  try {
    LevenbergMarquardtParams params;
    params.setVerbosity("ERROR");
    LevenbergMarquardtOptimizer lm(*graph_, initialEstimate, params);
    result = lm.optimize();
  } catch (exception &e) {
    LOG(INFO) << e.what();
  }
  LOG(INFO) << "final error: " << graph_->error(result) << endl;

  //  Values result = DoglegOptimizer(*graph_, initialEstimate).optimize();
  //  result.print("Final results:\n");
}
//------------------------------------------------------------------------------
void PoseOptimisation::CalculateResiduals() {}

} // namespace pose_optimisation
} // namespace core
} // namespace aru
