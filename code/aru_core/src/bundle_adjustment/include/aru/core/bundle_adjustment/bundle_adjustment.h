#ifndef ARU_CORE_BUNDLE_ADJUSTMENT_H_
#define ARU_CORE_BUNDLE_ADJUSTMENT_H_

#include <Eigen/Dense>
#include <glog/logging.h>
#include <iostream>

#include "aru/core/utilities/camera/camera.h"
#include "aru/core/utilities/image/feature_tracker.h"
#include "aru/core/utilities/image/image.h"
// #include "bal_problem.h"
#include "ceres/ceres.h"
#include <aru/core/utilities/transforms/transform_map.h>
#include <aru/core/utilities/viewer/viewer.h>
#include <aru/core/vo/vo.h>
#include <ceres/rotation.h>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>

namespace aru::core::bundle_adjustment {

// Templated pinhole camera model for used with Ceres.  The camera is
// parameterized using 6 parameters: 3 for rotation, 3 for translation
struct SnavelyReprojectionError {
  SnavelyReprojectionError(double observed_x, double observed_y,
                           const Eigen::Matrix3d &_intrinsics)
      : observed_x(observed_x), observed_y(observed_y),
        intrinsics(_intrinsics) {}
  template <typename T>
  bool operator()(const T *const camera, const T *const point,
                  T *residuals) const {
    // camera[0,1,2] are the angle-axis rotation.
    T p[3];
    ceres::AngleAxisRotatePoint(camera, point, p);
    // camera[3,4,5] are the translation.
    p[0] += camera[3];
    p[1] += camera[4];
    p[2] += camera[5];
    // Compute final projected point position.
    const T predicted_x = p[0] / p[2] * intrinsics(0, 0) + intrinsics(0, 2);
    const T predicted_y = p[1] / p[2] * intrinsics(1, 1) + intrinsics(1, 2);
    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - observed_x;
    residuals[1] = predicted_y - observed_y;
    return true;
  }
  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction *Create(const double observed_x,
                                     const double observed_y,
                                     const Eigen::Matrix3d &intrinsics) {
    return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 6, 3>(
        new SnavelyReprojectionError(observed_x, observed_y, intrinsics)));
  }
  double observed_x;
  double observed_y;
  const Eigen::Matrix3d intrinsics;
};
// Templated pinhole camera model for used with Ceres.  The camera is
// parameterized using 7 parameters. 4 for rotation, 3 for
// translation
struct SnavelyReprojectionErrorWithQuaternions {
  // (u, v): the position of the observation with respect to the image
  // center point.
  SnavelyReprojectionErrorWithQuaternions(double observed_x, double observed_y,
                                          const Eigen::Matrix3d &_intrinsics)
      : observed_x(observed_x), observed_y(observed_y),
        intrinsics(_intrinsics) {}
  template <typename T>
  bool operator()(const T *const camera, const T *const point,
                  T *residuals) const {
    // camera[0,1,2,3] is are the rotation of the camera as a quaternion.
    //
    // We use QuaternionRotatePoint as it does not assume that the
    // quaternion is normalized, since one of the ways to run the
    // bundle adjuster is to let Ceres optimize all 4 quaternion
    // parameters without a local parameterization.
    T p[3];
    ceres::QuaternionRotatePoint(camera, point, p);
    p[0] += camera[4];
    p[1] += camera[5];
    p[2] += camera[6];
    // Compute final projected point position.
    const T predicted_x = p[0] / p[2] * intrinsics(0, 0) + intrinsics(0, 2);
    const T predicted_y = p[1] / p[2] * intrinsics(0, 0) + intrinsics(0, 2);

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - observed_x;
    residuals[1] = predicted_y - observed_y;
    return true;
  }
  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction *Create(const double observed_x,
                                     const double observed_y,
                                     const Eigen::Matrix3d &intrinsics) {
    return (
        new ceres::AutoDiffCostFunction<SnavelyReprojectionErrorWithQuaternions,
                                        2, 7, 3>(
            new SnavelyReprojectionErrorWithQuaternions(observed_x, observed_y,
                                                        intrinsics)));
  }
  double observed_x;
  double observed_y;
  Eigen::Matrix3d intrinsics;
};

class BundleAdjustment {

public:
  explicit BundleAdjustment(std::string bundle_settings_file);

  ~BundleAdjustment() = default;

  void Optimise();

  void InitialFrame(aru::core::utilities::image::StereoImage dest_image_frame,
                    aru::core::utilities::image::StereoImage src_image_frame);

  void AddFrame(aru::core::utilities::image::StereoImage image_frame);

  void AddPose(aru::core::utilities::transform::TransformSPtr transform);

  std::vector<Eigen::Affine3f> GetPoses() { return init_poses; }

  // The reprojection error of the problem remains the same.
  void Normalize();
  // Perturb the camera pose and the geometry with random normal
  // numbers with corresponding standard deviations.
  void Perturb(const double rotation_sigma, const double translation_sigma,
               const double point_sigma);

  Eigen::Matrix3d GetIntrinsics() { return camera_params_.K; }

  int CameraBlockSize() const { return 6; }
  int PointBlockSize() const { return 3; }
  int NumCameras() const { return num_cameras_; }
  int NumPoints() const { return num_points_; }
  int NumObservations() const { return num_observations_; }
  int NumParameters() const { return num_parameters_; }
  const int *PointIndex() const { return point_index_; }
  const int *CameraIndex() const { return camera_index_; }
  const double *Observations() const { return observations_; }
  const double *Parameters() const { return parameters_; }
  const double *Cameras() const { return parameters_; }
  double *MutableCameras() { return parameters_; }

  double *MutablePoints() {
    return parameters_ + CameraBlockSize() * num_cameras_;
  }

  void CalculateResiduals();

private:
  void CameraToAngleAxisAndCenter(const double *camera, double *angle_axis,
                                  double *center) const;
  void AngleAxisAndCenterToCamera(const double *angle_axis,
                                  const double *center, double *camera) const;

  std::string bundle_settings_file_;

  boost::shared_ptr<utilities::image::VisoFeatureTracker> viso_extractor_;

  utilities::image::ExtractorParams extractor_params_;
  utilities::image::MatcherParams matcher_params_;
  utilities::camera::CameraParams camera_params_;

  std::vector<Eigen::Affine3f> init_poses;
  std::map<size_t, Sophus::SE3d> ba_poses;

  boost::shared_ptr<utilities::transform::TransformMap> transform_map_;

  int num_cameras_;
  int num_points_;
  int num_observations_;
  int num_parameters_;

  int *point_index_;
  int *camera_index_;
  double *observations_;
  // The parameter vector is laid out as follows
  // [camera_1, ..., camera_n, point_1, ..., point_m]
  double *parameters_;
};

class BASolver {
public:
  BASolver();
  ~BASolver() = default;
  void SetLinearSolver(ceres::Solver::Options *options);

  void SetOrdering(BundleAdjustment bal, ceres::Solver::Options *options);
  void SetMinimizerOptions(ceres::Solver::Options *options);
  void SetSolverOptions(BundleAdjustment bal, ceres::Solver::Options *options);
  void BuildProblem(BundleAdjustment bal, ceres::Problem *problem);

  std::vector<Eigen::Affine3f> SolveProblem(BundleAdjustment bal);

private:
  bool inner_iterations;

  std::string trust_region_strategy;
  std::string dogleg;
  std::string blocks_for_inner_iterations;
  std::string linear_solver;
  bool explicit_schur_complement;

  std::string preconditioner;
  std::string visibility_clustering;
  std::string sparse_linear_algebra_library;
  std::string dense_linear_algebra_library;
  std::string ordering_;

  // bool use_quaternions;
  bool use_local_parameterization;
  bool robustify;
  double eta;
  int num_threads;
  int num_iterations;
  double max_solver_time;
  double rotation_sigma;
  double translation_sigma;
  double point_sigma;
  int random_seed;
  bool line_search;
  bool mix_precision_solves;
  int max_num_refinement_iterations;
  bool nonmonotonic_steps;
  bool mixed_precision_solves;
};
} // namespace aru::core::bundle_adjustment

#endif // ARU_CORE_BUNDLE_ADJUSTMENT_H_
