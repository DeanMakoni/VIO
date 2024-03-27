#ifndef ARU_CORE_POSE_OPTIMISATION_H_
#define ARU_CORE_POSE_OPTIMISATION_H_

#include <Eigen/Dense>
#include <glog/logging.h>
#include <iostream>

#include "aru/core/localisation/localisation.h"
#include "aru/core/utilities/camera/camera.h"
#include "aru/core/utilities/image/feature_tracker.h"
#include "aru/core/utilities/image/image.h"
#include <aru/core/utilities/transforms/transform_map.h>
#include <aru/core/utilities/viewer/viewer.h>
#include <aru/core/vo/vo.h>
#include <ceres/rotation.h>
#include <opencv2/opencv.hpp>

#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/GeneralSFMFactor.h>
#include <gtsam/slam/dataset.h>

namespace aru {
namespace core {
namespace pose_optimisation {

class PoseOptimisation {

  typedef std::map<int64_t, utilities::image::StereoImage> KeyFrameMap;

public:
  explicit PoseOptimisation(std::string bundle_settings_file);

  ~PoseOptimisation() = default;

  void OptimiseBundle();

  void OptimisePose();

  void InitialFrame(aru::core::utilities::image::StereoImage dest_image_frame);

  void AddFrame(aru::core::utilities::image::StereoImage image_frame);

  void AddPose(aru::core::utilities::transform::TransformSPtr transform);

  std::vector<Eigen::Affine3f> GetPoses() { return init_poses; }

  std::vector<Eigen::Affine3f> GetKeyFramePoses() { return init_poses; }

  gtsam::Values GetOptPoses(){ return opt_result_;}

  std::vector<long> GetTimestamps() {return timestamps_vec_;}

  void CalculateResiduals();

private:
  void AddNewKeyframe();

  std::string bundle_settings_file_;

  boost::shared_ptr<aru::core::vo::VO> vo_;
  boost::shared_ptr<utilities::image::VisoFeatureTracker> viso_extractor_;

  utilities::image::ExtractorParams extractor_params_;
  utilities::image::MatcherParams matcher_params_;
  utilities::camera::CameraParams camera_params_;

  std::vector<Eigen::Affine3f> init_poses;
  std::vector<Eigen::Affine3f> key_frame_poses;

  boost::shared_ptr<utilities::transform::TransformMap> transform_map_;

  boost::shared_ptr<gtsam::NonlinearFactorGraph> graph_;

  boost::shared_ptr<KeyFrameMap> keyframe_map_;

  std::vector<aru::core::utilities::image::StereoImage> stereo_vector;

  std::vector<long> timestamps_vec_;

  boost::shared_ptr<aru::core::localisation::Localisation> localiser_;

  gtsam::Values init_estimate_;

  gtsam::Values opt_result_;

  long prev_timestamp;

  float min_distance_;
  float min_rotation_;

  int max_inter_frames_;

  int num_cameras_;
  int num_points_;
  int num_observations_;
  int num_parameters_;
};
} // namespace pose_optimisation
} // namespace core
} // namespace aru
#endif // ARU_CORE_BUNDLE_ADJUSTMENT_H_
