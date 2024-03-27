#ifndef ARU_CORE_REPEAT_H_
#define ARU_CORE_REPEAT_H_

#include <aru/core/utilities/image/feature_matcher.h>
#include <aru/core/utilities/image/image.h>
#include <aru/core/utilities/transforms/transforms.h>
#include <aru/core/vo/vo.h>

#include "aru/core/utilities/transforms/transform_map.h"
#include <Eigen/Dense>
#include <glog/logging.h>
#include <iostream>

namespace aru {
namespace core {
namespace repeat {

class Repeat {

public:
  explicit Repeat(std::string repeat_config_file, std::string vocab_file);

  void InitialiseMap(utilities::image::StereoImage image);

  void AddTeachKeyframe(utilities::image::StereoImage teach_keyframe);

  void AddTeachTransform(utilities::transform::TransformSPtr teach_transform);

  void QueryRepeatframe(utilities::image::StereoImage repeat_frame);

  aru::core::utilities::image::FeatureSPtrVectorSptr
  FeaturesFromMapPoints(ORB_SLAM3::KeyFrame *key_frame, ORB_SLAM3::Frame *frame,
                        std::vector<ORB_SLAM3::MapPoint *> map_matches);
                        
                        aru::core::utilities::transform::TransformSPtrVectorSptr RepeatPoseChain(){return repeat_pose_chain_;}
                        
  aru::core::utilities::transform::TransformSPtrVectorSptr TeachPoseChain(){return teach_pose_chain_;}

  ~Repeat() = default;

private:
  std::shared_ptr<utilities::transform::TransformMap> transform_map_;

  std::shared_ptr<aru::core::vo::VO> vo_;
  std::string vocab_file_;

  // ORB vocabulary used for place recognition and feature matching.
  ORB_SLAM3::ORBVocabulary *map_vocabulary_;

  ORB_SLAM3::GeometricCamera *map_camera_;

  utilities::image::ExtractorParams extractor_params_;

  ORB_SLAM3::ORBextractor *mpORBextractorLeft, *mpORBextractorRight;

  float baseline_;

  // Calibration matrix
  cv::Mat mK;
  cv::Mat mDistCoef;
  float mbf, fx, fy, cx, cy, mThDepth;

  // Atlas
  ORB_SLAM3::Atlas *map_atlas_;

  // KeyFrame database for place recognition (relocalization and loop
  // detection).
  ORB_SLAM3::KeyFrameDatabase *map_key_frame_database_;

  // Vector of actual teach images
  // TODO: Change this to a map of teach images linked to an indexed logger
  std::vector<utilities::image::StereoImage> image_vector_;
  
  // Pose chains
  aru::core::utilities::transform::TransformSPtrVectorSptr repeat_pose_chain_;
  aru::core::utilities::transform::TransformSPtrVectorSptr teach_pose_chain_;
  
  Eigen::Affine3f repeat_curr_position;
  Eigen::Affine3f teach_curr_position;
};
} // namespace repeat
} // namespace core
} // namespace aru

#endif // ARU_CORE_VO_H_
