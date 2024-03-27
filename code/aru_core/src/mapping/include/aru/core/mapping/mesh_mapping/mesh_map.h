#ifndef ARU_CORE_MAPPING_MESH_MAP_H_
#define ARU_CORE_MAPPING_MESH_MAP_H_

#include <Eigen/Dense>
#include <glog/logging.h>
#include <iostream>

#include "aru/core/utilities/image/feature_tracker.h"
#include "aru/core/utilities/image/image.h"
#include "voxblox/core/tsdf_map.h"
#include "voxblox/integrator/tsdf_integrator.h"
#include "voxblox/mesh/mesh.h"
#include "voxblox/mesh/mesh_integrator.h"
#include "voxblox/mesh/mesh_layer.h"
#include <aru/core/mesh/mesh.h>
#include <aru/core/utilities/laser/laser.h>
#include <aru/core/utilities/transforms/transform_map.h>
#include <aru/core/utilities/viewer/viewer.h>
#include <aru/core/vo/vo.h>
#include <opencv2/opencv.hpp>
#include <voxblox/alignment/icp.h>

namespace aru {
namespace core {
namespace mapping {
namespace mesh_map {

class MeshMap {

public:
  explicit MeshMap(std::string mesh_map_settings_file, std::string vocab_file);

  ~MeshMap() = default;

  void InitialiseMap(utilities::image::StereoImage image_init);

  void UpdateMap(utilities::image::StereoImage image_new);

  void ReadTransform(utilities::transform::TransformSPtr transform);

  void UpdateFrameMesh();

  void InsertDepthImage(utilities::image::Image depth_image,
                        utilities::image::Image rgb_image);

  void InsertDepthImage(cv::Mat depth_image, cv::Mat rgb_image,
                        Eigen::Affine3f position);

  void UpdateFramePose();

  void DrawCurrentTsdf();

  void SaveCurrentTsdf(std::string output_ply);

  utilities::transform::Transform GetCurrentPose() { return curr_pose_; }

  std::vector<Eigen::Vector3d> GetCurrentPointCloud() {
    return curr_pointcloud_;
  }

  std::vector<Eigen::Vector3d> GetColorPointCloud() {
    return color_pointcloud_;
  }

private:
  std::string mesh_map_settings_file_;
  std::string vocab_file_;
  utilities::image::ExtractorParams extractor_params_;
  utilities::image::MatcherParams matcher_params_;
  vo::SolverParams solver_params_;

  mesh::RegularisationParams reg_params_;
  mesh::ViewerParams viewer_params_;
  utilities::camera::CameraParams camera_params_;

  utilities::image::FeatureSPtrVectorSptr map_features_;
  utilities::image::FeatureSPtrVectorSptr frame_features_;

  boost::shared_ptr<vo::VO> vo_;
  boost::shared_ptr<mesh::Mesh> mesh_;

  boost::shared_ptr<voxblox::TsdfMap> tsdf_map_;
  boost::shared_ptr<voxblox::FastTsdfIntegrator> tsdf_integrator_;

  /// ICP matcher
  boost::shared_ptr<voxblox::ICP> icp_;
  boost::shared_ptr<voxblox::MeshLayer> mesh_layer_;
  boost::shared_ptr<voxblox::MeshIntegrator<voxblox::TsdfVoxel>>
      mesh_integrator_;

  boost::shared_ptr<utilities::image::VisoFeatureTracker> viso_extractor_;

  boost::shared_ptr<utilities::image::OrbFeatureMatcher> orb_matcher_;

  boost::shared_ptr<utilities::transform::TransformMap> transform_map_;

  cv::Mat curr_descriptor_;

  std::vector<utilities::transform::Transform> pose_chain_;

  utilities::transform::Transform curr_pose_;

  std::vector<Eigen::Vector3d> curr_pointcloud_;
  std::vector<Eigen::Vector3d> color_pointcloud_;

  Eigen::Affine3f current_position_;

  std::vector<Eigen::Affine3f> position_vector_;

  utilities::image::StereoImage curr_frame_;

  boost::shared_ptr<utilities::viewer::Viewer> viewer_;

  bool use_laser_;

  int frame_no;

};
} // namespace mesh_map
} // namespace mapping
} // namespace core
} // namespace aru

#endif // ARU_CORE_MAPPING_MESH_MAP_H_
