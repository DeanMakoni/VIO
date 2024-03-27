#ifndef ARU_CORE_MAPPING_MESH_LASER_MAP_H_
#define ARU_CORE_MAPPING_MESH_LASER_MAP_H_

#include <Eigen/Dense>
#include <glog/logging.h>
#include <iostream>

#include "voxblox/core/tsdf_map.h"
#include "voxblox/integrator/tsdf_integrator.h"
#include "voxblox/mesh/mesh.h"
#include "voxblox/mesh/mesh_integrator.h"
#include "voxblox/mesh/mesh_layer.h"
#include "aru/core/utilities/image/feature_tracker.h"
#include "aru/core/utilities/image/image.h"
#include <voxblox/alignment/icp.h>
#include <aru/core/mesh/mesh.h>
#include <aru/core/utilities/laser/laser.h>
#include <aru/core/utilities/viewer/viewer.h>
#include <aru/core/utilities/transforms/transform_map.h>
#include <aru/core/vo/vo.h>
#include <opencv2/opencv.hpp>

namespace aru {
namespace core {
namespace mapping {
namespace mesh_map {

class LaserMeshMap {

public:
  explicit LaserMeshMap(std::string mesh_map_settings_file);

  ~LaserMeshMap() = default;

  void ReadTransform(utilities::transform::TransformSPtr transform);

  void UpdateMap(utilities::image::Image image_new,
                 utilities::laser::Laser laser_new);

  void UpdateMap(utilities::laser::Laser laser_new);

  void DrawCurrentTsdf();

  void SaveCurrentTsdf(const std::string& output_ply);


private:
  std::string mesh_map_settings_file_;

  utilities::camera::CameraParams camera_params_;
  utilities::image::ExtractorParams extractor_params_;
  utilities::image::MatcherParams matcher_params_;
  vo::SolverParams solver_params_;

  mesh::RegularisationParams reg_params_;
  mesh::ViewerParams viewer_params_;

  boost::shared_ptr<voxblox::TsdfMap> tsdf_map_;
  boost::shared_ptr<voxblox::FastTsdfIntegrator> tsdf_integrator_;

  /// ICP matcher
  boost::shared_ptr<voxblox::ICP> icp_;
  boost::shared_ptr<voxblox::MeshLayer> mesh_layer_;
  boost::shared_ptr<
      voxblox::MeshIntegrator<voxblox::TsdfVoxel>>
  mesh_integrator_;

  // Mesh
  boost::shared_ptr<mesh::Mesh> mesh_;


  boost::shared_ptr<utilities::viewer::Viewer> viewer_;

  boost::shared_ptr<utilities::transform::TransformMap> transform_map_;

  bool use_laser_;

  int frame_no;
  int64_t initial_timestamp_;
};
} // namespace mesh_map
} // namespace mapping
} // namespace core
} // namespace aru

#endif // ARU_CORE_MAPPING_MESH_LASER_MAP_H_
