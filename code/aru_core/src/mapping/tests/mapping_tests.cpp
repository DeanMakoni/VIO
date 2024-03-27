#define BOOST_TEST_MODULE My Test

#include "glog/logging.h"
#include <boost/make_shared.hpp>
#include <boost/test/included/unit_test.hpp>

#include "aru/core/mapping/dense_reconstruction/core/tsdf_map.h"
#include "aru/core/mapping/dense_reconstruction/integrator/tsdf_integrator.h"
#include "aru/core/mapping/mesh_mapping/mesh_map.h"

using namespace aru::core::mapping::dense_reconstruction;
using namespace aru::core::mapping::mesh_map;
using namespace aru::core::utilities::image;
struct MappingFixture {

  const double baseline = 0.5372;
  Eigen::MatrixXd K;

  cv::Mat image_1_left_vo;
  cv::Mat image_2_left_vo;

  cv::Mat image_1_right_vo;
  cv::Mat image_2_right_vo;

  MappingFixture() {

    image_1_left_vo =
        cv::imread("/home/paulamayo/data/kitti/dataset/sequences/00/image_0"
                   "/000000.png");
    image_2_left_vo =
        cv::imread("/home/paulamayo/data/kitti/dataset/sequences/00/image_0"
                   "/000002.png");

    image_1_right_vo =
        cv::imread("/home/paulamayo/data/kitti/dataset/sequences/00/image_1"
                   "/000000.png");
    image_2_right_vo =
        cv::imread("/home/paulamayo/data/kitti/dataset/sequences/00/image_1"
                   "/000002.png");
  }
};

BOOST_FIXTURE_TEST_CASE(Init, MappingFixture) {

  aru::core::mapping::mesh_map::MeshMap mesh_map(
      "/home/paulamayo/code/aru-core/src/mapping/config/mesh_mapping.yaml");
}

struct VoxBloxFixture {

  const double baseline = 0.5372;
  Eigen::MatrixXd K;

  cv::Mat image_1_left_vo;
  cv::Mat image_2_left_vo;

  cv::Mat image_1_right_vo;
  cv::Mat image_2_right_vo;
  StereoImage image_init;
  StereoImage image_next;

  boost::shared_ptr<TsdfMap> tsdf_map_;
  boost::shared_ptr<SimpleTsdfIntegrator> tsdf_integrator_;

  VoxBloxFixture() {

    image_1_left_vo =
        cv::imread("/home/paulamayo/data/kitti/training/image_2/000000_10.png");
    image_2_left_vo =
        cv::imread("/home/paulamayo/data/kitti/training/image_2/000000_11.png");

    image_1_right_vo =
        cv::imread("/home/paulamayo/data/kitti/training/image_3/000000_10.png");
    image_2_right_vo =
        cv::imread("/home/paulamayo/data/kitti/training/image_3/000000_11.png");
    image_init.first = Image(0, image_1_left_vo);
    image_init.second = Image(0, image_1_right_vo);

    image_next.first = Image(0, image_2_left_vo);
    image_next.second = Image(0, image_2_right_vo);

    TsdfMap::Config config_;
    config_.tsdf_voxel_size = 0.2;
    config_.tsdf_voxels_per_side = 16;
    tsdf_map_ = boost::make_shared<TsdfMap>(config_);
    tsdf_integrator_ = boost::make_shared<SimpleTsdfIntegrator>(
        TsdfIntegratorBase::Config(), tsdf_map_->getTsdfLayerPtr());
  }
};

BOOST_FIXTURE_TEST_CASE(KittiDense, VoxBloxFixture) {

  aru::core::mapping::mesh_map::MeshMap mesh_map(
      "/home/paulamayo/code/aru-core/src/mapping/config/mesh_mapping.yaml");

  mesh_map.InitialiseMap(image_init);

  Pointcloud pointcloud_float;
  Colors colors;
  Eigen::Matrix3f rotation;
  rotation << -1, 0, 0, 0, -1, 0, 0, 0, 1;

  for (const auto &vertice : mesh_map.GetCurrentPointCloud()) {
    Eigen::Vector3f point_f = vertice.cast<float>();
    pointcloud_float.push_back(point_f);
  }
  for (const auto &point_color : mesh_map.GetColorPointCloud()) {
    Color color(point_color(0), point_color(1), point_color(2));
    colors.push_back(color);
  }
  aru::core::utilities::viewer::Viewer::ViewVoxelPointCloud(
      mesh_map.GetCurrentPointCloud(), mesh_map.GetColorPointCloud());
//  tsdf_integrator_->integratePointCloud(
//      mesh_map.GetCurrentPose().GetTransform(), pointcloud_float, colors);

  mesh_map.UpdateMap(image_next);
  pointcloud_float.clear();
  colors.clear();

  for (const auto &vertice : mesh_map.GetCurrentPointCloud()) {
    Eigen::Vector3f point_f = vertice.cast<float>();
    pointcloud_float.push_back(point_f);
    Color color(128, 128, 128);
    colors.push_back(color);
  }

//  tsdf_integrator_->integratePointCloud(
//      mesh_map.GetCurrentPose().GetTransform().inverse(), pointcloud_float,
//      colors);

  // Draw the output

  const Layer<TsdfVoxel> &layer = tsdf_map_->getTsdfLayer();
  std::vector<Eigen::Vector3d> pointcloud_vis;
  BlockIndexList blocks;
  tsdf_map_->getTsdfLayer().getAllAllocatedBlocks(&blocks);

  // Cache layer settings.
  size_t vps = tsdf_map_->getTsdfLayer().voxels_per_side();
  size_t num_voxels_per_block = vps * vps * vps;

  // Temp variables.
  double intensity = 0.0;
  // Iterate over all blocks.
  std::vector<Eigen::Vector3d> voxel_points_;
  std::vector<Eigen::Vector3d> color_points_;
  for (const BlockIndex &index : blocks) {
    // Iterate over all voxels in said blocks.
    const Block<TsdfVoxel> &block =
        tsdf_map_->getTsdfLayer().getBlockByIndex(index);

    for (size_t linear_index = 0; linear_index < num_voxels_per_block;
         ++linear_index) {
      Point coord = block.computeCoordinatesFromLinearIndex(linear_index);

      if (block.getVoxelByLinearIndex(linear_index).weight > 0 &&
          block.getVoxelByLinearIndex(linear_index).color.r > 0 &&
          block.getVoxelByLinearIndex(linear_index).color.g > 0 &&
          block.getVoxelByLinearIndex(linear_index).color.b > 0) {
        Eigen::Vector3d point(coord.x(), coord.y(), coord.z());
        Color color = block.getVoxelByLinearIndex(linear_index).color;
        Eigen::Vector3d color_point(color.r, color.g, color.b);
        voxel_points_.push_back(point);
        color_points_.push_back(color_point);
      }
    }
  }
  aru::core::utilities::viewer::Viewer::ViewVoxelPointCloud(voxel_points_,
                                                            color_points_);
}