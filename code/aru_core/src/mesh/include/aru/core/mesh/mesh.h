#ifndef ARU_CORE_MESH_H_
#define ARU_CORE_MESH_H_

#include <Eigen/Dense>
#include <glog/logging.h>
#include <iostream>

#include <opencv2/opencv.hpp>

#include "aru/core/mesh/delaunay.h"
#include "aru/core/mesh/depth_colour_map.h"
#include "aru/core/mesh/mesh_estimation.h"
#include "aru/core/utilities/image/feature_matcher.h"
#include "voxblox/core/common.h"

namespace aru {
namespace core {
namespace mesh {

class Mesh {

public:
  ///
  /// \param mesh_estimation_settings_file
  Mesh(std::string mesh_estimation_settings_file);

  ///
  /// \param reg_params
  /// \param viewer_params
  /// \param camera_params
  /// \param matcher_params
  /// \param extractor_params
  Mesh(RegularisationParams reg_params, ViewerParams viewer_params,
       const aru::core::utilities::camera::CameraParams &camera_params,
       utilities::image::MatcherParams matcher_params,
       utilities::image::ExtractorParams extractor_params);

  ~Mesh() = default;

  ///
  /// \param image_left
  /// \param image_right
  void EstimateMesh(const cv::Mat &image_left, const cv::Mat &image_right);

  void EstimateMesh(utilities::image::FeatureSPtrVectorSptr features,
                    bool triangulate);

  cv::Mat DepthToDisparity(const cv::Mat &depth);

  cv::Mat DisparityToDepth(cv::Mat disparity);

  ///
  /// \param image_left
  /// \param image_right
  /// \param image_gnd
  void EstimateDepthGnd(cv::Mat image_left, const cv::Mat &image_right,
                        cv::Mat image_gnd);

  static std::pair<cv::Mat, cv::Mat>
  CreateDenseDepthMap(cv::Mat sparse_depth_map);

  std::pair<voxblox::Pointcloud, voxblox::Colors>
  GetInterpolatedColorPointCloud(cv::Mat image_color);

  std::pair<voxblox::Pointcloud, voxblox::Colors>
  GetInterpolatedColorPointCloud(cv::Mat image_color, float max_depth);

  std::pair<voxblox::Pointcloud, voxblox::Colors>
  GetInterpolatedColorPointCloud(cv::Mat image_color, cv::Mat depth_image);

  std::pair<voxblox::Pointcloud, voxblox::Colors>
  GetInterpolatedColorPointCloud(cv::Mat image_color, cv::Mat depth_image,
                                 float max_depth);

  std::vector<Eigen::Vector3d> GetInterpolatedPointCloud();

  cv::Mat GetInterpolatedDepth() { return interpolated_depth_; }

  std::vector<Eigen::Vector3d> GetMeshFeatures();

  std::vector<Eigen::Vector3i> GetMeshTriangles();

  void InitXYZ();

  std::vector<double> GetVerticeDepths();

  std::vector<double> GetVerticeDisparity();

  std::vector<cv::KeyPoint> GetVerticeKeypoints();

  static cv::Mat DrawWireframe(const cv::Mat &img_color, cv::Mat image_depth);

  Eigen::Matrix3d GetCameraIntrinsic() { return camera_params_.K; }

  static void DrawShadedTriangleBarycentric(const cv::Point &p1,
                                            const cv::Point &p2,
                                            const cv::Point &p3, float v1,
                                            float v2, float v3, cv::Mat *img);

  static void InterpolateMesh(
      const std::vector<Triangle> &triangles,
      const utilities::image::FeatureSPtrVectorSptr &sparse_supports,
      cv::Mat &dense_map);

  static inline int min3(int x, int y, int z) {
    return x < y ? (x < z ? x : z) : (y < z ? y : z);
  }

  static inline int max3(int x, int y, int z) {
    return x > y ? (x > z ? x : z) : (y > z ? y : z);
  }

private:
  std::string mesh_estimation_settings_file_;
  RegularisationParams reg_params_;
  ViewerParams viewer_params_;
  aru::core::utilities::camera::CameraParams camera_params_;
  utilities::image::MatcherParams matcher_params_;
  utilities::image::ExtractorParams extractor_params_;

  boost::shared_ptr<MeshEstimation> mesh_estimator_;

  utilities::image::FeatureSPtrVectorSptr pixel_features_;
  std::vector<Triangle> delaunay_triangles_;
  cv::Mat interpolated_depth_;

  cv::Mat xyz_;
};
} // namespace mesh
} // namespace core
} // namespace aru

#endif // ARU_CORE_MESH_H_
