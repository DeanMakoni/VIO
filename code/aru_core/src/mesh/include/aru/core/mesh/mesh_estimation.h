#ifndef ARU_CORE_DEPTH_PIPELINE_H_
#define ARU_CORE_DEPTH_PIPELINE_H_

#include <glog/logging.h>
#include <iostream>

#include "aru/core/utilities/camera/camera.h"
#include "delaunay.h"
#include <aru/core/utilities/image/feature_matcher.h>

namespace aru {
namespace core {
namespace mesh {

struct RegularisationParams {
  // Regulariser choice
  bool use_tv = false;
  bool use_tgv = false;
  bool use_log_tv = false;
  bool use_log_tgv = true;

  // REGULARISATION
  float sigma = 0.125f;
  float tau = 0.125f;
  float lambda = 1.0f; // TGV:1.0, TV:0.5
  float theta = 1.0f;
  float alpha_1 = 0.3f;
  float alpha_2 = 0.8f;
  float beta = 1;
  int iterations = 500;
  int outer_iterations = 10;
};

struct ViewerParams {
  float max_depth = 64;
  float colour_scale = 1.0;
};

class MeshEstimation {

public:
  MeshEstimation(
      aru::core::utilities::camera::CameraParams camera_params,
      RegularisationParams reg_params,
      utilities::image::MatcherParams matcher_params,
      utilities::image::ExtractorParams extractor_params);

  ~MeshEstimation() = default;

  void EstimateMesh(const cv::Mat &image_left, const cv::Mat &image_right);

  void EstimateMesh(utilities::image::FeatureSPtrVectorSptr features,
                    bool triangulate_features);

  utilities::image::FeatureSPtrVectorSptr GetMeshFeatures() {
    return point_features_;
  }

  std::vector<Triangle> GetMeshTriangles() { return triangles_; }

  std::vector<Triangle> RemoveLongTriangles();
  std::vector<Triangle> RemoveObtuseTriangles();

private:
  cv::Mat image_gnd_;
  cv::Mat mesh_;

  float fx_;
  float stereo_baseline_fx_;

  RegularisationParams reg_params_;
  aru::core::utilities::camera::CameraParams cam_params_;
  utilities::image::MatcherParams matcher_params_;
  utilities::image::ExtractorParams extractor_params_;

  utilities::image::FeatureSPtrVectorSptr point_features_;
  std::vector<Triangle> triangles_;
  boost::shared_ptr<Matcher> feature_matcher_;

  cv::Mat T1_;
  cv::Mat T2_;
};
} // namespace mesh
} // namespace core
} // namespace aru

#endif // ARU_CORE_DEPTH_PIPELINE_H_
