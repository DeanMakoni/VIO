#ifndef ARU_CORE_DIMR_H_
#define ARU_CORE_DIMR_H_

#include <Eigen/Dense>
#include <glog/logging.h>
#include <iostream>

#include "aru/core/coral_cuda/coral_cuda_wrapper.h"
#include "voxblox/core/tsdf_map.h"
#include "voxblox/integrator/tsdf_integrator.h"
#include "voxblox/mesh/mesh.h"
#include "voxblox/mesh/mesh_integrator.h"
#include "voxblox/mesh/mesh_layer.h"
#include "aru/core/utilities/image/feature_tracker.h"
#include "aru/core/utilities/image/image.h"

#include <aru/core/coral/model_initialiser.h>
#include <aru/core/coral/models/coral_pnp_model.h>
#include <voxblox/alignment/icp.h>
#include <aru/core/mesh/mesh.h>
#include <aru/core/utilities/laser/laser.h>
#include <aru/core/utilities/viewer/viewer.h>
#include <aru/core/vo/vo.h>
#include <boost/graph/adjacency_list.hpp>
#include <opencv2/opencv.hpp>

namespace aru {
namespace core {
namespace dimr {

typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS>
    Graph;

class DiMR {

public:
  explicit DiMR(std::string dimr_settings_file);

  ~DiMR() = default;

  void InitialFrame(const cv::Mat& image_left, const cv::Mat& image_right);

  void UpdateFrame(const cv::Mat &image_left, const cv::Mat& image_right);

private:
  Graph FindFeatureTrackletNeighbours(
      std::vector<utilities::image::FeatureTrack> active_tracks_);

  Graph FindUnsortedFeatureTrackletNeighbours(
      std::vector<utilities::image::FeatureTrack> active_tracks_);

  static std::vector<utilities::image::FeatureSPtrVectorSptr> SubGraphFeatures(
      const utilities::image::FeatureSPtrVectorSptr &active_features,
      const Graph &neighbourhood);

  static std::vector<std::vector<utilities::image::FeatureTrack>>
  SubGraphTracks(const std::vector<utilities::image::FeatureTrack>& active_tracks_,
                 const Graph &neighbourhood);

  coral::models::ModelVectorSPtr
  InitialModels(const std::vector<utilities::image::FeatureSPtrVectorSptr>& features);

  std::vector<coral::models::ModelVectorSPtr> InitialModelsWindow(
      const std::vector<std::vector<utilities::image::FeatureTrack>>& active_tracks_);

  std::vector<coral::models::ModelVectorSPtr> InitialModelsBatch(
      const std::vector<std::vector<utilities::image::FeatureTrack>>& active_tracks_);

  std::vector<coral::models::ModelVectorSPtr>
  UpdateModelsWindow(const std::vector<utilities::image::FeatureTrack>& active_tracks_,
                     Eigen::MatrixXd labels, int num_models);

  std::vector<coral::models::ModelVectorSPtr>
  UpdateModelsBatch(const std::vector<utilities::image::FeatureTrack>& active_tracks_,
                     Eigen::MatrixXd labels, int num_models) const;

  void
  DrawLabelsWindow(const std::vector<utilities::image::FeatureTrack>& active_tracks_,
                   Eigen::MatrixXd labels);

  void
  DrawLabelsBatch(const std::vector<utilities::image::FeatureTrack>& active_tracks_,
                   const Eigen::MatrixXd& labels);

  void CreateMesh(std::vector<utilities::image::FeatureTrack> active_tracks_,
                  Eigen::MatrixXd labels, int num_models);

  void CreateDistractionFreeMesh(
      std::vector<utilities::image::FeatureTrack> active_tracks_,
      Eigen::MatrixXd labels, int num_models);

  void UpdateTSDF(
      std::pair<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3d>>
          color_pointcloud);

  void ViewTSDF();

  Eigen::MatrixXd ObtainGeometricCostBatch(
      std::vector<coral::models::ModelVectorSPtr> model_window,
      const std::vector<utilities::image::FeatureTrack>& track_window);

  Eigen::MatrixXd
  ObtainGeometricCost(std::vector<coral::models::ModelVectorSPtr> model_window,
                      const std::vector<utilities::image::FeatureTrack>& track_window);

  std::string dimr_settings_file_;
  utilities::image::ExtractorParams extractor_params_;
  utilities::image::MatcherParams matcher_params_;

  mesh::RegularisationParams reg_params_;
  mesh::ViewerParams viewer_params_;
  utilities::camera::CameraParams camera_params_;

  boost::shared_ptr<utilities::image::VisoFeatureTracker> viso_extractor_;

  boost::shared_ptr<mesh::Mesh> mesh_;

  boost::shared_ptr<
      coral::models::ModelInitialiser<coral::models::CoralPNPModel>>
      pnp_model_initialiser_;

  boost::shared_ptr<
      coral::optimiser::cuda::CoralCudaWrapper<coral::models ::CoralPNPModel>>
      optimiser_;

  Eigen::SparseMatrix<double> neighbour_index_;

  coral::models::ModelInitialiserParams mi_params_;

  coral::optimiser::CoralOptimiserParams coral_params_;

  boost::shared_ptr<voxblox::TsdfMap> tsdf_map_;
  boost::shared_ptr<voxblox::FastTsdfIntegrator>
      tsdf_integrator_;

  /// ICP matcher
  boost::shared_ptr<voxblox::ICP> icp_;

  boost::shared_ptr<voxblox::MeshLayer> mesh_layer_;
  boost::shared_ptr<voxblox::MeshIntegrator<
      voxblox::TsdfVoxel>>
      mesh_integrator_;

  std::vector<utilities::transform::Transform> pose_chain_;

  utilities::transform::Transform curr_pose_;

  std::vector<Eigen::Vector3d> curr_pointcloud_;
  std::vector<Eigen::Vector3d> color_pointcloud_;

  Eigen::Affine3f current_position_;

  std::vector<Eigen::Affine3f> position_vector_;

  utilities::image::StereoImage curr_frame_;

  boost::shared_ptr<utilities::viewer::Viewer> viewer_;

  // Vector of motion estimation and labels

  std::vector<coral::models::ModelVectorSPtr> model_vector_frames;

  std::vector<Eigen::MatrixXf> segmentation_frames_;

  std::vector<cv::Mat> image_vector_;

  int curr_frame_num_;

  int window_size_;

  std::vector<cv::Scalar> colour_map_;
};

} // namespace dimr
} // namespace core
} // namespace aru

#endif // ARU_CORE_DIMR_H_
