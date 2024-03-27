#include "aru/core/dimr/dimr.h"

#include <Eigen/Dense>
#include <boost/graph/connected_components.hpp>
#include <boost/make_shared.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/persistence.hpp>
#include <utility>

using namespace aru::core::utilities;
using namespace aru::core::utilities::image;
using namespace std;
namespace aru {
namespace core {
namespace dimr {
//------------------------------------------------------------------------------
DiMR::DiMR(std::string dimr_settings_file)
    : dimr_settings_file_(std::move(dimr_settings_file)) {

  Eigen::Affine3f identity_affine;
  identity_affine.matrix() = Eigen::MatrixXf::Identity(4, 4);
  curr_pose_ = transform::Transform(0, 0, identity_affine);
  cv::FileStorage fs;
  fs.open(dimr_settings_file_, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    LOG(ERROR) << "Could not open mesh map settings file: ";
  }
  // Regularisor choice
  fs["Regulariser"]["TV"] >> reg_params_.use_tv;
  fs["Regulariser"]["TGV"] >> reg_params_.use_tgv;
  fs["Regulariser"]["LOG_TV"] >> reg_params_.use_log_tv;
  fs["Regulariser"]["LOG_TGV"] >> reg_params_.use_log_tgv;

  // Regularisation parameters
  reg_params_.sigma = fs["Regulariser"]["sigma"];
  reg_params_.tau = fs["Regulariser"]["tau"];
  reg_params_.lambda = fs["Regulariser"]["lambda"];
  reg_params_.theta = fs["Regulariser"]["theta"];
  reg_params_.alpha_1 = fs["Regulariser"]["alpha_1"];
  reg_params_.alpha_2 = fs["Regulariser"]["alpha_2"];
  reg_params_.beta = fs["Regulariser"]["beta"];
  reg_params_.iterations = fs["Regulariser"]["iterations"];
  reg_params_.outer_iterations = fs["Regulariser"]["outer_iterations"];

  // Viewer params
  viewer_params_.max_depth = fs["Viewer"]["max_depth"];
  viewer_params_.colour_scale = fs["Viewer"]["colour_scale"];

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

  // Mesh parameters
  mesh_ = boost::make_shared<mesh::Mesh>(reg_params_, viewer_params_,
                                         camera_params_, matcher_params_,
                                         extractor_params_);

  viewer_ = boost::make_shared<utilities::viewer::Viewer>(
      camera_params_.image_height, camera_params_.image_width,
      camera_params_.K);

  colour_map_ = viewer_->GetColourMap();

  // TSDF parameters
  voxblox::TsdfMap::Config config_;
  config_.tsdf_voxel_size = 0.02;
  config_.tsdf_voxels_per_side = 32;

  tsdf_map_ = boost::make_shared<voxblox::TsdfMap>(config_);
  tsdf_integrator_ = boost::make_shared<voxblox::FastTsdfIntegrator>(
      voxblox::TsdfIntegratorBase::Config(), tsdf_map_->getTsdfLayerPtr());

  mesh_layer_ = boost::make_shared<voxblox::MeshLayer>(tsdf_map_->block_size());
  LOG(INFO) << "Block size is " << tsdf_map_->block_size();
  mesh_integrator_ =
      boost::make_shared<voxblox::MeshIntegrator<voxblox::TsdfVoxel>>(
          voxblox::MeshIntegratorConfig(), tsdf_map_->getTsdfLayer(),
          mesh_layer_.get());

  icp_ = boost::make_shared<voxblox::ICP>(voxblox::ICP::Config());

  // CORAL parameters
  mi_params_.ransac_max_iterations = 100;
  mi_params_.ransac_prob = 0.95;

  coral_params_.num_neighbours = 4;
  coral_params_.outlier_threshold = 4;
  coral_params_.num_iterations = 1000;
  coral_params_.num_loops = 8;
  coral_params_.lambda = 1.5;
  coral_params_.beta = 100;
  coral_params_.nu = 0.125;
  coral_params_.alpha = 0.125;
  coral_params_.tau = 0.125;
  coral_params_.update_models = true;

  optimiser_ = boost::make_shared<
      coral::optimiser::cuda::CoralCudaWrapper<coral::models::CoralPNPModel>>(
      coral_params_);
  pnp_model_initialiser_ = boost::make_shared<
      coral::models::ModelInitialiser<coral::models::CoralPNPModel>>(
      mi_params_);
  pnp_model_initialiser_->SetCameraMatrix(camera_params_.K);

  window_size_ = 5;
}
//------------------------------------------------------------------------------
void DiMR::InitialFrame(const cv::Mat &image_left, const cv::Mat &image_right) {

  cv::Mat image_left_grey, image_right_grey;
  cv::cvtColor(image_left, image_left_grey, cv::COLOR_BGR2GRAY);
  cv::cvtColor(image_right, image_right_grey, cv::COLOR_BGR2GRAY);

  // Obtain Features
  viso_extractor_->InitialiseFeatures(image_left_grey, image_right_grey,
                                      image_left_grey, image_right_grey);

  // Update the frame no
  curr_frame_num_ = 0;
}

//------------------------------------------------------------------------------
void DiMR::UpdateFrame(const cv::Mat &image_left, const cv::Mat &image_right) {

  cv::Mat image_left_grey, image_right_grey;
  cv::cvtColor(image_left, image_left_grey, cv::COLOR_BGR2GRAY);
  cv::cvtColor(image_right, image_right_grey, cv::COLOR_BGR2GRAY);

  // Obtain Features
  viso_extractor_->UpdateFeatures(image_left_grey, image_right_grey);
  // Update Image Vector
  image_vector_.push_back(image_left);
  // Update the frame no
  curr_frame_num_++;

  // Get Active Features
  utilities::image::FeatureSPtrVectorSptr active_features =
      viso_extractor_->GetActiveFeatures();
  LOG(INFO) << "Number of features is " << active_features->size();

  // Get Active Tracks
  std::vector<utilities::image::FeatureTrack> active_tracks =
      viso_extractor_->GetActiveTracks();

  // Get Neighbours
  if (curr_frame_num_ == 5) {
    int num_neighbours = 4;
    Graph neighbourhood = FindUnsortedFeatureTrackletNeighbours(active_tracks);
    // Get Subgraph Track
    std::vector<std::vector<utilities::image::FeatureTrack>> sub_graph_track =
        SubGraphTracks(active_tracks, neighbourhood);

    // Obtain initial models
    std::vector<coral::models::ModelVectorSPtr> initial_models_vector =
        InitialModelsBatch(sub_graph_track);

    // Obtain Geometric Cost
    Eigen::MatrixXd cost =
        ObtainGeometricCostBatch(initial_models_vector, active_tracks);

    // Optimisation
    coral::optimiser::EnergyMinimisationResult result;
    for (int i = 0; i < coral_params_.num_loops; ++i) {
      result = optimiser_->EnergyMinimisation(cost, neighbour_index_);
      viewer_->VisualiseLabels(image_left, active_features,
                               result.DiscreteLabel);
      cv::waitKey(0);

      // Update the models
      std::vector<coral::models::ModelVectorSPtr> models_vector =
          UpdateModelsBatch(active_tracks, result.DiscreteLabel,
                            optimiser_->NumLabels() - 1);

      // Update the costs
      cost = ObtainGeometricCostBatch(models_vector, active_tracks);
    }
    DrawLabelsBatch(active_tracks, result.DiscreteLabel);
  }
}
//------------------------------------------------------------------------------
Graph DiMR::FindUnsortedFeatureTrackletNeighbours(
    std::vector<utilities::image::FeatureTrack> active_tracks_) {
  LOG(INFO) << "Obtaining neighbours";
  cv::Mat curr_image = image_vector_.back().clone();
  // Slow slow neighbourhood max
  size_t num_tracks = active_tracks_.size();

  neighbour_index_ = Eigen::SparseMatrix<double>(
      coral_params_.num_neighbours * num_tracks, num_tracks);
  neighbour_index_.reserve(coral_params_.num_neighbours * num_tracks);
  Graph g(num_tracks);

  int track_no = 0;
  std::vector<double> query_dist;
  query_dist.push_back(0);
  int dim = 2;
  int neighbour_no = 0;
  for (const auto &track : active_tracks_) {
    int track_size = track.match_track_->size();
    int track_begin = track.frame_track_->front();
    int track_end = track.frame_track_->back();
    if (track_begin == track_end)
      track_end++;
    Eigen::MatrixXd feat_point(track_size, 2);
    for (int i = 0; i < track_size; ++i) {
      feat_point(i, 0) = track.match_track_->at(track_size - 1 - i).u1c;
      feat_point(i, 1) = track.match_track_->at(track_size - 1 - i).v1c;
    }
    Eigen::MatrixXd dist_features(active_tracks_.size(), 1);
    int comp_no = 0;
    for (const auto &comp_track : active_tracks_) {
      size_t comp_size = comp_track.match_track_->size();
      Eigen::MatrixXd comp_point(comp_size, dim);

      int comp_begin = comp_track.frame_track_->front();
      int comp_end = comp_track.frame_track_->back();
      if (comp_begin == comp_end)
        comp_end++;
      for (int i = 0; i < comp_size; ++i) {
        comp_point(i, 0) = comp_track.match_track_->at(comp_size - 1 - i).u1c;
        comp_point(i, 1) = comp_track.match_track_->at(comp_size - 1 - i).v1c;
      }

      float dist_prev = 0;
      float max_distance = 1e6;
      // Obtain the potential overlaps
      // If track begins before comp and ends after comp
      if (comp_begin >= track_begin && comp_end <= track_end) {
        int offset_begin = comp_begin - track_begin;
        int length = comp_end - comp_begin;

        for (int i = 0; i < length; ++i) {
          float dist_i =
              sqrt(pow(comp_point(i, 0) - feat_point(i + offset_begin, 0), 2) +
                   pow(comp_point(i, 1) - feat_point(i + offset_begin, 1), 2));
          max_distance = std::max(dist_i, dist_prev);
          dist_prev = max_distance;
        }
      }
      // If track begins before comp and ends before comp
      else if (comp_begin >= track_begin && comp_end > track_end) {
        int offset_begin = comp_begin - track_begin;
        int length = track_end - comp_begin;
        for (int i = 0; i < length; ++i) {
          float dist_i =
              sqrt(pow(comp_point(i, 0) - feat_point(i + offset_begin, 0), 2) +
                   pow(comp_point(i, 1) - feat_point(i + offset_begin, 1), 2));
          max_distance = std::max(dist_i, dist_prev);
          dist_prev = max_distance;
        }
      }
      // If track begins after comp and ends before comp
      else if (track_begin > comp_begin && comp_end >= track_end) {
        int offset_begin = track_begin - comp_begin;
        int length = track_end - track_begin;
        for (int i = 0; i < length; ++i) {
          float dist_i =
              sqrt(pow(comp_point(i + offset_begin, 0) - feat_point(i, 0), 2) +
                   pow(comp_point(i + offset_begin, 1) - feat_point(i, 1), 2));
          max_distance = std::max(dist_i, dist_prev);
          dist_prev = max_distance;
        }
      }
      // If track begins after comp and ends after comp
      else if (track_begin > comp_begin && comp_end < track_end) {
        int offset_begin = track_begin - comp_begin;
        int length = comp_end - track_begin;
        for (int i = 0; i < length; ++i) {
          float dist_i =
              sqrt(pow(comp_point(i + offset_begin, 0) - feat_point(i, 0), 2) +
                   pow(comp_point(i + offset_begin, 1) - feat_point(i, 1), 2));
          max_distance = std::max(dist_i, dist_prev);
          dist_prev = max_distance;
        }
      }
      dist_features(comp_no) = max_distance;
      comp_no++;
    }

    // Get the neighbours
    // Search for nn
    typedef nanoflann::KDTreeEigenMatrixAdaptor<Eigen::MatrixXd> my_kd_tree_t;

    my_kd_tree_t
        mat_index(1, std::cref(dist_features), 10 /* max
leaf */);
    mat_index.index->buildIndex();

    int nn = coral_params_.num_neighbours + 1;
    std::vector<size_t> ret_indexes(nn);
    std::vector<double> out_dists_sqr(nn);

    nanoflann::KNNResultSet<double> resultSet(nn);
    resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);

    mat_index.index->findNeighbors(resultSet, &query_dist[0],
                                   nanoflann::SearchParams(10));
    cv::circle(curr_image,
               cv::Point2f(active_tracks_[track_no].match_track_->back().u1c,
                           active_tracks_[track_no].match_track_->back().v1c),
               1, cv::Scalar(0, 255, 0), -1, cv::FILLED);

    for (int k = 1; k < nn; ++k) {
      // add the edges to the graph object
      add_edge(track_no, ret_indexes[k], g);

      // add connections to sparse matrix
      neighbour_index_.coeffRef(neighbour_no, track_no) = -1;
      neighbour_index_.coeffRef(neighbour_no, ret_indexes[k]) = 1;
      neighbour_no++;

      auto track_neighbour = active_tracks_[ret_indexes[k]];
      cv::circle(
          curr_image,
          cv::Point2f(active_tracks_[ret_indexes[k]].match_track_->back().u1c,
                      active_tracks_[ret_indexes[k]].match_track_->back().v1c),
          1, cv::Scalar(0, 255, 0), -1, cv::FILLED);
      cv::line(
          curr_image,
          cv::Point2f(active_tracks_[track_no].match_track_->back().u1c,
                      active_tracks_[track_no].match_track_->back().v1c),
          cv::Point2f(active_tracks_[ret_indexes[k]].match_track_->back().u1c,
                      active_tracks_[ret_indexes[k]].match_track_->back().v1c),
          cv::Scalar(0, 255, 0), 1);
    }

    track_no++;
  }
  LOG(INFO) << "Neighbours retrieved";

  return g;
}
//------------------------------------------------------------------------------
Graph DiMR::FindFeatureTrackletNeighbours(
    std::vector<utilities::image::FeatureTrack> active_tracks_) {

  cv::Mat curr_image = image_vector_.back().clone();
  // Slow slow neighbourhood max
  size_t num_tracks = active_tracks_.size();

  neighbour_index_ = Eigen::SparseMatrix<double>(
      coral_params_.num_neighbours * num_tracks, num_tracks);
  neighbour_index_.reserve(coral_params_.num_neighbours * num_tracks);

  Graph g(num_tracks);

  int track_no = 0;
  std::vector<double> query_dist;
  query_dist.push_back(0);
  int dim = 2;

  int neighbour_no = 0;
  for (const auto &track : active_tracks_) {
    size_t track_size = track.match_track_->size();
    Eigen::MatrixXd feat_point(track_size, 2);
    for (int i = 0; i < track_size; ++i) {
      feat_point(i, 0) = track.match_track_->at(track_size - 1 - i).u1c;
      feat_point(i, 1) = track.match_track_->at(track_size - 1 - i).v1c;
    }
    Eigen::MatrixXd dist_features(active_tracks_.size(), 1);
    int comp_no = 0;
    for (const auto &comp_track : active_tracks_) {
      size_t comp_size = comp_track.match_track_->size();
      Eigen::MatrixXd comp_point(comp_size, dim);
      for (int i = 0; i < comp_size; ++i) {
        comp_point(i, 0) = comp_track.match_track_->at(comp_size - 1 - i).u1c;
        comp_point(i, 1) = comp_track.match_track_->at(comp_size - 1 - i).v1c;
      }

      float dist_prev = 0;
      float max_distance = 0;
      if (comp_size > track_size) {
        for (int i = 0; i < track_size; ++i) {
          float dist_i = sqrt(pow(comp_point(i, 0) - feat_point(i, 0), 2) +
                              pow(comp_point(i, 1) - feat_point(i, 1), 2));
          max_distance = max(dist_i, dist_prev);
          dist_prev = max_distance;
        }
      } else {
        for (int i = 0; i < comp_size; ++i) {
          float dist_i = sqrt(pow(comp_point(i, 0) - feat_point(i, 0), 2) +
                              pow(comp_point(i, 1) - feat_point(i, 1), 2));
          max_distance = max(dist_i, dist_prev);
          dist_prev = max_distance;
        }
      }
      dist_features(comp_no) = max_distance;
      comp_no++;
    }
    // Search for nn
    typedef nanoflann::KDTreeEigenMatrixAdaptor<Eigen::MatrixXd> my_kd_tree_t;

    my_kd_tree_t
        mat_index(1, std::cref(dist_features), 10 /* max
leaf */);
    mat_index.index->buildIndex();

    int nn = coral_params_.num_neighbours + 1;
    std::vector<size_t> ret_indexes(nn);
    std::vector<double> out_dists_sqr(nn);

    nanoflann::KNNResultSet<double> resultSet(nn);
    resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);

    mat_index.index->findNeighbors(resultSet, &query_dist[0],
                                   nanoflann::SearchParams(10));
    cv::circle(curr_image,
               cv::Point2f(active_tracks_[track_no].match_track_->back().u1c,
                           active_tracks_[track_no].match_track_->back().v1c),
               1, cv::Scalar(0, 255, 0), -1, cv::FILLED);

    for (int k = 1; k < nn; ++k) {
      // add the edges to the graph object
      add_edge(track_no, ret_indexes[k], g);

      // add connections to sparse matrix
      neighbour_index_.coeffRef(neighbour_no, track_no) = -1;
      neighbour_index_.coeffRef(neighbour_no, ret_indexes[k]) = 1;
      neighbour_no++;

      cv::circle(
          curr_image,
          cv::Point2f(active_tracks_[ret_indexes[k]].match_track_->back().u1c,
                      active_tracks_[ret_indexes[k]].match_track_->back().v1c),
          1, cv::Scalar(0, 255, 0), -1, cv::FILLED);
      cv::line(
          curr_image,
          cv::Point2f(active_tracks_[track_no].match_track_->back().u1c,
                      active_tracks_[track_no].match_track_->back().v1c),
          cv::Point2f(active_tracks_[ret_indexes[k]].match_track_->back().u1c,
                      active_tracks_[ret_indexes[k]].match_track_->back().v1c),
          cv::Scalar(0, 255, 0), 1);
    }
    track_no++;
  }
  return g;
}

//------------------------------------------------------------------------------
std::vector<utilities::image::FeatureSPtrVectorSptr> DiMR::SubGraphFeatures(
    const utilities::image::FeatureSPtrVectorSptr &active_features,
    const Graph &neighbourhood) {

  std::vector<utilities::image::FeatureSPtrVectorSptr> sub_graph_features;
  std::vector<int> component(boost::num_vertices(neighbourhood));
  int num = connected_components(neighbourhood, &component[0]);

  int min_components = 10;
  std::vector<int>::size_type ii;
  LOG(INFO) << "Total number of components: " << num;
  std::vector<std::vector<int>> component_index_total;

  for (int jj = 0; jj < num; ++jj) {
    std::vector<int> component_index;
    int num_components = 0;
    for (ii = 0; ii != component.size(); ++ii) {
      if (jj == component[ii]) {
        num_components++;
        component_index.push_back(ii);
      }
    }
    if (num_components > min_components) {
      // Get a feature vector from this
      utilities::image::FeatureSPtrVectorSptr component_features =
          boost::make_shared<utilities::image::FeatureSPtrVector>();
      for (auto comp : component_index) {
        component_features->push_back(active_features->at(comp));
      }
      sub_graph_features.push_back(component_features);
      component_index_total.push_back(component_index);
    }
  }
  LOG(INFO) << "Number of valid components is " << component_index_total.size();
  return sub_graph_features;
}

//------------------------------------------------------------------------------
std::vector<std::vector<utilities::image::FeatureTrack>> DiMR::SubGraphTracks(
    const std::vector<utilities::image::FeatureTrack> &active_tracks_,
    const Graph &neighbourhood) {
  std::vector<std::vector<utilities::image::FeatureTrack>> sub_graph_tracks;
  std::vector<int> component(boost::num_vertices(neighbourhood));
  int num = connected_components(neighbourhood, &component[0]);

  int min_components = 30;
  std::vector<int>::size_type ii;
  LOG(INFO) << "Total number of components: " << num;
  std::vector<std::vector<int>> component_index_total;

  for (int jj = 0; jj < num; ++jj) {
    std::vector<int> component_index;
    int num_components = 0;
    for (ii = 0; ii != component.size(); ++ii) {
      if (jj == component[ii]) {
        num_components++;
        component_index.push_back(ii);
      }
    }
    if (num_components > min_components) {
      // Get a feature vector from this
      std::vector<utilities::image::FeatureTrack> component_track;
      for (auto comp : component_index) {
        component_track.push_back(active_tracks_[comp]);
      }
      sub_graph_tracks.push_back(component_track);
      component_index_total.push_back(component_index);
    }
  }
  LOG(INFO) << "Number of valid components is " << component_index_total.size();
  return sub_graph_tracks;
}
//------------------------------------------------------------------------------
Eigen::MatrixXd DiMR::ObtainGeometricCostBatch(
    std::vector<coral::models::ModelVectorSPtr> model_window,
    const std::vector<utilities::image::FeatureTrack> &track_window) {

  int num_labels = model_window[0]->size() + 1;
  int num_features = track_window.size();

  Eigen::MatrixXd costs = Eigen::MatrixXd::Zero(num_features, num_labels);

  // Scroll through all the features tracks
  int track_no = 0;
  for (const auto &tracks : track_window) {
    int track_begin = tracks.frame_track_->front();
    int track_end = tracks.frame_track_->back();
    int track_size = tracks.frame_track_->size();
    Eigen::MatrixXd curr_feat_cost = costs.row(track_no);

    // Scroll through the features in the track
    for (int feat_no = 0; feat_no < track_size; ++feat_no) {
      FeatureSPtrVectorSptr curr_features =
          boost::make_shared<FeatureSPtrVector>();
      FeatureSPtr curr_feature = boost::make_shared<Feature>(
          Eigen::Vector2d(tracks.match_track_->at(feat_no).u1c,
                          tracks.match_track_->at(feat_no).v1c));
      curr_feature->SetKeyPoint(
          cv::KeyPoint(tracks.match_track_->at(feat_no).u1c,
                       tracks.match_track_->at(feat_no).v1c, 1));
      curr_feature->SetMatchedKeypoint(
          cv::KeyPoint(tracks.match_track_->at(feat_no).u2c,
                       tracks.match_track_->at(feat_no).v2c, 1));
      curr_feature->SetSequentialKeyPoint(
          cv::KeyPoint(tracks.match_track_->at(feat_no).u1p,
                       tracks.match_track_->at(feat_no).v1p, 1));
      curr_feature->TriangulatePoint(camera_params_.K, camera_params_.baseline);
      curr_features->push_back(curr_feature);

      int curr_frame = track_begin + feat_no;

      // Obtain the costs for each model
      int model_no = 0;
      for (const auto &model : *model_window[curr_frame]) {
        Eigen::MatrixXd model_cost = model->EvaluateCost(curr_features);
        // Update the model costs
        curr_feat_cost(model_no) += model_cost(0);
        model_no++;
      }
      curr_feat_cost(model_no) += coral_params_.outlier_threshold;
    }
    curr_feat_cost = curr_feat_cost / track_size;

    costs.row(track_no) = curr_feat_cost;
    track_no++;
  }
  return costs;
}
//------------------------------------------------------------------------------
Eigen::MatrixXd DiMR::ObtainGeometricCost(
    std::vector<coral::models::ModelVectorSPtr> model_window,
    const std::vector<utilities::image::FeatureTrack> &track_window) {

  size_t num_labels = model_window[0]->size() + 1;
  size_t num_features = track_window.size();

  Eigen::MatrixXd costs = Eigen::MatrixXd::Zero(num_features, num_labels);

  // Scroll through all the windows
  for (int i = 0; i < window_size_; ++i) {
    // Obtain the features at each track
    FeatureSPtrVectorSptr curr_features =
        boost::make_shared<FeatureSPtrVector>();
    for (const auto &tracks : track_window) {
      size_t track_size = tracks.match_track_->size() - 1;
      FeatureSPtr curr_feature = boost::make_shared<Feature>(
          Eigen::Vector2d(tracks.match_track_->at(track_size - i).u1c,
                          tracks.match_track_->at(track_size - i).v1c));
      curr_feature->SetKeyPoint(
          cv::KeyPoint(tracks.match_track_->at(track_size - i).u1c,
                       tracks.match_track_->at(track_size - i).v1c, 1));
      curr_feature->SetMatchedKeypoint(
          cv::KeyPoint(tracks.match_track_->at(track_size - i).u2c,
                       tracks.match_track_->at(track_size - i).v2c, 1));
      curr_feature->SetSequentialKeyPoint(
          cv::KeyPoint(tracks.match_track_->at(track_size - i).u1p,
                       tracks.match_track_->at(track_size - i).v1p, 1));
      curr_feature->TriangulatePoint(camera_params_.K, camera_params_.baseline);
      curr_features->push_back(curr_feature);
    }

    // Obtain the costs for each model
    int model_no = 0;
    for (const auto &model : *model_window[i]) {
      Eigen::MatrixXd model_cost = model->EvaluateCost(curr_features);

      // Update the total model costs
      costs.col(model_no) += model_cost;
      model_no++;
    }
    costs.col(model_no) += Eigen::MatrixXd::Constant(
        num_features, 1, coral_params_.outlier_threshold);
  }

  costs = costs / window_size_;
  return costs;
}
//------------------------------------------------------------------------------
std::vector<coral::models::ModelVectorSPtr> DiMR::UpdateModelsBatch(
    const std::vector<utilities::image::FeatureTrack> &active_tracks_,
    Eigen::MatrixXd labels, int num_models) const {

  std::vector<coral::models::ModelVectorSPtr> models_batch_;
  LOG(INFO) << "Updating the Models";

  // Scroll through all the frames
  for (int frame = 0; frame < curr_frame_num_; ++frame) {
    // Set the model vector for this frame
    coral::models::ModelVectorSPtr curr_models(new coral::models::ModelVector);

    // Scroll through the models
    for (int model_no = 0; model_no < num_models; ++model_no) {
      // Create a new model
      boost::shared_ptr<coral::models::CoralPNPModel> new_model_ptr(
          new coral::models::CoralPNPModel(camera_params_.K));

      new_model_ptr->SetRotation(Eigen::MatrixXd::Identity(3, 3));
      new_model_ptr->SetTranslation(Eigen::VectorXd::Zero(3));
      // Scroll through the features
      FeatureSPtrVectorSptr curr_features =
          boost::make_shared<FeatureSPtrVector>();
      int feat_no = 0;
      for (const auto &tracks : active_tracks_) {
        if (labels(feat_no) == model_no) {
          int track_begin = tracks.frame_track_->front();
          int track_end = tracks.frame_track_->back();
          // if frame is within track add feature
          if (frame >= track_begin && frame <= track_end) {
            int frame_offset = frame - track_begin;
            FeatureSPtr curr_feature = boost::make_shared<Feature>(
                Eigen::Vector2d(tracks.match_track_->at(frame_offset).u1c,
                                tracks.match_track_->at(frame_offset).v1c));
            curr_feature->SetKeyPoint(
                cv::KeyPoint(tracks.match_track_->at(frame_offset).u1c,
                             tracks.match_track_->at(frame_offset).v1c, 1));
            curr_feature->SetMatchedKeypoint(
                cv::KeyPoint(tracks.match_track_->at(frame_offset).u2c,
                             tracks.match_track_->at(frame_offset).v2c, 1));
            curr_feature->SetSequentialKeyPoint(
                cv::KeyPoint(tracks.match_track_->at(frame_offset).u1p,
                             tracks.match_track_->at(frame_offset).v1p, 1));
            curr_feature->TriangulatePoint(camera_params_.K,
                                           camera_params_.baseline);
            curr_features->push_back(curr_feature);
          }
        }
        feat_no++;
      }
      //  Update the models
      if (curr_features->size() >= new_model_ptr->ModelDegreesOfFreedom())
        new_model_ptr->UpdateModel(curr_features);
      curr_models->push_back(new_model_ptr);
    }
    LOG(INFO) << "Number of models in frame " << frame << " is "
              << curr_models->size();
    models_batch_.push_back(curr_models);
  }
  return models_batch_;
}
//------------------------------------------------------------------------------
std::vector<coral::models::ModelVectorSPtr> DiMR::UpdateModelsWindow(
    const std::vector<utilities::image::FeatureTrack> &active_tracks_,
    Eigen::MatrixXd labels, int num_models) {
  std::vector<coral::models::ModelVectorSPtr> models_window_;

  LOG(INFO) << "Updating the Models";

  // Scroll through all the windows
  for (int i = 0; i < window_size_; ++i) {
    // Set the model vector for this window
    coral::models::ModelVectorSPtr curr_models(new coral::models::ModelVector);

    // Scroll through the models
    for (int model_no = 0; model_no < num_models; ++model_no) {
      // Create a new model
      boost::shared_ptr<coral::models::CoralPNPModel> new_model_ptr(
          new coral::models::CoralPNPModel(camera_params_.K));

      // Scroll through the features
      FeatureSPtrVectorSptr curr_features =
          boost::make_shared<FeatureSPtrVector>();
      int feat_no = 0;
      for (const auto &tracks : active_tracks_) {
        if (labels(feat_no) == model_no) {
          int track_size = tracks.match_track_->size() - 1;
          FeatureSPtr curr_feature = boost::make_shared<Feature>(
              Eigen::Vector2d(tracks.match_track_->at(track_size - i).u1c,
                              tracks.match_track_->at(track_size - i).v1c));
          curr_feature->SetKeyPoint(
              cv::KeyPoint(tracks.match_track_->at(track_size - i).u1c,
                           tracks.match_track_->at(track_size - i).v1c, 1));
          curr_feature->SetMatchedKeypoint(
              cv::KeyPoint(tracks.match_track_->at(track_size - i).u2c,
                           tracks.match_track_->at(track_size - i).v2c, 1));
          curr_feature->SetSequentialKeyPoint(
              cv::KeyPoint(tracks.match_track_->at(track_size - i).u1p,
                           tracks.match_track_->at(track_size - i).v1p, 1));
          curr_feature->TriangulatePoint(camera_params_.K,
                                         camera_params_.baseline);
          curr_features->push_back(curr_feature);
        }
        feat_no++;
      }

      //  Update the models
      if (curr_features->size() >= new_model_ptr->ModelDegreesOfFreedom()) {
        new_model_ptr->UpdateModel(curr_features);
        curr_models->push_back(new_model_ptr);
      }
    }
    LOG(INFO) << "Number of models in frame " << i << " is "
              << curr_models->size();
    models_window_.push_back(curr_models);
  }
  return models_window_;
}
//------------------------------------------------------------------------------
std::vector<coral::models::ModelVectorSPtr> DiMR::InitialModelsBatch(
    const std::vector<std::vector<utilities::image::FeatureTrack>>
        &active_tracks_) {
  std::vector<coral::models::ModelVectorSPtr> models_batch_;
  int num_models = 1;
  float threshold = 3.0;

  LOG(INFO) << "Obtaining models";
  // Scroll through all the frames
  for (int frame = 0; frame < curr_frame_num_; ++frame) {
    // Set the model vector
    coral::models::ModelVectorSPtr curr_models(new coral::models::ModelVector);

    // Get the tracks for each sub graph
    for (const auto &sub_graph_tracks : active_tracks_) {
      // Obtain the features at each track
      FeatureSPtrVectorSptr curr_features =
          boost::make_shared<FeatureSPtrVector>();

      for (const auto &tracks : sub_graph_tracks) {
        int track_begin = tracks.frame_track_->front();
        int track_end = tracks.frame_track_->back();
        // if frame is within track add feature
        if (frame >= track_begin && frame <= track_end) {
          int frame_offset = frame - track_begin;
          FeatureSPtr curr_feature = boost::make_shared<Feature>(
              Eigen::Vector2d(tracks.match_track_->at(frame_offset).u1c,
                              tracks.match_track_->at(frame_offset).v1c));
          curr_feature->SetKeyPoint(
              cv::KeyPoint(tracks.match_track_->at(frame_offset).u1c,
                           tracks.match_track_->at(frame_offset).v1c, 1));
          curr_feature->SetMatchedKeypoint(
              cv::KeyPoint(tracks.match_track_->at(frame_offset).u2c,
                           tracks.match_track_->at(frame_offset).v2c, 1));
          curr_feature->SetSequentialKeyPoint(
              cv::KeyPoint(tracks.match_track_->at(frame_offset).u1p,
                           tracks.match_track_->at(frame_offset).v1p, 1));
          curr_feature->TriangulatePoint(camera_params_.K,
                                         camera_params_.baseline);
          curr_features->push_back(curr_feature);
        }
      }
      coral::models::ModelVectorSPtr pnp_models(new coral::models::ModelVector);
      pnp_model_initialiser_->Initialise(curr_features, num_models, threshold,
                                         pnp_models);
      if (pnp_models->empty()) {
        // If no model found add an identity model
        boost::shared_ptr<coral::models::CoralPNPModel> static_model(
            new coral::models::CoralPNPModel(camera_params_.K));

        static_model->SetRotation(Eigen::MatrixXd::Identity(3, 3));
        static_model->SetTranslation(Eigen::VectorXd::Zero(3));

        curr_models->push_back(static_model);
      } else {
        for (const auto &model : *pnp_models) {
          curr_models->push_back(model);
        }
      }
    }
    LOG(INFO) << "Number models in frame " << frame << " is "
              << curr_models->size();
    models_batch_.push_back(curr_models);
  }
  return models_batch_;
}

//------------------------------------------------------------------------------
std::vector<coral::models::ModelVectorSPtr> DiMR::InitialModelsWindow(
    const std::vector<std::vector<utilities::image::FeatureTrack>>
        &active_tracks_) {

  std::vector<coral::models::ModelVectorSPtr> models_window_;
  int num_models = 1;
  float threshold = 3.0;

  // Scroll through all the windows
  for (int i = 0; i < window_size_; ++i) {
    // Set the model vector
    coral::models::ModelVectorSPtr curr_models(new coral::models::ModelVector);

    // Set a model for static features
    boost::shared_ptr<coral::models::CoralPNPModel> static_model(
        new coral::models::CoralPNPModel(camera_params_.K));

    static_model->SetRotation(Eigen::MatrixXd::Identity(3, 3));
    static_model->SetTranslation(Eigen::VectorXd::Zero(3));

    curr_models->push_back(static_model);

    // Get the tracks for each sub graph
    for (const auto &sub_graph_tracks : active_tracks_) {

      // Obtain the features at each track
      FeatureSPtrVectorSptr curr_features =
          boost::make_shared<FeatureSPtrVector>();

      for (const auto &tracks : sub_graph_tracks) {
        int track_size = tracks.match_track_->size() - 1;
        FeatureSPtr curr_feature = boost::make_shared<Feature>(
            Eigen::Vector2d(tracks.match_track_->at(track_size - i).u1c,
                            tracks.match_track_->at(track_size - i).v1c));
        curr_feature->SetKeyPoint(
            cv::KeyPoint(tracks.match_track_->at(track_size - i).u1c,
                         tracks.match_track_->at(track_size - i).v1c, 1));
        curr_feature->SetMatchedKeypoint(
            cv::KeyPoint(tracks.match_track_->at(track_size - i).u2c,
                         tracks.match_track_->at(track_size - i).v2c, 1));
        curr_feature->SetSequentialKeyPoint(
            cv::KeyPoint(tracks.match_track_->at(track_size - i).u1p,
                         tracks.match_track_->at(track_size - i).v1p, 1));
        curr_feature->TriangulatePoint(camera_params_.K,
                                       camera_params_.baseline);
        curr_features->push_back(curr_feature);
      }

      coral::models::ModelVectorSPtr pnp_models(new coral::models::ModelVector);
      pnp_model_initialiser_->Initialise(curr_features, num_models, threshold,
                                         pnp_models);
      for (const auto &model : *pnp_models) {
        curr_models->push_back(model);
      }
    }
    LOG(INFO) << "Number of models in frame " << i << " is "
              << curr_models->size();
    models_window_.push_back(curr_models);
  }
  return models_window_;
}
//------------------------------------------------------------------------------
coral::models::ModelVectorSPtr
DiMR::InitialModels(const std::vector<utilities::image::FeatureSPtrVectorSptr>
                        &feature_vector) {

  coral::models::ModelVectorSPtr all_models(new coral::models::ModelVector);

  // Set a model for static features
  boost::shared_ptr<coral::models::CoralPNPModel> static_model(
      new coral::models::CoralPNPModel(camera_params_.K));

  static_model->SetRotation(Eigen::MatrixXd::Identity(3, 3));
  static_model->SetTranslation(Eigen::VectorXd::Zero(3));

  all_models->push_back(static_model);

  for (const auto &features : feature_vector) {

    for (auto feature : *features) {
      feature->TriangulatePoint(camera_params_.K, camera_params_.baseline);
    }
    int num_models = 1;
    float threshold = 3.0;
    coral::models::ModelVectorSPtr pnp_models(new coral::models::ModelVector);
    pnp_model_initialiser_->Initialise(features, num_models, threshold,
                                       pnp_models);
    for (const auto &model : *pnp_models) {
      all_models->push_back(model);
    }
  }
  return all_models;
}
//------------------------------------------------------------------------------
void DiMR::DrawLabelsBatch(
    const std::vector<utilities::image::FeatureTrack> &active_tracks_,
    const Eigen::MatrixXd &labels) {
  // Scroll through all the frames
  for (int frame = 0; frame < curr_frame_num_; ++frame) {
    // Scroll through the features
    FeatureSPtrVectorSptr curr_features =
        boost::make_shared<FeatureSPtrVector>();
    cv::Mat curr_image = image_vector_[frame].clone();
    int feat_no = 0;
    for (const auto &tracks : active_tracks_) {
      int track_begin = tracks.frame_track_->front();
      int track_end = tracks.frame_track_->back();
      // if frame is within track add feature
      if (frame >= track_begin && frame <= track_end) {
        int frame_offset = frame - track_begin;
        FeatureSPtr curr_feature = boost::make_shared<Feature>(
            Eigen::Vector2d(tracks.match_track_->at(frame_offset).u1c,
                            tracks.match_track_->at(frame_offset).v1c));
        curr_feature->SetKeyPoint(
            cv::KeyPoint(tracks.match_track_->at(frame_offset).u1c,
                         tracks.match_track_->at(frame_offset).v1c, 1));
        curr_feature->SetMatchedKeypoint(
            cv::KeyPoint(tracks.match_track_->at(frame_offset).u2c,
                         tracks.match_track_->at(frame_offset).v2c, 1));
        curr_feature->SetSequentialKeyPoint(
            cv::KeyPoint(tracks.match_track_->at(frame_offset).u1p,
                         tracks.match_track_->at(frame_offset).v1p, 1));
        curr_feature->TriangulatePoint(camera_params_.K,
                                       camera_params_.baseline);

        cv::KeyPoint keypoint = curr_feature->GetKeyPoint();
        cv::circle(curr_image, keypoint.pt, 2, colour_map_[labels(feat_no)], -1,
                   cv::FILLED);
      }
      feat_no++;
    }
    cv::imshow("Feature Label", curr_image);
    cv::waitKey(0);
  }
}
//------------------------------------------------------------------------------
void DiMR::DrawLabelsWindow(
    const std::vector<utilities::image::FeatureTrack> &active_tracks_,
    Eigen::MatrixXd labels) {
  // Scroll through all the windows
  size_t num_frames = image_vector_.size() - 1;
  for (int i = 0; i < window_size_; ++i) {
    // Scroll through the features
    FeatureSPtrVectorSptr curr_features =
        boost::make_shared<FeatureSPtrVector>();
    for (const auto &tracks : active_tracks_) {
      int track_size = tracks.match_track_->size() - 1;
      FeatureSPtr curr_feature = boost::make_shared<Feature>(
          Eigen::Vector2d(tracks.match_track_->at(track_size - i).u1c,
                          tracks.match_track_->at(track_size - i).v1c));
      curr_feature->SetKeyPoint(
          cv::KeyPoint(tracks.match_track_->at(track_size - i).u1c,
                       tracks.match_track_->at(track_size - i).v1c, 1));
      curr_feature->SetMatchedKeypoint(
          cv::KeyPoint(tracks.match_track_->at(track_size - i).u2c,
                       tracks.match_track_->at(track_size - i).v2c, 1));
      curr_feature->SetSequentialKeyPoint(
          cv::KeyPoint(tracks.match_track_->at(track_size - i).u1p,
                       tracks.match_track_->at(track_size - i).v1p, 1));
      curr_feature->TriangulatePoint(camera_params_.K, camera_params_.baseline);
      curr_features->push_back(curr_feature);
    }

    cv::Mat curr_image = image_vector_[num_frames - i].clone();

    int feature_no = 0;
    for (const auto &feat : *curr_features) {
      cv::KeyPoint keypoint = feat->GetKeyPoint();
      cv::circle(curr_image, keypoint.pt, 2, colour_map_[labels(feature_no)],
                 -1, cv::FILLED);
      feature_no++;
    }
    cv::imwrite("/home/paulamayo/data/multi_vo/mesh/feature_" +
                    std::to_string(i) + ".png",
                curr_image);

    viewer_->VisualiseLabels(curr_image, curr_features, labels);
    cv::waitKey(0);
  }
}
//------------------------------------------------------------------------------
void DiMR::CreateDistractionFreeMesh(
    std::vector<utilities::image::FeatureTrack> active_tracks_,
    Eigen::MatrixXd labels, int num_models) {
  // Scroll through all the windows
  int num_frames = image_vector_.size() - 1;
  for (int i = 0; i < window_size_; ++i) {
    // Scroll through the features
    FeatureSPtrVectorSptr curr_features =
        boost::make_shared<FeatureSPtrVector>();
    for (const auto &tracks : active_tracks_) {
      int track_size = tracks.match_track_->size() - 1;
      FeatureSPtr curr_feature = boost::make_shared<Feature>(
          Eigen::Vector2d(tracks.match_track_->at(track_size - i).u1c,
                          tracks.match_track_->at(track_size - i).v1c));
      curr_feature->SetKeyPoint(
          cv::KeyPoint(tracks.match_track_->at(track_size - i).u1c,
                       tracks.match_track_->at(track_size - i).v1c, 1));
      curr_feature->SetMatchedKeypoint(
          cv::KeyPoint(tracks.match_track_->at(track_size - i).u2c,
                       tracks.match_track_->at(track_size - i).v2c, 1));
      curr_feature->SetSequentialKeyPoint(
          cv::KeyPoint(tracks.match_track_->at(track_size - i).u1p,
                       tracks.match_track_->at(track_size - i).v1p, 1));
      curr_feature->TriangulatePoint(camera_params_.K, camera_params_.baseline);
      curr_features->push_back(curr_feature);
    }

    cv::Mat curr_image = image_vector_[num_frames - i];
    cv::Mat image_mesh = curr_image.clone();

    // Full frame mesh
    mesh_->EstimateMesh(curr_features, true);
    cv::Mat original_depth_map = mesh_->GetInterpolatedDepth();

    // Label meshes
    std::pair<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3d>>
        label_pt_cloud;
    for (int label_no = 1; label_no < num_models; label_no++) {
      utilities::image::FeatureSPtrVectorSptr label_features =
          boost::make_shared<utilities::image::FeatureSPtrVector>();
      int feat_no = 0;
      for (auto feat : *curr_features) {
        if (labels(feat_no) == label_no)
          label_features->push_back(feat);
        feat_no++;
      }
      mesh_->EstimateMesh(label_features, true);
      // Draw the mesh
      cv::Mat disparity = mesh_->GetInterpolatedDepth();
      std::vector<cv::KeyPoint> keypoints = mesh_->GetVerticeKeypoints();
      std::vector<double> depths = mesh_->GetVerticeDepths();

      std::vector<Eigen::Vector3i> triangles = mesh_->GetMeshTriangles();

      viewer_->ViewMeshWireFrame(image_mesh, keypoints, depths, triangles, 10,
                                 colour_map_[label_no]);

      cv::Mat label_depth = mesh_->GetInterpolatedDepth();
      // viewer_->ViewDepthPointCloud(label_depth, 10);

      int32_t width = curr_image.cols;
      int32_t height = curr_image.rows;

      for (int u = 0; u < width; ++u) {
        for (int v = 0; v < height; ++v) {
          float depth = label_depth.at<float>(v, u);
          if (depth > 0) {
            original_depth_map.at<float>(v, u) = 0;
          }
        }
      }
    }
    cv::imwrite("/home/paulamayo/data/multi_vo/mesh/mesh_" + std::to_string(i) +
                    ".png",
                image_mesh);
    auto pt_cloud_colour =
        mesh_->GetInterpolatedColorPointCloud(curr_image, original_depth_map);
    // UpdateTSDF(pt_cloud_colour);
  }
  // ViewTSDF();
}
//------------------------------------------------------------------------------
void DiMR::CreateMesh(
    std::vector<utilities::image::FeatureTrack> active_tracks_,
    Eigen::MatrixXd labels, int num_models) {

  // Scroll through all the windows
  int num_frames = image_vector_.size() - 1;
  for (int i = 0; i < window_size_; ++i) {
    // Scroll through the features
    FeatureSPtrVectorSptr curr_features =
        boost::make_shared<FeatureSPtrVector>();
    for (const auto &tracks : active_tracks_) {
      int track_size = tracks.match_track_->size() - 1;
      FeatureSPtr curr_feature = boost::make_shared<Feature>(
          Eigen::Vector2d(tracks.match_track_->at(track_size - i).u1c,
                          tracks.match_track_->at(track_size - i).v1c));
      curr_feature->SetKeyPoint(
          cv::KeyPoint(tracks.match_track_->at(track_size - i).u1c,
                       tracks.match_track_->at(track_size - i).v1c, 1));
      curr_feature->SetMatchedKeypoint(
          cv::KeyPoint(tracks.match_track_->at(track_size - i).u2c,
                       tracks.match_track_->at(track_size - i).v2c, 1));
      curr_feature->SetSequentialKeyPoint(
          cv::KeyPoint(tracks.match_track_->at(track_size - i).u1p,
                       tracks.match_track_->at(track_size - i).v1p, 1));
      curr_feature->TriangulatePoint(camera_params_.K, camera_params_.baseline);
      curr_features->push_back(curr_feature);
    }

    cv::Mat curr_image = image_vector_[num_frames - i];
    cv::Mat image_mesh = curr_image.clone();

    // Full frame mesh
    mesh_->EstimateMesh(curr_features, true);
    auto pt_cloud_colour = mesh_->GetInterpolatedColorPointCloud(curr_image);
    // UpdateTSDF(pt_cloud_colour);
  }
  // ViewTSDF();
}
//------------------------------------------------------------------------------
void DiMR::UpdateTSDF(
    std::pair<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3d>>
        color_pointcloud) {
  voxblox::Pointcloud pointcloud_float;
  voxblox::Colors colors;
  Eigen::Matrix3f rotation;
  rotation << -1, 0, 0, 0, -1, 0, 0, 0, 1;

  for (const auto &vertice : color_pointcloud.first) {
    Eigen::Vector3f point_f = vertice.cast<float>();
    pointcloud_float.push_back(point_f);
  }
  for (const auto &point_color : color_pointcloud.second) {
    voxblox::Color color(point_color(0), point_color(1), point_color(2));
    colors.push_back(color);
  }
  Eigen::Matrix3f rot = Eigen::MatrixXf::Identity(3, 3);
  Eigen::Vector3f pos = Eigen::VectorXf::Zero(3);
  voxblox::Transformation icp_initial(voxblox::Rotation(rot), pos);

  tsdf_integrator_->integratePointCloud(icp_initial, pointcloud_float, colors);
}
//------------------------------------------------------------------------------
void DiMR::ViewTSDF() {
  constexpr bool kOnlyMeshUpdatedBlocks = false;
  constexpr bool kClearUpdatedFlag = false;
  mesh_integrator_->generateMesh(kOnlyMeshUpdatedBlocks, kClearUpdatedFlag);

  voxblox::BlockIndexList mesh_indices;
  mesh_layer_->getAllAllocatedMeshes(&mesh_indices);
  std::vector<Eigen::Vector3d> voxel_points_;
  std::vector<Eigen::Vector3d> color_points_;
  // Write to ply
  std::ofstream stream("/home/paulamayo/data/multi_vo/mesh"
                       "/mesh_dummy.ply");

  voxblox::Mesh combined_mesh(mesh_layer_->block_size(),
                              voxblox::Point::Zero());

  mesh_layer_->getConnectedMesh(&combined_mesh);

  size_t num_points = combined_mesh.vertices.size();
  stream << "ply" << std::endl;
  stream << "format ascii 1.0" << std::endl;
  stream << "element vertex " << num_points << std::endl;
  stream << "property float x" << std::endl;
  stream << "property float y" << std::endl;
  stream << "property float z" << std::endl;
  if (combined_mesh.hasNormals()) {
    stream << "property float normal_x" << std::endl;
    stream << "property float normal_y" << std::endl;
    stream << "property float normal_z" << std::endl;
  }
  if (combined_mesh.hasColors()) {
    stream << "property uchar red" << std::endl;
    stream << "property uchar green" << std::endl;
    stream << "property uchar blue" << std::endl;
    stream << "property uchar alpha" << std::endl;
  }
  if (combined_mesh.hasTriangles()) {
    stream << "element face " << combined_mesh.indices.size() / 3 << std::endl;
    stream << "property list uchar int vertex_indices"
           << std::endl; // pcl-1.7(ros::kinetic) breaks ply
    // convention by not
    // using "vertex_index"
  }
  stream << "end_header" << std::endl;
  size_t vert_idx = 0;
  for (const voxblox::Point &vert : combined_mesh.vertices) {
    stream << vert(0) << " " << vert(1) << " " << vert(2);
    Eigen::Vector3d point;
    point.x() = vert(0);
    point.y() = vert(1);
    point.z() = vert(2);
    voxel_points_.push_back(point);

    if (combined_mesh.hasNormals()) {
      const voxblox::Point &normal = combined_mesh.normals[vert_idx];
      stream << " " << normal.x() << " " << normal.y() << " " << normal.z();
    }
    if (combined_mesh.hasColors()) {
      const voxblox::Color &color = combined_mesh.colors[vert_idx];
      Eigen::Vector3d color_point(color.r, color.g, color.b);
      color_points_.push_back(color_point);
      int r = static_cast<int>(color.r);
      int g = static_cast<int>(color.g);
      int b = static_cast<int>(color.b);
      int a = static_cast<int>(color.a);
      // Uint8 prints as character otherwise. :(
      stream << " " << r << " " << g << " " << b << " " << a;
    }

    stream << std::endl;
    vert_idx++;
  }
  if (combined_mesh.hasTriangles()) {
    for (size_t i = 0; i < combined_mesh.indices.size(); i += 3) {
      stream << "3 ";

      for (int j = 0; j < 3; j++) {
        stream << combined_mesh.indices.at(i + j) << " ";
      }

      stream << std::endl;
    }
  }
  aru::core::utilities::viewer::Viewer::ViewVoxelPointCloud(voxel_points_,
                                                            color_points_);
}

} // namespace dimr
} // namespace core
} // namespace aru
