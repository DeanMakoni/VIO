

#include "aru/core/coral/model_initialiser.h"
#include "aru/core/coral/models/coral_pnp_model.h"
#include "aru/core/coral_cuda/coral_cuda_wrapper.h"

#include "aru/core/mapping/mesh_mapping/mesh_map.h"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/graph_traits.hpp>

#include "aru/core/utilities/image/feature_matcher.h"
#include "aru/core/utilities/image/feature_tracker.h"
#include "aru/core/utilities/viewer/viewer.h"
#include "opencv2/optflow.hpp"
#include "opencv2/ximgproc.hpp"
#include <Eigen/Dense>
#include <boost/make_shared.hpp>
#include <iostream>

#include <aru/core/mesh/mesh.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

using namespace aru::core;
using namespace cv;
using namespace aru::core::mapping;

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  LOG(INFO) << "This is an info  message";

  utilities::image::extractors::ExtractorParams params_ext{};
  params_ext.num_features = 5000;
  params_ext.num_levels = 8;
  params_ext.scale_factor = 1.2;
  params_ext.initial_fast_threshold = 20;
  params_ext.minimum_fast_threshold = 7;
  params_ext.patch_size = 31;
  params_ext.half_patch_size = 15;
  params_ext.edge_threshold = 19;

  utilities::image::MatcherParams params_match{};
  params_match.match_threshold_low = 50;
  params_match.match_threshold_high = 100;
  params_match.stereo_baseline = 0.24;
  params_match.focal_length = 969;

  LOG(INFO) << "Obtain the keypoints";

  boost::shared_ptr<utilities::image::VisoFeatureTracker> viso_extractor_(
      new utilities::image::VisoFeatureTracker(params_match, params_ext));

  boost::shared_ptr<mesh::Mesh> mesh_ = boost::make_shared<mesh::Mesh>(
      "/home/paulamayo/data/multi_vo/mesh/mvo_mesh_depth.yaml");

  const double baseline = 0.24;
  const double uc = 643;
  const double vc = 482;

  const double fu = 969.4750;
  const double fv = 969.475;
  Eigen::Matrix3d K;
  K << fu, 0, uc, 0, fv, vc, 0, 0, 1;
  boost::shared_ptr<utilities::viewer::Viewer> viewer_ =
      boost::make_shared<utilities::viewer::Viewer>(1280, 720, K);

  // Create the TSDF
  aru::core::mapping::dense_reconstruction::TsdfMap::Config config_;
  config_.tsdf_voxel_size = 0.05;
  config_.tsdf_voxels_per_side = 32;

  boost::shared_ptr<dense_reconstruction::TsdfMap> tsdf_map_ =
      boost::make_shared<aru::core::mapping::dense_reconstruction::TsdfMap>(
          config_);
  boost::shared_ptr<dense_reconstruction::FastTsdfIntegrator> tsdf_integrator_ =
      boost::make_shared<
          aru::core::mapping::dense_reconstruction::FastTsdfIntegrator>(
          aru::core::mapping::dense_reconstruction::TsdfIntegratorBase::
              Config(),
          tsdf_map_->getTsdfLayerPtr());

  boost::shared_ptr<dense_reconstruction::MeshLayer> mesh_layer_ =
      boost::make_shared<dense_reconstruction::MeshLayer>(
          tsdf_map_->block_size());
  LOG(INFO) << "Block size is " << tsdf_map_->block_size();
  boost::shared_ptr<
      dense_reconstruction::MeshIntegrator<dense_reconstruction ::TsdfVoxel>>
      mesh_integrator_ =
          boost::make_shared<dense_reconstruction::MeshIntegrator<
              dense_reconstruction::TsdfVoxel>>(
              dense_reconstruction::MeshIntegratorConfig(),
              tsdf_map_->getTsdfLayer(), mesh_layer_.get());

  boost::shared_ptr<dense_reconstruction::ICP> icp_ =
      boost::make_shared<dense_reconstruction::ICP>(
          dense_reconstruction::ICP::Config());

  int start = 380;
  for (int i = start; i < 700; ++i) {
    std::string directory_left(
        "/home/paulamayo/data/multi_vo/swinging_static/stereo/000");

    std::string directory_right(
        "/home/paulamayo/data/multi_vo/swinging_static/stereo/000");
    std::string zero_pad;
    if (i < 1000) {
      zero_pad = "";
    }
    if (i < 100) {
      zero_pad = "0";
    }

    if (i < 10) {
      zero_pad = "00";
    }

    int delta = 1;
    LOG(INFO) << directory_left + zero_pad + std::to_string(i) + "_left.png";
    cv::Mat image_1_left;
    float resize = 1;
    cv::resize(cv::imread(directory_left + zero_pad + std::to_string(i) +
                          "_left"
                          ".png"),
               image_1_left, cv::Size(), resize, resize);

    cv::Mat image_1_right;
    cv::resize(cv::imread(directory_right + zero_pad + std::to_string(i) +
                          "_right.png"),
               image_1_right, cv::Size(), resize, resize);

    cv::Mat image_1_left_grey, image_1_right_grey;
    cv::cvtColor(image_1_left, image_1_left_grey, cv::COLOR_BGR2GRAY);
    cv::cvtColor(image_1_right, image_1_right_grey, cv::COLOR_BGR2GRAY);

    if (i == start) {
      viso_extractor_->InitialiseFeatures(image_1_left_grey,
                                          image_1_right_grey);
    } else {
      viso_extractor_->UpdateFeatures(image_1_left_grey, image_1_right_grey);

      std::vector<utilities::image::FeatureTrack> feature_track =
          viso_extractor_->GetFeatureTrackTriplet();

      LOG(INFO) << "Number of features tracks is " << feature_track.size();

      // Demo code for neighbourhood function
      int nn = 4 + 1;
      int dim = 6;

      cv::Mat image_clone = image_1_left.clone();
      // Go into every row

      int num_features = feature_track.size();
      Eigen::MatrixXd target_features(feature_track.size(), dim);

      // Create the gradient variable

      Eigen::SparseMatrix<double> neighbour_index_ =
          Eigen::SparseMatrix<double>((nn - 1) * num_features, num_features);
      neighbour_index_.reserve((nn - 1) * num_features);

      // create a typedef for the Graph type
      typedef boost::adjacency_list<boost::vecS, boost::vecS,
                                    boost::bidirectionalS>
          Graph;

      int feat_no = 0;
      std::vector<std::vector<double>> query_points;
      // cv::Mat image_clone = image_1_left.clone();
      for (const auto &track : feature_track) {
        int track_size = track.match_track_->size() - 1;
        Eigen::VectorXd feat_point(dim);
        feat_point(0) = track.match_track_->at(track_size).u1c;
        feat_point(1) = track.match_track_->at(track_size).v1c;
        feat_point(2) = track.match_track_->at(track_size - 1).u1c;
        feat_point(3) = track.match_track_->at(track_size - 1).v1c;
        feat_point(4) = track.match_track_->at(track_size - 2).u1c;
        feat_point(5) = track.match_track_->at(track_size - 2).v1c;
        //        feat_point(6) = track.match_track_->at(track_size - 3).u1c;
        //        feat_point(7) = track.match_track_->at(track_size - 3).v1c;
        std::vector<double> current_point;
        current_point.push_back(feat_point(0));
        current_point.push_back(feat_point(1));
        current_point.push_back(feat_point(2));
        current_point.push_back(feat_point(3));
        current_point.push_back(feat_point(4));
        current_point.push_back(feat_point(5));
        //        current_point.push_back(feat_point(6));
        //        current_point.push_back(feat_point(7));
        query_points.push_back(current_point);
        target_features.row(feat_no) = feat_point;
        feat_no++;
      }

      // ------------------------------------------------------------
      // construct a kd-tree index:
      //    Some of the different possibilities (uncomment just one)
      // ------------------------------------------------------------
      // Dimensionality set at run-time (default: L2)
      typedef nanoflann::KDTreeEigenMatrixAdaptor<Eigen::MatrixXd> my_kd_tree_t;

      my_kd_tree_t mat_index(dim, std::cref(target_features),
                             10 /* max leaf */);
      mat_index.index->buildIndex();

      // Create the gradient variable
      // Make convenient labels for the vertices
      const int num_vertices = feature_track.size();
      // declare a graph object
      Graph g(num_vertices);

      uint cont = 0;
      for (int ii = 0; ii < feature_track.size(); ++ii) {
        std::vector<size_t> ret_indexes(nn);
        std::vector<double> out_dists_sqr(nn);

        nanoflann::KNNResultSet<double> resultSet(nn);
        resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);

        mat_index.index->findNeighbors(resultSet, &query_points[ii][0],
                                       nanoflann::SearchParams(10));
        cv::circle(image_clone,
                   cv::Point2f(feature_track[ii].match_track_->back().u1c,
                               feature_track[ii].match_track_->back().v1c),
                   1, cv::Scalar(0, 255, 0), -1, cv::FILLED);
        for (int k = 1; k < nn; ++k) {

          // add the edges to the graph object
          add_edge(ii, ret_indexes[k], g);

          neighbour_index_.coeffRef(cont, ii) = -1;
          neighbour_index_.coeffRef(cont, ret_indexes[k]) = 1;

          cv::circle(
              image_clone,
              cv::Point2f(
                  feature_track[ret_indexes[k]].match_track_->back().u1c,
                  feature_track[ret_indexes[k]].match_track_->back().v1c),
              1, cv::Scalar(0, 255, 0), -1, cv::FILLED);
          cv::line(image_clone,
                   cv::Point2f(feature_track[ii].match_track_->back().u1c,
                               feature_track[ii].match_track_->back().v1c),
                   cv::Point2f(
                       feature_track[ret_indexes[k]].match_track_->back().u1c,
                       feature_track[ret_indexes[k]].match_track_->back().v1c),
                   cv::Scalar(0, 255, 0), 1);
        }
      }

      utilities::image::FeatureSPtrVectorSptr active_features =
          viso_extractor_->GetActiveFeatures();
      coral::models::ModelVectorSPtr all_models(new coral::models::ModelVector);
      viso_extractor_->CullTracks();

      // Set a model for static features
      for (const auto &feature : *active_features) {
        feature->TriangulatePoint(K, baseline);
      }

      boost::shared_ptr<coral::models::CoralPNPModel> static_model(
          new coral::models::CoralPNPModel(K));

      static_model->SetRotation(Eigen::MatrixXd::Identity(3, 3));
      static_model->SetTranslation(Eigen::VectorXd::Zero(3));

      all_models->push_back(static_model);

      if (i > start + 5) {
        std::vector<int> component(boost::num_vertices(g));
        int num = connected_components(g, &component[0]);

        int min_components = 10;
        std::vector<int>::size_type ii;
        LOG(INFO) << "Total number of components: " << num;
        std::vector<std::vector<int>> component_index_total;

        for (int jj = 0; jj < num; ++jj) {
          cv::Mat component_image = image_1_left.clone();
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

            coral::models::ModelInitialiserParams mi_params(100, 0.95);
            coral::models::ModelInitialiser<coral::models::CoralPNPModel>
                pnp_model_initialiser(mi_params);
            pnp_model_initialiser.SetCameraMatrix(K);

            int num_models = 1;
            float threshold = 3.0;

            coral::models::ModelVectorSPtr pnp_models(
                new coral::models::ModelVector);
            pnp_model_initialiser.Initialise(component_features, num_models,
                                             threshold, pnp_models);

            for (const auto &model : *pnp_models) {
              all_models->push_back(model);
              model->ModelEquation();
            }
            component_index_total.push_back(component_index);
          }
        }
        LOG(INFO) << "Number of valid components is "
                  << component_index_total.size();
        coral::optimiser::CoralOptimiserParams params{};

        params.num_features = active_features->size();
        params.num_neighbours = 2;
        params.outlier_threshold = 4;

        params.num_labels = all_models->size() + 1;
        params.num_iterations = 1000;
        params.num_loops = 3;

        params.lambda = 2.0;
        params.beta = 10;

        params.nu = 0.125;
        params.alpha = 0.125;
        params.tau = 0.125;
        params.update_models = true;

        coral::optimiser::cuda::CoralCudaWrapper<coral::models::CoralPNPModel>
            optimiser(params);
        LOG(INFO) << "Number of  models is " << all_models->size();
        coral::optimiser::EnergyMinimisationResult result =
            optimiser.EnergyMinimisation(active_features, all_models);

        std::vector<cv::Scalar> colour_map;
        colour_map.push_back(cv::Scalar(0, 0, 255));
        colour_map.push_back(cv::Scalar(0, 255, 0));
        colour_map.push_back(cv::Scalar(255, 0, 0));
        colour_map.push_back(cv::Scalar(0, 255, 255));
        colour_map.push_back(cv::Scalar(255, 0, 255));
        colour_map.push_back(cv::Scalar(255, 255, 0));
        colour_map.push_back(cv::Scalar(0, 0, 128));
        colour_map.push_back(cv::Scalar(0, 128, 0));
        colour_map.push_back(cv::Scalar(128, 0, 255));
        colour_map.push_back(cv::Scalar(0, 128, 128));
        colour_map.push_back(cv::Scalar(128, 0, 128));
        colour_map.push_back(cv::Scalar(128, 128, 0));
        colour_map.push_back(cv::Scalar(0, 0, 64));
        colour_map.push_back(cv::Scalar(0, 64, 0));
        colour_map.push_back(cv::Scalar(64, 0, 0));
        colour_map.push_back(cv::Scalar(0, 64, 64));
        colour_map.push_back(cv::Scalar(64, 0, 64));
        colour_map.push_back(cv::Scalar(64, 64, 0));
        colour_map.push_back(cv::Scalar(128, 0, 255));
        colour_map.push_back(cv::Scalar(0, 128, 255));
        colour_map.push_back(cv::Scalar(0, 255, 128));
        colour_map.push_back(cv::Scalar(128, 0, 255));
        colour_map.push_back(cv::Scalar(255, 0, 128));
        colour_map.push_back(cv::Scalar(64, 255, 128));
        colour_map.push_back(cv::Scalar(128, 64, 255));
        colour_map.push_back(cv::Scalar(255, 64, 128));

        int feature_no = 0;
        cv::Mat image_label_feat = image_1_left.clone();
        for (auto &feature : *active_features) {
          cv::Point2f curr_point = feature->GetSequentialKeyPoint().pt;

          cv::circle(image_label_feat, curr_point, 2,
                     colour_map[result.DiscreteLabel(feature_no)], -1,
                     cv::FILLED);
          feature_no++;
        }

        cv::imshow("Label visualiser Feature", image_label_feat);
        // cv::imwrite("/home/paulamayo/Pictures/coral_vo.jpg",image_label_feat);
        // cv::waitKey(0);

        // Create a mesh  and do the TSDF

        cv::Mat image_frame = cv::imread(directory_left + zero_pad +
                                         std::to_string(i  - 1) +
                                         "_left.png");
        cv::Mat image_mesh = image_frame.clone();

        // Full frame mesh
        mesh_->EstimateFeatureMesh(active_features, true);
        cv::Mat original_depth_map = mesh_->GetInterpolatedDepth();

//        auto pt_cloud_colour = mesh_->GetInterpolatedColorPointCloud(
//            image_1_left, original_depth_map);

        // Label meshes
        std::pair<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3d>>
            label_pt_cloud;
        for (int label_no = 1; label_no < optimiser.NumLabels() - 1;
             label_no++) {
          utilities::image::FeatureSPtrVectorSptr label_features =
              boost::make_shared<utilities::image::FeatureSPtrVector>();
          int feat_no = 0;
          for (auto feat : *active_features) {
            if (result.DiscreteLabel(feat_no) == label_no)
              label_features->push_back(feat);
            feat_no++;
          }
          mesh_->EstimateFeatureMesh(label_features, true);
//          if (label_no == 1) {
//            pt_cloud_colour =
//                mesh_->GetInterpolatedColorPointCloud(image_1_left);
//          }
          // Draw the mesh
          cv::Mat disparity = mesh_->GetInterpolatedDepth();
          std::vector<cv::KeyPoint> keypoints = mesh_->GetVerticeKeypoints();
          std::vector<double> depths = mesh_->GetVerticeDepths();

          std::vector<Eigen::Vector3i> triangles = mesh_->GetMeshTriangles();

          viewer_->ViewMeshWireFrame(image_mesh, keypoints, depths, triangles,
                                     10, colour_map[label_no]);

          cv::Mat label_depth = mesh_->GetInterpolatedDepth();
          // viewer_->ViewDepthPointCloud(label_depth, 10);

          int32_t width = image_1_left.cols;
          int32_t height = image_1_left.rows;

          for (int u = 0; u < width; ++u) {
            for (int v = 0; v < height; ++v) {
              float depth = label_depth.at<float>(v, u);
              if (depth > 0) {
                original_depth_map.at<float>(v, u) = 0;
              }
            }
          }
          //          viewer_->ViewInterpolatedMesh(image_mesh, disparity, 10);

          // cv::imshow("Image Mesh", image_mesh);
          // cv::waitKey(15);
        }
        // cv::imwrite("/home/paulamayo/Pictures/coral_mesh.png",image_mesh);
        //cv::waitKey(0);
        mesh_->EstimateFeatureMesh(active_features, true);
        auto pt_cloud_colour = mesh_->GetInterpolatedColorPointCloud(
            image_frame);
        //        if (i == 260) {
        //          mesh_->EstimateFeatureMesh(active_features, true);
        //          pt_cloud_colour =
        //          mesh_->GetInterpolatedColorPointCloud(image_1_left);
        //        }

        dense_reconstruction::Pointcloud pointcloud_float;
        dense_reconstruction::Colors colors;
        Eigen::Matrix3f rotation;
        rotation << -1, 0, 0, 0, -1, 0, 0, 0, 1;

        for (const auto &vertice : pt_cloud_colour.first) {
          Eigen::Vector3f point_f = vertice.cast<float>();
          pointcloud_float.push_back(point_f);
        }
        for (const auto &point_color : pt_cloud_colour.second) {
          dense_reconstruction::Color color(point_color(0), point_color(1),
                                            point_color(2));
          colors.push_back(color);
        }
        Eigen::Matrix3f rot = Eigen::MatrixXf::Identity(3, 3);
        Eigen::Vector3f pos = Eigen::VectorXf::Zero(3);
        dense_reconstruction::Transformation icp_initial(
            dense_reconstruction::Rotation(rot), pos);

        tsdf_integrator_->integratePointCloud(icp_initial, pointcloud_float,
                                              colors);

        if (i % 10 == 0) {

          constexpr bool kOnlyMeshUpdatedBlocks = false;
          constexpr bool kClearUpdatedFlag = false;
          mesh_integrator_->generateMesh(kOnlyMeshUpdatedBlocks,
                                         kClearUpdatedFlag);

          dense_reconstruction::BlockIndexList mesh_indices;
          mesh_layer_->getAllAllocatedMeshes(&mesh_indices);
          std::vector<Eigen::Vector3d> voxel_points_;
          std::vector<Eigen::Vector3d> color_points_;
          // Write to ply
          std::ofstream stream("/home/paulamayo/data/multi_vo/mesh"
                               "/2_orig_mesh.ply");

          dense_reconstruction::Mesh combined_mesh(
              mesh_layer_->block_size(), dense_reconstruction::Point::Zero());

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
            stream << "element face " << combined_mesh.indices.size() / 3
                   << std::endl;
            stream << "property list uchar int vertex_indices"
                   << std::endl; // pcl-1.7(ros::kinetic) breaks ply
            // convention by not
            // using "vertex_index"
          }
          stream << "end_header" << std::endl;
          size_t vert_idx = 0;
          for (const dense_reconstruction::Point &vert :
               combined_mesh.vertices) {
            stream << vert(0) << " " << vert(1) << " " << vert(2);
            Eigen::Vector3d point;
            point.x() = vert(0);
            point.y() = vert(1);
            point.z() = vert(2);
            voxel_points_.push_back(point);

            if (combined_mesh.hasNormals()) {
              const dense_reconstruction::Point &normal =
                  combined_mesh.normals[vert_idx];
              stream << " " << normal.x() << " " << normal.y() << " "
                     << normal.z();
            }
            if (combined_mesh.hasColors()) {
              const dense_reconstruction::Color &color =
                  combined_mesh.colors[vert_idx];
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
          aru::core::utilities::viewer::Viewer::ViewVoxelPointCloud(
              voxel_points_, color_points_);
        }

        //        if (i == 257) {
        //
        //
        //
        //          // Draw from the previous 4 frames
        //          for (int frame = 4; frame > 0; --frame) {
        //            aru::core::utilities::image::FeatureSPtrVectorSptr
        //            frame_features =
        //                boost::make_shared<
        //                    aru::core::utilities::image::FeatureSPtrVector>();
        //            int feat_no = 0;
        //            for (const auto &tracks : feature_track) {
        //              int track_size = tracks.match_track_->size();
        //              aru::core::utilities::image::FeatureSPtr curr_feature =
        //                  boost::make_shared<aru::core::utilities::image::Feature>(
        //                      Eigen::Vector2d(
        //                          tracks.match_track_->at(track_size -
        //                          frame).u1p,
        //                          tracks.match_track_->at(track_size -
        //                          frame).v1p));
        //              curr_feature->SetKeyPoint(
        //                  KeyPoint(tracks.match_track_->at(track_size -
        //                  frame).u1p,
        //                           tracks.match_track_->at(track_size -
        //                           frame).v1p, 1));
        //              curr_feature->SetMatchedKeypoint(
        //                  KeyPoint(tracks.match_track_->at(track_size -
        //                  frame).u2p,
        //                           tracks.match_track_->at(track_size -
        //                           frame).v2p, 1));
        //              curr_feature->SetSequentialKeyPoint(
        //                  KeyPoint(tracks.match_track_->at(track_size -
        //                  frame).u1c,
        //                           tracks.match_track_->at(track_size -
        //                           frame).v1c, 1));
        //
        //              curr_feature->TriangulatePoint(K, baseline);
        //              if (result.DiscreteLabel(feat_no) == 0) {
        //                frame_features->push_back(curr_feature);
        //              }
        //              feat_no++;
        //            }
        //
        //            cv::Mat image_frame = cv::imread(directory_left + zero_pad
        //            +
        //                                             std::to_string(i - (frame
        //                                             - 1)) +
        //                                             "_left"
        //                                             ".png");
        //
        //            viewer_->ViewImageFeatures(image_frame, frame_features);
        //
        //            mesh_->EstimateFeatureMesh(frame_features, true);
        //            // Draw the mesh
        //            cv::Mat disparity = mesh_->GetInterpolatedDepth();
        //            std::vector<cv::KeyPoint> keypoints =
        //            mesh_->GetVerticeKeypoints(); std::vector<double> depths =
        //            mesh_->GetVerticeDepths();
        //
        //            std::vector<Eigen::Vector3i> triangles =
        //            mesh_->GetMeshTriangles();
        //
        //            //            viewer_->ViewMeshWireFrame(image_mesh,
        //            keypoints,
        //            //            depths, triangles,
        //            //                                       10);
        //
        //            cv::Mat original_depth_map =
        //            mesh_->GetInterpolatedDepth();
        //            //viewer_->ViewDepthPointCloud(original_depth_map, 10);
        //
        //            // Remove points that belong to a different label
        //            int num_labels = optimiser.NumLabels();
        //            for (int labels = 1; labels < num_labels-1; ++labels) {
        //              aru::core::utilities::image::FeatureSPtrVectorSptr
        //                  label_features = boost::make_shared<
        //                      aru::core::utilities::image::FeatureSPtrVector>();
        //              feat_no = 0;
        //              for (const auto &tracks : feature_track) {
        //                int track_size = tracks.match_track_->size();
        //                aru::core::utilities::image::FeatureSPtr curr_feature
        //                =
        //                    boost::make_shared<aru::core::utilities::image::Feature>(
        //                        Eigen::Vector2d(
        //                            tracks.match_track_->at(track_size -
        //                            frame).u1p,
        //                            tracks.match_track_->at(track_size -
        //                            frame).v1p));
        //                curr_feature->SetKeyPoint(KeyPoint(
        //                    tracks.match_track_->at(track_size - frame).u1p,
        //                    tracks.match_track_->at(track_size - frame).v1p,
        //                    1));
        //                curr_feature->SetMatchedKeypoint(KeyPoint(
        //                    tracks.match_track_->at(track_size - frame).u2p,
        //                    tracks.match_track_->at(track_size - frame).v2p,
        //                    1));
        //                curr_feature->SetSequentialKeyPoint(KeyPoint(
        //                    tracks.match_track_->at(track_size - frame).u1c,
        //                    tracks.match_track_->at(track_size - frame).v1c,
        //                    1));
        //
        //                curr_feature->TriangulatePoint(K, baseline);
        //                if (result.DiscreteLabel(feat_no) == labels) {
        //                  label_features->push_back(curr_feature);
        //                }
        //                feat_no++;
        //              }
        //              mesh_->EstimateFeatureMesh(label_features, true);
        //              // Draw the mesh
        //              cv::Mat label_depth = mesh_->GetInterpolatedDepth();
        //              //viewer_->ViewDepthPointCloud(label_depth, 10);
        //
        //              int32_t width = image_1_left.cols;
        //              int32_t height = image_1_left.rows;
        //
        //              for (int u = 0; u < width; ++u) {
        //                for (int v = 0; v < height; ++v) {
        //                  float depth = label_depth.at<float>(v, u);
        //                  if (depth > 0) {
        //                    original_depth_map.at<float>(v, u) = 0;
        //                  }
        //                }
        //              }
        //              //viewer_->ViewDepthPointCloud(original_depth_map, 10);
        //            }
        //            auto pt_cloud_colour =
        //            mesh_->GetInterpolatedColorPointCloud(
        //                image_frame, original_depth_map);
        //
        //            dense_reconstruction::Pointcloud pointcloud_float;
        //            dense_reconstruction::Colors colors;
        //            Eigen::Matrix3f rotation;
        //            rotation << -1, 0, 0, 0, -1, 0, 0, 0, 1;
        //
        //            for (const auto &vertice : pt_cloud_colour.first) {
        //              Eigen::Vector3f point_f = vertice.cast<float>();
        //              pointcloud_float.push_back(point_f);
        //            }
        //            for (const auto &point_color : pt_cloud_colour.second) {
        //              dense_reconstruction::Color color(point_color(0),
        //              point_color(1),
        //                                                point_color(2));
        //              colors.push_back(color);
        //            }
        //            Eigen::Matrix3f rot = Eigen::MatrixXf::Identity(3, 3);
        //            Eigen::Vector3f pos = Eigen::VectorXf::Zero(3);
        //            dense_reconstruction::Transformation icp_initial(
        //                dense_reconstruction::Rotation(rot), pos);
        //
        //            tsdf_integrator_->integratePointCloud(icp_initial,
        //            pointcloud_float,
        //                                                  colors);
        //
        //            constexpr bool kOnlyMeshUpdatedBlocks = false;
        //            constexpr bool kClearUpdatedFlag = false;
        //            mesh_integrator_->generateMesh(kOnlyMeshUpdatedBlocks,
        //                                           kClearUpdatedFlag);
        //
        //            dense_reconstruction::BlockIndexList mesh_indices;
        //            mesh_layer_->getAllAllocatedMeshes(&mesh_indices);
        //            std::vector<Eigen::Vector3d> voxel_points_;
        //            std::vector<Eigen::Vector3d> color_points_;
        //            // Write to ply
        //            std::ofstream stream(
        //                "/home/paulamayo/data/multi_vo/mesh/indoor_mesh.ply");
        //
        //            dense_reconstruction::Mesh combined_mesh(
        //                mesh_layer_->block_size(),
        //                dense_reconstruction::Point::Zero());
        //
        //            mesh_layer_->getConnectedMesh(&combined_mesh);
        //
        //            size_t num_points = combined_mesh.vertices.size();
        //            stream << "ply" << std::endl;
        //            stream << "format ascii 1.0" << std::endl;
        //            stream << "element vertex " << num_points << std::endl;
        //            stream << "property float x" << std::endl;
        //            stream << "property float y" << std::endl;
        //            stream << "property float z" << std::endl;
        //            if (combined_mesh.hasNormals()) {
        //              stream << "property float normal_x" << std::endl;
        //              stream << "property float normal_y" << std::endl;
        //              stream << "property float normal_z" << std::endl;
        //            }
        //            if (combined_mesh.hasColors()) {
        //              stream << "property uchar red" << std::endl;
        //              stream << "property uchar green" << std::endl;
        //              stream << "property uchar blue" << std::endl;
        //              stream << "property uchar alpha" << std::endl;
        //            }
        //            if (combined_mesh.hasTriangles()) {
        //              stream << "element face " <<
        //              combined_mesh.indices.size() / 3
        //                     << std::endl;
        //              stream << "property list uchar int vertex_indices"
        //                     << std::endl; // pcl-1.7(ros::kinetic) breaks ply
        //                                   // convention by not
        //              // using "vertex_index"
        //            }
        //            stream << "end_header" << std::endl;
        //            size_t vert_idx = 0;
        //            for (const dense_reconstruction::Point &vert :
        //                 combined_mesh.vertices) {
        //              stream << vert(0) << " " << vert(1) << " " << vert(2);
        //              Eigen::Vector3d point;
        //              point.x() = vert(0);
        //              point.y() = vert(1);
        //              point.z() = vert(2);
        //              voxel_points_.push_back(point);
        //
        //              if (combined_mesh.hasNormals()) {
        //                const dense_reconstruction::Point &normal =
        //                    combined_mesh.normals[vert_idx];
        //                stream << " " << normal.x() << " " << normal.y() << "
        //                "
        //                       << normal.z();
        //              }
        //              if (combined_mesh.hasColors()) {
        //                const dense_reconstruction::Color &color =
        //                    combined_mesh.colors[vert_idx];
        //                Eigen::Vector3d color_point(color.r, color.g,
        //                color.b); color_points_.push_back(color_point); int r
        //                = static_cast<int>(color.r); int g =
        //                static_cast<int>(color.g); int b =
        //                static_cast<int>(color.b); int a =
        //                static_cast<int>(color.a);
        //                // Uint8 prints as character otherwise. :(
        //                stream << " " << r << " " << g << " " << b << " " <<
        //                a;
        //              }
        //
        //              stream << std::endl;
        //              vert_idx++;
        //            }
        //            if (combined_mesh.hasTriangles()) {
        //              for (size_t i = 0; i < combined_mesh.indices.size(); i
        //              += 3) {
        //                stream << "3 ";
        //
        //                for (int j = 0; j < 3; j++) {
        //                  stream << combined_mesh.indices.at(i + j) << " ";
        //                }
        //
        //                stream << std::endl;
        //              }
        //            }
        //            aru::core::utilities::viewer::Viewer::ViewVoxelPointCloud(
        //                voxel_points_, color_points_);
        //
        //            cv::waitKey(0);
        //          }
        //        }
      }
    }
  }
  return 0;
}
