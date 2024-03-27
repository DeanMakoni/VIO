#include "aru/core/mapping/mesh_mapping/laser_mesh_map.h"

#include <Eigen/Dense>
#include <boost/make_shared.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/persistence.hpp>
#include <utility>

using namespace aru::core::utilities;
using namespace aru::core::utilities::image;
using namespace voxblox;
namespace aru {
namespace core {
namespace mapping {
namespace mesh_map {
//------------------------------------------------------------------------------
LaserMeshMap::LaserMeshMap(std::string mesh_map_settings_file)
    : mesh_map_settings_file_(std::move(mesh_map_settings_file)) {

  cv::FileStorage fs;
  fs.open(mesh_map_settings_file_, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    LOG(ERROR) << "Could not open mesh map settings file: ";
  }

  // Camera params
  LOG(INFO) << "Camera params found";
  camera_params_.baseline = fs["Camera"]["baseline"];
  camera_params_.image_height = fs["Camera"]["height"];
  camera_params_.image_width = fs["Camera"]["width"];
  cv::Mat camera_mat;
  fs["Camera"]["CameraMatrix"] >> camera_mat;
  cv::cv2eigen(camera_mat, camera_params_.K);

  viewer_ = boost::make_shared<utilities::viewer::Viewer>(
      camera_params_.image_height, camera_params_.image_width,
      camera_params_.K);

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

  mesh_ = boost::make_shared<mesh::Mesh>(reg_params_, viewer_params_,
                                         camera_params_, matcher_params_,
                                         extractor_params_);

  voxblox::TsdfMap::Config config_;
  config_.tsdf_voxel_size = 0.1;
  config_.tsdf_voxels_per_side = 32;

  voxblox::TsdfIntegratorBase::Config base_config{};
  base_config.max_ray_length_m = 15;

  tsdf_map_ = boost::make_shared<voxblox::TsdfMap>(config_);
  tsdf_integrator_ = boost::make_shared<voxblox::FastTsdfIntegrator>(
      base_config, tsdf_map_->getTsdfLayerPtr());

  mesh_layer_ = boost::make_shared<voxblox::MeshLayer>(tsdf_map_->block_size());
  LOG(INFO) << "Block size is " << tsdf_map_->block_size();
  mesh_integrator_ =
      boost::make_shared<voxblox::MeshIntegrator<voxblox::TsdfVoxel>>(
          MeshIntegratorConfig(), tsdf_map_->getTsdfLayer(), mesh_layer_.get());

  icp_ = boost::make_shared<voxblox::ICP>(voxblox::ICP::Config());

  use_laser_ = false;
  transform_map_ = boost::make_shared<utilities::transform::TransformMap>();
  frame_no = 0;

  // Initialise the mesh
  mesh_->InitXYZ();
}
//------------------------------------------------------------------------------
void LaserMeshMap::ReadTransform(
    utilities::transform::TransformSPtr transform) {
  transform_map_->AddTransform(std::move(transform));
}
//------------------------------------------------------------------------------
void LaserMeshMap::SaveCurrentTsdf(const std::string &output_ply) {
  LOG(INFO) << "Saving output mesh";
  constexpr bool kOnlyMeshUpdatedBlocks = false;
  constexpr bool kClearUpdatedFlag = false;
  mesh_integrator_->generateMesh(kOnlyMeshUpdatedBlocks, kClearUpdatedFlag);

  BlockIndexList mesh_indices;
  mesh_layer_->getAllAllocatedMeshes(&mesh_indices);
  std::vector<Eigen::Vector3d> voxel_points_;
  std::vector<Eigen::Vector3d> color_points_;
  // Write to ply
  std::ofstream stream(output_ply);

  Mesh combined_mesh(mesh_layer_->block_size(), Point::Zero());

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
           << std::endl; // pcl-1.7(ros::kinetic) breaks ply convention by not
    // using "vertex_index"
  }
  stream << "end_header" << std::endl;
  size_t vert_idx = 0;
  for (const Point &vert : combined_mesh.vertices) {
    stream << vert(0) << " " << vert(1) << " " << vert(2);
    Eigen::Vector3d point;
    point.x() = vert(0);
    point.y() = vert(1);
    point.z() = vert(2);
    voxel_points_.push_back(point);

    if (combined_mesh.hasNormals()) {
      const Point &normal = combined_mesh.normals[vert_idx];
      stream << " " << normal.x() << " " << normal.y() << " " << normal.z();
    }
    if (combined_mesh.hasColors()) {
      const Color &color = combined_mesh.colors[vert_idx];
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
}
//------------------------------------------------------------------------------
void LaserMeshMap::DrawCurrentTsdf() {

  constexpr bool kOnlyMeshUpdatedBlocks = false;
  constexpr bool kClearUpdatedFlag = false;
  mesh_integrator_->generateMesh(kOnlyMeshUpdatedBlocks, kClearUpdatedFlag);

  BlockIndexList mesh_indices;
  mesh_layer_->getAllAllocatedMeshes(&mesh_indices);
  std::vector<Eigen::Vector3d> voxel_points_;
  std::vector<Eigen::Vector3d> color_points_;
  Mesh combined_mesh(mesh_layer_->block_size(), Point::Zero());
  mesh_layer_->getConnectedMesh(&combined_mesh);

  size_t vert_idx = 0;
  for (const Point &vert : combined_mesh.vertices) {
    Eigen::Vector3d point;
    point.x() = vert(0);
    point.y() = vert(1);
    point.z() = vert(2);
    voxel_points_.push_back(point);
    if (combined_mesh.hasColors()) {
      const Color &color = combined_mesh.colors[vert_idx];
      Eigen::Vector3d color_point(color.r, color.g, color.b);
      color_points_.push_back(color_point);
    }
    vert_idx++;
  }
  aru::core::utilities::viewer::Viewer::ViewVoxelPointCloud(voxel_points_,
                                                            color_points_);
}
//------------------------------------------------------------------------------
void LaserMeshMap::UpdateMap(utilities::image::Image image_new,
                             utilities::laser::Laser laser_new) {

  //  cv::imshow("image in",image_new.GetImage());
  //  cv::waitKey(0);

  // Read the laser features
  FeatureSPtrVectorSptr laser_features =
      boost::make_shared<FeatureSPtrVector>();
  Eigen::MatrixXf point_cloud = laser_new.GetPoints();
  // TODO: Read this from YAML file somewhere
  Eigen::Affine3f laser_to_camera;
  laser_to_camera.matrix() << 0.9998, 0.0207, -0.0066, 0.0519, -0.0070, 0.0217,
      -0.9997, -0.2890, -0.0206, 0.9996, 0.0218, -0.0281, 0, 0, 0, 1.0000;

  utilities::transform::TransformSPtr curr_position =
      transform_map_->Interpolate(laser_new.GetTimeStamp(),
                                  image_new.GetTimeStamp());
  //  LOG(INFO) << "Time laser is " << laser_new.GetTimeStamp();
  //  LOG(INFO) << "Time image is " << image_new.GetTimeStamp();
  if (curr_position) {
    // LOG(INFO) << "Transform is \n" << curr_position->GetTransform().matrix();
    voxblox::Pointcloud pointcloud_float;
    voxblox::Colors colors;

    std::vector<Eigen::Vector3d> voxel_points_;
    std::vector<Eigen::Vector3d> color_points_;
    cv::Mat laser_depth =
        cv::Mat(camera_params_.image_height, camera_params_.image_width,
                CV_32FC1, cv::Scalar(0));
    VLOG(2) << "Curr position is \n" << curr_position->GetTransform().matrix();
    for (int i = 0; i < point_cloud.rows(); ++i) {
      Eigen::Vector3f curr_row = point_cloud.row(i);
      Eigen::Vector3f homogenous = laser_to_camera * curr_row;
      int down_sample = 1;
      if (homogenous(2) > 0) {
        homogenous = curr_position->GetTransform() * homogenous;
        Eigen::Vector3f points = camera_params_.K.cast<float>() * homogenous;
        float depth = homogenous(2);
        float u = camera_params_.K(0, 0) * homogenous(0) / homogenous(2) +
                  camera_params_.K(0, 2);
        float v = camera_params_.K(1, 1) * homogenous(1) / homogenous(2) +
                  camera_params_.K(1, 2);
        int pad = 5;
        if (u > pad && u < camera_params_.image_width - pad && v > pad &&
            v < camera_params_.image_height - pad) {
          pointcloud_float.push_back(homogenous);
          voxel_points_.emplace_back(homogenous.cast<double>());

          cv::Vec3b color_point = image_new.GetImage().at<cv::Vec3b>(v, u);
          laser_depth.at<float>(v, u) = depth;

          voxblox::Color color(color_point(2), color_point(1), color_point(0));
          colors.push_back(color);

          color_points_.emplace_back(color_point(2), color_point(1),
                                     color_point(0));
          FeatureSPtr curr_feature =
              boost::make_shared<Feature>(Eigen::Vector2d((float)u, (float)v));
          curr_feature->SetKeyPoint(cv::KeyPoint(u, v, 1));
          curr_feature->UpdateDepth(depth);
          laser_features->push_back(curr_feature);
        }
      }
    }
    int dilation_size = 3;
    cv::Mat element = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
        cv::Point(dilation_size, dilation_size));
    cv::dilate(laser_depth, laser_depth, element);
    cv::imshow("Dilation Demo", laser_depth);
    //
    //    bool triangulate = false;
    //    mesh_->EstimateMesh(laser_features, triangulate);
    //    // Draw the mesh
    //    cv::Mat disparity = mesh_->GetInterpolatedDepth();
    //    std::vector<cv::KeyPoint> keypoints = mesh_->GetVerticeKeypoints();
    //    std::vector<double> depths = mesh_->GetVerticeDepths();
    //    std::vector<Eigen::Vector3i> triangles = mesh_->GetMeshTriangles();

    cv::Mat image_clone = image_new.GetImage();
    float max_depth = 10;
    viewer_->ViewImageFeaturesDepth(image_clone, laser_features, max_depth);
    //    viewer_->ViewMeshWireFrame(image_clone, keypoints, depths, triangles,
    //                               viewer_params_.max_depth);
    //    viewer_->ViewInterpolatedMesh(image_clone, disparity,
    //                                  viewer_params_.max_depth);
    cv::waitKey(1);

    auto estimation_start = std::chrono::high_resolution_clock::now();
    auto pt_cloud_colour = mesh_->GetInterpolatedColorPointCloud(
        image_new.GetImage(), laser_depth, max_depth);
    pointcloud_float = pt_cloud_colour.first;
    colors = pt_cloud_colour.second;

    auto estimation_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = estimation_end - estimation_start;
    LOG(INFO) << "Interpolation takes " << elapsed.count() << " seconds";
    // float max_depth = 10;

    // cv::waitKey(15);
    // viewer_->ViewVoxelPointCloud(curr_pointcloud_, color_pointcloud_);

    utilities::transform::TransformSPtr image_position =
        transform_map_->Interpolate(image_new.GetTimeStamp());
    VLOG(2) << "Image position is "
            << image_position->GetTranslation().transpose();
    Eigen::Matrix3f rot = image_position->GetRotation();
    Eigen::Vector3f pos = image_position->GetTranslation();
    Transformation icp_initial(Rotation(rot), pos);
    Transformation icp_refine;
    icp_->runICP(tsdf_map_->getTsdfLayer(), pointcloud_float, icp_initial,
                 &icp_refine);

    //    LOG(INFO)<<"Icp refine is "<<icp_refine.asVector().transpose();
    //    LOG(INFO)<<"Icp init is "<<icp_initial.asVector().transpose();

    tsdf_integrator_->integratePointCloud(icp_refine, pointcloud_float, colors);
    //    //  current_position_.matrix() = icp_refine.getTransformationMatrix();
    //
    //    if (frame_no % 200 == 0)
    //      SaveCurrentTsdf();

    frame_no++;
    LOG(INFO) << "Frame no is " << frame_no;
  }
}

//------------------------------------------------------------------------------
void LaserMeshMap::UpdateMap(utilities::laser::Laser laser_new) {
  // Read the laser features
  FeatureSPtrVectorSptr laser_features =
      boost::make_shared<FeatureSPtrVector>();
  Eigen::MatrixXf point_cloud = laser_new.GetPoints();
  LOG(INFO) << "Number of points is " << point_cloud.rows();
  Eigen::Affine3f laser_to_camera;
  Eigen::Matrix4f transformation = Eigen::MatrixXf::Zero(4, 4);
  laser_to_camera.matrix() << 0.99996, 0.007019, 0.004319, 0.07807, 0.00419,
      0.0178, -0.9998, -0.294, -0.00709, 0.99981, 0.01782, 0.00373, 0, 0, 0, 1;
  utilities::transform::TransformSPtr curr_position =
      transform_map_->Interpolate(laser_new.GetTimeStamp());
  if (curr_position) {
    voxblox::Pointcloud pointcloud_float;
    voxblox::Colors colors;
    Eigen::Matrix3f rotation;
    rotation << -1, 0, 0, 0, -1, 0, 0, 0, 1;
    int laser_down = 0;
    cv::Mat laser_depth =
        cv::Mat(camera_params_.image_height, camera_params_.image_width,
                CV_32FC1, cv::Scalar(0));
    for (int i = 0; i < point_cloud.rows(); ++i) {
      Eigen::Vector3f curr_row = point_cloud.row(i);
      Eigen::Vector3f homogenous = laser_to_camera * curr_row;

      if (homogenous(2) > 0) {
        homogenous = curr_position->GetTransform() * homogenous;
        pointcloud_float.push_back(homogenous);
        voxblox::Color color(128, 128, 128);
        colors.push_back(color);
        Eigen::Vector3f points = camera_params_.K.cast<float>() * homogenous;
        float depth = homogenous(2);
        float u = camera_params_.K(0, 0) * homogenous(0) / homogenous(2) +
                  camera_params_.K(0, 2);
        float v = camera_params_.K(1, 1) * homogenous(1) / homogenous(2) +
                  camera_params_.K(1, 2);
      }
    }

    LOG(INFO) << "Curr position is "
              << curr_position->GetTranslation().transpose();
    Eigen::Matrix3f rot = curr_position->GetRotation();
    Eigen::Vector3f pos = curr_position->GetTranslation();
    Transformation icp_initial(Rotation(rot), pos);
    //    Transformation icp_refine;
    //    icp_->runICP(tsdf_map_->getTsdfLayer(), pointcloud_float, icp_initial,
    //                 &icp_refine);

    tsdf_integrator_->integratePointCloud(icp_initial, pointcloud_float,
                                          colors);
    //  current_position_.matrix() = icp_refine.getTransformationMatrix();

    if (frame_no % 50 == 0)
      DrawCurrentTsdf();

    frame_no++;
    LOG(INFO) << "Frame no is " << frame_no;
  }
}

} // namespace mesh_map
} // namespace mapping
} // namespace core
} // namespace aru
