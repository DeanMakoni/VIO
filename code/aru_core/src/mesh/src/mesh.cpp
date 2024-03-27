#include "aru/core/mesh/mesh.h"
#include "aru/core/utilities/image/point_feature.h"
#include "voxblox/core/common.h"

#include <Eigen/Dense>
#include <boost/make_shared.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/persistence.hpp>
#include <utility>

struct TriangleEdge {
  static const int stepXSize = 4;
  static const int stepYSize = 1;

  // __m128 is the SSE 128-bit packed float type (4 floats).
  __m128 oneStepX;
  __m128 oneStepY;

  __m128 init(const cv::Point &v0, const cv::Point &v1,
              const cv::Point &origin) {
    // Edge setup
    float A = v1.y - v0.y;
    float B = v0.x - v1.x;
    float C = v1.x * v0.y - v0.x * v1.y;

    // Step deltas
    // __m128i y = _mm_set1_ps(x) sets y[0..3] = x.
    oneStepX = _mm_set1_ps(A * stepXSize);
    oneStepY = _mm_set1_ps(B * stepYSize);

    // x/y values for initial pixel block
    // NOTE: Set operations have arguments in reverse order!
    // __m128 y = _mm_set_epi32(x3, x2, x1, x0) sets y0 = x0, etc.
    __m128 x = _mm_set_ps(origin.x + 3, origin.x + 2, origin.x + 1, origin.x);
    __m128 y = _mm_set1_ps(origin.y);

    // Edge function values at origin
    // A*x + B*y + C.
    __m128 A4 = _mm_set1_ps(A);
    __m128 B4 = _mm_set1_ps(B);
    __m128 C4 = _mm_set1_ps(C);

    return _mm_add_ps(_mm_add_ps(_mm_mul_ps(A4, x), _mm_mul_ps(B4, y)), C4);
  }
};
namespace aru {
namespace core {
namespace mesh {
//------------------------------------------------------------------------------
Mesh::Mesh(std::string mesh_estimation_settings_file)
    : mesh_estimation_settings_file_(mesh_estimation_settings_file) {

  cv::FileStorage fs;
  fs.open(mesh_estimation_settings_file_, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    LOG(ERROR) << "Could not open camera model file: ";
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

  // Create the mesh estimator
  mesh_estimator_ = boost::make_shared<MeshEstimation>(
      camera_params_, reg_params_, matcher_params_, extractor_params_);
}
//------------------------------------------------------------------------------
Mesh::Mesh(const RegularisationParams reg_params,
           const ViewerParams viewer_params,
           const utilities::camera::CameraParams &camera_params,
           utilities::image::MatcherParams matcher_params,
           utilities::image::ExtractorParams extractor_params)
    : reg_params_(reg_params), viewer_params_(viewer_params),
      camera_params_(camera_params), matcher_params_(matcher_params),
      extractor_params_(extractor_params) {
  // Create the mesh estimator
  mesh_estimator_ = boost::make_shared<MeshEstimation>(
      camera_params_, reg_params_, matcher_params_, extractor_params_);

  pixel_features_ = boost::make_shared<utilities::image::FeatureSPtrVector>();
}
//------------------------------------------------------------------------------
void Mesh::EstimateMesh(utilities::image::FeatureSPtrVectorSptr features,
                        bool triangulate) {
  mesh_estimator_->EstimateMesh((features), triangulate);

  pixel_features_ = mesh_estimator_->GetMeshFeatures();
  delaunay_triangles_ = mesh_estimator_->GetMeshTriangles();

  interpolated_depth_ =
      cv::Mat(camera_params_.image_height, camera_params_.image_width, CV_32FC1,
              cv::Scalar(0));

  //InterpolateMesh(delaunay_triangles_, pixel_features_, interpolated_depth_);
}
//------------------------------------------------------------------------------
void Mesh::InitXYZ() {
  xyz_ = cv::Mat(camera_params_.image_height, camera_params_.image_width,
                 CV_32FC3, cv::Scalar(0, 0, 0));

  for (int u = 0; u < camera_params_.image_width; ++u) {
    for (int v = 0; v < camera_params_.image_height; ++v) {
      Eigen::Vector3d uv(u, v, 1);
      Eigen::Vector3d xyz = camera_params_.K.inverse() * uv;
      xyz_.at<cv::Vec3f>(v, u) = cv::Vec3f(xyz(0), xyz(1), xyz(2));
    }
  }
}
//------------------------------------------------------------------------------
cv::Mat Mesh::DisparityToDepth(cv::Mat disparity) {
  cv::Mat depth = cv::Mat(camera_params_.image_height,
                          camera_params_.image_width, CV_32FC1, cv::Scalar(0));
  for (int u = 0; u < camera_params_.image_width; ++u) {
    for (int v = 0; v < camera_params_.image_height; ++v) {
      float disp = disparity.at<uchar>(v, u);
      if (disp > 0) {
        float baseline = camera_params_.baseline;
        float fx = camera_params_.K(0, 0);
        float depth_pix = baseline / disp * fx;
        depth.at<float>(v, u) = depth_pix;
      }
    }
  }
  return depth;
}

//------------------------------------------------------------------------------
cv::Mat Mesh::DepthToDisparity(const cv::Mat &depth) {
  cv::Mat disparity =
      cv::Mat(camera_params_.image_height, camera_params_.image_width, CV_8UC1,
              cv::Scalar(0));
  for (int u = 0; u < camera_params_.image_width; ++u) {
    for (int v = 0; v < camera_params_.image_height; ++v) {
      float depth_pix = depth.at<float>(v, u);
      if (depth_pix > 0) {
        float baseline = camera_params_.baseline;
        float fx = camera_params_.K(0, 0);
        int disp = baseline / depth_pix * fx;
        disparity.at<uchar>(v, u) = disp;
      }
    }
  }
  return disparity;
}
//------------------------------------------------------------------------------
std::pair<voxblox::Pointcloud, voxblox::Colors>
Mesh::GetInterpolatedColorPointCloud(cv::Mat image_left, float max_depth) {

  long count = 0;

  const float *depth_row =
      reinterpret_cast<const float *>(&interpolated_depth_.data[0]);
  int row_depth_step = interpolated_depth_.step / sizeof(float);

  std::vector<float> x(camera_params_.image_height *
                       camera_params_.image_width);
  auto it_x = x.begin();
  std::vector<float> y(camera_params_.image_height *
                       camera_params_.image_width);
  auto it_y = y.begin();
  std::vector<float> z(camera_params_.image_height *
                       camera_params_.image_width);
  auto it_z = z.begin();
  std::vector<float> r(camera_params_.image_height *
                       camera_params_.image_width);
  auto it_r = r.begin();
  std::vector<float> g(camera_params_.image_height *
                       camera_params_.image_width);
  auto it_g = g.begin();
  std::vector<float> b(camera_params_.image_height *
                       camera_params_.image_width);
  auto it_b = b.begin();
  for (int v = 0; v < camera_params_.image_height;
       ++v, depth_row += row_depth_step) {
    for (int u = 0; u < camera_params_.image_width; ++u) {
      float depth = depth_row[u];
      if (depth > 0 && depth < max_depth) {
        cv::Vec3b color = image_left.at<cv::Vec3b>(v, u);
        cv::Vec3f xyz = xyz_.at<cv::Vec3f>(v, u) * depth;
        *it_x = xyz(0);
        *it_y = xyz(1);
        *it_z = xyz(2);

        *it_r = color(2);
        *it_g = color(1);
        *it_b = color(0);
        ++it_x, ++it_y, ++it_z, ++it_r, ++it_g, ++it_b;
        ++count;

        //        *it_points=Eigen::Vector3d(xyz(0), xyz(1), xyz(2));
        //        *it_colors=Eigen::Vector3d(color(2), color(1), color(0));
        //        colors.emplace_back(color(2), color(1), color(0));
      }
    }
  }
  voxblox::Pointcloud points(count);
  voxblox::Colors colors(count);

  for (long i = 0; i < count; ++i) {
    points[i] = Eigen::Vector3f(x[i], y[i], z[i]);
    colors[i] = voxblox::Color(r[i], g[i], b[i]);
  }

  // for (long i=0;i<)
  return std::make_pair(points, colors);
}
//------------------------------------------------------------------------------
std::pair<voxblox::Pointcloud, voxblox::Colors>
Mesh::GetInterpolatedColorPointCloud(cv::Mat image_left) {

  voxblox::Pointcloud points;
  voxblox::Colors colors;
  for (int u = 0; u < camera_params_.image_width; ++u) {
    for (int v = 0; v < camera_params_.image_height; ++v) {
      float depth = interpolated_depth_.at<float>(v, u);
      if (depth > 0) {
        cv::Vec3b color = image_left.at<cv::Vec3b>(v, u);
        cv::Vec3f xyz = xyz_.at<cv::Vec3f>(v, u) * depth;
        Eigen::Vector3d uv(u, v, 1);
        // Eigen::Vector3d xyz(0, 0,0); // = camera_params_.K.inverse() * uv *
        //  depth;
        points.emplace_back(xyz(0), xyz(1), xyz(2));
        colors.emplace_back(color(2), color(1), color(0));
      }
    }
  }
  return std::make_pair(points, colors);
}

//------------------------------------------------------------------------------
std::pair<voxblox::Pointcloud, voxblox::Colors>
Mesh::GetInterpolatedColorPointCloud(cv::Mat image_left, cv::Mat image_depth) {

  voxblox::Pointcloud points;
  voxblox::Colors colors;

  const float *depth_row =
      reinterpret_cast<const float *>(&image_depth.data[0]);
  int row_depth_step = image_depth.step / sizeof(float);
  const cv::Vec3b *color_row =
      reinterpret_cast<const cv::Vec3b *>(&image_left.data[0]);
  int color_depth_step = image_left.step / sizeof(cv::Vec3b);
  for (int v = 0; v < camera_params_.image_height;
       ++v, depth_row += row_depth_step, color_row += color_depth_step) {
    for (int u = 0; u < camera_params_.image_width; ++u) {
      float depth = depth_row[u];
      if (depth > 0) {
        cv::Vec3b color = color_row[u];
        cv::Vec3f xyz = xyz_.at<cv::Vec3f>(v, u) * depth;
        points.emplace_back(xyz(0), xyz(1), xyz(2));
        colors.emplace_back(color(2), color(1), color(0));
      }
    }
  }
  return std::make_pair(points, colors);
}
//------------------------------------------------------------------------------
std::pair<voxblox::Pointcloud, voxblox::Colors>
Mesh::GetInterpolatedColorPointCloud(cv::Mat image_left, cv::Mat image_depth,
                                     float max_depth) {

  voxblox::Pointcloud points;
  voxblox::Colors colors;
  for (int u = 0; u < camera_params_.image_width; ++u) {
    for (int v = 0; v < camera_params_.image_height; ++v) {
      float depth = image_depth.at<float>(v, u);
      if (depth > 0 && depth < max_depth) {
        cv::Vec3b color = image_left.at<cv::Vec3b>(v, u);
        cv::Vec3f xyz = xyz_.at<cv::Vec3f>(v, u) * depth;
        points.emplace_back(xyz(0), xyz(1), xyz(2));
        colors.emplace_back(color(2), color(1), color(0));
      }
    }
  }
  return std::make_pair(points, colors);
}

//------------------------------------------------------------------------------
std::vector<Eigen::Vector3d> Mesh::GetInterpolatedPointCloud() {
  std::vector<Eigen::Vector3d> points;
  for (int u = 0; u < camera_params_.image_width; ++u) {
    for (int v = 0; v < camera_params_.image_height; ++v) {
      float depth = interpolated_depth_.at<float>(v, u);
      if (depth > 0) {
        Eigen::Vector3d uv(u, v, 1);
        Eigen::Vector3d xyz = camera_params_.K.inverse() * uv * depth;
        points.emplace_back(xyz);
      }
    }
  }
  return points;
}
//------------------------------------------------------------------------------
cv::Mat Mesh::DrawWireframe(const cv::Mat &img_color, cv::Mat image_depth) {
  std::vector<cv::Point2f> points;
  std::vector<float> depth;
  // Scroll through the sparse depth map
  for (int u = 0; u < image_depth.cols; ++u) {
    for (int v = 0; v < image_depth.rows; ++v) {
      float curr_depth = image_depth.at<float>(v, u);
      if (curr_depth > 0) {
        cv::Point2f curr_point(u, v);
        points.push_back(curr_point);
        depth.push_back(curr_depth);
      }
    }
  }

  Delaunay delaunay;
  std::vector<Triangle> triangles = delaunay.Triangulate(points);

  float max_depth = 38;
  cv::Mat img = img_color.clone();
  for (auto t : triangles) {
    cv::Point2f vert1 = points[t[0]];
    cv::Point2f vert2 = points[t[1]];
    cv::Point2f vert3 = points[t[2]];

    float depth1 =
        std::max(0.0f, std::min(depth[t[0]] / max_depth * 255, 255.0f));
    float depth2 =
        std::max(0.0f, std::min(depth[t[1]] / max_depth * 255, 255.0f));
    float depth3 =
        std::max(0.0f, std::min(depth[t[2]] / max_depth * 255, 255.0f));

    cv::Vec3b c1, c2, c3;
    COLOUR_MAP.LookUp(depth1, c1);
    COLOUR_MAP.LookUp(depth2, c2);
    COLOUR_MAP.LookUp(depth3, c3);

    Delaunay::drawLineColourMap(img, vert1, vert2, depth1, depth2);
    Delaunay::drawLineColourMap(img, vert2, vert3, depth2, depth3);
    Delaunay::drawLineColourMap(img, vert3, vert1, depth3, depth1);
  }
  return img;
}
//------------------------------------------------------------------------------
void Mesh::EstimateMesh(const cv::Mat &image_left, const cv::Mat &image_right) {

  mesh_estimator_->EstimateMesh(image_left, image_right);
  pixel_features_ = mesh_estimator_->GetMeshFeatures();
  delaunay_triangles_ = mesh_estimator_->GetMeshTriangles();

  cv::Mat delaunay_smooth_img = image_left.clone();
  Delaunay::DrawWireframe(pixel_features_, delaunay_triangles_,
                          delaunay_smooth_img, viewer_params_.max_depth);
  cv::Mat disp_dense_smooth, disp_dense_smooth_show;

  interpolated_depth_ =
      cv::Mat(image_left.size().height, image_left.size().width, CV_32FC1,
              cv::Scalar(0));

  VLOG(2) << "Number of features is " << pixel_features_->size()
          << " and triangles is " << delaunay_triangles_.size();

  InterpolateMesh(delaunay_triangles_, pixel_features_, interpolated_depth_);

  disp_dense_smooth_show = (interpolated_depth_ / viewer_params_.max_depth *
                            viewer_params_.colour_scale) *
                           255.0f;
  ViewerParams params;

  std::transform(disp_dense_smooth_show.begin<float>(),
                 disp_dense_smooth_show.end<float>(),
                 disp_dense_smooth_show.begin<float>(),
                 [&params](float f) -> float {
                   return std::max(0.0f, std::min(f, 255.0f));
                 }); // clamp between 0 and 255
  disp_dense_smooth_show.convertTo(disp_dense_smooth_show, CV_8U);
  COLOUR_MAP.ApplyColourMap(disp_dense_smooth_show, disp_dense_smooth_show);
}
//------------------------------------------------------------------------------
void Mesh::EstimateDepthGnd(cv::Mat image_left, const cv::Mat &image_right,
                            cv::Mat image_gnd) {

  LOG(INFO) << "Mesh estimation";
  mesh_estimator_->EstimateMesh(image_left, image_right);
  utilities::image::FeatureSPtrVectorSptr point_features =
      mesh_estimator_->GetMeshFeatures();
  std::vector<Triangle> triangles = mesh_estimator_->GetMeshTriangles();

  cv::Mat delaunay_smooth_img = image_left.clone();
  Delaunay::DrawWireframe(point_features, triangles, delaunay_smooth_img,
                          viewer_params_.max_depth);
  cv::Mat disp_dense_smooth, disp_dense_smooth_show;
  disp_dense_smooth = cv::Mat(image_left.size().height, image_left.size().width,
                              CV_32FC1, cv::Scalar(0));

  LOG(INFO) << "Number of feaures is " << point_features->size()
            << " and triangles is " << triangles.size();
  InterpolateMesh(triangles, point_features, disp_dense_smooth);
  disp_dense_smooth_show = (disp_dense_smooth / viewer_params_.max_depth *
                            viewer_params_.colour_scale) *
                           255.0f;
  ViewerParams params;
  std::transform(disp_dense_smooth_show.begin<float>(),
                 disp_dense_smooth_show.end<float>(),
                 disp_dense_smooth_show.begin<float>(),
                 [&params](float f) -> float {
                   return std::max(0.0f, std::min(f, 255.0f));
                 }); // clamp between 0 and 255
  disp_dense_smooth_show.convertTo(disp_dense_smooth_show, CV_8U);
  COLOUR_MAP.ApplyColourMap(disp_dense_smooth_show, disp_dense_smooth_show);

  cv::imshow("Wireframe", delaunay_smooth_img);
  cv::imshow("Dense frame", disp_dense_smooth_show);

  std::vector<cv::KeyPoint> key_points_cv_left;
  for (const auto &feature : *point_features) {
    if (feature->GetMatchedKeyPoint().pt.x > 0) {
      cv::Point2f vert = feature->GetKeyPoint().pt;
      int depth_norm = static_cast<int>(
          feature->GetDisparity() /
          (viewer_params_.max_depth * viewer_params_.colour_scale) * 255.0f);
      cv::Vec3b c1;
      COLOUR_MAP.LookUp(depth_norm, c1);
      float radius = 2;
      circle(image_left, vert, radius, c1, cv::FILLED);
      key_points_cv_left.push_back(feature->GetKeyPoint());
    }
  }

  image_gnd.convertTo(image_gnd, CV_8UC1);
  cv::cvtColor(image_gnd, image_gnd, cv::COLOR_BGR2GRAY);

  // Convert disparity to depth
  aru::core::utilities::camera::CameraParams cam_params = camera_params_;
  std::transform(image_gnd.begin<char>(), image_gnd.end<char>(),
                 image_gnd.begin<char>(), [&cam_params](char f) -> char {
                   return (char)std::round(cam_params.baseline / (float)f *
                                           cam_params.K(0, 0));
                 });

  std::transform(image_gnd.begin<char>(), image_gnd.end<char>(),
                 image_gnd.begin<char>(), [&params](char f) -> char {
                   return (char)std::max(0, std::min((int)f, 64 - 1));
                 });

  // clamp between 0 and 255
  cv::Mat img_gnd_show =
      image_gnd * (256.0 / (float)64 * viewer_params_.colour_scale);
  img_gnd_show.convertTo(img_gnd_show, CV_8U);
  COLOUR_MAP.ApplyColourMap(img_gnd_show, img_gnd_show);

  cv::imshow("Ground Truth", img_gnd_show);

  cv::imshow("Depth image", image_left);
  cv::waitKey(0);
}
//------------------------------------------------------------------------------
// NB image must be type CV32FC1
void Mesh::DrawShadedTriangleBarycentric(const cv::Point &p1,
                                         const cv::Point &p2,
                                         const cv::Point &p3, float v1,
                                         float v2, float v3, cv::Mat *img) {
  // Compute triangle bounding box
  int xmin = min3(p1.x, p2.x, p3.x);
  int ymin = min3(p1.y, p2.y, p3.y);
  int xmax = max3(p1.x, p2.x, p3.x);
  int ymax = max3(p1.y, p2.y, p3.y);

  int min_bounding = 4;

  if (abs(xmax - xmin) > min_bounding && abs(ymax - ymin) > min_bounding) {

    cv::Point p(xmin, ymin);
    TriangleEdge e12, e23, e31;

    // __m128 is the SSE 128-bit packed float type (4 floats).
    __m128 w1_row = e23.init(p2, p3, p);
    __m128 w2_row = e31.init(p3, p1, p);
    __m128 w3_row = e12.init(p1, p2, p);

    // Values as 4 packed floats.
    __m128 v14 = _mm_set1_ps(v1);
    __m128 v24 = _mm_set1_ps(v2);
    __m128 v34 = _mm_set1_ps(v3);

    // Rasterize
    for (p.y = ymin; p.y <= ymax; p.y += TriangleEdge::stepYSize) {
      // Determine barycentric coordinates
      __m128 w1 = w1_row;
      __m128 w2 = w2_row;
      __m128 w3 = w3_row;

      for (p.x = xmin; p.x <= xmax; p.x += TriangleEdge::stepXSize) {
        // If p is on or inside all edges, render pixel.
        __m128 zero = _mm_set1_ps(0.0f);

        // (w1 >= 0) && (w2 >= 0) && (w3 >= 0)
        // mask tells whether we should set the pixel.
        __m128 mask = _mm_and_ps(
            _mm_cmpge_ps(w1, zero),
            _mm_and_ps(_mm_cmpge_ps(w2, zero), _mm_cmpge_ps(w3, zero)));

        // w1 + w2 + w3
        __m128 norm = _mm_add_ps(w1, _mm_add_ps(w2, w3));

        // v1*w1 + v2*w2 + v3*w3 / norm
        __m128 vals = _mm_div_ps(
            _mm_add_ps(_mm_mul_ps(v14, w1),
                       _mm_add_ps(_mm_mul_ps(v24, w2), _mm_mul_ps(v34, w3))),
            norm);

        // Grab original data.  We need to use different store/load functions
        // if the address is not aligned to 16-bytes.
        uint32_t addr = sizeof(float) * (p.y * img->cols + p.x);
        if (addr % 16 == 0) {
          float *img_ptr = reinterpret_cast<float *>(&(img->data[addr]));
          __m128 data = _mm_load_ps(img_ptr);

          // Set values using mask.
          // If mask is true, use vals, otherwise use data.
          __m128 res =
              _mm_or_ps(_mm_and_ps(mask, vals), _mm_andnot_ps(mask, data));
          _mm_store_ps(img_ptr, res);
        } else {
          // Address is not 16-byte aligned. Need to use special functions to
          // load/store.
          float *img_ptr = reinterpret_cast<float *>(&(img->data[addr]));
          __m128 data = _mm_loadu_ps(img_ptr);

          // Set values using mask.
          // If mask is true, use vals, otherwise use data.
          __m128 res =
              _mm_or_ps(_mm_and_ps(mask, vals), _mm_andnot_ps(mask, data));
          _mm_storeu_ps(img_ptr, res);
        }

        // One step to the right.
        w1 = _mm_add_ps(w1, e23.oneStepX);
        w2 = _mm_add_ps(w2, e31.oneStepX);
        w3 = _mm_add_ps(w3, e12.oneStepX);
      }

      // Row step.
      w1_row = _mm_add_ps(w1_row, e23.oneStepY);
      w2_row = _mm_add_ps(w2_row, e31.oneStepY);
      w3_row = _mm_add_ps(w3_row, e12.oneStepY);
    }
  }
}
//------------------------------------------------------------------------------
std::vector<Eigen::Vector3i> Mesh::GetMeshTriangles() {
  std::vector<Eigen::Vector3i> mesh_triangles;

  for (auto t : delaunay_triangles_) {
    mesh_triangles.emplace_back(t[0], t[1], t[2]);
  }
  return mesh_triangles;
}
//------------------------------------------------------------------------------
std::vector<double> Mesh::GetVerticeDepths() {
  std::vector<double> depths;

  for (auto &feature : *pixel_features_) {
    depths.push_back(feature->GetDepth());
  }
  return depths;
}
//------------------------------------------------------------------------------
std::vector<cv::KeyPoint> Mesh::GetVerticeKeypoints() {
  std::vector<cv::KeyPoint> keypoints;

  for (auto &feature : *pixel_features_) {
    cv::KeyPoint key_curr = feature->GetKeyPoint();
    keypoints.emplace_back(key_curr);
  }
  return keypoints;
}
//------------------------------------------------------------------------------
std::vector<Eigen::Vector3d> Mesh::GetMeshFeatures() {
  std::vector<Eigen::Vector3d> mesh_features;

  for (auto &feature : *pixel_features_) {
    feature->UpdateTriangulatedPointDepth();
    mesh_features.push_back(feature->GetTriangulatedPoint());
  }
  return mesh_features;
}
//------------------------------------------------------------------------------
std::pair<cv::Mat, cv::Mat>
Mesh::CreateDenseDepthMap(cv::Mat sparse_depth_map) {

  std::vector<cv::Point2f> points;
  std::vector<float> depth;
  // Scroll through the sparse depth map
  for (int u = 0; u < sparse_depth_map.cols; ++u) {
    for (int v = 0; v < sparse_depth_map.rows; ++v) {
      float curr_depth = sparse_depth_map.at<float>(v, u);
      if (curr_depth > 0) {
        cv::Point2f curr_point(u, v);
        points.push_back(curr_point);
        depth.push_back(curr_depth);
      }
    }
  }

  Delaunay delaunay;
  std::vector<Triangle> triangles = delaunay.Triangulate(points);

  cv::Mat dense_map = cv::Mat(sparse_depth_map.rows, sparse_depth_map.cols,
                              CV_32FC1, cv::Scalar(-1));

  for (auto t : triangles) {
    cv::Point2f vert1 = points[t[0]];
    cv::Point2f vert2 = points[t[1]];
    cv::Point2f vert3 = points[t[2]];

    float depth1 = depth[t[0]];
    float depth2 = depth[t[1]];
    float depth3 = depth[t[2]];

    // points are expected CCW, NB image must be type CV32FC1
    DrawShadedTriangleBarycentric(vert3, vert2, vert1, depth3, depth2, depth1,
                                  &dense_map);
  }

  cv::Mat disp_dense_smooth_show = (dense_map / 38.0f) * 255.0f;
  ViewerParams params;

  std::transform(disp_dense_smooth_show.begin<float>(),
                 disp_dense_smooth_show.end<float>(),
                 disp_dense_smooth_show.begin<float>(),
                 [&params](float f) -> float {
                   return std::max(0.0f, std::min(f, 255.0f));
                 }); // clamp between 0 and 255
  disp_dense_smooth_show.convertTo(disp_dense_smooth_show, CV_8U);
  COLOUR_MAP.ApplyColourMap(disp_dense_smooth_show, disp_dense_smooth_show);

  return std::make_pair(disp_dense_smooth_show, dense_map);
}
//------------------------------------------------------------------------------
void Mesh::InterpolateMesh(
    const std::vector<Triangle> &triangles,
    const utilities::image::FeatureSPtrVectorSptr &sparse_supports,
    cv::Mat &dense_map) {

  for (auto t : triangles) {
    cv::Point2f vert1 = sparse_supports->at(t[0])->GetKeyPoint().pt;
    cv::Point2f vert2 = sparse_supports->at(t[1])->GetKeyPoint().pt;
    cv::Point2f vert3 = sparse_supports->at(t[2])->GetKeyPoint().pt;
    float depth1 = sparse_supports->at(t[0])->GetDepth();
    float depth2 = sparse_supports->at(t[1])->GetDepth();
    float depth3 = sparse_supports->at(t[2])->GetDepth();

    // points are expected CCW, NB image must be type CV32FC1
    DrawShadedTriangleBarycentric(vert3, vert2, vert1, depth3, depth2, depth1,
                                  &dense_map);

    //        cv::Mat disp_dense_smooth_show = (dense_map / 80) * 255.0f;
    //        std::transform(disp_dense_smooth_show.begin<float>(),
    //                       disp_dense_smooth_show.end<float>(),
    //                       disp_dense_smooth_show.begin<float>(), [](float f)
    //                       -> float {
    //              return std::max(0.0f, std::min(f, 255.0f));
    //            }); // clamp between 0 and 255
    //        disp_dense_smooth_show.convertTo(disp_dense_smooth_show, CV_8U);
    //        COLOUR_MAP.ApplyColourMap(disp_dense_smooth_show,
    //        disp_dense_smooth_show);
    //
    //        cv::imshow("Interpolated Mesh", disp_dense_smooth_show);
    //        cv::waitKey(0);
  }
}

} // namespace mesh
} // namespace core
} // namespace aru
