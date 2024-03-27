#include <Eigen/Dense>
#include <aru/core/utilities/viewer/viewer.h>
#include <boost/make_shared.hpp>
#include <chrono>
#include <pangolin/gl/gl.h>
#include <utility>

namespace aru {
namespace core {
namespace utilities {
namespace viewer {

//------------------------------------------------------------------------------
PoseViewer::PoseViewer(utilities::transform::TransformSPtrVectorSptr pose_chain)
    : pose_chain_(pose_chain) {}
//------------------------------------------------------------------------------
void PoseViewer::DrawCurrentPose(pangolin::OpenGlMatrix matrix) {
  float mCameraSize = 0.7;
  const float &w = mCameraSize;
  const float h = w * 0.75;
  const float z = w * 0.6;

  glPushMatrix();

#ifdef HAVE_GLES
  glMultMatrixf(Twc.m);
#else
  glMultMatrixd(matrix.m);
#endif

  float mCameraLineWidth = 3;
  glLineWidth(mCameraLineWidth);
  glColor3f(0.0f, 1.0f, 0.0f);
  glBegin(GL_LINES);
  glVertex3f(0, 0, 0);
  glVertex3f(w, h, z);
  glVertex3f(0, 0, 0);
  glVertex3f(w, -h, z);
  glVertex3f(0, 0, 0);
  glVertex3f(-w, -h, z);
  glVertex3f(0, 0, 0);
  glVertex3f(-w, h, z);

  glVertex3f(w, h, z);
  glVertex3f(w, -h, z);

  glVertex3f(-w, h, z);
  glVertex3f(-w, -h, z);

  glVertex3f(-w, h, z);
  glVertex3f(w, h, z);

  glVertex3f(-w, -h, z);
  glVertex3f(w, -h, z);
  glEnd();

  glPopMatrix();
}
//------------------------------------------------------------------------------
void PoseViewer::DrawCurrentPose(pangolin::OpenGlMatrix matrix,
                                 Eigen::Vector3f color) {
  float mCameraSize = 0.7;
  const float &w = mCameraSize;
  const float h = w * 0.75;
  const float z = w * 0.6;

  glPushMatrix();

#ifdef HAVE_GLES
  glMultMatrixf(Twc.m);
#else
  glMultMatrixd(matrix.m);
#endif

  float mCameraLineWidth = 3;
  glLineWidth(mCameraLineWidth);
  glColor3f(color(0), color(1), color(2));
  glBegin(GL_LINES);
  glVertex3f(0, 0, 0);
  glVertex3f(w, h, z);
  glVertex3f(0, 0, 0);
  glVertex3f(w, -h, z);
  glVertex3f(0, 0, 0);
  glVertex3f(-w, -h, z);
  glVertex3f(0, 0, 0);
  glVertex3f(-w, h, z);

  glVertex3f(w, h, z);
  glVertex3f(w, -h, z);

  glVertex3f(-w, h, z);
  glVertex3f(-w, -h, z);

  glVertex3f(-w, h, z);
  glVertex3f(w, h, z);

  glVertex3f(-w, -h, z);
  glVertex3f(w, -h, z);
  glEnd();

  glPopMatrix();
}
//------------------------------------------------------------------------------
pangolin::OpenGlMatrix PoseViewer::GetCurrentPose() {
  pangolin::OpenGlMatrix curr_matrix =
      GetOpenGLPoseMatrix(pose_chain_->back()->GetTransform().matrix());
  return curr_matrix;
}
//------------------------------------------------------------------------------
void PoseViewer::DrawPoses() {
  // Scroll through the poses
  int num = 0;
  for (auto pose : *pose_chain_) {
    if (pose) {
      pangolin::OpenGlMatrix curr_matrix =
          GetOpenGLPoseMatrix(pose->GetTransform().matrix());
      DrawCurrentPose(curr_matrix);
    }
  }
}
//------------------------------------------------------------------------------
void PoseViewer::DrawPoses(Eigen::Vector3f color) {
  // Scroll through the poses
  int num = 0;
  for (auto pose : *pose_chain_) {
    if (pose) {
      pangolin::OpenGlMatrix curr_matrix =
          GetOpenGLPoseMatrix(pose->GetTransform().matrix());
      DrawCurrentPose(curr_matrix, color);
    }
  }
}
//------------------------------------------------------------------------------
pangolin::OpenGlMatrix PoseViewer::GetOpenGLPoseMatrix(Eigen::MatrixXf pose) {

  pangolin::OpenGlMatrix mat;
  mat.m[0] = pose(0, 0);
  mat.m[1] = pose(1, 0);
  mat.m[2] = pose(2, 0);
  mat.m[3] = 0.0;

  mat.m[4] = pose(0, 1);
  mat.m[5] = pose(1, 1);
  mat.m[6] = pose(2, 1);
  mat.m[7] = 0.0;

  mat.m[8] = pose(0, 2);
  mat.m[9] = pose(1, 2);
  mat.m[10] = pose(2, 2);
  mat.m[11] = 0.0;

  mat.m[12] = pose(0, 3);
  mat.m[13] = pose(1, 3);
  mat.m[14] = pose(2, 3);
  mat.m[15] = 1.0;
  return mat;
}
//------------------------------------------------------------------------------
void setImageData(unsigned char *imageArray, int size) {
  for (int i = 0; i < size; i++) {
    imageArray[i] = (unsigned char)(rand() / (RAND_MAX / 255.0));
  }
}
//------------------------------------------------------------------------------
Viewer::Viewer(int height, int width, Eigen::Matrix3d camera_intrinsic)
    : height_(height), width_(width),
      camera_intrinsic_(std::move(camera_intrinsic)) {

  colour_map_.clear();
  colour_map_.emplace_back(0, 0, 255);
  colour_map_.emplace_back(0, 255, 0);
  colour_map_.emplace_back(255, 0, 0);
  colour_map_.emplace_back(0, 255, 255);
  colour_map_.emplace_back(255, 0, 255);
  colour_map_.emplace_back(255, 255, 0);
  colour_map_.emplace_back(0, 0, 128);
  colour_map_.emplace_back(0, 128, 0);
  colour_map_.emplace_back(128, 0, 255);
  colour_map_.emplace_back(0, 128, 128);
  colour_map_.emplace_back(128, 0, 128);
  colour_map_.emplace_back(128, 128, 0);
  colour_map_.emplace_back(0, 0, 64);
  colour_map_.emplace_back(0, 64, 0);
  colour_map_.emplace_back(64, 0, 0);
  colour_map_.emplace_back(0, 64, 64);
  colour_map_.emplace_back(64, 0, 64);
  colour_map_.emplace_back(64, 64, 0);
  colour_map_.emplace_back(128, 0, 255);
  colour_map_.emplace_back(0, 128, 255);
  colour_map_.emplace_back(0, 255, 128);
  colour_map_.emplace_back(128, 0, 255);
  colour_map_.emplace_back(255, 0, 128);
}
//------------------------------------------------------------------------------
void Viewer::ViewPoseChain(const std::vector<Eigen::MatrixXf> &poses) {
  // Create OpenGL window in single line
  pangolin::CreateWindowAndBind("Main", 640, 480);

  // 3D Mouse handler requires depth testing to be enabled
  glEnable(GL_DEPTH_TEST);

  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  // Define Projection and initial ModelView matrix
  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100),
      pangolin::ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin::AxisY));

  pangolin::Renderable tree;
  for (const auto &pose : poses) {
    auto axis_i = std::make_shared<pangolin::Axis>();
    axis_i->T_pc = GetOpenGLPoseMatrix(pose);
    // axis_i->T_pc = pangolin::OpenGlMatrix::Translate(i * 2.0, i * 0.1,
    // 0.0);
    tree.Add(axis_i);
  }

  // Create Interactive View in window
  pangolin::SceneHandler handler(tree, s_cam);
  pangolin::View &d_cam = pangolin::CreateDisplay()
                              .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f / 480.0f)
                              .SetHandler(&handler);

  d_cam.SetDrawFunction([&](pangolin::View &view) {
    view.Activate(s_cam);
    tree.Render();
  });

  while (!pangolin::ShouldQuit()) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Swap frames and Process Events
    pangolin::FinishFrame();
  }
  pangolin::DestroyWindow("Main");
}
//------------------------------------------------------------------------------
pangolin::OpenGlMatrix Viewer::GetOpenGLPoseMatrix(Eigen::MatrixXf pose) {

  pangolin::OpenGlMatrix mat;
  mat.m[0] = pose(0, 0);
  mat.m[1] = pose(1, 0);
  mat.m[2] = pose(2, 0);
  mat.m[3] = 0.0;

  mat.m[4] = pose(0, 1);
  mat.m[5] = pose(1, 1);
  mat.m[6] = pose(2, 1);
  mat.m[7] = 0.0;

  mat.m[8] = pose(0, 2);
  mat.m[9] = pose(1, 2);
  mat.m[10] = pose(2, 2);
  mat.m[11] = 0.0;

  mat.m[12] = pose(0, 3);
  mat.m[13] = pose(1, 3);
  mat.m[14] = pose(2, 3);
  mat.m[15] = 1.0;
  return mat;
}
//------------------------------------------------------------------------------
void Viewer::ViewPointCloud(const std::vector<Eigen::Vector3d> &points) {
  // Create OpenGL window in single line
  pangolin::CreateWindowAndBind("Main", 640, 480);

  // 3D Mouse handler requires depth testing to be enabled
  glEnable(GL_DEPTH_TEST);

  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  // Define Projection and initial ModelView matrix
  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100),
      pangolin::ModelViewLookAt(0, 0, -1, 0, 0, 0, pangolin::AxisY));

  pangolin::Renderable tree;
  for (size_t i = 0; i < 1; ++i) {
    auto axis_i = std::make_shared<pangolin::Axis>();
    // axis_i->T_pc = pangolin::OpenGlMatrix::Translate(i * 2.0, i * 0.1,
    // 0.0);
    tree.Add(axis_i);
  }

  // Create Interactive View in window
  pangolin::SceneHandler handler(tree, s_cam);
  pangolin::View &d_cam = pangolin::CreateDisplay()
                              .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f / 480.0f)
                              .SetHandler(&handler);

  d_cam.SetDrawFunction([&](pangolin::View &view) {
    view.Activate(s_cam);
    tree.Render();
  });

  while (!pangolin::ShouldQuit()) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Render
    glPointSize(1);
    glBegin(GL_POINTS);
    glColor3f(0.0, 1.0, 0.0);

    Eigen::Matrix3d rotation;
    rotation << -1, 0, 0, 0, -1, 0, 0, 0, 1;

    for (auto point : points) {
      cv::Vec3b c1;
      COLOUR_MAP.LookUp(point(2) / 50 * 255, c1);
      Eigen::Vector3d rot_point = rotation * point;
      glColor3f(c1[2], c1[1], c1[0]);
      glVertex3f(rot_point(0), rot_point(1), rot_point(2));
    }
    glEnd();
    // Swap frames and Process Events
    pangolin::FinishFrame();
  }
  pangolin::DestroyWindow("Main");
}
//------------------------------------------------------------------------------
void Viewer::ViewVoxelPointCloud(const std::vector<Eigen::Vector3d> &points,
                                 const std::vector<Eigen::Vector3d> &colors) {
  // Create OpenGL window in single line
  pangolin::CreateWindowAndBind("Main", 640, 480);

  // 3D Mouse handler requires depth testing to be enabled
  glEnable(GL_DEPTH_TEST);

  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  // Define Projection and initial ModelView matrix
  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100),
      pangolin::ModelViewLookAt(0, 0, -1, 0, 0, 0, pangolin::AxisY));

  pangolin::Renderable tree;
  for (size_t i = 0; i < 1; ++i) {
    auto axis_i = std::make_shared<pangolin::Axis>();
    // axis_i->T_pc = pangolin::OpenGlMatrix::Translate(i * 2.0, i * 0.1,
    // 0.0);
    tree.Add(axis_i);
  }

  // Create Interactive View in window
  pangolin::SceneHandler handler(tree, s_cam);
  pangolin::View &d_cam = pangolin::CreateDisplay()
                              .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f / 480.0f)
                              .SetHandler(&handler);

  d_cam.SetDrawFunction([&](pangolin::View &view) {
    view.Activate(s_cam);
    tree.Render();
  });

  while (!pangolin::ShouldQuit()) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Render
    glPointSize(1);
    glBegin(GL_POINTS);
    glColor3f(0.0, 1.0, 0.0);

    Eigen::Matrix3d rotation;
    rotation << -1, 0, 0, 0, -1, 0, 0, 0, 1;

    int32_t point_no = 0;
    for (const auto &point : points) {
      cv::Vec3b c1;
      COLOUR_MAP.LookUp(point(2) / 50 * 255, c1);
      Eigen::Vector3d rot_point = rotation * point;
      Eigen::Vector3d color_gl = colors[point_no];
      // LOG(INFO) << "Color is " << c1;
      // glColor3f(c1[2], c1[1], c1[0]);
      glColor3f(color_gl(0) / 255, color_gl(1) / 255, color_gl(2) / 255);
      glVertex3f(rot_point(0), rot_point(1), rot_point(2));
      point_no++;
    }
    glEnd();
    // Swap frames and Process Events
    pangolin::FinishFrame();
  }
  pangolin::DestroyWindow("Main");
}

//------------------------------------------------------------------------------
void Viewer::ViewVoxelPointCloud(const std::vector<Eigen::Vector3d> &points,
                                 const std::vector<Eigen::Vector3d> &colors,
                                 const std::vector<Eigen::Affine3f> &poses) {
  // Create OpenGL window in single line
  pangolin::CreateWindowAndBind("Main", 640, 480);

  // 3D Mouse handler requires depth testing to be enabled
  glEnable(GL_DEPTH_TEST);

  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  // Define Projection and initial ModelView matrix
  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100),
      pangolin::ModelViewLookAt(0, 0, -1, 0, 0, 0, pangolin::AxisY));

  pangolin::Renderable tree;
  for (const auto &pose : poses) {
    auto axis_i = std::make_shared<pangolin::Axis>();
    Eigen::Matrix4f rotation;
    rotation << -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;
    axis_i->T_pc = GetOpenGLPoseMatrix(rotation * pose.matrix());
    // axis_i->T_pc = pangolin::OpenGlMatrix::Translate(i * 2.0, i * 0.1,
    // 0.0);
    tree.Add(axis_i);
  }

  // Create Interactive View in window
  pangolin::SceneHandler handler(tree, s_cam);
  pangolin::View &d_cam = pangolin::CreateDisplay()
                              .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f / 480.0f)
                              .SetHandler(&handler);

  d_cam.SetDrawFunction([&](pangolin::View &view) {
    view.Activate(s_cam);
    tree.Render();
  });

  while (!pangolin::ShouldQuit()) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Render
    glPointSize(1);
    glBegin(GL_POINTS);
    glColor3f(0.0, 1.0, 0.0);

    Eigen::Matrix3d rotation;
    rotation << -1, 0, 0, 0, -1, 0, 0, 0, 1;

    int32_t point_no = 0;
    for (const auto &point : points) {
      cv::Vec3b c1;
      COLOUR_MAP.LookUp(point(2) / 50 * 255, c1);
      Eigen::Vector3d rot_point = rotation * point;
      Eigen::Vector3d color_gl = colors[point_no];
      // LOG(INFO) << "Color is " << c1;
      // glColor3f(c1[2], c1[1], c1[0]);
      glColor3f(color_gl(0) / 255, color_gl(1) / 255, color_gl(2) / 255);
      glVertex3f(rot_point(0), rot_point(1), rot_point(2));
      point_no++;
    }
    glEnd();
    // Swap frames and Process Events
    pangolin::FinishFrame();
  }
  pangolin::DestroyWindow("Main");
}
//------------------------------------------------------------------------------
void Viewer::Run() {}
//------------------------------------------------------------------------------
void Viewer::ViewDepthPointCloud(cv::Mat depth, float max_depth) {
  cv::Mat disp_dense_smooth_show = (depth / max_depth) * 255.0f;
  // ViewerParams params;

  std::transform(disp_dense_smooth_show.begin<float>(),
                 disp_dense_smooth_show.end<float>(),
                 disp_dense_smooth_show.begin<float>(), [](float f) -> float {
                   return std::max(0.0f, std::min(f, 255.0f));
                 }); // clamp between 0 and 255
  disp_dense_smooth_show.convertTo(disp_dense_smooth_show, CV_8U);
  COLOUR_MAP.ApplyColourMap(disp_dense_smooth_show, disp_dense_smooth_show);

  cv::imshow("Depth", disp_dense_smooth_show);
  cv::waitKey(0);
}
//------------------------------------------------------------------------------
void Viewer::ViewImageFeatures(
    cv::Mat image_1, utilities::image::FeatureSPtrVectorSptr &features) {
  cv::Mat image_show = image_1.clone();
  for (const auto &feat : *features) {
    cv::KeyPoint keypoint = feat->GetKeyPoint();
    cv::circle(image_show, keypoint.pt, 1, cv::Scalar(0, 255, 0), -1,
               cv::FILLED);
  }
  cv::resize(image_show, image_show, cv::Size(), 0.5, 0.5);
  cv::imshow("Matched features", image_show);
  cv::waitKey(15);
}

//------------------------------------------------------------------------------
void Viewer::ViewImageFeaturesDepth(
    cv::Mat image_1, utilities::image::FeatureSPtrVectorSptr &features,
    float max_depth) {
  cv::Mat image_show = image_1.clone();
  for (const auto &feat : *features) {
    cv::KeyPoint keypoint = feat->GetKeyPoint();
    int depth = (feat->GetDepth() / max_depth) * 255;
    cv::Vec3b color;
    COLOUR_MAP.LookUp(depth, color);
    cv::circle(image_show, keypoint.pt, 1, color, -1, cv::FILLED);
  }
  cv::imshow("Matched features", image_show);
  cv::waitKey(15);
}
//------------------------------------------------------------------------------
void Viewer::ViewStereoImageFeatures(
    const cv::Mat &image_1, cv::Mat image_2, float resize,
    utilities::image::FeatureSPtrVectorSptr &features) {
  cv::Mat image_1_show = image_1.clone();
  cv::Mat image_2_show = image_2.clone();
  cv::Mat combined = image_1_show.clone();
  cv::vconcat(image_1_show, image_2_show, combined);
  for (const auto &feat : *features) {
    cv::KeyPoint keypoint = feat->GetKeyPoint();
    cv::circle(combined, keypoint.pt, 1, cv::Scalar(0, 255, 0), 3, cv::FILLED);
    cv::KeyPoint keypoint_right = feat->GetMatchedKeyPoint();
    // keypoint_right.pt.y = keypoint_right.pt.y + image_1_show.rows;

    cv::line(combined, keypoint.pt, keypoint_right.pt, cv::Scalar(255, 0, 0));
    keypoint_right.pt.y = keypoint_right.pt.y + image_1_show.rows;
    cv::circle(combined, keypoint_right.pt, 1, cv::Scalar(0, 0, 255), 3,
               cv::FILLED);
  }
  cv::Mat img_resize;
  cv::resize(combined, img_resize, cv::Size(), resize, resize);
  cv::imshow("Stereo features", img_resize);
  // cv::waitKey(0);
}
//------------------------------------------------------------------------------
void Viewer::ViewSequentialImageFeatures(
    const cv::Mat &image_1, cv::Mat image_2, float resize,
    utilities::image::FeatureSPtrVectorSptr &features) {
  cv::Mat image_1_show = image_1.clone();
  cv::Mat image_2_show = image_2.clone();
  cv::Mat combined;
  cv::vconcat(image_1_show, image_2_show, combined);
  for (const auto &feat : *features) {
    cv::KeyPoint keypoint = feat->GetKeyPoint();
    cv::circle(combined, keypoint.pt, 1, cv::Scalar(0, 255, 0), -1, cv::FILLED);
  }
  cv::Mat img_resize;
  cv::resize(combined, img_resize, cv::Size(), resize, resize);
  cv::imshow("Seq features", img_resize);
  cv::waitKey(0);
}
//------------------------------------------------------------------------------
void Viewer::VisualiseLabels(cv::Mat image,
                             utilities::image::FeatureSPtrVectorSptr &features,
                             Eigen::MatrixXd labels) {
  cv::Mat image_show = image.clone();
  int feature_no = 0;
  for (const auto &feat : *features) {
    cv::KeyPoint keypoint = feat->GetKeyPoint();
    cv::circle(image_show, keypoint.pt, 2, colour_map_[labels(feature_no)], -1,
               cv::FILLED);
    feature_no++;
  }
  cv::imshow("Feature Labels", image_show);
  cv::waitKey(15);
}
//------------------------------------------------------------------------------
void Viewer::ViewInterpolatedMesh(const cv::Mat &image, const cv::Mat &depth,
                                  float max_depth) {
  cv::Mat disp_dense_smooth_show = (depth / max_depth) * 255.0f;
  std::transform(disp_dense_smooth_show.begin<float>(),
                 disp_dense_smooth_show.end<float>(),
                 disp_dense_smooth_show.begin<float>(), [](float f) -> float {
                   return std::max(0.0f, std::min(f, 255.0f));
                 }); // clamp between 0 and 255
  disp_dense_smooth_show.convertTo(disp_dense_smooth_show, CV_8U);
  COLOUR_MAP.ApplyColourMap(disp_dense_smooth_show, disp_dense_smooth_show);

  cv::imshow("Interpolated Mesh", disp_dense_smooth_show);
  cv::waitKey(5);
}
//------------------------------------------------------------------------------
void Viewer::ViewDisparity(const cv::Mat &disparity, float max_disp) {
  cv::Mat disp_dense_smooth_show = disparity / max_disp * 255;
  COLOUR_MAP.ApplyColourMap(disp_dense_smooth_show, disp_dense_smooth_show);
  cv::imshow("Disparity", disp_dense_smooth_show);
  cv::waitKey(5);
}
//------------------------------------------------------------------------------
void Viewer::ViewMeshWireFrame(cv::Mat &image,
                               std::vector<cv::KeyPoint> uv_points,
                               std::vector<double> depths,
                               const std::vector<Eigen::Vector3i> &triangles,
                               float max_depth, cv::Scalar colour) {
  cv::Mat image_grey = image;
  if (image_grey.channels() == 1) {
    cv::cvtColor(image, image_grey, cv::COLOR_GRAY2BGR);
  }

  for (auto t : triangles) {

    cv::Point2f vert1 = uv_points.at(t[0]).pt;
    cv::Point2f vert2 = uv_points.at(t[1]).pt;
    cv::Point2f vert3 = uv_points.at(t[2]).pt;

    cv::line(image_grey, vert1, vert2, colour, 2);
    cv::line(image_grey, vert3, vert2, colour, 2);
    cv::line(image_grey, vert1, vert3, colour, 2);
  }
  cv::imshow("WireFrame", image_grey);
  cv::waitKey(5);
}
//------------------------------------------------------------------------------
void Viewer::ViewMeshWireFrame(cv::Mat &image,
                               std::vector<cv::KeyPoint> uv_points,
                               std::vector<double> depths,
                               const std::vector<Eigen::Vector3i> &triangles,
                               float max_depth) {

  cv::Mat image_grey = image.clone();
  if (image_grey.channels() == 1) {
    cv::cvtColor(image, image_grey, cv::COLOR_GRAY2BGR);
  }

  float COLOUR_SCALE = 1.0; // 1.12

  int thickness = 1;
  int lineType = cv::LINE_8;

  cv::Mat depth_colour(1, depths.size(), CV_32FC1);
  cv::Mat depths_norm(1, depths.size(), CV_32FC1);

  std::transform(depths.begin(), depths.end(), depth_colour.begin<float>(),
                 [](double const &p) -> float { return p; });

  // cv::normalize(depths,depths_norm,0, 255, cv::NORM_MINMAX, -1);

  depth_colour = (depth_colour / (float)max_depth * COLOUR_SCALE) * 255.0f;
  std::transform(depth_colour.begin<float>(), depth_colour.end<float>(),
                 depths_norm.begin<float>(), [](float f) -> float {
                   return std::max(0.0f, std::min(f, 255.0f));
                 }); // clamp between 0 and 255

  for (auto t : triangles) {

    cv::Point2f vert1 = uv_points.at(t[0]).pt;
    cv::Point2f vert2 = uv_points.at(t[1]).pt;
    cv::Point2f vert3 = uv_points.at(t[2]).pt;

    int depth1_norm = static_cast<int>(depths_norm.at<float>(0, t[0]));
    int depth2_norm = static_cast<int>(depths_norm.at<float>(0, t[1]));
    int depth3_norm = static_cast<int>(depths_norm.at<float>(0, t[2]));

    cv::Vec3b c1, c2, c3;
    COLOUR_MAP.LookUp(depth1_norm, c1);
    COLOUR_MAP.LookUp(depth2_norm, c2);
    COLOUR_MAP.LookUp(depth3_norm, c3);

    COLOUR_MAP.DrawLineColourMap(image_grey, vert1, vert2, depth1_norm,
                                 depth2_norm);
    COLOUR_MAP.DrawLineColourMap(image_grey, vert2, vert3, depth2_norm,
                                 depth3_norm);
    COLOUR_MAP.DrawLineColourMap(image_grey, vert3, vert1, depth3_norm,
                                 depth1_norm);
  }
  cv::imshow("WireFrame", image_grey);
  cv::waitKey(5);
  //  cv::imwrite("/home/paulamayo/data/husky_data/training/mesh_" +
  //                  std::to_string(0) + ".png",
  //              image_grey);
}
//------------------------------------------------------------------------------
static cv::Mat LinSpace(float x0, float x1, int n) {
  cv::Mat pts(n, 1, CV_32FC1);
  float step = (x1 - x0) / (n - 1);
  for (int i = 0; i < n; i++)
    pts.at<float>(i, 0) = x0 + i * step;
  return pts;
}
//------------------------------------------------------------------------------
static cv::Mat ArgSort(cv::InputArray &_src, bool ascending) {
  cv::Mat src = _src.getMat();
  if (src.rows != 1 && src.cols != 1)
    CV_Error(cv::Error::StsBadArg, "cv::argsort only sorts 1D matrices.");
  int flags = cv::SORT_EVERY_ROW |
              (ascending ? cv::SORT_ASCENDING : cv::SORT_DESCENDING);
  cv::Mat sorted_indices;
  sortIdx(src.reshape(1, 1), sorted_indices, flags);
  return sorted_indices;
}
//------------------------------------------------------------------------------
static void SortMatrixRowsByIndices(cv::InputArray &_src,
                                    cv::InputArray &_indices,
                                    cv::OutputArray &_dst) {
  if (_indices.getMat().type() != CV_32SC1)
    CV_Error(cv::Error::StsUnsupportedFormat,
             "cv::sortRowsByIndices only works on integer indices!");
  cv::Mat src = _src.getMat();
  std::vector<int> indices = _indices.getMat();
  _dst.create(src.rows, src.cols, src.type());
  cv::Mat dst = _dst.getMat();
  for (size_t idx = 0; idx < indices.size(); idx++) {
    cv::Mat originalRow = src.row(indices[idx]);
    cv::Mat sortedRow = dst.row((int)idx);
    originalRow.copyTo(sortedRow);
  }
}
//------------------------------------------------------------------------------
static cv::Mat SortMatrixRowsByIndices(cv::InputArray &src,
                                       cv::InputArray &indices) {
  cv::Mat dst;
  SortMatrixRowsByIndices(src, indices, dst);
  return dst;
}
//------------------------------------------------------------------------------
template <typename _Tp>
static cv::Mat Interp1_(const cv::Mat &X_, const cv::Mat &Y_,
                        const cv::Mat &XI) {
  int n = XI.rows;
  // sort input table
  std::vector<int> sort_indices = ArgSort(X_);

  cv::Mat X = SortMatrixRowsByIndices(X_, sort_indices);
  cv::Mat Y = SortMatrixRowsByIndices(Y_, sort_indices);
  // interpolated values
  cv::Mat yi = cv::Mat::zeros(XI.size(), XI.type());
  for (int i = 0; i < n; i++) {
    int low = 0;
    int high = X.rows - 1;
    // set bounds
    if (XI.at<_Tp>(i, 0) < X.at<_Tp>(low, 0))
      high = 1;
    if (XI.at<_Tp>(i, 0) > X.at<_Tp>(high, 0))
      low = high - 1;
    // binary search
    while ((high - low) > 1) {
      const int c = low + ((high - low) >> 1);
      if (XI.at<_Tp>(i, 0) > X.at<_Tp>(c, 0)) {
        low = c;
      } else {
        high = c;
      }
    }
    // linear interpolation
    yi.at<_Tp>(i, 0) +=
        Y.at<_Tp>(low, 0) + (XI.at<_Tp>(i, 0) - X.at<_Tp>(low, 0)) *
                                (Y.at<_Tp>(high, 0) - Y.at<_Tp>(low, 0)) /
                                (X.at<_Tp>(high, 0) - X.at<_Tp>(low, 0));
  }
  return yi;
}
//------------------------------------------------------------------------------
static cv::Mat Interp1(cv::InputArray &_x, cv::InputArray &_Y,
                       cv::InputArray &_xi) {
  // get matrices
  cv::Mat x = _x.getMat();
  cv::Mat Y = _Y.getMat();
  cv::Mat xi = _xi.getMat();
  // check types & alignment
  CV_Assert((x.type() == Y.type()) && (Y.type() == xi.type()));
  CV_Assert((x.cols == 1) && (x.rows == Y.rows) && (x.cols == Y.cols));
  // call templated interp1
  switch (x.type()) {
  case CV_8SC1:
    return Interp1_<char>(x, Y, xi);
    break;
  case CV_8UC1:
    return Interp1_<unsigned char>(x, Y, xi);
    break;
  case CV_16SC1:
    return Interp1_<short>(x, Y, xi);
    break;
  case CV_16UC1:
    return Interp1_<unsigned short>(x, Y, xi);
    break;
  case CV_32SC1:
    return Interp1_<int>(x, Y, xi);
    break;
  case CV_32FC1:
    return Interp1_<float>(x, Y, xi);
    break;
  case CV_64FC1:
    return Interp1_<double>(x, Y, xi);
    break;
  }
  CV_Error(cv::Error::StsUnsupportedFormat, "");
}
//------------------------------------------------------------------------------
ColourMap::ColourMap() : ColourMap(256) {}
//------------------------------------------------------------------------------
ColourMap::ColourMap(const int n) {

  static const float r[] = {0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0.00588235294117645f,
                            0.02156862745098032f,
                            0.03725490196078418f,
                            0.05294117647058827f,
                            0.06862745098039214f,
                            0.084313725490196f,
                            0.1000000000000001f,
                            0.115686274509804f,
                            0.1313725490196078f,
                            0.1470588235294117f,
                            0.1627450980392156f,
                            0.1784313725490196f,
                            0.1941176470588235f,
                            0.2098039215686274f,
                            0.2254901960784315f,
                            0.2411764705882353f,
                            0.2568627450980392f,
                            0.2725490196078431f,
                            0.2882352941176469f,
                            0.303921568627451f,
                            0.3196078431372549f,
                            0.3352941176470587f,
                            0.3509803921568628f,
                            0.3666666666666667f,
                            0.3823529411764706f,
                            0.3980392156862744f,
                            0.4137254901960783f,
                            0.4294117647058824f,
                            0.4450980392156862f,
                            0.4607843137254901f,
                            0.4764705882352942f,
                            0.4921568627450981f,
                            0.5078431372549019f,
                            0.5235294117647058f,
                            0.5392156862745097f,
                            0.5549019607843135f,
                            0.5705882352941174f,
                            0.5862745098039217f,
                            0.6019607843137256f,
                            0.6176470588235294f,
                            0.6333333333333333f,
                            0.6490196078431372f,
                            0.664705882352941f,
                            0.6803921568627449f,
                            0.6960784313725492f,
                            0.7117647058823531f,
                            0.7274509803921569f,
                            0.7431372549019608f,
                            0.7588235294117647f,
                            0.7745098039215685f,
                            0.7901960784313724f,
                            0.8058823529411763f,
                            0.8215686274509801f,
                            0.8372549019607844f,
                            0.8529411764705883f,
                            0.8686274509803922f,
                            0.884313725490196f,
                            0.8999999999999999f,
                            0.9156862745098038f,
                            0.9313725490196076f,
                            0.947058823529412f,
                            0.9627450980392158f,
                            0.9784313725490197f,
                            0.9941176470588236f,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            0.9862745098039216f,
                            0.9705882352941178f,
                            0.9549019607843139f,
                            0.93921568627451f,
                            0.9235294117647062f,
                            0.9078431372549018f,
                            0.892156862745098f,
                            0.8764705882352941f,
                            0.8607843137254902f,
                            0.8450980392156864f,
                            0.8294117647058825f,
                            0.8137254901960786f,
                            0.7980392156862743f,
                            0.7823529411764705f,
                            0.7666666666666666f,
                            0.7509803921568627f,
                            0.7352941176470589f,
                            0.719607843137255f,
                            0.7039215686274511f,
                            0.6882352941176473f,
                            0.6725490196078434f,
                            0.6568627450980391f,
                            0.6411764705882352f,
                            0.6254901960784314f,
                            0.6098039215686275f,
                            0.5941176470588236f,
                            0.5784313725490198f,
                            0.5627450980392159f,
                            0.5470588235294116f,
                            0.5313725490196077f,
                            0.5156862745098039f,
                            0.5f};
  static const float g[] = {0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0.001960784313725483f,
                            0.01764705882352935f,
                            0.03333333333333333f,
                            0.0490196078431373f,
                            0.06470588235294117f,
                            0.08039215686274503f,
                            0.09607843137254901f,
                            0.111764705882353f,
                            0.1274509803921569f,
                            0.1431372549019607f,
                            0.1588235294117647f,
                            0.1745098039215687f,
                            0.1901960784313725f,
                            0.2058823529411764f,
                            0.2215686274509804f,
                            0.2372549019607844f,
                            0.2529411764705882f,
                            0.2686274509803921f,
                            0.2843137254901961f,
                            0.3f,
                            0.3156862745098039f,
                            0.3313725490196078f,
                            0.3470588235294118f,
                            0.3627450980392157f,
                            0.3784313725490196f,
                            0.3941176470588235f,
                            0.4098039215686274f,
                            0.4254901960784314f,
                            0.4411764705882353f,
                            0.4568627450980391f,
                            0.4725490196078431f,
                            0.4882352941176471f,
                            0.503921568627451f,
                            0.5196078431372548f,
                            0.5352941176470587f,
                            0.5509803921568628f,
                            0.5666666666666667f,
                            0.5823529411764705f,
                            0.5980392156862746f,
                            0.6137254901960785f,
                            0.6294117647058823f,
                            0.6450980392156862f,
                            0.6607843137254901f,
                            0.6764705882352942f,
                            0.692156862745098f,
                            0.7078431372549019f,
                            0.723529411764706f,
                            0.7392156862745098f,
                            0.7549019607843137f,
                            0.7705882352941176f,
                            0.7862745098039214f,
                            0.8019607843137255f,
                            0.8176470588235294f,
                            0.8333333333333333f,
                            0.8490196078431373f,
                            0.8647058823529412f,
                            0.8803921568627451f,
                            0.8960784313725489f,
                            0.9117647058823528f,
                            0.9274509803921569f,
                            0.9431372549019608f,
                            0.9588235294117646f,
                            0.9745098039215687f,
                            0.9901960784313726f,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            0.9901960784313726f,
                            0.9745098039215687f,
                            0.9588235294117649f,
                            0.943137254901961f,
                            0.9274509803921571f,
                            0.9117647058823528f,
                            0.8960784313725489f,
                            0.8803921568627451f,
                            0.8647058823529412f,
                            0.8490196078431373f,
                            0.8333333333333335f,
                            0.8176470588235296f,
                            0.8019607843137253f,
                            0.7862745098039214f,
                            0.7705882352941176f,
                            0.7549019607843137f,
                            0.7392156862745098f,
                            0.723529411764706f,
                            0.7078431372549021f,
                            0.6921568627450982f,
                            0.6764705882352944f,
                            0.6607843137254901f,
                            0.6450980392156862f,
                            0.6294117647058823f,
                            0.6137254901960785f,
                            0.5980392156862746f,
                            0.5823529411764707f,
                            0.5666666666666669f,
                            0.5509803921568626f,
                            0.5352941176470587f,
                            0.5196078431372548f,
                            0.503921568627451f,
                            0.4882352941176471f,
                            0.4725490196078432f,
                            0.4568627450980394f,
                            0.4411764705882355f,
                            0.4254901960784316f,
                            0.4098039215686273f,
                            0.3941176470588235f,
                            0.3784313725490196f,
                            0.3627450980392157f,
                            0.3470588235294119f,
                            0.331372549019608f,
                            0.3156862745098041f,
                            0.2999999999999998f,
                            0.284313725490196f,
                            0.2686274509803921f,
                            0.2529411764705882f,
                            0.2372549019607844f,
                            0.2215686274509805f,
                            0.2058823529411766f,
                            0.1901960784313728f,
                            0.1745098039215689f,
                            0.1588235294117646f,
                            0.1431372549019607f,
                            0.1274509803921569f,
                            0.111764705882353f,
                            0.09607843137254912f,
                            0.08039215686274526f,
                            0.06470588235294139f,
                            0.04901960784313708f,
                            0.03333333333333321f,
                            0.01764705882352935f,
                            0.001960784313725483f,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0};
  static const float b[] = {0.5f,
                            0.5156862745098039f,
                            0.5313725490196078f,
                            0.5470588235294118f,
                            0.5627450980392157f,
                            0.5784313725490196f,
                            0.5941176470588235f,
                            0.6098039215686275f,
                            0.6254901960784314f,
                            0.6411764705882352f,
                            0.6568627450980392f,
                            0.6725490196078432f,
                            0.6882352941176471f,
                            0.7039215686274509f,
                            0.7196078431372549f,
                            0.7352941176470589f,
                            0.7509803921568627f,
                            0.7666666666666666f,
                            0.7823529411764706f,
                            0.7980392156862746f,
                            0.8137254901960784f,
                            0.8294117647058823f,
                            0.8450980392156863f,
                            0.8607843137254902f,
                            0.8764705882352941f,
                            0.892156862745098f,
                            0.907843137254902f,
                            0.9235294117647059f,
                            0.9392156862745098f,
                            0.9549019607843137f,
                            0.9705882352941176f,
                            0.9862745098039216f,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            0.9941176470588236f,
                            0.9784313725490197f,
                            0.9627450980392158f,
                            0.9470588235294117f,
                            0.9313725490196079f,
                            0.915686274509804f,
                            0.8999999999999999f,
                            0.884313725490196f,
                            0.8686274509803922f,
                            0.8529411764705883f,
                            0.8372549019607844f,
                            0.8215686274509804f,
                            0.8058823529411765f,
                            0.7901960784313726f,
                            0.7745098039215685f,
                            0.7588235294117647f,
                            0.7431372549019608f,
                            0.7274509803921569f,
                            0.7117647058823531f,
                            0.696078431372549f,
                            0.6803921568627451f,
                            0.6647058823529413f,
                            0.6490196078431372f,
                            0.6333333333333333f,
                            0.6176470588235294f,
                            0.6019607843137256f,
                            0.5862745098039217f,
                            0.5705882352941176f,
                            0.5549019607843138f,
                            0.5392156862745099f,
                            0.5235294117647058f,
                            0.5078431372549019f,
                            0.4921568627450981f,
                            0.4764705882352942f,
                            0.4607843137254903f,
                            0.4450980392156865f,
                            0.4294117647058826f,
                            0.4137254901960783f,
                            0.3980392156862744f,
                            0.3823529411764706f,
                            0.3666666666666667f,
                            0.3509803921568628f,
                            0.335294117647059f,
                            0.3196078431372551f,
                            0.3039215686274508f,
                            0.2882352941176469f,
                            0.2725490196078431f,
                            0.2568627450980392f,
                            0.2411764705882353f,
                            0.2254901960784315f,
                            0.2098039215686276f,
                            0.1941176470588237f,
                            0.1784313725490199f,
                            0.1627450980392156f,
                            0.1470588235294117f,
                            0.1313725490196078f,
                            0.115686274509804f,
                            0.1000000000000001f,
                            0.08431372549019622f,
                            0.06862745098039236f,
                            0.05294117647058805f,
                            0.03725490196078418f,
                            0.02156862745098032f,
                            0.00588235294117645f,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0};

  cv::Mat X = LinSpace(0, 1, 256);
  this->_lut = ColourMap::LinearColormap(
      X, cv::Mat(256, 1, CV_32FC1, (void *)r).clone(), // red
      cv::Mat(256, 1, CV_32FC1, (void *)g).clone(),    // green
      cv::Mat(256, 1, CV_32FC1, (void *)b).clone(),    // blue
      n);
}
//------------------------------------------------------------------------------
cv::Mat ColourMap::LinearColormap(cv::InputArray &X, cv::InputArray &r,
                                  cv::InputArray &g, cv::InputArray &b,
                                  cv::InputArray &xi) {

  cv::Mat lut, lut8;
  cv::Mat planes[] = {Interp1(X, b, xi), Interp1(X, g, xi), Interp1(X, r, xi)};
  merge(planes, 3, lut);
  lut.convertTo(lut8, CV_8U, 255.);
  return lut8;
}

//------------------------------------------------------------------------------
// Interpolates from a base colormap.
cv::Mat ColourMap::LinearColormap(cv::InputArray &X, cv::InputArray &r,
                                  cv::InputArray &g, cv::InputArray &b,
                                  const int n) {
  return LinearColormap(X, r, g, b, LinSpace(0, 1, n));
}
//------------------------------------------------------------------------------
// Interpolates from a base colormap.
cv::Mat ColourMap::LinearColormap(cv::InputArray &X, cv::InputArray &r,
                                  cv::InputArray &g, cv::InputArray &b,
                                  const float begin, const float end,
                                  const float n) {
  return LinearColormap(X, r, g, b, LinSpace(begin, end, cvRound(n)));
}
//------------------------------------------------------------------------------
// use default cv colour map for dense images
void ColourMap::ApplyColourMap(cv::InputArray &src,
                               cv::OutputArray &dst) const {
  cv::applyColorMap(src, dst, cv::COLORMAP_JET);
}
//------------------------------------------------------------------------------
void ColourMap::DrawLineColourMap(cv::Mat &img, const cv::Point &start,
                                  const cv::Point &end, const int depth_norm1,
                                  const int depth_norm2) {
  cv::Vec3b c1;
  cv::LineIterator iter(img, start, end, cv::LINE_8);

  for (int i = 0; i < iter.count; i++, iter++) {
    double alpha = double(i) / iter.count;

    COLOUR_MAP.LookUp(depth_norm1 * (1.0 - alpha) + depth_norm2 * alpha, c1);

    (*iter)[0] = (uint8_t)(c1[0]);
    (*iter)[1] = (uint8_t)(c1[1]);
    (*iter)[2] = (uint8_t)(c1[2]);
  }
}
//------------------------------------------------------------------------------
// lookup_value must be normalised between 0 and 255
void ColourMap::LookUp(const int lookup_value, cv::Vec3b &colour_out) const {
  colour_out = _lut.at<cv::Vec3b>(lookup_value);
}
//------------------------------------------------------------------------------
void ColourMap::LookUp2(const float lookup_value, cv::Vec3b &c) const {
  c = cv::Vec3b(255, 255, 255); // white
  float dv;
  float vmin = 0.0f;
  float vmax = 255.0f;

  float v = lookup_value;

  if (v < vmin)
    v = vmin;
  if (v > vmax)
    v = vmax;
  dv = vmax - vmin;

  if (v < (vmin + 0.25 * dv)) {
    c[2] = 0;
    c[1] = static_cast<uint8_t>(255 * (4 * (v - vmin) / dv));
  } else if (v < (vmin + 0.5 * dv)) {
    c[2] = 0;
    c[0] = static_cast<uint8_t>(255 * (1 + 4 * (vmin + 0.25 * dv - v) / dv));
    ;
  } else if (v < (vmin + 0.75 * dv)) {
    c[2] = static_cast<uint8_t>(255 * (4 * (v - vmin - 0.5 * dv) / dv));
    c[0] = 0;
  } else {
    c[1] = static_cast<uint8_t>(255 * (1 + 4 * (vmin + 0.75 * dv - v) / dv));
    c[0] = 0;
  }
}
//------------------------------------------------------------------------------
void ColourMap::LookUpAlt(int lookup_value, cv::Vec3b &colour_out) const {

  int c1R, c1G, c1B, c2R, c2G, c2B;
  float R, G, B;

  float thresh = 0.1;
  float fraction = lookup_value / 255.0;

  if (fraction < thresh) {
    c1R = 255;
    c1G = 0;
    c1B = 0;
    c2R = 180;
    c2G = 255;
    c2B = 0;

    R = (c2R - c1R) * fraction * (1 / thresh) + c1R;
    G = (c2G - c1G) * fraction * (1 / thresh) + c1G;
    B = (c2B - c1B) * fraction * (1 / thresh) + c1B;

    std::cout << fraction << std::endl;
  } else {
    c1R = 180;
    c1G = 255;
    c1B = 0; // 29, 221, 26
    c2R = 81;
    c2G = 103;
    c2B = 206; // 37; 65; 206;

    R = (c2R - c1R) * fraction * thresh + c1R;
    G = (c2G - c1G) * fraction * thresh + c1G;
    B = (c2B - c1B) * fraction * thresh + c1B;
  }

  colour_out[0] = static_cast<char>(B);
  colour_out[1] = static_cast<char>(G);
  colour_out[2] = static_cast<char>(R);
}

} // namespace viewer
} // namespace utilities
} // namespace core
} // namespace aru
