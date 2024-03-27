#ifndef ARU_UTILITIES_VIEWER_H_
#define ARU_UTILITIES_VIEWER_H_

#include <Eigen/Dense>
#include <boost/shared_ptr.hpp>
#include <glog/logging.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <utility>

#include <aru/core/utilities/image/point_feature.h>
#include <aru/core/utilities/transforms/transforms.h>
#include <pangolin/display/display.h>
#include <pangolin/display/view.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/handler/handler.h>
#include <pangolin/scene/axis.h>
#include <pangolin/scene/scenehandler.h>

namespace aru {
namespace core {
namespace utilities {
namespace viewer {

class PoseViewer {
public:
  PoseViewer(utilities::transform::TransformSPtrVectorSptr pose_chain);

  ~PoseViewer() = default;

  PoseViewer() = default;

  void DrawPoses();

  void DrawPoses(Eigen::Vector3f color);

  void DrawCurrentPose(pangolin::OpenGlMatrix matrix);

  void DrawCurrentPose(pangolin::OpenGlMatrix matrix, Eigen::Vector3f color);

  pangolin::OpenGlMatrix GetCurrentPose();

private:
  static pangolin::OpenGlMatrix GetOpenGLPoseMatrix(Eigen::MatrixXf pose);

  utilities::transform::TransformSPtrVectorSptr pose_chain_;
};

class Viewer {
public:
  Viewer(int height, int width, Eigen::Matrix3d camera_intrinsic);

  Viewer() = default;

  ~Viewer() = default;

  void Run();

  static void ViewPointCloud(const std::vector<Eigen::Vector3d> &points);

  static void ViewVoxelPointCloud(const std::vector<Eigen::Vector3d> &points,
                                  const std::vector<Eigen::Vector3d> &colors);

  static void ViewVoxelPointCloud(const std::vector<Eigen::Vector3d> &points,
                                  const std::vector<Eigen::Vector3d> &colors,
                                  const std::vector<Eigen::Affine3f> &poses);

  static void ViewMeshWireFrame(cv::Mat &image,
                                std::vector<cv::KeyPoint> uv_points,
                                std::vector<double> depths,
                                const std::vector<Eigen::Vector3i> &triangles,
                                float max_depth);

  static void ViewMeshWireFrame(cv::Mat &image,
                                std::vector<cv::KeyPoint> uv_points,
                                std::vector<double> depths,
                                const std::vector<Eigen::Vector3i> &triangles,
                                float max_depth, cv::Scalar colour);

  static void ViewInterpolatedMesh(const cv::Mat &image, const cv::Mat &depth,
                                   float max_depth);

  static void ViewDisparity(const cv::Mat &disparity, float max_disp);

  void ViewDepthPointCloud(cv::Mat depth, float max_depth);

  void ViewPoseChain(const std::vector<Eigen::MatrixXf> &poses);

  std::vector<cv::Scalar> GetColourMap() { return colour_map_; }

  void VisualiseLabels(cv::Mat image,
                       utilities::image::FeatureSPtrVectorSptr &features,
                       Eigen::MatrixXd labels);

  static void
  ViewImageFeatures(cv::Mat image_1,
                    utilities::image::FeatureSPtrVectorSptr &features);

  void ViewImageFeaturesDepth(cv::Mat image_1,
                              utilities::image::FeatureSPtrVectorSptr &features,
                              float max_depth);

  static void ViewSequentialImageFeatures(
      const cv::Mat &image_1, cv::Mat image_2, float resize,
      utilities::image::FeatureSPtrVectorSptr &features);

  static void
  ViewStereoImageFeatures(const cv::Mat &image_1, cv::Mat image_2, float resize,
                          utilities::image::FeatureSPtrVectorSptr &features);

private:
  static pangolin::OpenGlMatrix GetOpenGLPoseMatrix(Eigen::MatrixXf pose);
  int height_;
  int width_;

  Eigen::Matrix3d camera_intrinsic_;

  std::vector<cv::Scalar> colour_map_;
};

static cv::Mat LinSpace(float x0, float x1, int n);
static cv::Mat ArgSort(cv::InputArray &_src, bool ascending = true);
static void SortMatrixRowsByIndices(cv::InputArray &_src,
                                    cv::InputArray &_indices,
                                    cv::OutputArray &_dst);
static cv::Mat SortMatrixRowsByIndices(cv::InputArray &src,
                                       cv::InputArray &indices);

template <typename _Tp>
static cv::Mat Interp1_(const cv::Mat &X_, const cv::Mat &Y_,
                        const cv::Mat &XI);
static cv::Mat Interp1(cv::InputArray &_x, cv::InputArray &_Y,
                       cv::InputArray &_xi);

class ColourMap {

private:
  cv::Mat _lut;

public:
  ColourMap();
  ColourMap(const int n);
  static cv::Mat LinearColormap(cv::InputArray &X, cv::InputArray &r,
                                cv::InputArray &g, cv::InputArray &b,
                                cv::InputArray &xi);
  cv::Mat LinearColormap(cv::InputArray &X, cv::InputArray &r,
                         cv::InputArray &g, cv::InputArray &b, const int n);
  cv::Mat LinearColormap(cv::InputArray &X, cv::InputArray &r,
                         cv::InputArray &g, cv::InputArray &b,
                         const float begin, const float end, const float n);

  void DrawLineColourMap(cv::Mat &img, const cv::Point &start,
                         const cv::Point &end, const int depth_norm1,
                         const int depth_norm2);

  void ApplyColourMap(cv::InputArray &src, cv::OutputArray &dst) const;
  void LookUp(int lookup_value, cv::Vec3b &colour_out) const;
  void LookUp2(const float v, cv::Vec3b &c) const;
  void LookUpAlt(int lookup_value, cv::Vec3b &colour_out) const;
};

static ColourMap COLOUR_MAP;

} // namespace viewer
} // namespace utilities
} // namespace core
} // namespace aru

#endif // ARU_UTILITIES_VIEWER_H_
